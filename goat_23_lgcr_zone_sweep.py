"""
LGCR Zone Pivot Sweep — Standalone Backtest
============================================
Model overview:
  1. Resample M5 OHLCV → M30 candles internally (no extra download).
  2. Compute Heikin-Ashi on M30, detect LGCR patterns.
     • BEAR LGCR zone: [HA_Close, HA_High] of the bearish LGCR candle.
     • BULL LGCR zone: [HA_Low,   HA_Close] of the bullish LGCR candle.
  3. Zone stays active as long as at LEAST ONE of its two lines has not been
     closed through by a subsequent M5 HA close.
  4. On M5 HA, look for 1+1+1 pivot highs (BEAR) / pivot lows (BULL) whose
     wick touches inside an active zone that lies above (BEAR) / below (BULL)
     the current price.
  5. Entry trigger: first M5 bar AFTER the pivot that closes below the pivot's
     HA_Low (BEAR) / above the pivot's HA_High (BULL).
  6. SL = pivot HA_High (BEAR) / pivot HA_Low (BULL).
  7. TP = entry ± RR × |entry − SL|.
  8. Optional break-even: when price moves be_trigger_r × risk in our favour,
     move SL to entry.

Usage:
  python3 goat_23_lgcr_zone_sweep.py --symbol COMP/USDT:USDT --tf 5m \\
      --days 180 --rr 3.0 --be 1.5 --start 2025-11-01 --end 2026-03-25 --plot
"""

import argparse
import math
import os
import warnings
import logging

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time as time_module

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from goat_15_data_manager import get_ohlcv


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: HEIKIN-ASHI
# ═══════════════════════════════════════════════════════════════════

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi columns on top of a standard OHLCV dataframe."""
    ha = df.copy()
    o  = ha["open"].values.astype(np.float64)
    h  = ha["high"].values.astype(np.float64)
    lo = ha["low"].values.astype(np.float64)
    c  = ha["close"].values.astype(np.float64)

    ha_close = (o + h + lo + c) / 4.0
    ha_open  = np.empty(len(ha), dtype=np.float64)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_body_max = np.maximum(ha_open, ha_close)
    ha_body_min = np.minimum(ha_open, ha_close)
    ha_high = np.maximum(h, ha_body_max)
    ha_low  = np.minimum(lo, ha_body_min)

    ha["HA_Close"] = ha_close
    ha["HA_Open"]  = ha_open
    ha["HA_High"]  = ha_high
    ha["HA_Low"]   = ha_low
    return ha


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: LGCR PATTERN DETECTION (pure-Python, no numba dep)
# ═══════════════════════════════════════════════════════════════════

def detect_lgcr(ha_open: np.ndarray, ha_close: np.ndarray,
                ha_high: np.ndarray, ha_low: np.ndarray):
    """Detect LGCR (LG + LGC + LGCR) on Heikin-Ashi arrays.

    Returns
    -------
    bull_lgcr, bear_lgcr : bool arrays of length n
    """
    n = len(ha_open)
    bull_lg   = np.zeros(n, dtype=bool)
    bear_lg   = np.zeros(n, dtype=bool)
    bull_lgcr = np.zeros(n, dtype=bool)
    bear_lgcr = np.zeros(n, dtype=bool)

    # ── LG ──
    for i in range(1, n):
        if ha_high[i - 1] < ha_close[i]:
            bull_lg[i] = True
        if ha_low[i - 1] > ha_close[i]:
            bear_lg[i] = True

    # ── LGC is implicit; LGCR uses LG reference levels ──
    ref_bull = -1.0   # last bear_lg bar's HA_High (LGCR bull target)
    ref_bear = -1.0   # last bull_lg bar's HA_Low  (LGCR bear target)
    bull_flag = False
    bear_flag = False

    for i in range(n):
        if bull_lg[i]:
            ref_bear = ha_low[i]
            bear_flag = False
        if bear_lg[i]:
            ref_bull = ha_high[i]
            bull_flag = False
        if bull_lg[i] and ref_bull > 0 and ha_close[i] > ref_bull and not bull_flag:
            bull_lgcr[i] = True
            bull_flag = True
        if bear_lg[i] and ref_bear > 0 and ha_close[i] < ref_bear and not bear_flag:
            bear_lgcr[i] = True
            bear_flag = True

    return bull_lgcr, bear_lgcr


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: M30 ZONE BUILDER
# ═══════════════════════════════════════════════════════════════════

_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080,
}

HTF_TF = "30m"   # fixed context timeframe


def _resample_to_htf(df_m5: pd.DataFrame, htf_str: str = HTF_TF) -> pd.DataFrame:
    """Resample M5 OHLCV to a higher timeframe."""
    minutes = _TF_MINUTES[htf_str]
    offset  = f"{minutes}min"

    df = df_m5.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    htf = (df.resample(offset, closed="left", label="left")
             .agg(agg)
             .dropna(subset=["open", "close"]))
    return htf


def build_m30_zones(df_m5: pd.DataFrame):
    """Compute M30 HA LGCR zones and map them to the M5 bar index.

    Returns
    -------
    zones : list of dict, one entry per LGCR candle, with keys:
        side        'BEAR' | 'BULL'
        line_lo     lower level of the zone
        line_hi     upper level of the zone
        formed_ts   pandas Timestamp when the M30 candle closed (= start of NEXT M30 bar)
    """
    htf = _resample_to_htf(df_m5)
    if len(htf) < 3:
        return []

    ha_htf = calculate_heikin_ashi(htf)
    ha_open_h  = ha_htf["HA_Open"].values.astype(np.float64)
    ha_close_h = ha_htf["HA_Close"].values.astype(np.float64)
    ha_high_h  = ha_htf["HA_High"].values.astype(np.float64)
    ha_low_h   = ha_htf["HA_Low"].values.astype(np.float64)

    bull_lgcr, bear_lgcr = detect_lgcr(ha_open_h, ha_close_h, ha_high_h, ha_low_h)

    htf_ts = htf.index  # DatetimeIndex — start of each M30 bar
    minutes = _TF_MINUTES[HTF_TF]

    zones = []
    for i in range(len(htf)):
        if bear_lgcr[i]:
            # Zone between HA_Close (bottom) and HA_High (top)
            zones.append({
                "side":      "BEAR",
                "line_lo":   ha_close_h[i],
                "line_hi":   ha_high_h[i],
                "formed_ts": htf_ts[i] + pd.Timedelta(minutes=minutes),
            })
        if bull_lgcr[i]:
            # Zone between HA_Low (bottom) and HA_Close (top)
            zones.append({
                "side":      "BULL",
                "line_lo":   ha_low_h[i],
                "line_hi":   ha_close_h[i],
                "formed_ts": htf_ts[i] + pd.Timedelta(minutes=minutes),
            })

    return zones


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ZONE VALIDITY ON M5
# ═══════════════════════════════════════════════════════════════════

def zone_is_active(zone: dict, ha_close_m5: np.ndarray,
                   from_bar: int, to_bar: int) -> tuple:
    """Check whether a zone is still active after processing M5 bars [from_bar, to_bar).

    A zone stays active as long as at least one of its two lines has NOT been
    closed through by any M5 HA close.

    Returns
    -------
    active : bool
    lo_valid : bool   (line_lo not yet closed through)
    hi_valid : bool   (line_hi not yet closed through)
    """
    lo = zone["line_lo"]
    hi = zone["line_hi"]
    lo_valid = True
    hi_valid = True

    for i in range(from_bar, to_bar):
        c = ha_close_m5[i]
        if zone["side"] == "BEAR":
            # Bear zone: a line is invalidated if M5 HA_Close goes ABOVE it
            if lo_valid and c > lo:
                lo_valid = False
            if hi_valid and c > hi:
                hi_valid = False
        else:
            # Bull zone: a line is invalidated if M5 HA_Close goes BELOW it
            if lo_valid and c < lo:
                lo_valid = False
            if hi_valid and c < hi:
                hi_valid = False

        if not lo_valid and not hi_valid:
            break

    active = lo_valid or hi_valid
    return active, lo_valid, hi_valid


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df_m5: pd.DataFrame, rr_ratio: float = 3.0,
                 be_trigger_r: float = 0.0, warmup: int = 300,
                 quiet: bool = False):
    """Run the LGCR Zone Pivot Sweep backtest.

    Parameters
    ----------
    df_m5        : M5 OHLCV dataframe (must have 'timestamp' column)
    rr_ratio     : risk-reward ratio for TP
    be_trigger_r : break-even trigger in R (0 = disabled)
    warmup       : number of leading bars to skip for signal detection
    quiet        : suppress per-bar output

    Returns
    -------
    trades : list of dicts
    """
    t0 = time_module.perf_counter()

    if not quiet:
        print(f"\n{'='*60}")
        print(f"LGCR Zone Pivot Sweep | RR={rr_ratio} | "
              f"BE={'disabled' if not be_trigger_r else f'{be_trigger_r}R'}")
        print(f"Warmup: {warmup} bars")
        print(f"{'='*60}")
        print("  Building M30 HA LGCR zones...")

    # ── Step 1: build zones ──
    all_zones = build_m30_zones(df_m5)
    if not quiet:
        n_bear_z = sum(1 for z in all_zones if z["side"] == "BEAR")
        n_bull_z = sum(1 for z in all_zones if z["side"] == "BULL")
        print(f"    {len(all_zones)} zones ({n_bear_z} BEAR, {n_bull_z} BULL)")

    # ── Step 2: compute M5 HA ──
    if not quiet:
        print("  Computing M5 Heikin-Ashi...")
    ha_m5 = calculate_heikin_ashi(df_m5).reset_index(drop=True)

    n = len(ha_m5)
    ha_close = ha_m5["HA_Close"].values.astype(np.float64)
    ha_high  = ha_m5["HA_High"].values.astype(np.float64)
    ha_low   = ha_m5["HA_Low"].values.astype(np.float64)
    raw_high = df_m5["high"].values.astype(np.float64)
    raw_low  = df_m5["low"].values.astype(np.float64)
    timestamps = ha_m5["timestamp"].values

    # ── Step 3: map each zone's formed_ts to its first visible M5 bar ──
    # A zone is visible on bar i only if zone.formed_ts <= timestamps[i]
    # (the M30 bar has fully closed before bar i starts).
    ts_arr = pd.DatetimeIndex(timestamps)  # for searchsorted

    zone_first_bar = []
    for z in all_zones:
        ft = pd.Timestamp(z["formed_ts"])
        # first M5 bar whose timestamp >= formed_ts
        idx = np.searchsorted(ts_arr.asi8, ft.value, side="left")
        zone_first_bar.append(int(idx))

    # ── Step 4: iterate M5 bars ──
    if not quiet:
        print("  Scanning pivot sweeps...")

    trades = []
    active_trades = []

    # We track, per zone, up to which bar we have confirmed validity (optimisation):
    # zone_lo_valid[z_idx], zone_hi_valid[z_idx] track invalidation incrementally.
    zone_lo_valid = [True] * len(all_zones)
    zone_hi_valid = [True] * len(all_zones)

    # open_pivot stores a pending pivot sweep waiting for a confirming close.
    # Format: {"side": ..., "pivot_bar": i, "pivot_high": ..., "pivot_low": ...,
    #          "zone_lo": ..., "zone_hi": ..., "confirmed": False}
    open_pivots = []

    # Debug counters
    n_pivot_high = 0
    n_pivot_low = 0
    n_pivot_high_in_zone = 0
    n_pivot_low_in_zone = 0
    n_open_pivots_created = 0
    n_confirming_closes = 0
    n_trades_opened = 0

    for bar in range(1, n):
        bar_ts = pd.Timestamp(timestamps[bar])

        # ── A) Update zone validity up to (not including) this bar ──
        for zi, z in enumerate(all_zones):
            if zone_first_bar[zi] > bar:
                continue  # zone not yet visible
            if not zone_lo_valid[zi] and not zone_hi_valid[zi]:
                continue  # already dead

            # Invalidate lines based on previous bar's close (bar-1)
            prev = bar - 1
            c = ha_close[prev]
            if z["side"] == "BEAR":
                if zone_lo_valid[zi] and c > z["line_lo"]:
                    zone_lo_valid[zi] = False
                if zone_hi_valid[zi] and c > z["line_hi"]:
                    zone_hi_valid[zi] = False
            else:
                if zone_lo_valid[zi] and c < z["line_lo"]:
                    zone_lo_valid[zi] = False
                if zone_hi_valid[zi] and c < z["line_hi"]:
                    zone_hi_valid[zi] = False

        # ── B) Check active trades (SL / TP / BE) ──
        to_close = []
        for t in active_trades:
            c_hi = raw_high[bar]
            c_lo = raw_low[bar]

            if t["side"] == "BULL":
                if t["risk"] > 0:
                    t["max_r"] = max(t["max_r"],
                                     (c_hi - t["entry"]) / t["risk"])
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = t["entry"]

                # SL hit check (conservative: SL before TP on same bar)
                if c_lo <= t["sl"]:
                    result = "BE" if (t["be_active"] and t["sl"] == t["entry"]) else "SL"
                    pnl = 0.0 if result == "BE" else -1.0
                    to_close.append((t, result, pnl))
                elif c_hi >= t["tp"]:
                    to_close.append((t, "TP", float(rr_ratio)))

            else:  # BEAR
                if t["risk"] > 0:
                    t["max_r"] = max(t["max_r"],
                                     (t["entry"] - c_lo) / t["risk"])
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = t["entry"]

                if c_hi >= t["sl"]:
                    result = "BE" if (t["be_active"] and t["sl"] == t["entry"]) else "SL"
                    pnl = 0.0 if result == "BE" else -1.0
                    to_close.append((t, result, pnl))
                elif c_lo <= t["tp"]:
                    to_close.append((t, "TP", float(rr_ratio)))

        for t, result, pnl_r in to_close:
            t["result"]   = result
            t["pnl_r"]    = pnl_r
            t["exit_bar"] = bar
            t["exit_ts"]  = timestamps[bar]
            t["duration_bars"] = bar - t["entry_bar"]
            trades.append(t)
            active_trades.remove(t)

        # ── C) Check pending pivot confirmations (bar i+2 onwards) ──
        # A pivot at bar p is confirmed on bar p+1.
        # A confirming close can happen on bar p+2 at the earliest.
        still_pending = []
        for piv in open_pivots:
            pb = piv["pivot_bar"]
            if bar < pb + 2:
                still_pending.append(piv)
                continue

            if piv["side"] == "BEAR":
                # Confirming close: HA_Close < pivot's HA_Low
                if ha_close[bar] < piv["pivot_low"]:
                    n_confirming_closes += 1
                    # Entry on this bar's HA_Close (realistic: same bar confirmed)
                    entry = ha_close[bar]
                    sl    = piv["pivot_high"]  # SL at pivot high
                    risk  = abs(entry - sl)
                    if risk == 0 or sl <= entry:
                        # Invalid geometry
                        continue
                    tp = entry - rr_ratio * risk

                    if not quiet:
                        ts_str = str(timestamps[bar])[:19]
                        print(f"  [BEAR] {ts_str} | entry={entry:.6f} "
                              f"SL={sl:.6f} TP={tp:.6f} risk={risk:.6f}")

                    n_trades_opened += 1
                    active_trades.append({
                        "side": "BEAR", "entry": entry, "sl": sl, "tp": tp,
                        "risk": risk, "original_sl": sl,
                        "entry_bar": bar, "entry_ts": timestamps[bar],
                        "pivot_bar": pb, "pivot_ts": timestamps[pb],
                        "zone_lo": piv["zone_lo"], "zone_hi": piv["zone_hi"],
                        "max_r": 0.0, "be_active": False,
                        "result": None, "pnl_r": None,
                        "exit_bar": None, "exit_ts": None, "duration_bars": None,
                    })
                    # Once confirmed, remove this pivot candidate
                    # (don't push to still_pending)
                else:
                    still_pending.append(piv)

            else:  # BULL
                if ha_close[bar] > piv["pivot_high"]:
                    n_confirming_closes += 1
                    entry = ha_close[bar]
                    sl    = piv["pivot_low"]
                    risk  = abs(entry - sl)
                    if risk == 0 or sl >= entry:
                        continue
                    tp = entry + rr_ratio * risk

                    if not quiet:
                        ts_str = str(timestamps[bar])[:19]
                        print(f"  [BULL] {ts_str} | entry={entry:.6f} "
                              f"SL={sl:.6f} TP={tp:.6f} risk={risk:.6f}")

                    n_trades_opened += 1
                    active_trades.append({
                        "side": "BULL", "entry": entry, "sl": sl, "tp": tp,
                        "risk": risk, "original_sl": sl,
                        "entry_bar": bar, "entry_ts": timestamps[bar],
                        "pivot_bar": pb, "pivot_ts": timestamps[pb],
                        "zone_lo": piv["zone_lo"], "zone_hi": piv["zone_hi"],
                        "max_r": 0.0, "be_active": False,
                        "result": None, "pnl_r": None,
                        "exit_bar": None, "exit_ts": None, "duration_bars": None,
                    })
                else:
                    still_pending.append(piv)

        open_pivots = still_pending

        # ── D) Detect new 1+1+1 pivots (confirmed on THIS bar = bar i+1 relative to pivot bar i) ──
        # Pivot bar is bar-1; we need bar-2 and bar (=i+1) to confirm.
        if bar < warmup or bar < 2:
            continue

        pivot_bar = bar - 1  # the candidate pivot bar

        # 1+1+1 pivot HIGH (BEAR setup)
        if (ha_high[pivot_bar] > ha_high[pivot_bar - 1] and
                ha_high[pivot_bar] > ha_high[bar]):
            pivot_high = ha_high[pivot_bar]
            n_pivot_high += 1

            # Find an active BEAR zone that either contains current price or is
            # above current price, and that contains pivot_high inside [line_lo, line_hi].
            # Priority: zones containing current price first; otherwise closest above.
            cur_price = ha_close[bar]
            best_zone = None
            best_dist = math.inf
            best_contains = False  # does the best candidate contain cur_price?

            for zi, z in enumerate(all_zones):
                if z["side"] != "BEAR":
                    continue
                if zone_first_bar[zi] > pivot_bar:
                    continue  # zone not yet visible at pivot time
                if not (zone_lo_valid[zi] or zone_hi_valid[zi]):
                    continue  # dead zone

                # Allow zones that contain current price OR are above current price.
                # Reject zones that are entirely below current price.
                if z["line_hi"] < cur_price:
                    continue

                # pivot high must be inside the zone (Mode A)
                if not (z["line_lo"] <= pivot_high <= z["line_hi"]):
                    continue

                contains = z["line_lo"] <= cur_price <= z["line_hi"]
                if contains:
                    dist = 0.0
                else:
                    # Zone is above price (line_lo > cur_price guaranteed by filter above)
                    dist = z["line_lo"] - cur_price  # positive: zone above price

                # Prefer zones that contain price; among equal, pick closest.
                if best_zone is None:
                    best_zone = z
                    best_dist = dist
                    best_contains = contains
                elif contains and not best_contains:
                    best_zone = z
                    best_dist = dist
                    best_contains = True
                elif contains == best_contains and dist < best_dist:
                    best_zone = z
                    best_dist = dist

            if best_zone is not None:
                n_pivot_high_in_zone += 1
                n_open_pivots_created += 1
                open_pivots.append({
                    "side":        "BEAR",
                    "pivot_bar":   pivot_bar,
                    "pivot_high":  ha_high[pivot_bar],
                    "pivot_low":   ha_low[pivot_bar],
                    "zone_lo":     best_zone["line_lo"],
                    "zone_hi":     best_zone["line_hi"],
                })

        # 1+1+1 pivot LOW (BULL setup)
        if (ha_low[pivot_bar] < ha_low[pivot_bar - 1] and
                ha_low[pivot_bar] < ha_low[bar]):
            pivot_low = ha_low[pivot_bar]
            n_pivot_low += 1

            cur_price = ha_close[bar]
            best_zone = None
            best_dist = math.inf
            best_contains = False

            for zi, z in enumerate(all_zones):
                if z["side"] != "BULL":
                    continue
                if zone_first_bar[zi] > pivot_bar:
                    continue
                if not (zone_lo_valid[zi] or zone_hi_valid[zi]):
                    continue

                # Allow zones that contain current price OR are below current price.
                # Reject zones that are entirely above current price.
                if z["line_lo"] > cur_price:
                    continue

                # pivot low must be inside the zone (Mode A)
                if not (z["line_lo"] <= pivot_low <= z["line_hi"]):
                    continue

                contains = z["line_lo"] <= cur_price <= z["line_hi"]
                if contains:
                    dist = 0.0
                else:
                    # Zone is below price (line_hi < cur_price guaranteed by filter above)
                    dist = cur_price - z["line_hi"]  # positive: zone below price

                if best_zone is None:
                    best_zone = z
                    best_dist = dist
                    best_contains = contains
                elif contains and not best_contains:
                    best_zone = z
                    best_dist = dist
                    best_contains = True
                elif contains == best_contains and dist < best_dist:
                    best_zone = z
                    best_dist = dist

            if best_zone is not None:
                n_pivot_low_in_zone += 1
                n_open_pivots_created += 1
                open_pivots.append({
                    "side":        "BULL",
                    "pivot_bar":   pivot_bar,
                    "pivot_high":  ha_high[pivot_bar],
                    "pivot_low":   ha_low[pivot_bar],
                    "zone_lo":     best_zone["line_lo"],
                    "zone_hi":     best_zone["line_hi"],
                })

    # Mark still-open trades
    for t in active_trades:
        t["result"]  = "OPEN"
        t["pnl_r"]   = 0.0
        trades.append(t)

    el = time_module.perf_counter() - t0
    if not quiet:
        print(f"  ✅ Done in {el:.1f}s — {len(trades)} trades")
        print(f"  📊 Counters: pivot_high={n_pivot_high} | pivot_low={n_pivot_low} | "
              f"ph_in_zone={n_pivot_high_in_zone} | pl_in_zone={n_pivot_low_in_zone} | "
              f"open_pivots={n_open_pivots_created} | conf_closes={n_confirming_closes} | "
              f"trades_opened={n_trades_opened}")

    return trades


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: RESULTS + TRADE TABLE
# ═══════════════════════════════════════════════════════════════════

def _fmt(v):
    if v is None:
        return "—"
    v = float(v)
    if abs(v) < 0.01:
        return f"{v:.8f}"
    elif abs(v) < 1:
        return f"{v:.6f}"
    elif abs(v) < 100:
        return f"{v:.4f}"
    return f"{v:.2f}"


def calc_max_dd_r(trades):
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        if t["pnl_r"] is not None:
            cumulative += t["pnl_r"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
    return max_dd


def print_trade_table(trades, label=""):
    print(f"\n{'='*170}")
    print(f"  TRADE TABLE: {label}")
    print(f"{'='*170}")
    hdr = (f"{'#':>3} {'Side':<5} {'Res':<5} {'PnL':>7}  "
           f"{'Entry':>14} {'SL':>14} {'TP':>14} {'Risk':>12}  "
           f"{'MaxR':>5}  {'Dur':>4}  {'Pivot Time':<22} {'Entry Time':<22} {'Exit Time':<22}")
    print(hdr)
    print("-" * 170)

    for i, t in enumerate(trades):
        piv_ts  = str(t.get("pivot_ts",  ""))[:19]
        ent_ts  = str(t.get("entry_ts",  ""))[:19]
        exit_ts = str(t.get("exit_ts",   ""))[:19]
        r = t.get("result", "?")
        m = {"TP": "✅", "SL": "❌", "BE": "🔄", "OPEN": "⏳"}.get(r, "?")
        dur = t.get("duration_bars")
        dur_s = str(dur) if dur is not None else ""

        print(f"{i+1:>3} {t['side']:<5} {m} {r:<4} "
              f"{(t.get('pnl_r') or 0):>+6.2f}R  "
              f"{_fmt(t.get('entry')):>14} "
              f"{_fmt(t.get('original_sl', t.get('sl'))):>14} "
              f"{_fmt(t.get('tp')):>14} "
              f"{_fmt(t.get('risk')):>12}  "
              f"{t.get('max_r', 0):>5.1f}  "
              f"{dur_s:>4}  "
              f"{piv_ts:<22} {ent_ts:<22} {exit_ts:<22}")

    print("-" * 170)
    wins  = sum(1 for t in trades if (t.get("pnl_r") or 0) > 0)
    loss  = sum(1 for t in trades if (t.get("pnl_r") or 0) < 0)
    zeros = sum(1 for t in trades if
                t.get("pnl_r") is not None and t["pnl_r"] == 0.0
                and t.get("result") != "OPEN")
    net   = sum(t.get("pnl_r") or 0 for t in trades)
    print(f"    Total: {len(trades)} | 💚 {wins} | ❌ {loss} | 🔄 {zeros} | Net: {net:+.1f}R")


def print_results(trades, label="", rr_ratio=3.0):
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")

    if not trades:
        print("  No trades.")
        return

    closed = [t for t in trades if t.get("result") not in ("OPEN", None)]
    opens  = [t for t in trades if t.get("result") == "OPEN"]
    wins   = [t for t in closed if t.get("result") == "TP"]
    losses = [t for t in closed if t.get("result") == "SL"]
    bes    = [t for t in closed if t.get("result") == "BE"]
    total  = len(closed)

    print(f"\n📊 TRADES: {len(trades)} total | {total} closed | ⏳ {len(opens)} open")
    print(f"   ✅ {len(wins)} TP | ❌ {len(losses)} SL | 🔄 {len(bes)} BE")

    if total > 0:
        wr = len(wins) / total * 100
        print(f"   Win Rate: {wr:.1f}%")

    pnl    = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
    max_dd = calc_max_dd_r(closed)
    rpt    = pnl / total if total > 0 else 0.0

    print(f"\n💰 Net: {pnl:+.1f}R")
    print(f"📉 Max Drawdown: {max_dd:.1f}R")
    if max_dd > 0:
        print(f"📊 Return/DD ratio: {pnl/max_dd:.2f}x")
    print(f"📊 R per trade: {rpt:+.2f}R")

    gross_win  = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
    gross_loss = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    print(f"📊 Profit Factor: {pf:.2f}")

    print(f"\n📊 BY SIDE:")
    for s in ["BULL", "BEAR"]:
        st = [t for t in closed if t["side"] == s]
        if st:
            net_s = sum(t["pnl_r"] for t in st if t["pnl_r"] is not None)
            w_s   = sum(1 for t in st if (t["pnl_r"] or 0) > 0)
            print(f"   {s}: {len(st)} trades | {w_s}W | Net: {net_s:+.1f}R | "
                  f"R/trade: {net_s/len(st):+.2f}R")

    closed_dur = [t for t in closed if t.get("duration_bars")]
    if closed_dur:
        durs = [t["duration_bars"] for t in closed_dur]
        print(f"\n⏱️  DURATION (bars): avg={np.mean(durs):.0f} med={np.median(durs):.0f}")


def plot_equity_curve(trades, symbol, rr_ratio):
    """Plot a simple equity curve using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed. Skipping plot.")
        return

    closed = [t for t in trades if t.get("result") not in ("OPEN", None)]
    if not closed:
        print("  No closed trades to plot.")
        return

    cumulative = [0.0]
    for t in closed:
        cumulative.append(cumulative[-1] + (t["pnl_r"] or 0))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative, linewidth=1.5, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"LGCR Zone Pivot Sweep — {symbol} | RR={rr_ratio} | "
                 f"{len(closed)} closed trades")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=[c >= 0 for c in cumulative], alpha=0.15, color="green")
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=[c < 0 for c in cumulative], alpha=0.15, color="red")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CLI
# ═══════════════════════════════════════════════════════════════════

def _compute_warmup(tf: str) -> int:
    """Auto-scale warmup bars to roughly 1 week of M30 context."""
    minutes = _TF_MINUTES.get(tf, 5)
    m30_bars_per_week = 7 * 24 * 2  # 336 M30 bars in a week
    return max(300, m30_bars_per_week * (30 // minutes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LGCR Zone Pivot Sweep — Standalone Backtest")
    parser.add_argument("--symbol",  default="BTC/USDT:USDT",
                        help="Trading pair (default: BTC/USDT:USDT)")
    parser.add_argument("--tf",      default="5m",
                        help="Base timeframe (default: 5m). Context is always M30.")
    parser.add_argument("--days",    type=int, default=180,
                        help="Days of data to fetch (default: 180)")
    parser.add_argument("--rr",      type=float, default=3.0,
                        help="Risk-reward ratio for TP (default: 3.0)")
    parser.add_argument("--be",      type=float, default=0.0,
                        help="Break-even trigger in R (default: 0 = disabled)")
    parser.add_argument("--start",   default=None,
                        help="Start date (YYYY-MM-DD). Alternative to --days.")
    parser.add_argument("--end",     default=None,
                        help="End date (YYYY-MM-DD).")
    parser.add_argument("--warmup",  type=int, default=None,
                        help="Number of leading bars to skip (default: auto)")
    parser.add_argument("--force",   action="store_true",
                        help="Force re-download of data")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-trade output")
    parser.add_argument("--plot",    action="store_true",
                        help="Show equity curve plot after backtest")
    args = parser.parse_args()

    symbol = args.symbol
    tf     = args.tf
    rr     = args.rr
    be     = args.be

    warmup = args.warmup if args.warmup is not None else _compute_warmup(tf)

    if args.start or args.end:
        date_label = f"{args.start or 'start'} → {args.end or 'now'}"
    else:
        date_label = f"{args.days}d"

    print("=" * 60)
    print("LGCR Zone Pivot Sweep Backtest")
    print(f"Symbol:  {symbol}  |  TF: {tf}  |  Context: {HTF_TF}")
    print(f"RR: {rr}  |  BE: {'disabled' if not be else f'{be}R'}")
    print(f"Range:   {date_label}")
    print(f"Warmup:  {warmup} bars")
    print("=" * 60)

    df_raw = get_ohlcv(symbol, tf, args.days,
                       force_download=args.force,
                       start_date=args.start,
                       end_date=args.end)

    if df_raw is None or len(df_raw) == 0:
        print(f"\n❌ No data for {symbol} {tf} {date_label}")
        raise SystemExit(1)

    actual_start = df_raw["timestamp"].iloc[0].strftime("%Y-%m-%d") \
        if "timestamp" in df_raw.columns else "?"
    actual_end   = df_raw["timestamp"].iloc[-1].strftime("%Y-%m-%d") \
        if "timestamp" in df_raw.columns else "?"
    print(f"  Data: {actual_start} → {actual_end} ({len(df_raw)} bars)")

    trades = run_backtest(df_raw, rr_ratio=rr, be_trigger_r=be,
                          warmup=warmup, quiet=args.quiet)

    label = f"{symbol} {tf} {date_label} RR={rr}"
    if not args.quiet:
        print_trade_table(trades, label)
    print_results(trades, label, rr_ratio=rr)

    if args.plot:
        plot_equity_curve(trades, symbol, rr)
