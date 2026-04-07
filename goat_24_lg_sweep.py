"""
LG Line Sweep — Standalone Backtest (Raw Candles)
==================================================
Strategy overview:
  1. Detect LG and LGC patterns on raw OHLCV candles (no Heikin-Ashi).
  2. For each LGC bar, the LGC line is:
       BEAR LGC: raw_low[lgc_bar - 1]  (low of bar two before the 3rd candle)
       BULL LGC: raw_high[lgc_bar - 1]
  3. Wait for Sweep Bar (Candle 1):
       BEAR: first bar after LGC whose HIGH > LGC line,
             must be a BEAR candle (close < open),
             must NOT close through the line (close <= line),
             must have highest HIGH of all bars from sweep bar → conf bar.
       BULL: first bar after LGC whose LOW < LGC line,
             must be a BULL candle (close > open),
             must NOT close through the line (close >= line),
             must have lowest LOW of all bars from sweep bar → conf bar.
  4. Confirmation Bar (Candle 2):
       BEAR: closes below raw_low[sweep_bar]
       BULL: closes above raw_high[sweep_bar]
  5. Entry at the open of the bar after Candle 2.
  6. SL: nearest valid 2+1+2 raw pivot above entry (BEAR) / below entry (BULL).
  7. TP = entry ± RR × risk.
  8. BE: when max favorable excursion ≥ BE_R, move SL to entry.

Usage:
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 30m --days 180
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 5m --days 180 --rr 3 --be 2
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 4h --days 365
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 1d --days 730
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 30m --days 180 \\
      --partial 1.5 --partial-pct 50
  python3 goat_24_lg_sweep.py --symbol BTC/USDT:USDT --tf 30m --days 180 \\
      --optimize-bayesian --n-trials 100
"""

import os
import subprocess
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import warnings
import time as time_module
import argparse
from numba import njit

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from goat_15_data_manager import get_ohlcv


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: PATTERN DETECTION — LG + LGC on raw candles
# ═══════════════════════════════════════════════════════════════════

@njit
def detect_lg_lgc_raw(n, raw_open, raw_high, raw_low, raw_close):
    """Detect LG and LGC patterns on raw OHLCV candles.

    LG (at bar i):
      Bull LG: raw_high[i-1] < raw_close[i]
      Bear LG: raw_low[i-1]  > raw_close[i]

    LGC (stored at bar i-1, the middle candle, detected at bar i):
      Bull LGC: raw_high[i-2] < raw_low[i]  AND raw_close[i-1] > raw_open[i-1]
                → line = raw_high[i-2]
      Bear LGC: raw_low[i-2]  > raw_high[i] AND raw_close[i-1] < raw_open[i-1]
                → line = raw_low[i-2]

    Returns arrays of length n.
    """
    bull_lgc      = np.zeros(n, dtype=np.bool_)
    bear_lgc      = np.zeros(n, dtype=np.bool_)
    bull_lgc_line = np.full(n, np.nan)
    bear_lgc_line = np.full(n, np.nan)

    for i in range(2, n):
        # Bull LGC stored at i-1
        if raw_high[i - 2] < raw_low[i] and raw_close[i - 1] > raw_open[i - 1]:
            bull_lgc[i - 1]      = True
            bull_lgc_line[i - 1] = raw_high[i - 2]
        # Bear LGC stored at i-1
        if raw_low[i - 2] > raw_high[i] and raw_close[i - 1] < raw_open[i - 1]:
            bear_lgc[i - 1]      = True
            bear_lgc_line[i - 1] = raw_low[i - 2]

    return bull_lgc, bear_lgc, bull_lgc_line, bear_lgc_line


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: PIVOT COMPUTATION — 2+1+2 on raw candles
# ═══════════════════════════════════════════════════════════════════

@njit
def precompute_pivots_raw(raw_low, raw_high, n, pivot_len):
    """Compute 2+1+2 (or 1+1+1) pivot highs and lows on raw candles.

    pivot_len=2 → 5-candle window (default)
    pivot_len=1 → 3-candle window
    """
    max_pivots = n
    pli = np.empty(max_pivots, dtype=np.int64)
    plv = np.empty(max_pivots, dtype=np.float64)
    phi = np.empty(max_pivots, dtype=np.int64)
    phv = np.empty(max_pivots, dtype=np.float64)
    nl = 0
    nh = 0

    for i in range(pivot_len, n - pivot_len):
        is_low = True
        for off in range(1, pivot_len + 1):
            if not (raw_low[i] < raw_low[i - off] and raw_low[i] < raw_low[i + off]):
                is_low = False
                break
        if is_low:
            pli[nl] = i
            plv[nl] = raw_low[i]
            nl += 1

        is_high = True
        for off in range(1, pivot_len + 1):
            if not (raw_high[i] > raw_high[i - off] and raw_high[i] > raw_high[i + off]):
                is_high = False
                break
        if is_high:
            phi[nh] = i
            phv[nh] = raw_high[i]
            nh += 1

    return pli[:nl], plv[:nl], phi[:nh], phv[:nh]


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: SPARSE TABLES for O(1) range max/min on raw close
# ═══════════════════════════════════════════════════════════════════

def build_sparse_table(arr, func):
    n = len(arr)
    LOG = max(1, int(np.log2(max(n, 1))) + 1)
    sp = np.zeros((LOG, n), dtype=np.float64)
    sp[0] = arr.copy()
    for k in range(1, LOG):
        half = 1 << (k - 1)
        limit = n - (1 << k) + 1
        if limit <= 0:
            break
        if func == "max":
            sp[k, :limit] = np.maximum(sp[k - 1, :limit], sp[k - 1, half:half + limit])
        else:
            sp[k, :limit] = np.minimum(sp[k - 1, :limit], sp[k - 1, half:half + limit])
    return sp


@njit
def sparse_max(sp, l, r):
    if l > r:
        return -1e18
    k = np.int64(np.log2(r - l + 1))
    a = sp[k, l]
    b = sp[k, r - (1 << k) + 1]
    return a if a > b else b


@njit
def sparse_min(sp, l, r):
    if l > r:
        return 1e18
    k = np.int64(np.log2(r - l + 1))
    a = sp[k, l]
    b = sp[k, r - (1 << k) + 1]
    return a if a < b else b


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: SIGNAL SCANNER (Numba JIT)
# ═══════════════════════════════════════════════════════════════════

@njit
def scan_lg_sweep_signals(n, warmup,
                          raw_open, raw_high, raw_low, raw_close,
                          bear_lgc, bear_lgc_line,
                          bull_lgc, bull_lgc_line):
    """Scan for LG Line Sweep signals on raw candles.

    For each LGC bar, finds:
      1. Sweep bar (Candle 1): first bar wicking beyond LGC line with correct color.
      2. Confirmation bar (Candle 2): first bar closing beyond sweep bar extreme,
         where sweep bar still has the highest high (BEAR) / lowest low (BULL)
         in the range [sweep_bar, conf_bar].

    Returns parallel arrays for each signal found.
    Side: 0 = BEAR, 1 = BULL.
    """
    max_sigs = n * 4
    sig_conf_bar  = np.empty(max_sigs, dtype=np.int64)
    sig_sweep_bar = np.empty(max_sigs, dtype=np.int64)
    sig_lgc_bar   = np.empty(max_sigs, dtype=np.int64)
    sig_side      = np.empty(max_sigs, dtype=np.int64)   # 0=BEAR, 1=BULL
    sig_line      = np.empty(max_sigs, dtype=np.float64)
    ns = 0

    # ── BEAR signals ──────────────────────────────────────────────
    # LGC stored at i-1 (range 1..n-2), so start lgc_bar scan at 1
    for lgc_bar in range(1, n - 2):
        if not bear_lgc[lgc_bar]:
            continue
        lgc_line = bear_lgc_line[lgc_bar]
        if np.isnan(lgc_line) or lgc_line <= 0.0:
            continue

        # Find sweep bar: first bar after lgc_bar with wick above line
        sw = -1
        for j in range(lgc_bar + 1, n - 1):
            if raw_high[j] > lgc_line:
                # First wick found at bar j
                # Must be bear candle (close < open) AND wick-only (close <= line)
                if raw_close[j] < raw_open[j] and raw_close[j] <= lgc_line:
                    sw = j
                # Either valid sweep or line consumed — stop scanning
                break

        if sw < 0:
            continue
        if sw < warmup:
            continue  # Signal must be in post-warmup zone

        # Find confirmation bar: first bar after sw that:
        # (a) Has highest high in [sw..conf_bar] (check incrementally)
        # (b) Closes below raw_low[sw]
        sw_high = raw_high[sw]
        for j in range(sw + 1, n - 1):   # leave room for entry bar
            if raw_high[j] > sw_high:
                # Highest-high condition violated, no valid conf_bar from here
                break
            if raw_close[j] < raw_low[sw]:
                # Valid confirmation bar
                conf_bar = j
                if conf_bar >= warmup and ns < max_sigs:
                    sig_conf_bar[ns]  = conf_bar
                    sig_sweep_bar[ns] = sw
                    sig_lgc_bar[ns]   = lgc_bar
                    sig_side[ns]      = 0   # BEAR
                    sig_line[ns]      = lgc_line
                    ns += 1
                break

    # ── BULL signals ──────────────────────────────────────────────
    for lgc_bar in range(1, n - 2):
        if not bull_lgc[lgc_bar]:
            continue
        lgc_line = bull_lgc_line[lgc_bar]
        if np.isnan(lgc_line) or lgc_line <= 0.0:
            continue

        # Find sweep bar: first bar after lgc_bar with wick below line
        sw = -1
        for j in range(lgc_bar + 1, n - 1):
            if raw_low[j] < lgc_line:
                # First wick found at bar j
                # Must be bull candle (close > open) AND wick-only (close >= line)
                if raw_close[j] > raw_open[j] and raw_close[j] >= lgc_line:
                    sw = j
                break

        if sw < 0:
            continue
        if sw < warmup:
            continue

        # Find confirmation bar: first bar closing above raw_high[sw],
        # with lowest low maintained in [sw..conf_bar]
        sw_low = raw_low[sw]
        for j in range(sw + 1, n - 1):
            if raw_low[j] < sw_low:
                break
            if raw_close[j] > raw_high[sw]:
                conf_bar = j
                if conf_bar >= warmup and ns < max_sigs:
                    sig_conf_bar[ns]  = conf_bar
                    sig_sweep_bar[ns] = sw
                    sig_lgc_bar[ns]   = lgc_bar
                    sig_side[ns]      = 1   # BULL
                    sig_line[ns]      = lgc_line
                    ns += 1
                break

    return (sig_conf_bar[:ns], sig_sweep_bar[:ns], sig_lgc_bar[:ns],
            sig_side[:ns], sig_line[:ns], ns)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: SL CALCULATION (Numba JIT)
# ═══════════════════════════════════════════════════════════════════

@njit
def calc_sl_raw(raw_close_arr, raw_low_arr, raw_high_arr,
                piv_low_idx, piv_low_lvl, n_pl,
                piv_high_idx, piv_high_lvl, n_ph,
                conf_bar, is_bear):
    """Find nearest valid 2+1+2 raw pivot for SL.

    For BULL: nearest valid pivot LOW below raw_close[conf_bar].
    For BEAR: nearest valid pivot HIGH above raw_close[conf_bar].

    A pivot is valid if no subsequent bar (pivot_bar+1 .. conf_bar) has
    made a lower low (for pivot low) / higher high (for pivot high) than the pivot.

    Returns (valid, sl_level).
    """
    ref = raw_close_arr[conf_bar]

    if not is_bear:
        # BULL: pivot low below ref
        best_idx = -1
        best_bar = -1
        for p in range(n_pl):
            pi  = piv_low_idx[p]
            plv = piv_low_lvl[p]
            if pi > conf_bar:
                continue
            if plv >= ref:
                continue  # pivot must be below reference price
            valid = True
            for j in range(pi + 1, conf_bar + 1):
                if raw_low_arr[j] < plv:
                    valid = False
                    break
            if valid and (best_idx < 0 or pi > best_bar):
                best_idx = p
                best_bar = pi
        if best_idx < 0:
            return False, 0.0
        return True, piv_low_lvl[best_idx]
    else:
        # BEAR: pivot high above ref
        best_idx = -1
        best_bar = -1
        for p in range(n_ph):
            pi  = piv_high_idx[p]
            phv = piv_high_lvl[p]
            if pi > conf_bar:
                continue
            if phv <= ref:
                continue  # pivot must be above reference price
            valid = True
            for j in range(pi + 1, conf_bar + 1):
                if raw_high_arr[j] > phv:
                    valid = False
                    break
            if valid and (best_idx < 0 or pi > best_bar):
                best_idx = p
                best_bar = pi
        if best_idx < 0:
            return False, 0.0
        return True, piv_high_lvl[best_idx]


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: PRECOMPUTE ALL
# ═══════════════════════════════════════════════════════════════════

def precompute_all(df_raw, pivot_len=2):
    """Precompute all arrays needed for the backtest."""
    t0 = time_module.perf_counter()
    print("  Computing patterns + pivots + sparse tables (raw candles)...")

    n = len(df_raw)
    raw_open  = df_raw['open'].values.astype(np.float64)
    raw_high  = df_raw['high'].values.astype(np.float64)
    raw_low   = df_raw['low'].values.astype(np.float64)
    raw_close = df_raw['close'].values.astype(np.float64)

    if 'timestamp' in df_raw.columns:
        timestamps = df_raw['timestamp'].values
    else:
        timestamps = np.arange(n)

    bull_lgc, bear_lgc, bull_lgc_line, bear_lgc_line = \
        detect_lg_lgc_raw(n, raw_open, raw_high, raw_low, raw_close)

    n_bull_lgc = int(np.sum(bull_lgc))
    n_bear_lgc = int(np.sum(bear_lgc))
    print(f"    LGC patterns: {n_bear_lgc} BEAR, {n_bull_lgc} BULL")

    piv_low_idx, piv_low_lvl, piv_high_idx, piv_high_lvl = \
        precompute_pivots_raw(raw_low, raw_high, n, pivot_len)
    print(f"    {len(piv_low_idx)} pivot lows, {len(piv_high_idx)} pivot highs")

    # Sparse tables on raw close for range validity checks
    sp_max_close = build_sparse_table(raw_close, "max")
    sp_min_close = build_sparse_table(raw_close, "min")

    el = time_module.perf_counter() - t0
    print(f"    Prep done in {el:.1f}s")

    return {
        "n": n,
        "raw_open": raw_open, "raw_high": raw_high,
        "raw_low": raw_low,   "raw_close": raw_close,
        "timestamps": timestamps,
        "bull_lgc": bull_lgc, "bear_lgc": bear_lgc,
        "bull_lgc_line": bull_lgc_line, "bear_lgc_line": bear_lgc_line,
        "piv_low_idx": piv_low_idx,   "piv_low_lvl": piv_low_lvl,
        "piv_high_idx": piv_high_idx, "piv_high_lvl": piv_high_lvl,
        "sp_max_close": sp_max_close,  "sp_min_close": sp_min_close,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_backtest(pre, rr_ratio=3.0, be_trigger_r=2.0, warmup=300,
                 partial_tp_r=0.0, partial_tp_pct=50.0, quiet=False,
                 pivot_len=2, debug_bars=None):
    """Run the LG Line Sweep backtest.

    Parameters
    ----------
    pre           : precomputed dict from precompute_all()
    rr_ratio      : risk-reward ratio for TP
    be_trigger_r  : break-even trigger in R (0 = disabled)
    warmup        : leading bars to skip for signal emission
    partial_tp_r  : R-multiple for partial TP (0 = disabled)
    partial_tp_pct: % of position to close at partial TP (default 50%)
    quiet         : suppress verbose output
    pivot_len     : pivot window length (1 or 2)
    debug_bars    : list of timestamp strings for diagnostics

    Returns
    -------
    trades : list of dicts
    """
    t0 = time_module.perf_counter()

    be_label      = f"{be_trigger_r}R BE" if be_trigger_r else "NO BE"
    partial_label = f" | Partial: {partial_tp_pct:.0f}%@{partial_tp_r}R" if partial_tp_r > 0 else ""

    if not quiet:
        print(f"\n{'='*60}")
        print(f"LG SWEEP BACKTEST: {be_label} | RR={rr_ratio}{partial_label}")
        print(f"Entry: next raw open after conf bar | SL/TP vs raw high/low")
        print(f"Warmup: {warmup} bars | Pivot: {pivot_len}+1+{pivot_len}")
        print(f"{'='*60}")

    use_partial  = partial_tp_r > 0
    partial_frac = partial_tp_pct / 100.0
    remain_frac  = 1.0 - partial_frac

    n            = pre["n"]
    raw_open     = pre["raw_open"]
    raw_high     = pre["raw_high"]
    raw_low      = pre["raw_low"]
    raw_close    = pre["raw_close"]
    timestamps   = pre["timestamps"]
    piv_low_idx  = pre["piv_low_idx"]
    piv_low_lvl  = pre["piv_low_lvl"]
    piv_high_idx = pre["piv_high_idx"]
    piv_high_lvl = pre["piv_high_lvl"]
    n_pl         = len(piv_low_idx)
    n_ph         = len(piv_high_idx)

    if not quiet:
        print("  Scanning LG sweep signals (Numba JIT)...")
    t1 = time_module.perf_counter()
    (sig_conf, sig_sw, sig_lgc, sig_side, sig_line, ns) = scan_lg_sweep_signals(
        n, warmup,
        raw_open, raw_high, raw_low, raw_close,
        pre["bear_lgc"], pre["bear_lgc_line"],
        pre["bull_lgc"], pre["bull_lgc_line"],
    )
    t_scan = time_module.perf_counter() - t1
    if not quiet:
        print(f"    Found {ns} raw signals in {t_scan:.1f}s")

    if debug_bars:
        _debug_diagnostic(pre, warmup, sig_conf, sig_sw, sig_lgc, sig_side, sig_line, ns, debug_bars)

    # Build signal lookup by conf_bar
    sig_by_conf = {}
    for s in range(ns):
        cb = sig_conf[s]
        if cb not in sig_by_conf:
            sig_by_conf[cb] = []
        sig_by_conf[cb].append(s)

    trades  = []
    active  = []
    pending = []
    skipped = 0

    if not quiet:
        print("  Simulating trades...")

    for bar in range(warmup, n):

        # ── Execute pending entries (from previous conf_bar) ──────
        new_pending = []
        for pend in pending:
            conf_bar  = pend["conf_bar"]
            sw_bar    = pend["sw_bar"]
            lgc_bar_p = pend["lgc_bar"]
            is_bear   = pend["is_bear"]

            valid_sl, sl = calc_sl_raw(
                raw_close, raw_low, raw_high,
                piv_low_idx, piv_low_lvl, n_pl,
                piv_high_idx, piv_high_lvl, n_ph,
                conf_bar, is_bear,
            )
            if not valid_sl:
                skipped += 1
                continue

            entry = raw_open[bar]

            # Validate SL geometry vs actual entry
            if not is_bear and sl >= entry:
                skipped += 1
                continue
            if is_bear and sl <= entry:
                skipped += 1
                continue

            risk = abs(entry - sl)
            if risk == 0.0:
                skipped += 1
                continue

            if not is_bear:
                tp  = entry + rr_ratio * risk
                tp1 = entry + partial_tp_r * risk if use_partial else 0.0
            else:
                tp  = entry - rr_ratio * risk
                tp1 = entry - partial_tp_r * risk if use_partial else 0.0

            side_str = "BEAR" if is_bear else "BULL"
            trade = {
                "side":           side_str,
                "case":           "LGS",
                "entry":          entry,
                "sl":             sl,
                "tp":             tp,
                "tp1":            tp1,
                "risk":           risk,
                "ha_entry":       entry,           # same as entry for raw
                "original_sl":    sl,
                "entry_bar":      bar,
                "entry_ts":       timestamps[bar],
                "signal_bar":     conf_bar,
                "signal_ts":      timestamps[conf_bar],
                "sweep_bar":      sw_bar,
                "sweep_ts":       timestamps[sw_bar],
                "lgc_bar":        lgc_bar_p,
                "lgc_ts":         timestamps[lgc_bar_p],
                "swept_val":      pend["lgc_line"],
                "max_r":          0.0,
                "be_active":      False,
                "partial_filled": False,
                "partial_pnl_r":  0.0,
                "result":         None,
                "pnl_r":          None,
                "exit_bar":       None,
                "exit_ts":        None,
                "duration_bars":  None,
            }
            active.append(trade)
        pending = new_pending

        # ── Check active trades (SL / TP / BE) ───────────────────
        to_close = []
        for t in active:
            c_hi  = raw_high[bar]
            c_lo  = raw_low[bar]
            risk  = t["risk"]
            entry = t["entry"]

            if t["side"] == "BULL":
                if risk > 0:
                    t["max_r"] = max(t["max_r"], (c_hi - entry) / risk)

                # Partial TP
                if use_partial and not t["partial_filled"] and c_hi >= t["tp1"]:
                    t["partial_filled"] = True
                    t["partial_pnl_r"]  = partial_frac * partial_tp_r

                # BE trigger
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"]        = entry

                # SL hit
                if c_lo <= t["sl"]:
                    if t["partial_filled"]:
                        if t["be_active"] and t["sl"] == entry:
                            remaining = 0.0
                            r = "P+BE"
                        else:
                            remaining = -remain_frac
                            r = "P+SL"
                        p = t["partial_pnl_r"] + remaining
                    else:
                        r = "BE" if (t["be_active"] and t["sl"] == entry) else "SL"
                        p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))

                # Full TP hit
                elif c_hi >= t["tp"]:
                    if t["partial_filled"]:
                        p = t["partial_pnl_r"] + remain_frac * rr_ratio
                        r = "P+TP"
                    else:
                        p = float(rr_ratio)
                        r = "TP"
                    to_close.append((t, r, p))

            else:  # BEAR
                if risk > 0:
                    t["max_r"] = max(t["max_r"], (entry - c_lo) / risk)

                # Partial TP
                if use_partial and not t["partial_filled"] and c_lo <= t["tp1"]:
                    t["partial_filled"] = True
                    t["partial_pnl_r"]  = partial_frac * partial_tp_r

                # BE trigger
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"]        = entry

                # SL hit
                if c_hi >= t["sl"]:
                    if t["partial_filled"]:
                        if t["be_active"] and t["sl"] == entry:
                            remaining = 0.0
                            r = "P+BE"
                        else:
                            remaining = -remain_frac
                            r = "P+SL"
                        p = t["partial_pnl_r"] + remaining
                    else:
                        r = "BE" if (t["be_active"] and t["sl"] == entry) else "SL"
                        p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))

                # Full TP hit
                elif c_lo <= t["tp"]:
                    if t["partial_filled"]:
                        p = t["partial_pnl_r"] + remain_frac * rr_ratio
                        r = "P+TP"
                    else:
                        p = float(rr_ratio)
                        r = "TP"
                    to_close.append((t, r, p))

        for t, result, pnl_r in to_close:
            t["result"]        = result
            t["pnl_r"]         = pnl_r
            t["exit_bar"]      = bar
            t["exit_ts"]       = timestamps[bar]
            t["duration_bars"] = bar - t["entry_bar"]
            trades.append(t)
            active.remove(t)

        # ── Queue new signals from this conf_bar ──────────────────
        if bar in sig_by_conf:
            for s in sig_by_conf[bar]:
                if bar + 1 >= n:
                    continue   # No room for entry bar

                is_bear_s = (sig_side[s] == 0)
                pending.append({
                    "conf_bar":  bar,
                    "sw_bar":    int(sig_sw[s]),
                    "lgc_bar":   int(sig_lgc[s]),
                    "is_bear":   is_bear_s,
                    "lgc_line":  float(sig_line[s]),
                })

    # Mark still-open trades
    for t in active:
        t["result"] = "OPEN"
        t["pnl_r"]  = 0.0
        trades.append(t)

    el = time_module.perf_counter() - t0
    if not quiet:
        if skipped > 0:
            print(f"    Skipped {skipped} invalid (SL wrong side of entry or zero risk)")
        print(f"  ✅ Done in {el:.1f}s — {len(trades)} trades")

    return trades


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: DEBUG DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════

def _debug_diagnostic(pre, warmup, sig_conf, sig_sw, sig_lgc, sig_side, sig_line, ns,
                      debug_ts_list):
    """Print diagnostic info for requested timestamps."""
    timestamps = pre["timestamps"]
    ts_strs = [str(ts)[:16].replace("T", " ") for ts in timestamps]
    n = pre["n"]

    print(f"\n{'='*60}")
    print(f"  LG SWEEP DEBUG DIAGNOSTIC")
    print(f"{'='*60}")

    for dbg_ts in debug_ts_list:
        dbg_ts = dbg_ts.strip().replace("T", " ")
        matches = [i for i, s in enumerate(ts_strs) if s == dbg_ts]
        if not matches:
            print(f"\n⚠️  {dbg_ts}: no matching bar found")
            continue

        for ci in matches:
            ts_label = str(timestamps[ci])[:19].replace("T", " ")
            in_warmup = ci < warmup
            print(f"\n── bar {ci} | {ts_label} {'[WARMUP]' if in_warmup else '[SCANNABLE]'} ──")
            print(f"   raw O={pre['raw_open'][ci]:.4f} H={pre['raw_high'][ci]:.4f}"
                  f" L={pre['raw_low'][ci]:.4f} C={pre['raw_close'][ci]:.4f}")

            for side_label, is_bear, lgc_f, lgc_l in [
                ("BEAR", True,  pre["bear_lgc"],  pre["bear_lgc_line"]),
                ("BULL", False, pre["bull_lgc"],  pre["bull_lgc_line"]),
            ]:
                if lgc_f[ci]:
                    print(f"   {side_label} LGC at this bar! Line = {lgc_l[ci]:.4f}")

            # Show any signals where this bar is conf_bar, sweep_bar, or lgc_bar
            for s in range(ns):
                cb = int(sig_conf[s])
                sw = int(sig_sw[s])
                lb = int(sig_lgc[s])
                if ci not in (cb, sw, lb):
                    continue
                side_str = "BEAR" if sig_side[s] == 0 else "BULL"
                cb_ts = str(timestamps[cb])[:19].replace("T", " ")
                sw_ts = str(timestamps[sw])[:19].replace("T", " ")
                lb_ts = str(timestamps[lb])[:19].replace("T", " ")
                print(f"   → {side_str} signal: LGC@{lb_ts} | Sweep@{sw_ts} | Conf@{cb_ts}"
                      f" | line={sig_line[s]:.4f}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: EQUITY CURVE + RISK METRICS
# ═══════════════════════════════════════════════════════════════════

_DAYS_PER_YEAR   = 365.25
# Used for annualising Sharpe/Sortino (finance convention: 365 calendar days)
_ANNUALIZE_DAYS  = 365.25


def compute_equity_curve(trades, starting_capital, risk_pct, rr_ratio,
                         taker_fee=0.0, maker_fee=0.0, total_bars=None):
    """Compute compounded equity curve from closed trades (same logic as goat_20)."""
    closed = [t for t in trades if t["result"] not in ("OPEN", None)]
    if not closed:
        comm_stats = {"total_taker": 0.0, "total_maker": 0.0, "total": 0.0}
        return [], starting_capital, starting_capital, 0, 0, {}, comm_stats

    balance   = starting_capital
    peak      = starting_capital
    max_dd_pct = 0.0
    max_dd_usd = 0.0

    equity_points = [{"trade": 0, "balance": balance, "result": "START",
                      "time": str(closed[0].get("entry_ts", ""))[:19],
                      "taker_fee_usd": 0.0, "maker_fee_usd": 0.0, "total_fee_usd": 0.0,
                      "total_bars": total_bars}]
    monthly_pnl       = {}
    total_taker_fees  = 0.0
    total_maker_fees  = 0.0

    for i, t in enumerate(closed):
        risk_usd   = balance * (risk_pct / 100.0)
        pnl_r      = t.get("pnl_r", 0) or 0
        pnl_usd    = risk_usd * pnl_r
        entry_price = t.get("entry", 0) or 0
        raw_risk    = t.get("risk", 0) or 0

        if raw_risk > 0 and entry_price > 0 and (taker_fee > 0 or maker_fee > 0):
            position_usd = risk_usd * entry_price / raw_risk
            fee_taker    = position_usd * (taker_fee / 100.0)
            fee_maker    = position_usd * (maker_fee / 100.0)
            total_fee    = fee_taker + fee_maker
        else:
            fee_taker = fee_maker = total_fee = 0.0

        total_taker_fees += fee_taker
        total_maker_fees += fee_maker
        pnl_usd          -= total_fee
        balance          += pnl_usd

        if balance > peak:
            peak = balance
        dd_usd  = peak - balance
        dd_pct  = dd_usd / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd

        exit_ts = str(t.get("exit_ts", ""))[:19]
        equity_points.append({
            "trade":         i + 1,
            "balance":       round(balance, 2),
            "pnl_usd":       round(pnl_usd, 2),
            "pnl_r":         round(pnl_r, 3),
            "risk_usd":      round(risk_usd, 2),
            "result":        t["result"],
            "side":          t["side"],
            "case":          t["case"],
            "time":          exit_ts,
            "dd_pct":        round(dd_pct, 2),
            "taker_fee_usd": round(fee_taker, 4),
            "maker_fee_usd": round(fee_maker, 4),
            "total_fee_usd": round(total_fee, 4),
            "total_bars":    total_bars,
        })

        if len(exit_ts) >= 7:
            month_key = exit_ts[:7]
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + pnl_usd

    comm_stats = {
        "total_taker": total_taker_fees,
        "total_maker": total_maker_fees,
        "total":       total_taker_fees + total_maker_fees,
    }
    return equity_points, balance, peak, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats


def export_equity_csv(equity_points, filename):
    pd.DataFrame(equity_points).to_csv(filename, index=False)
    print(f"  💾 {len(equity_points)} equity points → {filename}")


def calc_max_dd(trades):
    cumulative = peak = max_dd = 0.0
    for t in trades:
        if t["pnl_r"] is not None:
            cumulative += t["pnl_r"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
    return max_dd


def calc_risk_metrics_r(closed):
    r_series = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
    if len(r_series) < 2:
        return {"sharpe_r": None, "sortino_r": None, "vol_r": None, "trades_per_year": None}

    first_ts = closed[0].get("entry_ts") or closed[0].get("exit_ts")
    last_ts  = closed[-1].get("exit_ts") or closed[-1].get("entry_ts")
    num_years = 1.0
    if first_ts and last_ts:
        try:
            t0_ = pd.to_datetime(str(first_ts))
            t1_ = pd.to_datetime(str(last_ts))
            num_years = max((t1_ - t0_).days / _DAYS_PER_YEAR, 1 / _DAYS_PER_YEAR)
        except Exception:
            num_years = 1.0

    trades_per_year = len(r_series) / num_years
    mean_r = np.mean(r_series)
    std_r  = np.std(r_series)
    sharpe_r = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0

    neg_r       = [r for r in r_series if r < 0]
    down_std    = np.std(neg_r) if len(neg_r) >= 2 else 0.0
    sortino_r   = (mean_r / down_std) * np.sqrt(trades_per_year) if down_std > 0 else 999.0
    vol_r       = std_r * np.sqrt(trades_per_year)

    return {"sharpe_r": sharpe_r, "sortino_r": sortino_r, "vol_r": vol_r,
            "trades_per_year": trades_per_year}


def calc_risk_metrics_pct(equity_points):
    if not equity_points or len(equity_points) < 3:
        return {"sharpe_pct": None, "sortino_pct": None, "vol_pct": None}

    eq_df = pd.DataFrame([
        {"time": p.get("time", ""), "balance": p["balance"]}
        for p in equity_points if p.get("time")
    ])
    if len(eq_df) < 3:
        return {"sharpe_pct": None, "sortino_pct": None, "vol_pct": None}

    eq_df["time"] = pd.to_datetime(eq_df["time"], errors="coerce")
    eq_df = eq_df.dropna(subset=["time"]).sort_values("time")
    eq_df["date"] = eq_df["time"].dt.date
    daily = eq_df.groupby("date")["balance"].last()
    if len(daily) < 2:
        return {"sharpe_pct": None, "sortino_pct": None, "vol_pct": None}

    daily_ret = daily.pct_change().dropna().values
    if len(daily_ret) < 2:
        return {"sharpe_pct": None, "sortino_pct": None, "vol_pct": None}

    mean_d  = np.mean(daily_ret)
    std_d   = np.std(daily_ret)
    sharpe_pct  = (mean_d / std_d) * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0
    neg_d       = daily_ret[daily_ret < 0]
    down_d      = np.std(neg_d) if len(neg_d) >= 2 else 0.0
    sortino_pct = (mean_d / down_d) * np.sqrt(_ANNUALIZE_DAYS) if down_d > 0 else 999.0
    vol_pct     = std_d * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0

    return {"sharpe_pct": sharpe_pct, "sortino_pct": sortino_pct, "vol_pct": vol_pct}


def calc_bh_metrics(raw_close, timestamps):
    if raw_close is None or len(raw_close) < 2:
        return {"return_pct": None, "max_dd_pct": None, "sharpe": None, "sortino": None}

    bh_return_pct = (raw_close[-1] / raw_close[0] - 1) * 100

    peak_bh   = raw_close[0]
    max_dd_bh = 0.0
    for c in raw_close:
        if c > peak_bh:
            peak_bh = c
        if peak_bh > 0:
            dd = (peak_bh - c) / peak_bh * 100
            if dd > max_dd_bh:
                max_dd_bh = dd

    try:
        ts_pd        = pd.to_datetime(pd.Series(timestamps).astype(str), errors="coerce")
        close_series = pd.Series(raw_close, index=ts_pd)
        daily_close  = close_series.resample("D").last().dropna()
        if len(daily_close) >= 2:
            daily_ret = daily_close.pct_change().dropna().values
            if len(daily_ret) >= 2:
                mean_d  = np.mean(daily_ret)
                std_d   = np.std(daily_ret)
                sharpe  = (mean_d / std_d) * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0
                neg_d   = daily_ret[daily_ret < 0]
                down_d  = np.std(neg_d) if len(neg_d) >= 2 else 0.0
                sortino = (mean_d / down_d) * np.sqrt(_ANNUALIZE_DAYS) if down_d > 0 else 999.0
            else:
                sharpe = sortino = None
        else:
            sharpe = sortino = None
    except Exception:
        sharpe = sortino = None

    return {"return_pct": bh_return_pct, "max_dd_pct": max_dd_bh,
            "sharpe": sharpe, "sortino": sortino}


def enrich_equity_with_bh(equity_points, raw_close, timestamps, initial_capital):
    if not equity_points or raw_close is None or len(raw_close) < 2:
        return equity_points
    try:
        ts_pd      = pd.to_datetime(pd.Series(timestamps).astype(str), errors="coerce")
        bh_values  = initial_capital * raw_close / raw_close[0]
        bh_series  = pd.Series(bh_values, index=ts_pd)
        bh_series  = bh_series[~bh_series.index.isna()].sort_index()
        for pt in equity_points:
            pt_ts = pd.to_datetime(str(pt.get("time", "")), errors="coerce")
            if pd.isna(pt_ts):
                pt["bh_balance"] = None
                continue
            idx = bh_series.index.get_indexer([pt_ts], method="nearest")[0]
            pt["bh_balance"] = round(float(bh_series.iloc[idx]), 2) if 0 <= idx < len(bh_series) else None
    except Exception:
        for pt in equity_points:
            pt["bh_balance"] = None
    return equity_points


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: RESULTS + TRADE TABLE + CSV
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


def _sharpe_rating(v):
    if v is None:
        return ""
    if v < 1:   return "[POOR]"
    if v < 2:   return "[OK]"
    if v < 3:   return "[GOOD]"
    return "[EXCELLENT]"


def _sortino_rating(v):
    if v is None:
        return ""
    if v < 1.5: return "[POOR]"
    if v < 3:   return "[OK]"
    if v < 5:   return "[GOOD]"
    return "[EXCELLENT]"


def print_results(trades, label, rr_ratio=3, partial_tp_r=0, partial_tp_pct=50):
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    if not trades:
        print("  No trades.")
        return

    closed      = [t for t in trades if t["result"] not in ("OPEN", None)]
    wins_full   = [t for t in closed if t["result"] == "TP"]
    wins_p      = [t for t in closed if t["result"] == "P+TP"]
    losses      = [t for t in closed if t["result"] == "SL"]
    bes         = [t for t in closed if t["result"] == "BE"]
    p_be        = [t for t in closed if t["result"] == "P+BE"]
    p_sl        = [t for t in closed if t["result"] == "P+SL"]
    opens       = [t for t in trades if t["result"] == "OPEN"]
    total       = len(closed)

    print(f"\n📊 TRADES: {len(trades)} total | {total} closed | ⏳ {len(opens)} open")
    if partial_tp_r > 0:
        print(f"   ✅ TP: {len(wins_full)} | P+TP: {len(wins_p)} | "
              f"❌ SL: {len(losses)} | P+SL: {len(p_sl)} | "
              f"🔄 BE: {len(bes)} | P+BE: {len(p_be)}")
    else:
        print(f"   ✅ {len(wins_full)} TP | ❌ {len(losses)} SL | 🔄 {len(bes)} BE")
    if total > 0:
        pure_wins = len(wins_full) + len(wins_p)
        print(f"   Win Rate (TP only): {pure_wins/total*100:.1f}%")

    pnl    = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
    max_dd = calc_max_dd(closed)
    rpt    = pnl / total if total > 0 else 0

    print(f"\n💰 Net: {pnl:+.1f}R (${pnl*1000:+,.0f} at $1k/R)")
    print(f"📉 Max Drawdown: {max_dd:.1f}R")
    if max_dd > 0:
        print(f"📊 Return/DD ratio: {pnl/max_dd:.2f}x")
    print(f"📊 R per trade: {rpt:+.2f}R")

    gw = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
    gl = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
    pf = gw / gl if gl > 0 else 999
    print(f"📊 Profit Factor: {pf:.2f}")

    rm_r = calc_risk_metrics_r(closed)
    print(f"\n── Risk Metrics (flat R) ──────────────────")
    if rm_r["sharpe_r"] is not None:
        print(f"📐 Trades/year:   {rm_r['trades_per_year']:.1f}")
        print(f"📐 Sharpe  (R):   {rm_r['sharpe_r']:.2f}  {_sharpe_rating(rm_r['sharpe_r'])}")
        print(f"📐 Sortino (R):   {rm_r['sortino_r']:.2f}  {_sortino_rating(rm_r['sortino_r'])}")
        print(f"📐 Vol (R, ann):  {rm_r['vol_r']:.2f}R")
    else:
        print(f"📐 (insufficient trades for risk metrics)")

    if partial_tp_r > 0:
        print(f"\n📊 PARTIAL TP BREAKDOWN:")
        for rtype in ["TP", "P+TP", "SL", "P+SL", "BE", "P+BE"]:
            grp = [t for t in closed if t["result"] == rtype]
            if grp:
                avg_r = np.mean([t["pnl_r"] for t in grp if t["pnl_r"] is not None])
                tot_r = sum(t["pnl_r"] for t in grp if t["pnl_r"] is not None)
                print(f"   {rtype:<5}: {len(grp):>3} | avg {avg_r:+.2f}R | total {tot_r:+.1f}R")

    print(f"\n📊 BY SIDE:")
    for s in ['BULL', 'BEAR']:
        st = [t for t in closed if t["side"] == s]
        if st:
            net = sum(t["pnl_r"] for t in st if t["pnl_r"] is not None)
            w   = sum(1 for t in st if (t["pnl_r"] or 0) > 0)
            print(f"   {s}: {len(st)} trades | {w}W | Net: {net:+.1f}R | "
                  f"R/trade: {net/len(st):+.2f}R")

    if losses:
        mrs = [t["max_r"] for t in losses]
        print(f"\n📐 LOSS MAX-R: ", end="")
        for th in [0.5, 1.0, 1.5, 2.0]:
            c = sum(1 for r in mrs if r >= th)
            print(f"≥{th}R:{c}/{len(mrs)}({c/len(mrs)*100:.0f}%) ", end="")
        print()

    closed_dur = [t for t in closed if t.get("duration_bars")]
    if closed_dur:
        durs = [t["duration_bars"] for t in closed_dur]
        print(f"\n⏱️  DURATION (bars): avg={np.mean(durs):.0f} med={np.median(durs):.0f}")


def print_equity_curve(equity_points, starting_capital, final_balance, peak,
                       max_dd_pct, max_dd_usd, monthly_pnl, risk_pct, rr_ratio, label):
    print(f"\n{'='*70}")
    print(f"  EQUITY CURVE: {label}")
    print(f"  Starting: ${starting_capital:,.2f} | Risk: {risk_pct}% per trade | RR: {rr_ratio}")
    print(f"{'='*70}")

    total_return = (final_balance - starting_capital) / starting_capital * 100
    print(f"\n  💰 Final Balance:  ${final_balance:,.2f}")
    print(f"  📈 Total Return:   {total_return:+.1f}% (${final_balance - starting_capital:+,.2f})")
    print(f"  🏔️  Peak Balance:   ${peak:,.2f}")
    print(f"  📉 Max Drawdown:   {max_dd_pct:.1f}% (${max_dd_usd:,.2f})")
    if max_dd_pct > 0:
        print(f"  📊 Return/DD:      {total_return/max_dd_pct:.2f}x")

    rm_pct = calc_risk_metrics_pct(equity_points)
    print(f"\n  ── Risk Metrics (compounded %) ─────────────")
    if rm_pct["sharpe_pct"] is not None:
        print(f"  📐 Sharpe  (%):   {rm_pct['sharpe_pct']:.2f}  {_sharpe_rating(rm_pct['sharpe_pct'])}")
        print(f"  📐 Sortino (%):   {rm_pct['sortino_pct']:.2f}  {_sortino_rating(rm_pct['sortino_pct'])}")
        print(f"  📐 Vol (%, ann):  {rm_pct['vol_pct']*100:.1f}%")
    else:
        print(f"  📐 (insufficient data)")

    results = [p["result"] for p in equity_points if p["result"] not in ("START", None)]
    cur_loss = cur_win = max_loss = max_win = 0
    for r in results:
        if r in ("SL",):
            cur_loss += 1; cur_win = 0
            max_loss = max(max_loss, cur_loss)
        elif r in ("TP", "P+TP"):
            cur_win += 1; cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss = cur_win = 0
    print(f"\n  🔥 Max consecutive wins:   {max_win}")
    print(f"  💀 Max consecutive losses: {max_loss}")

    if monthly_pnl:
        print(f"\n  📅 MONTHLY P&L:")
        print(f"  {'Month':<10} {'P&L':>12} {'Cum':>12}")
        print("  " + "-" * 36)
        cum = 0
        max_abs = max(abs(v) for v in monthly_pnl.values()) if monthly_pnl else 1
        for month in sorted(monthly_pnl.keys()):
            pnl = monthly_pnl[month]
            cum += pnl
            bar_s = "█" * max(1, int(abs(pnl) / max_abs * 20)) if pnl != 0 else ""
            sign  = "🟢" if pnl >= 0 else "🔴"
            print(f"  {month:<10} ${pnl:>+11,.2f} ${cum:>+11,.2f}  {sign} {bar_s}")

    if len(equity_points) > 2:
        balances = [p["balance"] for p in equity_points]
        n_pts    = min(50, len(balances))
        step     = max(1, len(balances) // n_pts)
        sampled  = balances[::step]
        if sampled[-1] != balances[-1]:
            sampled.append(balances[-1])
        min_b = min(sampled); max_b = max(sampled)
        rng   = max_b - min_b if max_b > min_b else 1
        height = 12
        print(f"\n  📈 EQUITY CURVE:")
        print(f"  ${max_b:>10,.0f} ┤")
        for row in range(height - 1, -1, -1):
            line = "  " + " " * 12 + "│"
            for val in sampled:
                bh = int((val - min_b) / rng * height)
                if bh == row:
                    line += "●"
                elif bh > row:
                    line += "│"
                else:
                    line += " "
            if row == height // 2:
                mid = min_b + 0.5 * rng
                print(f"  ${mid:>10,.0f} ┤{line[15:]}")
            else:
                print(f"  {'':>11} │{line[15:]}")
        print(f"  ${min_b:>10,.0f} ┤{'─' * len(sampled)}")
        print(f"  {'':>12} Trade 1{' ' * max(0, len(sampled) - 15)}Trade {len(balances)-1}")


def print_bh_comparison(bh_metrics, initial_capital, final_balance):
    if bh_metrics.get("return_pct") is None:
        return
    bh_ret  = bh_metrics["return_pct"]
    bh_dd   = bh_metrics.get("max_dd_pct", 0) or 0
    bh_sh   = bh_metrics.get("sharpe")
    bh_so   = bh_metrics.get("sortino")
    strat_r = (final_balance / initial_capital - 1) * 100

    print(f"\n  ── Buy & Hold Benchmark ────────────────────────────────")
    print(f"  B&H Return:       {bh_ret:+.1f}%")
    print(f"  B&H Max DD:       {bh_dd:.1f}%")
    if bh_sh is not None:
        print(f"  B&H Sharpe:       {bh_sh:.2f}")
    if bh_so is not None:
        so_str = f"{bh_so:.2f}" if bh_so < 999 else "999.00"
        print(f"  B&H Sortino:      {so_str}")
    print(f"\n  ── Strategy vs B&H ─────────────────────────────────────")
    print(f"  Strategy Return:  {strat_r:+.1f}%")
    if bh_ret > 0.01:
        print(f"  Outperformance:   {strat_r/bh_ret:.1f}×")
    elif bh_ret < -0.01:
        print(f"  Outperformance:   {strat_r-bh_ret:+.1f}pp (B&H was negative)")
    else:
        print(f"  Outperformance:   ∞ (B&H return ≈ 0)")


def print_trade_table(trades, label=""):
    print(f"\n{'='*200}")
    print(f"  LG SWEEP TRADE TABLE: {label}")
    print(f"{'='*200}")
    print(f"{'#':>3} {'Side':<5} {'Res':<5} {'PnL':>7}  "
          f"{'Entry':>14} {'SL':>14} {'TP1':>14} {'TP2':>14} {'Risk':>12}  "
          f"{'LGCLine':>12} {'MaxR':>5} {'Dur':>4}  "
          f"{'Sweep Time':<22} {'Conf Time':<22} {'Exit Time':<22}")
    print("-" * 200)

    for i, t in enumerate(trades):
        sw_ts  = str(t.get("sweep_ts", ""))[:19]
        sg_ts  = str(t.get("signal_ts", ""))[:19]
        ex_ts  = str(t.get("exit_ts", ""))[:19]
        r      = t["result"]
        m      = {"TP": "✅", "P+TP": "✅", "SL": "❌", "P+SL": "⚠️",
                  "BE": "🔄", "P+BE": "💰", "OPEN": "⏳"}.get(r, "?")
        dur    = t.get("duration_bars")
        dur_s  = str(dur) if dur is not None else ""
        tp1    = t.get("tp1", 0)
        tp1_s  = _fmt(tp1) if tp1 and tp1 > 0 else "—"
        sv     = t.get("swept_val")
        sv_s   = _fmt(sv) if sv else ""

        print(f"{i+1:>3} {t['side']:<5} {m} {r:<4} "
              f"{(t.get('pnl_r') or 0):>+6.2f}R  "
              f"{_fmt(t.get('entry')):>14} "
              f"{_fmt(t.get('original_sl', t.get('sl'))):>14} "
              f"{tp1_s:>14} "
              f"{_fmt(t.get('tp')):>14} "
              f"{_fmt(t.get('risk')):>12}  "
              f"{sv_s:>12} "
              f"{t.get('max_r', 0):>5.1f} "
              f"{dur_s:>4}  "
              f"{sw_ts:<22} {sg_ts:<22} {ex_ts:<22}")

    print("-" * 200)
    w   = sum(1 for t in trades if (t.get("pnl_r") or 0) > 0)
    l   = sum(1 for t in trades if (t.get("pnl_r") or 0) < 0)
    b   = sum(1 for t in trades if t.get("pnl_r") is not None
              and t["pnl_r"] == 0.0 and t["result"] != "OPEN")
    net = sum(t.get("pnl_r", 0) for t in trades if t.get("pnl_r") is not None)
    print(f"    Total: {len(trades)} | 💚 {w} positive | ❌ {l} negative | "
          f"🔄 {b} zero | Net: {net:+.1f}R (${net*1000:+,.0f})")


def export_csv(trades, filename):
    rows = []
    for i, t in enumerate(trades):
        rows.append({
            "#":            i + 1,
            "side":         t["side"],
            "case":         t["case"],
            "result":       t["result"],
            "pnl_r":        round(t.get("pnl_r", 0) or 0, 3),
            "entry":        round(t.get("entry", 0), 8),
            "sl":           round(t.get("original_sl", t.get("sl", 0)), 8),
            "tp1":          round(t.get("tp1", 0) or 0, 8),
            "tp":           round(t.get("tp", 0), 8),
            "risk":         round(t.get("risk", 0), 8),
            "lgc_line":     t.get("swept_val", ""),
            "max_r":        round(t.get("max_r", 0), 2),
            "partial_filled": t.get("partial_filled", False),
            "duration_bars":  t.get("duration_bars", ""),
            "lgc_time":     str(t.get("lgc_ts", ""))[:19],
            "sweep_time":   str(t.get("sweep_ts", ""))[:19],
            "conf_time":    str(t.get("signal_ts", ""))[:19],
            "entry_time":   str(t.get("entry_ts", ""))[:19],
            "exit_time":    str(t.get("exit_ts", ""))[:19],
        })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  💾 {len(rows)} trades → {filename}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: BAYESIAN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def run_bayesian_optimizer(pre_pv1, pre_pv2, warmup, capital, risk_pct,
                           n_trials=200, objective_name="return_dd",
                           sym_safe="BTC_USDT", timeframe="30m",
                           date_tag="", plot=False,
                           taker_fee=0.0, maker_fee=0.0,
                           study_name="goat24_opt", storage="",
                           worker=False, _prebuilt_study=None):
    """Bayesian optimization with Optuna (TPE sampler)."""
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pre_map = {1: pre_pv1, 2: pre_pv2}

    if _prebuilt_study is not None:
        study = _prebuilt_study
    else:
        if not worker:
            print(f"\n{'='*80}")
            print(f"  BAYESIAN OPTIMIZER (Optuna TPE) — {n_trials} trials "
                  f"| objective: {objective_name}")
            print(f"{'='*80}")

        def objective(trial):
            rr          = trial.suggest_float("rr", 2.0, 6.0, step=0.5)
            be          = trial.suggest_float("be", 0.0, 3.5, step=0.5)
            partial_r   = trial.suggest_float("partial_r", 0.0, 2.5, step=0.5)
            partial_pct = trial.suggest_categorical("partial_pct", [25, 33, 50])
            pivot_len   = trial.suggest_categorical("pivot_len", [1, 2])

            pre = pre_map[pivot_len]
            trades = run_backtest(pre, rr_ratio=rr, be_trigger_r=be,
                                  warmup=warmup,
                                  partial_tp_r=partial_r,
                                  partial_tp_pct=float(partial_pct),
                                  quiet=True, pivot_len=pivot_len)

            closed = [t for t in trades if t["result"] not in ("OPEN", None)]
            if len(closed) < 5:
                return -999.0

            pnl_r_list = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
            net_r      = sum(pnl_r_list)

            if taker_fee > 0 or maker_fee > 0:
                adj_list   = []
                sim_bal    = capital
                for t in closed:
                    raw_r   = t.get("pnl_r", 0) or 0
                    r_usd   = sim_bal * (risk_pct / 100.0)
                    entry_p = t.get("entry", 0) or 0
                    ha_r    = t.get("risk", 0) or 0
                    if ha_r > 0 and entry_p > 0 and r_usd > 0:
                        pos_usd = r_usd * entry_p / ha_r
                        fee_r   = pos_usd * (taker_fee + maker_fee) / 100.0 / r_usd
                    else:
                        fee_r = 0.0
                    adj_r = raw_r - fee_r
                    adj_list.append(adj_r)
                    sim_bal += r_usd * adj_r
                pnl_r_list = adj_list
                net_r      = sum(pnl_r_list)

            rm_r = calc_risk_metrics_r(closed)
            trial.set_user_attr("sharpe_r",
                                round(rm_r["sharpe_r"], 3) if rm_r["sharpe_r"] is not None else 0.0)
            trial.set_user_attr("sortino_r",
                                round(rm_r["sortino_r"], 3) if rm_r["sortino_r"] is not None else 0.0)

            if objective_name == "net_r":
                return net_r
            elif objective_name in ("return_dd", "calmar"):
                _arr    = np.array(pnl_r_list)
                _cum    = _arr.cumsum()
                max_dd  = abs(min(0.0, min(_cum - np.maximum.accumulate(_cum))))
                return net_r / max_dd if max_dd > 0 else 0.0
            elif objective_name == "sharpe":
                mean_r = np.mean(pnl_r_list)
                std_r  = np.std(pnl_r_list)
                return mean_r / std_r if std_r > 0 else 0.0
            elif objective_name == "profit_factor":
                gw = sum(v for v in pnl_r_list if v > 0)
                gl = abs(sum(v for v in pnl_r_list if v < 0))
                return gw / gl if gl > 0 else 999.0
            return 0.0

        def _progress_cb(study, trial):
            if trial.number % 10 == 0:
                try:
                    bv = study.best_value
                    bp = study.best_params
                    print(f"  Trial {trial.number}/{n_trials} | Best: {bv:.3f} | {bp}")
                except Exception:
                    print(f"  Trial {trial.number}/{n_trials} | (no completed trials yet)")

        def _create_storage(storage_str):
            if not storage_str:
                return None
            if storage_str.startswith("journal:"):
                from optuna.storages import JournalStorage, JournalFileBackend
                path = storage_str[len("journal:"):]
                return JournalStorage(JournalFileBackend(path))
            return storage_str

        optuna_storage = _create_storage(storage)
        use_persistent = optuna_storage is not None
        study = optuna.create_study(
            study_name=study_name if use_persistent else None,
            storage=optuna_storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=use_persistent,
        )
        cbs = [] if worker else [_progress_cb]
        study.optimize(objective, n_trials=n_trials, callbacks=cbs)

        if worker:
            return

    best     = study.best_params
    best_val = study.best_value

    print(f"\n{'='*80}")
    print(f"  🏆 BEST PARAMETERS (objective={objective_name}={best_val:.4f})")
    print(f"{'='*80}")
    for k, v in best.items():
        print(f"    {k}: {v}")

    trials_data = []
    for t in study.trials:
        if t.value is not None:
            row = {"trial": t.number, "value": t.value}
            row.update(t.params)
            row["sharpe_r"]  = t.user_attrs.get("sharpe_r", 0.0)
            row["sortino_r"] = t.user_attrs.get("sortino_r", 0.0)
            trials_data.append(row)
    top10_df = (pd.DataFrame(trials_data)
                .sort_values("value", ascending=False)
                .head(10))
    print(f"\n  TOP 10 TRIALS:")
    print(top10_df.to_string(index=False))

    trials_df = study.trials_dataframe()
    csv_name  = (f"optuna24_{sym_safe}_{timeframe}_{objective_name}"
                 f"_{n_trials}trials{date_tag}.csv")
    trials_df.to_csv(csv_name, index=False)
    print(f"\n  💾 All trials → {csv_name}")

    # Re-run best config
    print(f"\n{'='*80}")
    print(f"  BEST CONFIG — FULL BACKTEST")
    print(f"{'='*80}")
    b_rr          = best["rr"]
    b_be          = best["be"]
    b_partial_r   = best["partial_r"]
    b_partial_pct = float(best["partial_pct"])
    b_pivot_len   = best["pivot_len"]

    best_pre    = pre_map[b_pivot_len]
    best_trades = run_backtest(
        best_pre, rr_ratio=b_rr, be_trigger_r=b_be, warmup=warmup,
        partial_tp_r=b_partial_r, partial_tp_pct=b_partial_pct,
        quiet=False, pivot_len=b_pivot_len,
    )

    p_label       = f" | Partial: {b_partial_pct:.0f}%@{b_partial_r}R" if b_partial_r > 0 else ""
    run_label     = f"{b_be}R BE | RR={b_rr}{p_label} | pv{b_pivot_len}"
    run_label_full = f"{run_label} | ${capital:,.0f} @ {risk_pct}%"

    print_results(best_trades, run_label, rr_ratio=b_rr,
                  partial_tp_r=b_partial_r, partial_tp_pct=b_partial_pct)

    eq_pts, final_bal, peak_bal, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats = \
        compute_equity_curve(best_trades, capital, risk_pct, b_rr,
                             taker_fee=taker_fee, maker_fee=maker_fee,
                             total_bars=best_pre["n"])

    print_equity_curve(eq_pts, capital, final_bal, peak_bal, max_dd_pct, max_dd_usd,
                       monthly_pnl, risk_pct, b_rr, run_label_full)

    if comm_stats["total"] > 0:
        print(f"\n  💸 Commissions (best run):")
        print(f"    Taker fees:  ${comm_stats['total_taker']:,.2f}")
        print(f"    Maker fees:  ${comm_stats['total_maker']:,.2f}")
        print(f"    Total fees:  ${comm_stats['total']:,.2f}")

    bh_metrics = calc_bh_metrics(best_pre["raw_close"], best_pre["timestamps"])
    print_bh_comparison(bh_metrics, capital, final_bal)
    enrich_equity_with_bh(eq_pts, best_pre["raw_close"], best_pre["timestamps"], capital)

    be_tag    = f"be{b_be}".replace(".", "")
    p_tag     = f"_p{b_partial_r}".replace(".", "") if b_partial_r > 0 else ""
    pv_tag    = f"_pv{b_pivot_len}"
    tr_csv    = (f"optuna24_best_{sym_safe}_{timeframe}_{be_tag}"
                 f"{p_tag}{date_tag}{pv_tag}_trades.csv")
    eq_csv    = (f"optuna24_best_{sym_safe}_{timeframe}_{be_tag}"
                 f"{p_tag}{date_tag}{pv_tag}_equity.csv")
    export_csv(best_trades, tr_csv)
    export_equity_csv(eq_pts, eq_csv)

    if plot:
        try:
            from goat_21_plot import load_data as load_plot_data, plot_all
            print("📈 Generating chart...")
            plot_trades_df, plot_equity_df = load_plot_data(tr_csv, eq_csv)
            plot_all(plot_trades_df, plot_equity_df, title=run_label_full,
                     trades_file=tr_csv, bh_metrics=bh_metrics)
        except ImportError as e:
            print(f"⚠️  Chart unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 12: WARMUP COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_warmup(timeframe, override=None):
    """Auto-scale warmup bars to approximately 1 week of 30m context."""
    if override is not None:
        return override
    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }
    mins = tf_minutes.get(timeframe, 5)
    warmup = max(50, int(1440 / mins))  # ~1 calendar day in bars, minimum 50
    return warmup


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LG Line Sweep — Standalone Backtest (Raw Candles)")
    parser.add_argument("--symbol",       default="BTC/USDT:USDT")
    parser.add_argument("--tf",           default="30m",
                        help="Timeframe: 5m, 30m, 4h, 1d (default: 30m)")
    parser.add_argument("--days",         type=int, default=180)
    parser.add_argument("--rr",           type=float, default=3.0,
                        help="Risk-reward ratio (default 3.0)")
    parser.add_argument("--be",           type=float, default=2.0,
                        help="Break-even trigger in R (default 2.0, 0 = disabled)")
    parser.add_argument("--partial",      type=float, default=0.0,
                        help="Partial TP at this R (0 = disabled)")
    parser.add_argument("--partial-pct",  type=float, default=50.0,
                        help="Percentage to close at partial TP (default 50)")
    parser.add_argument("--capital",      type=float, default=50000)
    parser.add_argument("--risk-pct",     type=float, default=2.0,
                        help="Risk %% per trade (default 2.0)")
    parser.add_argument("--warmup",       type=int,   default=None,
                        help="Leading bars to skip (default: auto)")
    parser.add_argument("--pivot-len",    type=int,   default=2, choices=[1, 2],
                        help="Pivot window: 1=1+1+1, 2=2+1+2 (default 2)")
    parser.add_argument("--force",        action="store_true",
                        help="Force re-download of data")
    parser.add_argument("--start",        type=str,   default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",          type=str,   default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--plot",         action="store_true",
                        help="Generate chart after backtest")
    parser.add_argument("--taker-fee",    type=float, default=0.055,
                        help="Taker fee %% (default 0.055)")
    parser.add_argument("--maker-fee",    type=float, default=0.01,
                        help="Maker fee %% (default 0.01)")
    parser.add_argument("--optimize-bayesian", action="store_true",
                        help="Enable Bayesian optimization (Optuna TPE)")
    parser.add_argument("--n-trials",     type=int,   default=200,
                        help="Optuna trials (default 200)")
    parser.add_argument("--objective",    type=str,   default="return_dd",
                        choices=["return_dd", "sharpe", "profit_factor", "net_r", "calmar"],
                        help="Optimization objective (default return_dd)")
    parser.add_argument("--study-name",   type=str,   default="goat24_opt")
    parser.add_argument("--storage",      type=str,   default="",
                        help="Optuna storage (default in-memory; "
                             "'journal:<path>' for file-backed)")
    parser.add_argument("--worker",       action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-bars",   type=str,   default="",
                        help="Comma-separated timestamps for signal diagnostics "
                             "(e.g. '2026-04-01 00:50,2026-04-01 03:05')")
    args = parser.parse_args()

    symbol      = args.symbol
    timeframe   = args.tf
    days        = args.days
    rr          = args.rr
    be          = args.be
    partial_r   = args.partial
    partial_pct = args.partial_pct
    capital     = args.capital
    risk_pct    = args.risk_pct
    taker_fee   = args.taker_fee
    maker_fee   = args.maker_fee
    start_date  = args.start
    end_date    = args.end
    pivot_len   = args.pivot_len

    sym_safe    = symbol.split(":")[0].replace("/", "_")
    warmup      = compute_warmup(timeframe, args.warmup)

    if start_date or end_date:
        date_range_label = f"{start_date or 'start'} → {end_date or 'now'}"
        date_tag         = (f"_{(start_date or '').replace('-', '')}"
                            f"_{(end_date or 'now').replace('-', '')}")
    else:
        date_range_label = f"{days}d"
        date_tag         = f"_{days}d"

    partial_label = f" | Partial: {partial_pct:.0f}%@{partial_r}R" if partial_r > 0 else ""
    mode = "BAYESIAN OPTIMIZER" if args.optimize_bayesian else "BACKTEST"

    print("=" * 60)
    print("LG Line Sweep (Raw Candles) — Backtest")
    print(f"{symbol} — {timeframe} — {date_range_label}")
    print(f"BE: {be}R | RR: {rr}{partial_label}")
    print(f"Capital: ${capital:,.0f} | Risk: {risk_pct}%/trade")
    print(f"Fees: taker={taker_fee}% / maker={maker_fee}%")
    print(f"Mode: {mode} | Pivot: {pivot_len}+1+{pivot_len}")
    print(f"Warmup: {warmup} bars {'(auto)' if args.warmup is None else '(manual)'}")
    print()
    print("  EXECUTION MODEL:")
    print("  1. LGC bar forms on raw candles")
    print("  2. Sweep bar: first wick beyond LGC line + correct color + wick-only")
    print("     + highest high (BEAR) / lowest low (BULL) through conf bar")
    print("  3. Conf bar: closes beyond sweep bar extreme")
    print("  4. Entry: open of bar after conf bar")
    print(f"  5. SL: nearest valid {pivot_len}+1+{pivot_len} raw pivot")
    print(f"  6. TP: entry ± {rr}R × risk")
    if be > 0:
        print(f"  7. BE: when max gain ≥ {be}R, SL moves to entry")
    print("=" * 60)

    df_raw = get_ohlcv(symbol, timeframe, days,
                       force_download=args.force,
                       start_date=start_date,
                       end_date=end_date)
    if df_raw is None or len(df_raw) == 0:
        print(f"\n❌ No data for {symbol} {timeframe} {date_range_label}")
        sys.exit(1)

    actual_start = (df_raw['timestamp'].iloc[0].strftime('%Y-%m-%d')
                    if 'timestamp' in df_raw.columns else '?')
    actual_end   = (df_raw['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                    if 'timestamp' in df_raw.columns else '?')
    print(f"  Data range: {actual_start} → {actual_end} ({len(df_raw)} bars)")

    debug_bars_list = ([ts.strip() for ts in args.debug_bars.split(",") if ts.strip()]
                       if args.debug_bars else None)

    print("\n⚡ Numba JIT warmup (first run compiles, be patient)...")

    if args.optimize_bayesian:
        pre_pv1 = precompute_all(df_raw, pivot_len=1)
        pre_pv2 = precompute_all(df_raw, pivot_len=2)

        run_bayesian_optimizer(
            pre_pv1=pre_pv1, pre_pv2=pre_pv2,
            warmup=warmup, capital=capital, risk_pct=risk_pct,
            n_trials=args.n_trials, objective_name=args.objective,
            sym_safe=sym_safe, timeframe=timeframe, date_tag=date_tag,
            plot=args.plot, taker_fee=taker_fee, maker_fee=maker_fee,
            study_name=args.study_name, storage=args.storage,
            worker=args.worker,
        )
        sys.exit(0)

    # ── SINGLE BACKTEST MODE ────────────────────────────────────────
    pre    = precompute_all(df_raw, pivot_len=pivot_len)
    trades = run_backtest(pre, rr_ratio=rr, be_trigger_r=be, warmup=warmup,
                          partial_tp_r=partial_r, partial_tp_pct=partial_pct,
                          pivot_len=pivot_len, debug_bars=debug_bars_list)

    run_label      = f"{be}R BE | RR={rr}{partial_label} | pv{pivot_len}"
    run_label_full = f"{run_label} | ${capital:,.0f} @ {risk_pct}%"

    print_results(trades, run_label, rr_ratio=rr,
                  partial_tp_r=partial_r, partial_tp_pct=partial_pct)
    print_trade_table(trades, run_label)

    eq_pts, final_bal, peak_bal, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats = \
        compute_equity_curve(trades, capital, risk_pct, rr,
                             taker_fee=taker_fee, maker_fee=maker_fee,
                             total_bars=pre["n"])

    print_equity_curve(eq_pts, capital, final_bal, peak_bal, max_dd_pct, max_dd_usd,
                       monthly_pnl, risk_pct, rr, run_label_full)

    if comm_stats["total"] > 0:
        print(f"\n  💸 Commissions:")
        print(f"    Taker fees:  ${comm_stats['total_taker']:,.2f}")
        print(f"    Maker fees:  ${comm_stats['total_maker']:,.2f}")
        print(f"    Total fees:  ${comm_stats['total']:,.2f}")

    bh_metrics = calc_bh_metrics(pre["raw_close"], pre["timestamps"])
    print_bh_comparison(bh_metrics, capital, final_bal)
    enrich_equity_with_bh(eq_pts, pre["raw_close"], pre["timestamps"], capital)

    pnl_r = sum(t["pnl_r"] for t in trades if t["pnl_r"] is not None)
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Fixed 1R = $1,000:")
    print(f"    Net: {pnl_r:+.1f}R = ${pnl_r*1000:+,.0f}")
    print(f"\n  Compounding {risk_pct}% of ${capital:,.0f}:")
    print(f"    Final: ${final_bal:,.2f} ({(final_bal-capital)/capital*100:+.1f}%)")
    print(f"    Max DD: {max_dd_pct:.1f}% (${max_dd_usd:,.2f})")
    if comm_stats["total"] > 0:
        print(f"    Fees: ${comm_stats['total']:,.2f}")

    print(f"\n📁 Exporting CSVs...")
    be_tag    = f"be{be}".replace(".", "")
    p_tag     = f"_p{partial_r}".replace(".", "") if partial_r > 0 else ""
    pv_tag    = f"_pv{pivot_len}"
    trades_csv = (f"goat24_{sym_safe}_{timeframe}_{be_tag}"
                  f"{p_tag}{date_tag}{pv_tag}_trades.csv")
    equity_csv = (f"goat24_{sym_safe}_{timeframe}_{be_tag}"
                  f"{p_tag}{date_tag}{pv_tag}_equity.csv")
    export_csv(trades, trades_csv)
    export_equity_csv(eq_pts, equity_csv)

    if args.plot:
        try:
            from goat_21_plot import load_data as load_plot_data, plot_all
            print("📈 Generating chart...")
            plot_trades_df, plot_equity_df = load_plot_data(trades_csv, equity_csv)
            plot_all(plot_trades_df, plot_equity_df, title=run_label_full,
                     trades_file=trades_csv, bh_metrics=bh_metrics)
        except ImportError as e:
            print(f"⚠️  Chart unavailable: {e} "
                  "— ensure goat_21_plot.py and dependencies are installed.")

    print(f"\n{'='*60}")
    print(f"  📋 EXECUTION RECAP:")
    print(f"     LGC + sweep (raw) + confirmation (raw)")
    print(f"     SL: nearest valid {pivot_len}+1+{pivot_len} raw pivot")
    print(f"     TP: entry ± {rr}R × |entry − SL|")
    if be > 0:
        print(f"     BE: max gain ≥ {be}R → SL to entry")
    print(f"{'='*60}")
