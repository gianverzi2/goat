"""
GOATv2 Common Functions
Shared across all timeframes (5m, 30m, 4h, 1D, 1W).
All GOAT logic, trade management, and alerting lives here.
"""

import os
import csv
import time
import asyncio
import logging
import pandas as pd
import aiohttp
import ccxt.async_support as ccxt

from function_LG_major_functions import calculate_heikin_ashi, detect_lg, detect_lgc, detect_lgcr

# ─── Shared State ───────────────────────────────────────────────
alerted_goats = set()
alerted_provisional = set()
bull_goat_count = 0
bear_goat_count = 0
active_trades = {}
trade_counter = 0
trade_results = {"wins": 0, "losses": 0}

HA_PIVOT_LENGTH = 2


# ─── Utility ────────────────────────────────────────────────────

def fmt(price):
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


def body_intersects_level(df, idx, level):
    body_low = min(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    body_high = max(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    return body_low <= level <= body_high


def load_symbols(filepath, max_symbols=0):
    symbols = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if row[0] == "symbol":
                    continue
                symbols.append(row[0])
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        logging.info(f"Loaded {len(symbols)} symbols from {filepath}")
    except FileNotFoundError:
        logging.error(f"Symbols file not found: {filepath}. Run fetch_bybit_perps.py first.")
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    return symbols


# ─── Data Fetching ──────────────────────────────────────────────

async def fetch_with_retry(exchange, symbol, timeframe, fetch_limit, max_retries=3, rate_limit_delay=0.2):
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(rate_limit_delay)
            return await exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit, params={"category": "linear"})
        except ccxt.RateLimitExceeded:
            wait_time = (attempt + 1) * 5
            logging.warning(f"Rate limited on {symbol}. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Retry {attempt+1}/{max_retries} for {symbol}: {e}")
            await asyncio.sleep(2 * (attempt + 1))
    return None


# ─── Pattern Detection ──────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["bullish_LG", "bearish_LG", "bullish_LGC", "bearish_LGC",
                "bullish_LGCR", "bearish_LGCR"]:
        if col not in df.columns:
            df[col] = False

    for col in ["bullish_LGC_line", "bearish_LGC_line"]:
        if col not in df.columns:
            df[col] = float('nan')

    lg_levels = detect_lg(df)
    lgc_levels = detect_lgc(df)
    lgcr_levels = detect_lgcr(df, lg_levels)

    for lg in lg_levels:
        df.loc[lg["index"], "bullish_LG" if lg["type"] == "bullish" else "bearish_LG"] = True
    for lgc in lgc_levels:
        col_flag = "bullish_LGC" if lgc["type"] == "bullish" else "bearish_LGC"
        col_line = "bullish_LGC_line" if lgc["type"] == "bullish" else "bearish_LGC_line"
        df.loc[lgc["index"], col_flag] = True
        df.loc[lgc["index"], col_line] = lgc["line_level"]
    for lgcr in lgcr_levels:
        df.loc[lgcr["index"], "bullish_LGCR" if lgcr["type"] == "bullish" else "bearish_LGCR"] = True
    return df


# ─── HA Pivot Detection (for GOAT Case 3) ───────────────────────

def find_ha_pivot_lows(df, start_idx, end_idx):
    pivots = []
    for i in range(start_idx, end_idx):
        if i - HA_PIVOT_LENGTH < 0 or i + HA_PIVOT_LENGTH >= len(df):
            continue
        is_pivot = True
        for offset in range(1, HA_PIVOT_LENGTH + 1):
            if not (df.loc[i, 'HA_Low'] < df.loc[i - offset, 'HA_Low'] and
                    df.loc[i, 'HA_Low'] < df.loc[i + offset, 'HA_Low']):
                is_pivot = False
                break
        if is_pivot:
            pivots.append((i, df.loc[i, 'HA_Low']))
    return pivots


def find_ha_pivot_highs(df, start_idx, end_idx):
    pivots = []
    for i in range(start_idx, end_idx):
        if i - HA_PIVOT_LENGTH < 0 or i + HA_PIVOT_LENGTH >= len(df):
            continue
        is_pivot = True
        for offset in range(1, HA_PIVOT_LENGTH + 1):
            if not (df.loc[i, 'HA_High'] > df.loc[i - offset, 'HA_High'] and
                    df.loc[i, 'HA_High'] > df.loc[i + offset, 'HA_High']):
                is_pivot = False
                break
        if is_pivot:
            pivots.append((i, df.loc[i, 'HA_High']))
    return pivots


# ─── Pivot Detection (for SL/TP on regular candles) ─────────────

def find_pivot_low(df_regular, before_idx, entry_price, pivot_length):
    for i in range(before_idx - 1, pivot_length - 1, -1):
        if i + pivot_length >= len(df_regular):
            continue
        is_pivot = True
        for offset in range(1, pivot_length + 1):
            if not (df_regular.loc[i, 'low'] < df_regular.loc[i - offset, 'low'] and
                    df_regular.loc[i, 'low'] < df_regular.loc[i + offset, 'low']):
                is_pivot = False
                break
        if is_pivot and df_regular.loc[i, 'low'] < entry_price:
            return df_regular.loc[i, 'low'], i
    return None, None


def find_pivot_high(df_regular, before_idx, entry_price, pivot_length):
    for i in range(before_idx - 1, pivot_length - 1, -1):
        if i + pivot_length >= len(df_regular):
            continue
        is_pivot = True
        for offset in range(1, pivot_length + 1):
            if not (df_regular.loc[i, 'high'] > df_regular.loc[i - offset, 'high'] and
                    df_regular.loc[i, 'high'] > df_regular.loc[i + offset, 'high']):
                is_pivot = False
                break
        if is_pivot and df_regular.loc[i, 'high'] > entry_price:
            return df_regular.loc[i, 'high'], i
    return None, None


def calculate_trade_levels(ha_df, trigger_idx, side, rr_ratio):
    entry = ha_df.loc[trigger_idx, 'HA_Close']

    if side == "BULL":
        pivots = find_ha_pivot_lows(ha_df, 0, trigger_idx + 1)
        sl_candidates = [(idx, lvl) for idx, lvl in pivots if lvl < entry]
        if not sl_candidates:
            return None
        pivot_idx, sl = max(sl_candidates, key=lambda x: x[0])
        risk = abs(entry - sl)
        tp = entry + rr_ratio * risk
    else:
        pivots = find_ha_pivot_highs(ha_df, 0, trigger_idx + 1)
        sh_candidates = [(idx, lvl) for idx, lvl in pivots if lvl > entry]
        if not sh_candidates:
            return None
        pivot_idx, sl = max(sh_candidates, key=lambda x: x[0])
        risk = abs(sl - entry)
        tp = entry - rr_ratio * risk

    return {
        "HA_entry": entry,
        "HA_sl": sl,
        "HA_tp": tp,
        "HA_risk": risk,
        "rr": rr_ratio,
        "pivot_idx": pivot_idx,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk,
    }


# ─── Trade ID ───────────────────────────────────────────────────

def generate_trade_id(symbol, side, timestamp):
    global trade_counter
    trade_counter += 1
    sym_short = symbol.replace("/USDT:USDT", "").replace("/", "")
    ts_short = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
    return f"GOATv2_{side}_{sym_short}_{ts_short}_{trade_counter:04d}"


# ─── LGCR selectors ─────────────────────────────────────────────

def _select_prior_lgcr_bear(df, cur):
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bearish_LGCR']:
            if df.loc[i, 'HA_Close'] > trigger_close or df.loc[i, 'HA_High'] > trigger_close:
                prior.append(i)
    if not prior:
        return None
    return min(prior, key=lambda i: min(abs(df.loc[i, 'HA_Close'] - trigger_close),
                                        abs(df.loc[i, 'HA_High'] - trigger_close)))


def _select_prior_lgcr_bull(df, cur):
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bullish_LGCR']:
            if df.loc[i, 'HA_Close'] < trigger_close or df.loc[i, 'HA_Low'] < trigger_close:
                prior.append(i)
    if not prior:
        return None
    return min(prior, key=lambda i: min(abs(df.loc[i, 'HA_Close'] - trigger_close),
                                        abs(df.loc[i, 'HA_Low'] - trigger_close)))


# ─── GOAT Checks ────────────────────────────────────────────────

def check_goat_bear(df, symbol="?"):
    """
    Bear GOAT Case 1/2/3 with consistent body requirement on sweep:
      - Sweep bar LOW must lie inside the body of the reference candle (current LGC+LGCR bar).
    """
    n = len(df)
    if n < 5:
        return False, None, None, None

    cur = n - 1

    # ── DIAGNOSTIC: LGC+LGCR gate ──
    has_lgc = df.loc[cur, 'bearish_LGC']
    has_lgcr = df.loc[cur, 'bearish_LGCR']
    if not (has_lgc and has_lgcr):
        if has_lgc or has_lgcr:
            logging.info(
                f"[DIAG_BEAR_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
                f"bearish_LGC={has_lgc}, bearish_LGCR={has_lgcr} — GATE FAILED (need both True)"
            )
        return False, None, None, None

    logging.info(
        f"[DIAG_BEAR_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
        f"bearish_LGC={has_lgc}, bearish_LGCR={has_lgcr} — GATE PASSED ✅, entering case checks"
    )

    # ─── Case 1: LGCR Sweep (first sweep; lines must stay valid through trigger) ───
    prior_idx = _select_prior_lgcr_bear(df, cur)
    if prior_idx is not None:
        bear_close_line = df.loc[prior_idx, 'HA_Close']
        bear_high_line = df.loc[prior_idx, 'HA_High']

        logging.info(
            f"[DIAG_BEAR_C1] {symbol} bar {cur}: prior LGCR at bar {prior_idx} "
            f"({df.loc[prior_idx,'timestamp']}), "
            f"lines=[close={fmt(bear_close_line)}, high={fmt(bear_high_line)}], "
            f"bars_between={cur - prior_idx - 1}"
        )

        sweep_processed = False

        for k in range(prior_idx + 1, cur):
            # Per-line validity BEFORE sweep
            high_valid = True
            close_valid = True
            for j in range(prior_idx + 1, k):
                c = df.loc[j, 'HA_Close']
                if c > bear_high_line:
                    high_valid = False
                if c < bear_close_line:
                    close_valid = False
                if not high_valid and not close_valid:
                    break

            # Bear: sweep must tag by HA_High
            wick_close_line = df.loc[k, 'HA_High'] >= bear_close_line * 0.999999
            wick_high_line = df.loc[k, 'HA_High'] >= bear_high_line * 0.999999
            swept_close_ok = wick_close_line and close_valid
            swept_high_ok = wick_high_line and high_valid
            if not (swept_close_ok or swept_high_ok):
                continue

            sweep_processed = True
            sweep_level = df.loc[k, 'HA_Low']

            # Reject if a prior body intersects the sweep level
            if any(body_intersects_level(df, j, sweep_level) for j in range(prior_idx + 1, k)):
                break  # first sweep fails; stop

            # Projection inside current LGC+LGCR body
            body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            if not (body_low <= sweep_level <= body_high):
                break  # first sweep fails; stop

            # Post-sweep line validity: all closes after sweep up to trigger must respect lines
            line_valid_after = True
            for j in range(k + 1, cur + 1):
                c = df.loc[j, 'HA_Close']
                if c > bear_high_line or c < bear_close_line:
                    line_valid_after = False
                    break
            if not line_valid_after:
                break  # first sweep fails; stop

            swept_label = "LGCR_HA_high_swept" if swept_high_ok else "LGCR_HA_close_swept"
            swept_value = bear_high_line if swept_high_ok else bear_close_line

            logging.info(
                f"[GOATv2_BEAR_C1] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                f"LGCR prior {prior_idx} ({df.loc[prior_idx,'timestamp']}), "
                f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                f"lines=[close={fmt(bear_close_line)}, high={fmt(bear_high_line)}], "
                f"{swept_label}={fmt(swept_value)}"
            )
            return True, "LGCR", swept_label, swept_value

        if not sweep_processed:
            logging.info(f"[DIAG_BEAR_C1_NO_SWEEP] {symbol}: no sweep candidate found.")
        else:
            logging.info(f"[DIAG_BEAR_C1_NO_SWEEP] {symbol}: first sweep candidate failed.")
    else:
        logging.info(
            f"[DIAG_BEAR_C1] {symbol} bar {cur}: no prior bearish LGCR found "
            f"with HA_Close or HA_High > trigger_close={fmt(df.loc[cur,'HA_Close'])} — "
            f"Case 1 skipped (no candidates)"
        )

    logging.info(f"[GOATv2_BEAR_C1] {symbol}: Case 1 (LGCR sweep) did not trigger")

    # ─── Case 2: LG Line Sweep (first sweep; line must stay valid through trigger) ───
    candidates_lg = []
    cur_price = df.loc[cur, 'HA_Close']
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bearish_LGC']:
            line_level = df.loc[i, 'bearish_LGC_line']
            if pd.isna(line_level):
                continue
            if line_level >= cur_price:
                candidates_lg.append((i, line_level))

    if candidates_lg:
        lgc_idx, line_level = min(candidates_lg, key=lambda x: abs(x[1] - cur_price))
        logging.info(
            f"[DIAG_BEAR_C2] {symbol} bar {cur}: best LG line candidate at bar {lgc_idx} "
            f"({df.loc[lgc_idx,'timestamp']}), line_level={fmt(line_level)}, "
            f"total_candidates={len(candidates_lg)}"
        )

        sweep_processed = False

        for k in range(lgc_idx + 1, cur):
            wicked_to_line = df.loc[k, 'HA_High'] >= line_level
            if not wicked_to_line:
                continue

            # Line validity before sweep
            line_valid = True
            for j in range(lgc_idx + 1, k):
                if df.loc[j, 'HA_Close'] > line_level:
                    line_valid = False
                    break
            if not line_valid:
                sweep_processed = True
                break  # first sweep fails

            sweep_processed = True
            sweep_level = df.loc[k, 'HA_Low']

            # Prior body intersect sweep level
            if any(body_intersects_level(df, j, sweep_level) for j in range(lgc_idx + 1, k)):
                break

            # Projection inside current LGC+LGCR body
            body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            if not (body_low <= sweep_level <= body_high):
                break

            # Post-sweep line validity up to trigger
            line_valid_after = True
            for j in range(k + 1, cur + 1):
                if df.loc[j, 'HA_Close'] > line_level:
                    line_valid_after = False
                    break
            if not line_valid_after:
                break

            logging.info(f"[GOATv2_BEAR_C2] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                         f"LG line from {lgc_idx} ({df.loc[lgc_idx,'timestamp']}), "
                         f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                         f"HA_Low={fmt(df.loc[k,'HA_Low'])}, HA_High={fmt(df.loc[k,'HA_High'])}, "
                         f"LG_line_swept={fmt(line_level)}")
            return True, "LG_LINE", "LG_line_swept", line_level

        if not sweep_processed:
            logging.info(f"[DIAG_BEAR_C2_NO_SWEEP] {symbol}: no sweep candidate found.")
        else:
            logging.info(f"[DIAG_BEAR_C2_NO_SWEEP] {symbol}: first sweep candidate failed.")
    else:
        logging.info(
            f"[DIAG_BEAR_C2] {symbol} bar {cur}: no bearish_LGC with line_level >= "
            f"cur_price={fmt(cur_price)} — Case 2 skipped (no candidates)"
        )

    logging.info(f"[GOATv2_BEAR_C2] {symbol}: Case 2 (LG line) did not trigger")

    # ─── Case 3: Pivot Sweep (single level, validity only before sweep, take the LAST sweep) ───
    pivots = find_ha_pivot_highs(df, 0, cur)
    pivot_candidates = [(idx, level) for idx, level in pivots if level >= cur_price]

    if not pivot_candidates:
        logging.info(
            f"[DIAG_BEAR_C3] {symbol} bar {cur}: no HA pivot highs >= "
            f"cur_price={fmt(cur_price)}, total_pivots_found={len(pivots)} — "
            f"Case 3 skipped (no candidates)"
        )
        return False, None, None, None

    pivot_idx, pivot_level = min(pivot_candidates, key=lambda x: abs(x[1] - cur_price))
    logging.info(
        f"[DIAG_BEAR_C3] {symbol} bar {cur}: best pivot high at bar {pivot_idx} "
        f"({df.loc[pivot_idx,'timestamp']}), pivot_level={fmt(pivot_level)}, "
        f"total_pivot_candidates={len(pivot_candidates)}"
    )

    last_sweep = None  # store the last (most recent) sweep candidate after the pivot
    for k in range(pivot_idx + 1, cur):
        wicked_to_pivot = df.loc[k, 'HA_High'] >= pivot_level * 0.999999
        if not wicked_to_pivot:
            continue

        # Pivot validity only on bars BEFORE the sweep bar
        pivot_valid = True
        for j in range(pivot_idx + 1, k):
            if df.loc[j, 'HA_Close'] > pivot_level:
                pivot_valid = False
                break
        if not pivot_valid:
            continue

        sweep_level = df.loc[k, 'HA_Low']

        if any(body_intersects_level(df, j, sweep_level) for j in range(pivot_idx + 1, k)):
            continue

        body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        if not (body_low <= sweep_level <= body_high):
            continue

        # keep the last (most recent) valid sweep candidate
        last_sweep = (k, sweep_level)

    if last_sweep:
        k, sweep_level = last_sweep
        logging.info(f"[GOATv2_BEAR_C3] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                     f"pivot from {pivot_idx} ({df.loc[pivot_idx,'timestamp']}), "
                     f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                     f"pivot_high_swept={fmt(pivot_level)}")
        return True, "PIVOT", "pivot_high_swept", pivot_level

    logging.info(f"[DIAG_BEAR_C3_NO_SWEEP] {symbol}: no sweep candidate passed after pivot")
    return False, None, None, None


def check_goat_bull(df, symbol="?"):
    """
    Bull GOAT Case 1/2/3 with consistent body requirement on sweep:
      - Sweep bar HIGH must lie inside the body of the reference candle (current LGC+LGCR bar).
    """
    n = len(df)
    if n < 5:
        return False, None, None, None

    cur = n - 1

    # ── DIAGNOSTIC: LGC+LGCR gate ──
    has_lgc = df.loc[cur, 'bullish_LGC']
    has_lgcr = df.loc[cur, 'bullish_LGCR']
    if not (has_lgc and has_lgcr):
        if has_lgc or has_lgcr:
            logging.info(
                f"[DIAG_BULL_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
                f"bullish_LGC={has_lgc}, bullish_LGCR={has_lgcr} — GATE FAILED (need both True)"
            )
        return False, None, None, None

    logging.info(
        f"[DIAG_BULL_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
        f"bullish_LGC={has_lgc}, bullish_LGCR={has_lgcr} — GATE PASSED ✅, entering case checks"
    )

    # ─── Case 1: LGCR Sweep (first sweep; lines must stay valid through trigger) ───
    prior_idx = _select_prior_lgcr_bull(df, cur)
    if prior_idx is not None:
        bull_low_line = df.loc[prior_idx, 'HA_Low']
        bull_close_line = df.loc[prior_idx, 'HA_Close']

        logging.info(
            f"[DIAG_BULL_C1] {symbol} bar {cur}: prior LGCR at bar {prior_idx} "
            f"({df.loc[prior_idx,'timestamp']}), "
            f"lines=[low={fmt(bull_low_line)}, close={fmt(bull_close_line)}], "
            f"bars_between={cur - prior_idx - 1}"
        )

        sweep_processed = False

        for k in range(prior_idx + 1, cur):
            # Per-line validity BEFORE sweep
            low_valid = True
            close_valid = True
            for j in range(prior_idx + 1, k):
                c = df.loc[j, 'HA_Close']
                if c < bull_low_line:
                    low_valid = False
                if c > bull_close_line:
                    close_valid = False
                if not low_valid and not close_valid:
                    break

            # Bull: sweep must tag by HA_Low
            wick_low_line = df.loc[k, 'HA_Low'] <= bull_low_line * 1.000001
            wick_close_line = df.loc[k, 'HA_Low'] <= bull_close_line * 1.000001
            swept_low_ok = wick_low_line and low_valid
            swept_close_ok = wick_close_line and close_valid
            if not (swept_low_ok or swept_close_ok):
                continue

            sweep_processed = True
            sweep_level = df.loc[k, 'HA_High']

            # Reject if a prior body intersects the sweep level
            if any(body_intersects_level(df, j, sweep_level) for j in range(prior_idx + 1, k)):
                break  # first sweep fails; stop

            # Projection inside current LGC+LGCR body
            body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            if not (body_low <= sweep_level <= body_high):
                break  # first sweep fails; stop

            # Post-sweep line validity up to trigger
            line_valid_after = True
            for j in range(k + 1, cur + 1):
                c = df.loc[j, 'HA_Close']
                if c < bull_low_line or c > bull_close_line:
                    line_valid_after = False
                    break
            if not line_valid_after:
                break

            swept_label = "LGCR_HA_low_swept" if swept_low_ok else "LGCR_HA_close_swept"
            swept_value = bull_low_line if swept_low_ok else bull_close_line

            logging.info(
                f"[GOATv2_BULL_C1] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                f"LGCR prior {prior_idx} ({df.loc[prior_idx,'timestamp']}), "
                f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                f"lines=[low={fmt(bull_low_line)}, close={fmt(bull_close_line)}], "
                f"{swept_label}={fmt(swept_value)}"
            )
            return True, "LGCR", swept_label, swept_value

        if not sweep_processed:
            logging.info(f"[DIAG_BULL_C1_NO_SWEEP] {symbol}: no sweep candidate found.")
        else:
            logging.info(f"[DIAG_BULL_C1_NO_SWEEP] {symbol}: first sweep candidate failed.")
    else:
        logging.info(
            f"[DIAG_BULL_C1] {symbol} bar {cur}: no prior bullish LGCR found "
            f"with HA_Close or HA_Low < trigger_close={fmt(df.loc[cur,'HA_Close'])} — "
            f"Case 1 skipped (no candidates)"
        )

    logging.info(f"[GOATv2_BULL_C1] {symbol}: Case 1 (LGCR sweep) did not trigger")

    # ─── Case 2: LG Line Sweep (first sweep; line must stay valid through trigger) ───
    candidates_lg = []
    cur_price = df.loc[cur, 'HA_Close']
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bullish_LGC']:
            line_level = df.loc[i, 'bullish_LGC_line']
            if pd.isna(line_level):
                continue
            if line_level <= cur_price:
                candidates_lg.append((i, line_level))

    if candidates_lg:
        lgc_idx, line_level = min(candidates_lg, key=lambda x: abs(cur_price - x[1]))
        logging.info(
            f"[DIAG_BULL_C2] {symbol} bar {cur}: best LG line candidate at bar {lgc_idx} "
            f"({df.loc[lgc_idx,'timestamp']}), line_level={fmt(line_level)}, "
            f"total_candidates={len(candidates_lg)}"
        )

        sweep_processed = False

        for k in range(lgc_idx + 1, cur):
            wicked_to_line = df.loc[k, 'HA_Low'] <= line_level
            if not wicked_to_line:
                continue

            # Line validity before sweep
            line_valid = True
            for j in range(lgc_idx + 1, k):
                if df.loc[j, 'HA_Close'] < line_level:
                    line_valid = False
                    break
            if not line_valid:
                sweep_processed = True
                break  # first sweep fails

            sweep_processed = True
            sweep_level = df.loc[k, 'HA_High']

            # Prior body intersect sweep level
            if any(body_intersects_level(df, j, sweep_level) for j in range(lgc_idx + 1, k)):
                break

            # Projection inside current LGC+LGCR body
            body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            if not (body_low <= sweep_level <= body_high):
                break

            # Post-sweep line validity up to trigger
            line_valid_after = True
            for j in range(k + 1, cur + 1):
                if df.loc[j, 'HA_Close'] < line_level:
                    line_valid_after = False
                    break
            if not line_valid_after:
                break

            logging.info(f"[GOATv2_BULL_C2] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                         f"LG line from {lgc_idx} ({df.loc[lgc_idx,'timestamp']}), "
                         f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                         f"HA_Low={fmt(df.loc[k,'HA_Low'])}, HA_High={fmt(df.loc[k,'HA_High'])}, "
                         f"LG_line_swept={fmt(line_level)}")
            return True, "LG_LINE", "LG_line_swept", line_level

        if not sweep_processed:
            logging.info(f"[DIAG_BULL_C2_NO_SWEEP] {symbol}: no sweep candidate found.")
        else:
            logging.info(f"[DIAG_BULL_C2_NO_SWEEP] {symbol}: first sweep candidate failed.")
    else:
        logging.info(
            f"[DIAG_BULL_C2] {symbol} bar {cur}: no bullish_LGC with line_level <= "
            f"cur_price={fmt(cur_price)} — Case 2 skipped (no candidates)"
        )

    logging.info(f"[GOATv2_BULL_C2] {symbol}: Case 2 (LG line) did not trigger")

    # ─── Case 3: Pivot Sweep (single level, validity only before sweep, take the LAST sweep) ───
    pivots = find_ha_pivot_lows(df, 0, cur)
    pivot_candidates = [(idx, level) for idx, level in pivots if level <= cur_price]

    if not pivot_candidates:
        logging.info(
            f"[DIAG_BULL_C3] {symbol} bar {cur}: no HA pivot lows <= "
            f"cur_price={fmt(cur_price)}, total_pivots_found={len(pivots)} — "
            f"Case 3 skipped (no candidates)"
        )
        return False, None, None, None

    pivot_idx, pivot_level = min(pivot_candidates, key=lambda x: abs(cur_price - x[1]))
    logging.info(
        f"[DIAG_BULL_C3] {symbol} bar {cur}: best pivot low at bar {pivot_idx} "
        f"({df.loc[pivot_idx,'timestamp']}), pivot_level={fmt(pivot_level)}, "
        f"total_pivot_candidates={len(pivot_candidates)}"
    )

    last_sweep = None  # store the last (most recent) sweep candidate after the pivot
    for k in range(pivot_idx + 1, cur):
        wicked_to_pivot = df.loc[k, 'HA_Low'] <= pivot_level * 1.000001
        if not wicked_to_pivot:
            continue

        pivot_valid = True
        for j in range(pivot_idx + 1, k):
            if df.loc[j, 'HA_Close'] < pivot_level:
                pivot_valid = False
                break
        if not pivot_valid:
            continue

        sweep_level = df.loc[k, 'HA_High']

        if any(body_intersects_level(df, j, sweep_level) for j in range(pivot_idx + 1, k)):
            continue

        body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        if not (body_low <= sweep_level <= body_high):
            continue

        # keep the last (most recent) valid sweep candidate
        last_sweep = (k, sweep_level)

    if last_sweep:
        k, sweep_level = last_sweep
        logging.info(f"[GOATv2_BULL_C3] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                     f"pivot from {pivot_idx} ({df.loc[pivot_idx,'timestamp']}), "
                     f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                     f"pivot_low_swept={fmt(pivot_level)}")
        return True, "PIVOT", "pivot_low_swept", pivot_level

    logging.info(f"[DIAG_BULL_C3_NO_SWEEP] {symbol}: no sweep candidate passed after pivot")
    return False, None, None, None


# ─── Trade Monitoring ────────────────────────────────────────────

async def check_active_trades(exchange, cfg):
    if not active_trades:
        return

    symbols_to_check = set(t["symbol"] for t in active_trades.values())
    latest_candles = {}

    for symbol in symbols_to_check:
        try:
            ohlcv = await fetch_with_retry(exchange, symbol, cfg["timeframe"],
                                           cfg["fetch_limit"], cfg["max_retries"], cfg["rate_limit_delay"])
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                latest_candles[symbol] = df
        except Exception as e:
            logging.error(f"Error fetching candles for trade check on {symbol}: {e}")

    trades_to_close = []

    for trade_id, trade in active_trades.items():
        symbol = trade["symbol"]
        if symbol not in latest_candles:
            continue

        df = latest_candles[symbol]
        entry_ts = trade["entry_timestamp"]

        new_candles = df[df["timestamp"] > entry_ts]
        if new_candles.empty:
            continue

        side = trade["side"]
        entry = trade["entry"]
        sl = trade["sl"]
        tp = trade["tp"]
        risk = trade["risk"]
        current_close = df.iloc[-1]["close"]

        hit_tp = False
        hit_sl = False
        hit_ts = None
        hit_price = None
        max_favorable = 0.0

        for _, candle in new_candles.iterrows():
            if side == "BULL":
                if risk > 0:
                    candle_best = (candle["high"] - entry) / risk
                    max_favorable = max(max_favorable, candle_best)
                if candle["low"] <= sl:
                    hit_sl = True
                    hit_ts = candle["timestamp"]
                    hit_price = sl
                    break
                if candle["high"] >= tp:
                    hit_tp = True
                    hit_ts = candle["timestamp"]
                    hit_price = tp
                    break
            else:
                if risk > 0:
                    candle_best = (entry - candle["low"]) / risk
                    max_favorable = max(max_favorable, candle_best)
                if candle["high"] >= sl:
                    hit_sl = True
                    hit_ts = candle["timestamp"]
                    hit_price = sl
                    break
                if candle["low"] <= tp:
                    hit_tp = True
                    hit_ts = candle["timestamp"]
                    hit_price = tp
                    break

        if hit_tp or hit_sl:
            result = "TP ✅" if hit_tp else "SL ❌"
            pnl = trade["risk"] * cfg["rr_ratio"] if hit_tp else -trade["risk"]
            pnl_sign = "+" if hit_tp else ""

            if hit_tp:
                trade_results["wins"] += 1
            else:
                trade_results["losses"] += 1

            total = trade_results["wins"] + trade_results["losses"]
            win_rate = (trade_results["wins"] / total * 100) if total > 0 else 0

            emoji = "🎯" if hit_tp else "🛑"
            msg = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{emoji} **Trade Closed: {result}**\n"
                f"🆔 `{trade_id}`\n"
                f"📊 {trade['symbol']} {side} {cfg['timeframe']}\n"
                f"💰 Entry: {fmt(entry)} | Hit: {fmt(hit_price)} | PnL: {pnl_sign}{fmt(pnl)}\n"
            )

            if hit_sl:
                msg += f"📐 Max R before reversal: {max_favorable:.1f}R\n"

            msg += (
                f"\n📈 **Stats:** {trade_results['wins']}W / {trade_results['losses']}L "
                f"(Win Rate: {win_rate:.1f}%)"
            )

            logging.info(msg.replace("**", "").replace("`", ""))
            await send_discord_notification(msg, cfg["webhook_url"])
            trades_to_close.append(trade_id)
        else:
            if side == "BULL":
                unrealized = current_close - entry
                dist_to_tp = tp - current_close
                dist_to_sl = current_close - sl
                current_r = unrealized / risk if risk > 0 else 0
            else:
                unrealized = entry - current_close
                dist_to_tp = current_close - tp
                dist_to_sl = sl - current_close
                current_r = unrealized / risk if risk > 0 else 0

            logging.info(f"[TRADE OPEN] {trade_id} | {symbol} {side} {cfg['timeframe']} | "
                         f"Entry={fmt(entry)} Current={fmt(current_close)} | "
                         f"PnL={unrealized:+.6f} ({current_r:+.1f}R) | "
                         f"MaxR={max_favorable:.1f}R | "
                         f"Dist→TP={fmt(dist_to_tp)} Dist→SL={fmt(dist_to_sl)}")

    for trade_id in trades_to_close:
        del active_trades[trade_id]


# ─── Discord ────────────────────────────────────────────────────

DASHBOARD_URL = "http://localhost:8080"

_WEBHOOK_TF_MAP = {
    "1472893353048801340": "m5",
    "1471942081177190647": "m30",
    "1472619891168247950": "h4",
    "1472894304249974881": "d",
}

def _guess_tf(webhook_url: str) -> str:
    for key, tf in _WEBHOOK_TF_MAP.items():
        if key in webhook_url:
            return tf
    return "m30"

async def send_discord_notification(message: str, webhook_url: str):
    if not webhook_url:
        logging.error("Webhook URL not set.")
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json={"content": message}) as resp:
                if resp.status in (200, 204):
                    logging.info("Notification sent")
                else:
                    logging.error(f"Discord error status {resp.status}")
            try:
                tf = _guess_tf(webhook_url)
                await session.post(
                    f"{DASHBOARD_URL}/webhook?tf={tf}",
                    data=message,
                    headers={"Content-Type": "text/plain"},
                )
                logging.info(f"Dashboard webhook sent (tf={tf})")
            except Exception:
                pass
    except Exception as e:
        logging.error(f"Error sending notification: {e}")


# ─── Symbol Analysis ────────────────────────────────────────────

async def analyze_symbol(exchange, symbol, df_regular_map, cfg, run_type="final"):
    t0 = time.perf_counter()

    ohlcv = await fetch_with_retry(exchange, symbol, cfg["timeframe"],
                                   cfg["fetch_limit"], cfg["max_retries"], cfg["rate_limit_delay"])
    if not ohlcv:
        logging.info(f"{symbol}: no OHLCV fetched")
        return False
    if len(ohlcv) < 6:
        logging.info(f"{symbol}: only {len(ohlcv)} bars fetched; need >=6 (5 closed + 1 forming)")
        return False

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    last_candle_open = df.iloc[-1]["timestamp"]
    last_candle_close = last_candle_open + pd.Timedelta(minutes=cfg["tf_minutes"])
    forming = last_candle_close > pd.Timestamp.now(tz="UTC")

    drop_forming = (run_type == "final") or (not cfg.get("allow_forming_on_provisional", False))
    if forming and drop_forming:
        logging.debug(f"{symbol}: dropping forming candle {last_candle_open} (run_type={run_type})")
        df = df.iloc[:-1]

    if len(df) < 5:
        logging.info(f"{symbol}: not enough closed bars after forming-filter (len={len(df)})")
        return False

    df_regular = df.copy().reset_index(drop=True)
    df_regular_map[symbol] = df_regular

    ha_df = calculate_heikin_ashi(df)
    if ha_df.empty or len(ha_df) < 5:
        logging.info(f"{symbol}: HA DF empty or <5 rows")
        return False

    ha_df = ha_df.reset_index(drop=True)
    ha_df = detect_patterns(ha_df)

    tail = ha_df.tail(5).copy()
    debug_cols = ["timestamp", "HA_Open", "HA_High", "HA_Low", "HA_Close",
                  "bullish_LGC", "bearish_LGC", "bullish_LGCR", "bearish_LGCR"]
    logging.info(f"[DBG TABLE] {symbol}\n{tail[debug_cols].to_string(index=False)}")

    recent = ha_df.tail(10)
    logging.info(
        f"[FLAGS] {symbol} last10 "
        f"LGC(bull/bear)={recent['bullish_LGC'].sum()}/{recent['bearish_LGC'].sum()} "
        f"LGCR(bull/bear)={recent['bullish_LGCR'].sum()}/{recent['bearish_LGCR'].sum()}"
    )

    global bull_goat_count, bear_goat_count
    last_idx = len(ha_df) - 1
    LOOKBACK_BARS = cfg.get("lookback_bars", 3)
    goat_found = False

    lookback_start = max(last_idx - LOOKBACK_BARS, 0)
    bear_lgc_in_window = sum(1 for i in range(lookback_start, last_idx + 1) if ha_df.loc[i, 'bearish_LGC'])
    bear_lgcr_in_window = sum(1 for i in range(lookback_start, last_idx + 1) if ha_df.loc[i, 'bearish_LGCR'])
    bull_lgc_in_window = sum(1 for i in range(lookback_start, last_idx + 1) if ha_df.loc[i, 'bullish_LGC'])
    bull_lgcr_in_window = sum(1 for i in range(lookback_start, last_idx + 1) if ha_df.loc[i, 'bullish_LGCR'])
    bear_both_in_window = sum(1 for i in range(lookback_start, last_idx + 1)
                              if ha_df.loc[i, 'bearish_LGC'] and ha_df.loc[i, 'bearish_LGCR'])
    bull_both_in_window = sum(1 for i in range(lookback_start, last_idx + 1)
                              if ha_df.loc[i, 'bullish_LGC'] and ha_df.loc[i, 'bullish_LGCR'])
    logging.info(
        f"[DIAG_LOOKBACK] {symbol} bars {lookback_start}-{last_idx} (window={LOOKBACK_BARS}): "
        f"BEAR LGC={bear_lgc_in_window} LGCR={bear_lgcr_in_window} BOTH={bear_both_in_window} | "
        f"BULL LGC={bull_lgc_in_window} LGCR={bull_lgcr_in_window} BOTH={bull_both_in_window}"
    )

    for check_idx in range(last_idx, max(last_idx - LOOKBACK_BARS, 0) - 1, -1):
        is_bear = ha_df.loc[check_idx, "bearish_LGC"] and ha_df.loc[check_idx, "bearish_LGCR"]
        is_bull = ha_df.loc[check_idx, "bullish_LGC"] and ha_df.loc[check_idx, "bullish_LGCR"]

        if not (is_bear or is_bull):
            continue

        logging.info(f"[CANDIDATE] {symbol} bar {check_idx} {ha_df.loc[check_idx,'timestamp']} "
                     f"is_bear={is_bear} is_bull={is_bull}")

        bar_ts = ha_df.loc[check_idx, "timestamp"]
        side_label = "BEAR" if is_bear else "BULL"
        if check_idx < last_idx:
            logging.info(f"[RECHECK] {symbol}: {side_label} LGC+LGCR on bar {check_idx} "
                         f"({bar_ts}), re-running GOATv2 check...")

        sub_df = ha_df.iloc[:check_idx + 1].reset_index(drop=True)

        goat_bear, bear_case, bear_swept_label, bear_swept_value = check_goat_bear(sub_df, symbol=symbol) if is_bear else (False, None, None, None)
        goat_bull, bull_case, bull_swept_label, bull_swept_value = check_goat_bull(sub_df, symbol=symbol) if is_bull else (False, None, None, None)

        triggered = goat_bear or goat_bull

        if not triggered:
            logging.info(
                f"[DIAG_ALL_CASES_FAILED] {symbol} bar {check_idx} ({bar_ts}): "
                f"LGC+LGCR gate passed (bear={is_bear}, bull={is_bull}) but ALL 3 cases "
                f"failed to produce a GOAT signal — candidates were found and rejected"
            )
            continue

        ts_str = ha_df.loc[check_idx, "timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        side = "BEAR" if goat_bear else "BULL"
        case_label = bear_case if goat_bear else bull_case
        swept_label = bear_swept_label if goat_bear else bull_swept_label
        swept_value = bear_swept_value if goat_bear else bull_swept_value
        alert_key = (symbol, ts_str, side)

        if run_type == "provisional":
            if alert_key in alerted_goats or alert_key in alerted_provisional:
                logging.info(f"{symbol}: provisional GOATv2_{side} already alerted for {ts_str}, skipping")
                goat_found = True
                continue

            alerted_provisional.add(alert_key)
            emoji = "🟠"
            msg = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{emoji} **PROVISIONAL — {cfg['exchange_name']} {cfg['timeframe']} {symbol}**\n"
                f"⏰ {ts_str} (forming bar)\n"
                f"📊 GOATv2_{side}_{case_label}\n"
            )
            if swept_label and swept_value is not None:
                msg += f"🔖 {swept_label}: {fmt(swept_value)}\n"
            logging.info(msg.replace("**", "").replace("`", ""))
            await send_discord_notification(msg, cfg["webhook_url"])
            goat_found = True
            continue

        if alert_key in alerted_goats:
            logging.info(f"{symbol}: GOATv2_{side} already alerted for {ts_str}, skipping")
            goat_found = True
            continue

        alerted_goats.add(alert_key)
        alerted_provisional.add(alert_key)

        trade = calculate_trade_levels(ha_df, check_idx, side, cfg["rr_ratio"])

        if side == "BULL":
            bull_goat_count += 1
        else:
            bear_goat_count += 1

        trade_id = generate_trade_id(symbol, side, ts_str)

        emoji = "🟢" if side == "BULL" else "🔴"
        msg = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} **{cfg['exchange_name']} {cfg['timeframe']} {symbol}**\n"
            f"🆔 `{trade_id}`\n"
            f"⏰ {ts_str}\n"
            f"📊 GOATv2_{side}_{case_label}\n"
        )
        if swept_label and swept_value is not None:
            msg += f"🔖 {swept_label}: {fmt(swept_value)}\n"

        if trade:
            msg += (
                f"\n💰 HA Entry: {fmt(trade['HA_entry'])} | "
                f"HA SL: {fmt(trade['HA_sl'])} | "
                f"HA TP({cfg['rr_ratio']}R): {fmt(trade['HA_tp'])} | "
                f"HA Risk: {fmt(trade['HA_risk'])}\n"
            )

            active_trades[trade_id] = {
                "symbol": symbol,
                "side": side,
                "entry": trade["HA_entry"],
                "sl": trade["HA_sl"],
                "tp": trade["HA_tp"],
                "risk": trade["HA_risk"],
                "entry_timestamp": ha_df.loc[check_idx, "timestamp"],
                "trade_id": trade_id,
            }
            logging.info(f"[TRADE REGISTERED] {trade_id} | {symbol} {side} {cfg['timeframe']} | "
                         f"Entry={fmt(trade['entry'])} SL={fmt(trade['sl'])} TP={fmt(trade['tp'])}")
        else:
            msg += "\n⚠️ Could not find pivot for SL calculation\n"

        msg += (
            f"\n🟢 Bull GOATv2 signals: {bull_goat_count}\n"
            f"🔴 Bear GOATv2 signals: {bear_goat_count}"
        )

        if active_trades:
            msg += f"\n📋 Active trades: {len(active_trades)}"

        logging.info(msg.replace("**", "").replace("`", ""))
        await send_discord_notification(msg, cfg["webhook_url"])
        goat_found = True

    if not goat_found:
        logging.info(f"{symbol}: no GOATv2 in last {LOOKBACK_BARS} bars")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logging.info(f"[SCAN TIME] {symbol} run_type={run_type} took {elapsed_ms:.1f} ms")
    return goat_found