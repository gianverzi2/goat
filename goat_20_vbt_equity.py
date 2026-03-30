"""
GOATv2 Numba Backtest — Configurable BE + Equity Curve + Case Selection + Partial TP
Signals from HA. SL from HA pivot (2+1+2). TP from HA close ± RR × HA_risk.
Entry on NEXT raw candle open (realistic execution).
SL/TP checked against raw high/low.
Warmup auto-scales with timeframe.

Usage:
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --partial 1.5
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --partial 1.5 --partial-pct 50
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --rr 3 --partial 2.0
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --optimize-cases
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --optimize-partial

  # 75% off at 1.5R — lock +1.125R, 25% rides
  python3 goat_20_vbt_equity.py --symbol SOL/USDT:USDT --tf 5m --days 180 --partial 1.5 --partial-pct 75

  # With AO filter (only long if AO<0, only short if AO>0)
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --filters ao

  # With MTF HA-LGCR direction filter (stateful HTF bias, no lookahead)
  # Valid pairs: 5m/30m, 30m/4h, 4h/1d, 1d/1w  (LTF must equal --tf)
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --filters mtf_lgcr --mtf-lgcr 5m/30m
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 30m --days 180 --filters mtf_lgcr --mtf-lgcr 30m/4h

  # With Donchian Channel touch-based bias filter (block signals against channel-touch bias)
  python3 goat_20_vbt_equity.py --symbol BTC/USDT:USDT --tf 5m --days 180 --filters donchian --donchian-period 200

  # Parallel Bayesian optimization (6 workers, journal file storage — no Postgres needed)
  python3 goat_20_vbt_equity.py --symbol ONDO/USDT:USDT --tf 5m \\
      --start 2025-11-01 --end 2026-03-01 \\
      --optimize-bayesian --n-trials 100 --n-jobs 6 \\
      --storage journal:optuna_journal.log --study-name my_study --plot
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
# SECTION 1: HEIKIN ASHI + PATTERNS
# ═══════════════════════════════════════════════════════════════════

def calculate_heikin_ashi(df):
    ha = df.copy()
    o = ha['open'].values.astype(np.float64)
    h = ha['high'].values.astype(np.float64)
    lo = ha['low'].values.astype(np.float64)
    c = ha['close'].values.astype(np.float64)

    ha_close = (o + h + lo + c) / 4.0
    ha_open = np.empty(len(ha))
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

    ha_body_max = np.maximum(ha_open, ha_close)
    ha_body_min = np.minimum(ha_open, ha_close)
    ha_high = np.maximum(h, ha_body_max)
    ha_low = np.minimum(lo, ha_body_min)

    ha['HA_Close'] = ha_close
    ha['HA_Open'] = ha_open
    ha['HA_High'] = ha_high
    ha['HA_Low'] = ha_low
    return ha


@njit
def detect_all_patterns_numba(n, ha_open, ha_close, ha_high, ha_low):
    bull_lg = np.zeros(n, dtype=np.bool_)
    bear_lg = np.zeros(n, dtype=np.bool_)
    bull_lgc = np.zeros(n, dtype=np.bool_)
    bear_lgc = np.zeros(n, dtype=np.bool_)
    bull_lgc_line = np.full(n, np.nan)
    bear_lgc_line = np.full(n, np.nan)
    bull_lgcr = np.zeros(n, dtype=np.bool_)
    bear_lgcr = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        if ha_high[i-1] < ha_close[i]:
            bull_lg[i] = True
        if ha_low[i-1] > ha_close[i]:
            bear_lg[i] = True

    for i in range(2, n):
        if ha_high[i-2] < ha_low[i] and ha_close[i-1] > ha_open[i-1]:
            bull_lgc[i-1] = True
            bull_lgc_line[i-1] = ha_high[i-2]
        if ha_low[i-2] > ha_high[i] and ha_close[i-1] < ha_open[i-1]:
            bear_lgc[i-1] = True
            bear_lgc_line[i-1] = ha_low[i-2]

    ref_bull = -1.0
    ref_bear = -1.0
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

    return (bull_lg, bear_lg, bull_lgc, bear_lgc,
            bull_lgc_line, bear_lgc_line, bull_lgcr, bear_lgcr)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: PIVOTS (2+1+2 = 5 candles on HA)
# ═══════════════════════════════════════════════════════════════════

HA_PIVOT_LENGTH = 2

@njit
def precompute_pivots_numba(ha_low, ha_high, n, pivot_len):
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
            if not (ha_low[i] < ha_low[i - off] and ha_low[i] < ha_low[i + off]):
                is_low = False
                break
        if is_low:
            pli[nl] = i
            plv[nl] = ha_low[i]
            nl += 1
        is_high = True
        for off in range(1, pivot_len + 1):
            if not (ha_high[i] > ha_high[i - off] and ha_high[i] > ha_high[i + off]):
                is_high = False
                break
        if is_high:
            phi[nh] = i
            phv[nh] = ha_high[i]
            nh += 1
    return pli[:nl], plv[:nl], phi[:nh], phv[:nh]


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: SPARSE TABLES
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
            sp[k, :limit] = np.maximum(sp[k-1, :limit], sp[k-1, half:half+limit])
        else:
            sp[k, :limit] = np.minimum(sp[k-1, :limit], sp[k-1, half:half+limit])
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
# SECTION 4: NUMBA-JIT CASE CHECKS (FIRST-SWEEP RULE)
# ═══════════════════════════════════════════════════════════════════

@njit
def _any_body_intersects(body_low, body_high, l, r, level):
    for i in range(l, r + 1):
        if body_low[i] <= level <= body_high[i]:
            return True
    return False


@njit
def check_case1_jit(ha_close, ha_open, ha_high, ha_low,
                    body_low, body_high,
                    lgcr_flags, cur, is_bear,
                    sp_max, sp_min):
    """Case 1: LGCR Sweep. First-sweep rule per line."""
    trigger_close = ha_close[cur]
    body_lo_cur = body_low[cur]
    body_hi_cur = body_high[cur]

    n_prior = 0
    for i in range(cur):
        if lgcr_flags[i]:
            n_prior += 1
    if n_prior == 0:
        return False, 0.0

    prior_idx_arr = np.empty(n_prior, dtype=np.int64)
    prior_dist = np.empty(n_prior, dtype=np.float64)
    j = 0
    for i in range(cur):
        if lgcr_flags[i]:
            if is_bear:
                ok = ha_close[i] > trigger_close or ha_high[i] > trigger_close
                d = min(abs(ha_close[i] - trigger_close), abs(ha_high[i] - trigger_close))
            else:
                ok = ha_close[i] < trigger_close or ha_low[i] < trigger_close
                d = min(abs(ha_close[i] - trigger_close), abs(ha_low[i] - trigger_close))
            if ok:
                prior_idx_arr[j] = i
                prior_dist[j] = d
                j += 1
    n_valid = j
    if n_valid == 0:
        return False, 0.0

    for a in range(n_valid):
        for b in range(a + 1, n_valid):
            if prior_dist[b] < prior_dist[a]:
                prior_dist[a], prior_dist[b] = prior_dist[b], prior_dist[a]
                prior_idx_arr[a], prior_idx_arr[b] = prior_idx_arr[b], prior_idx_arr[a]

    for pi in range(n_valid):
        prior_idx = prior_idx_arr[pi]
        line1 = ha_close[prior_idx]
        if is_bear:
            line2 = ha_high[prior_idx]
        else:
            line2 = ha_low[prior_idx]

        line1_valid = True
        line2_valid = True
        if prior_idx + 1 <= cur - 1:
            if is_bear:
                if sparse_max(sp_max, prior_idx + 1, cur - 1) > line1:
                    line1_valid = False
                if sparse_max(sp_max, prior_idx + 1, cur - 1) > line2:
                    line2_valid = False
            else:
                if sparse_min(sp_min, prior_idx + 1, cur - 1) < line1:
                    line1_valid = False
                if sparse_min(sp_min, prior_idx + 1, cur - 1) < line2:
                    line2_valid = False

        if not line1_valid and not line2_valid:
            continue

        line1_already_swept = False
        line2_already_swept = False

        for k in range(prior_idx + 1, cur):
            l1_ok = line1_valid and not line1_already_swept
            l2_ok = line2_valid and not line2_already_swept
            if k > prior_idx + 1:
                if l1_ok:
                    if is_bear:
                        if sparse_max(sp_max, prior_idx + 1, k - 1) > line1:
                            l1_ok = False
                    else:
                        if sparse_min(sp_min, prior_idx + 1, k - 1) < line1:
                            l1_ok = False
                if l2_ok:
                    if is_bear:
                        if sparse_max(sp_max, prior_idx + 1, k - 1) > line2:
                            l2_ok = False
                    else:
                        if sparse_min(sp_min, prior_idx + 1, k - 1) < line2:
                            l2_ok = False

            if is_bear:
                wick = ha_high[k]
                sw2 = (wick >= line2 * 0.999999) and l2_ok
                sw1 = (wick >= line1 * 0.999999) and l1_ok
            else:
                wick = ha_low[k]
                sw2 = (wick <= line2 * 1.000001) and l2_ok
                sw1 = (wick <= line1 * 1.000001) and l1_ok

            if not (sw1 or sw2):
                continue

            # ── Skip bars with no actual wick (flat-top/bottom body) ──
            # Do NOT mark lines as swept — bar didn't truly spike through.
            if is_bear:
                body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
                if wick <= body_top:
                    continue
            else:
                body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
                if wick >= body_bot:
                    continue

            if sw2:
                line2_already_swept = True
            if sw1:
                line1_already_swept = True

            sweep_close = ha_close[k]
            swept_ref = line2 if sw2 else line1
            if is_bear and sweep_close > swept_ref:
                continue
            if not is_bear and sweep_close < swept_ref:
                continue

            if is_bear:
                sweep_level = ha_low[k]
            else:
                sweep_level = ha_high[k]

            if not (body_lo_cur <= sweep_level <= body_hi_cur):
                continue

            if k + 1 <= cur - 1:
                if _any_body_intersects(body_low, body_high, k + 1, cur - 1, sweep_level):
                    continue

            if sw2:
                return True, line2
            else:
                return True, line1

    return False, 0.0


@njit
def check_case2_jit(ha_close, ha_open, ha_high, ha_low,
                    body_low, body_high,
                    lgc_flags, lgc_lines, cur, is_bear,
                    sp_max, sp_min):
    """Case 2: LG Line Sweep. First-sweep rule."""
    cur_price = ha_close[cur]
    body_lo_cur = body_low[cur]
    body_hi_cur = body_high[cur]

    best_idx = -1
    best_dist = 1e18
    for i in range(cur):
        if not lgc_flags[i]:
            continue
        ll = lgc_lines[i]
        if np.isnan(ll):
            continue
        if is_bear and ll < cur_price:
            continue
        if not is_bear and ll > cur_price:
            continue
        d = abs(ll - cur_price)
        if d < best_dist:
            best_dist = d
            best_idx = i

    if best_idx < 0:
        return False, 0.0

    line_level = lgc_lines[best_idx]

    if best_idx + 1 <= cur - 1:
        if is_bear:
            if sparse_max(sp_max, best_idx + 1, cur - 1) > line_level:
                return False, 0.0
        else:
            if sparse_min(sp_min, best_idx + 1, cur - 1) < line_level:
                return False, 0.0

    for k in range(best_idx + 1, cur):
        if is_bear:
            wick = ha_high[k]
            if wick < line_level:
                continue
        else:
            wick = ha_low[k]
            if wick > line_level:
                continue

        # ── Skip bars with no actual wick — level consumed, no valid sweep ──
        if is_bear:
            body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
            if wick <= body_top:
                return False, 0.0
        else:
            body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
            if wick >= body_bot:
                return False, 0.0

        sweep_close = ha_close[k]
        if is_bear and sweep_close > line_level:
            return False, 0.0
        if not is_bear and sweep_close < line_level:
            return False, 0.0

        if k > best_idx + 1:
            if is_bear:
                if sparse_max(sp_max, best_idx + 1, k - 1) > line_level:
                    return False, 0.0
            else:
                if sparse_min(sp_min, best_idx + 1, k - 1) < line_level:
                    return False, 0.0

        if is_bear:
            sweep_level = ha_low[k]
        else:
            sweep_level = ha_high[k]

        if not (body_lo_cur <= sweep_level <= body_hi_cur):
            return False, 0.0

        if k + 1 <= cur - 1:
            if _any_body_intersects(body_low, body_high, k + 1, cur - 1, sweep_level):
                return False, 0.0

        return True, line_level

    return False, 0.0


@njit
def check_case3_jit(ha_close, ha_open, ha_high, ha_low,
                    body_low, body_high,
                    piv_idx_arr, piv_lvl_arr, n_pivots, cur, is_bear,
                    sp_max, sp_min, pivot_len):
    """Case 3: Pivot Sweep. First-sweep rule."""
    cur_price = ha_close[cur]
    body_lo_cur = body_low[cur]
    body_hi_cur = body_high[cur]

    best_pi = -1
    best_dist = 1e18
    for p in range(n_pivots):
        pi = piv_idx_arr[p]
        pl = piv_lvl_arr[p]
        if pi >= cur:
            continue
        if is_bear and pl < cur_price:
            continue
        if not is_bear and pl > cur_price:
            continue
        d = abs(pl - cur_price)
        if d < best_dist:
            best_dist = d
            best_pi = p

    if best_pi < 0:
        return False, 0.0

    piv_idx = piv_idx_arr[best_pi]
    piv_level = piv_lvl_arr[best_pi]
    n = ha_high.shape[0]

    for k in range(piv_idx + 1, cur):
        if is_bear:
            wick = ha_high[k]
            if wick < piv_level * 0.999999:
                continue
        else:
            wick = ha_low[k]
            if wick > piv_level * 1.000001:
                continue

        # ── Skip bars with no actual wick — level consumed, no valid sweep ──
        if is_bear:
            body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
            if wick <= body_top:
                return False, 0.0
        else:
            body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
            if wick >= body_bot:
                return False, 0.0

        sweep_close = ha_close[k]
        if is_bear and sweep_close > piv_level:
            return False, 0.0
        if not is_bear and sweep_close < piv_level:
            return False, 0.0

        # ── Sweep bar must itself be a pivot (parameterized by pivot_len) ──
        if not (k >= pivot_len and k + pivot_len < n):
            return False, 0.0
        is_pivot = True
        for off in range(1, pivot_len + 1):
            if is_bear:
                if not (ha_high[k] > ha_high[k - off] and ha_high[k] > ha_high[k + off]):
                    is_pivot = False
                    break
            else:
                if not (ha_low[k] < ha_low[k - off] and ha_low[k] < ha_low[k + off]):
                    is_pivot = False
                    break
        if not is_pivot:
            return False, 0.0

        if k > piv_idx + 1:
            if is_bear:
                if sparse_max(sp_max, piv_idx + 1, k - 1) > piv_level:
                    return False, 0.0
            else:
                if sparse_min(sp_min, piv_idx + 1, k - 1) < piv_level:
                    return False, 0.0

        if is_bear:
            sweep_level = ha_low[k]
        else:
            sweep_level = ha_high[k]

        if not (body_lo_cur <= sweep_level <= body_hi_cur):
            return False, 0.0

        if k + 1 <= cur - 1:
            if _any_body_intersects(body_low, body_high, k + 1, cur - 1, sweep_level):
                return False, 0.0

        return True, piv_level

    return False, 0.0


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: SIGNAL SCANNER (Numba)
# ═══════════════════════════════════════════════════════════════════

@njit
def scan_all_signals(n, warmup, lookback,
                     ha_close, ha_open, ha_high, ha_low, body_low, body_high,
                     bull_lgc, bear_lgc, bull_lgcr, bear_lgcr,
                     bull_lgc_line, bear_lgc_line,
                     piv_low_idx, piv_low_lvl, n_piv_low,
                     piv_high_idx, piv_high_lvl, n_piv_high,
                     sp_max, sp_min,
                     enable_c1, enable_c2, enable_c3, pivot_len):
    max_sigs = n * 2
    sig_bar = np.empty(max_sigs, dtype=np.int64)
    sig_trigger = np.empty(max_sigs, dtype=np.int64)
    sig_side = np.empty(max_sigs, dtype=np.int64)
    sig_case = np.empty(max_sigs, dtype=np.int64)
    sig_swept = np.empty(max_sigs, dtype=np.float64)
    ns = 0

    seen = np.zeros(n * 2, dtype=np.bool_)

    for bar in range(warmup, n):
        lo = bar - lookback
        if lo < 0:
            lo = 0

        for ci in range(bar, lo - 1, -1):
            for side_val in range(2):
                if side_val == 0:
                    if not (bear_lgc[ci] and bear_lgcr[ci]):
                        continue
                    is_bear = True
                else:
                    if not (bull_lgc[ci] and bull_lgcr[ci]):
                        continue
                    is_bear = False

                key = ci * 2 + side_val
                if seen[key]:
                    continue

                if enable_c1:
                    lgcr_f = bear_lgcr if is_bear else bull_lgcr
                    ok, sv = check_case1_jit(ha_close, ha_open, ha_high, ha_low,
                                             body_low, body_high,
                                             lgcr_f, ci, is_bear, sp_max, sp_min)
                    if ok:
                        seen[key] = True
                        sig_bar[ns] = bar
                        sig_trigger[ns] = ci
                        sig_side[ns] = side_val
                        sig_case[ns] = 1
                        sig_swept[ns] = sv
                        ns += 1
                        continue

                if enable_c2:
                    lgc_f = bear_lgc if is_bear else bull_lgc
                    lgc_l = bear_lgc_line if is_bear else bull_lgc_line
                    ok, sv = check_case2_jit(ha_close, ha_open, ha_high, ha_low,
                                             body_low, body_high,
                                             lgc_f, lgc_l, ci, is_bear, sp_max, sp_min)
                    if ok:
                        seen[key] = True
                        sig_bar[ns] = bar
                        sig_trigger[ns] = ci
                        sig_side[ns] = side_val
                        sig_case[ns] = 2
                        sig_swept[ns] = sv
                        ns += 1
                        continue

                if enable_c3:
                    if is_bear:
                        pi = piv_high_idx
                        pl = piv_high_lvl
                        np_ = n_piv_high
                    else:
                        pi = piv_low_idx
                        pl = piv_low_lvl
                        np_ = n_piv_low

                    ok, sv = check_case3_jit(ha_close, ha_open, ha_high, ha_low,
                                             body_low, body_high,
                                             pi, pl, np_, ci, is_bear, sp_max, sp_min,
                                             pivot_len)
                    if ok:
                        seen[key] = True
                        sig_bar[ns] = bar
                        sig_trigger[ns] = ci
                        sig_side[ns] = side_val
                        sig_case[ns] = 3
                        sig_swept[ns] = sv
                        ns += 1

    return sig_bar[:ns], sig_trigger[:ns], sig_side[:ns], sig_case[:ns], sig_swept[:ns], ns


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: TRADE LEVELS
# ═══════════════════════════════════════════════════════════════════

@njit
def calc_sl_and_ha_risk(ha_close, ha_low_arr, ha_high_arr,
                        piv_low_idx, piv_low_lvl, n_pl,
                        piv_high_idx, piv_high_lvl, n_ph,
                        trigger_idx, is_bear, signal_bar):
    ha_entry = ha_close[trigger_idx]
    if not is_bear:
        best_idx = -1
        best_bar = -1
        for p in range(n_pl):
            if piv_low_idx[p] <= signal_bar and piv_low_lvl[p] < ha_entry:
                pbar = piv_low_idx[p]
                plvl = piv_low_lvl[p]
                valid = True
                for j in range(pbar + 1, signal_bar + 1):
                    if ha_low_arr[j] < plvl:
                        valid = False
                        break
                if valid and (best_idx < 0 or pbar > best_bar):
                    best_idx = p
                    best_bar = pbar
        if best_idx < 0:
            return False, 0.0, 0.0, 0.0
        sl = piv_low_lvl[best_idx]
    else:
        best_idx = -1
        best_bar = -1
        for p in range(n_ph):
            if piv_high_idx[p] <= signal_bar and piv_high_lvl[p] > ha_entry:
                pbar = piv_high_idx[p]
                plvl = piv_high_lvl[p]
                valid = True
                for j in range(pbar + 1, signal_bar + 1):
                    if ha_high_arr[j] > plvl:
                        valid = False
                        break
                if valid and (best_idx < 0 or pbar > best_bar):
                    best_idx = p
                    best_bar = pbar
        if best_idx < 0:
            return False, 0.0, 0.0, 0.0
        sl = piv_high_lvl[best_idx]

    ha_risk = abs(ha_entry - sl)
    if ha_risk == 0:
        return False, 0.0, 0.0, 0.0
    return True, sl, ha_entry, ha_risk


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: MTF LGCR BIAS + PRE-COMPUTE + BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def _tf_to_resample_offset(tf_str):
    """Convert a timeframe string (e.g. '30m', '4h') to a pandas resample offset."""
    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
        "1d": 1440, "1w": 10080,
    }
    if tf_str not in tf_minutes:
        raise ValueError(f"Unsupported timeframe for MTF LGCR resampling: '{tf_str}'. "
                         f"Valid: {', '.join(sorted(tf_minutes))}")
    return f"{tf_minutes[tf_str]}min"


def compute_mtf_lgcr_bias(df_raw, htf_str):
    """Compute stateful MTF HA-LGCR bias aligned to the LTF bar array.

    Returns an int8 numpy array of length len(df_raw) with values:
      +1  bull bias (last closed HTF bar was a bullish LGCR, or bias persists)
      -1  bear bias (last closed HTF bar was a bearish LGCR, or bias persists)
       0  neutral  (no LGCR seen yet on HTF)

    No lookahead: each LTF bar receives the bias of the most recent HTF bar
    that has *already fully closed* before that LTF bar starts.
    """
    offset = _tf_to_resample_offset(htf_str)

    df = df_raw.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns:
        agg['volume'] = 'sum'
    htf = df.resample(offset, closed='left', label='left').agg(agg).dropna(
        subset=['open', 'close'])

    n_htf = len(htf)
    if n_htf < 3:
        return np.zeros(len(df_raw), dtype=np.int8)

    # Compute Heikin-Ashi on HTF candles
    ha_htf = calculate_heikin_ashi(htf)
    ha_open_htf = ha_htf['HA_Open'].values.astype(np.float64)
    ha_close_htf = ha_htf['HA_Close'].values.astype(np.float64)
    ha_high_htf = ha_htf['HA_High'].values.astype(np.float64)
    ha_low_htf = ha_htf['HA_Low'].values.astype(np.float64)

    # Detect LGCR patterns on HTF HA (reuse existing numba function)
    (_, _, _, _, _, _, bull_lgcr_htf, bear_lgcr_htf) = detect_all_patterns_numba(
        n_htf, ha_open_htf, ha_close_htf, ha_high_htf, ha_low_htf)

    # Build stateful bias: persists on non-LGCR bars (forward-fill logic)
    htf_bias = np.zeros(n_htf, dtype=np.int8)
    current_bias = 0
    for i in range(n_htf):
        if bull_lgcr_htf[i]:
            current_bias = 1
        elif bear_lgcr_htf[i]:
            current_bias = -1
        htf_bias[i] = current_bias

    # Shift by 1: LTF bars inside HTF bar i use the bias of HTF bar i-1 (already closed).
    # htf_bias_shifted[0] = 0 (no prior closed HTF bar yet).
    htf_bias_shifted = np.zeros(n_htf, dtype=np.int8)
    htf_bias_shifted[1:] = htf_bias[:-1]

    # Map HTF bias to every LTF bar.
    # For a LTF bar at time t: searchsorted(htf_ts, t, 'right') - 1 gives the index
    # of the HTF bar whose period contains t (htf_ts[i] <= t < htf_ts[i+1]).
    if 'timestamp' in df_raw.columns:
        ltf_ts = pd.to_datetime(df_raw['timestamp']).values.astype('int64')
    else:
        ltf_ts = pd.to_datetime(df_raw.index).values.astype('int64')
    htf_ts = htf.index.values.astype('int64')

    indices = np.searchsorted(htf_ts, ltf_ts, side='right') - 1
    # Clip handles LTF bars that precede the first HTF bar (indices == -1 → 0).
    # htf_bias_shifted[0] is always 0 (neutral) by construction, which is correct:
    # there is no prior closed HTF bar before the dataset start.
    np.clip(indices, 0, n_htf - 1, out=indices)

    return htf_bias_shifted[indices].astype(np.int8)


def precompute_all(df_raw, pivot_len=2, mtf_lgcr_htf=None, donchian_period=None):
    # donchian_period=None means skip Donchian computation (filter not active).
    # Pass the desired lookback (e.g. 200) to enable it, matching --donchian-period.
    t0 = time_module.perf_counter()
    print("  Computing HA + patterns + pivots + sparse tables...")

    ha = calculate_heikin_ashi(df_raw).reset_index(drop=True)
    n = len(ha)

    ha_close = ha['HA_Close'].values.astype(np.float64)
    ha_open = ha['HA_Open'].values.astype(np.float64)
    ha_high = ha['HA_High'].values.astype(np.float64)
    ha_low = ha['HA_Low'].values.astype(np.float64)
    raw_open = df_raw['open'].values.astype(np.float64)
    raw_high = df_raw['high'].values.astype(np.float64)
    raw_low = df_raw['low'].values.astype(np.float64)
    raw_close = df_raw['close'].values.astype(np.float64)
    timestamps = ha['timestamp'].values

    (bull_lg, bear_lg, bull_lgc, bear_lgc,
     bull_lgc_line, bear_lgc_line, bull_lgcr, bear_lgcr) = \
        detect_all_patterns_numba(n, ha_open, ha_close, ha_high, ha_low)

    piv_low_idx, piv_low_lvl, piv_high_idx, piv_high_lvl = \
        precompute_pivots_numba(ha_low, ha_high, n, pivot_len)

    sp_max = build_sparse_table(ha_close, "max")
    sp_min = build_sparse_table(ha_close, "min")
    body_low = np.minimum(ha_open, ha_close)
    body_high = np.maximum(ha_open, ha_close)

    # AO = SMA(median_price, 5) - SMA(median_price, 34), computed on raw candles
    raw_median = (df_raw['high'].values.astype(np.float64) + df_raw['low'].values.astype(np.float64)) / 2.0
    ao_fast = pd.Series(raw_median).rolling(5).mean().values
    ao_slow = pd.Series(raw_median).rolling(34).mean().values
    ao = ao_fast - ao_slow
    ao = np.nan_to_num(ao, nan=0.0)

    # MTF LGCR bias (optional — computed only when filter is active)
    mtf_lgcr_bias = None
    if mtf_lgcr_htf:
        print(f"    Computing MTF LGCR bias ({mtf_lgcr_htf} HTF)...")
        mtf_lgcr_bias = compute_mtf_lgcr_bias(df_raw, mtf_lgcr_htf)
        n_bull = int(np.sum(mtf_lgcr_bias == 1))
        n_bear = int(np.sum(mtf_lgcr_bias == -1))
        n_neut = int(np.sum(mtf_lgcr_bias == 0))
        total_b = len(mtf_lgcr_bias)
        print(f"    MTF LGCR bias: bull={n_bull} ({100*n_bull/total_b:.1f}%), "
              f"bear={n_bear} ({100*n_bear/total_b:.1f}%), "
              f"neutral={n_neut} ({100*n_neut/total_b:.1f}%)")

    # Donchian Channel bias (touch-based, on raw candles — computed only when filter is active)
    dc_bias = None
    if donchian_period is not None:
        raw_high_s = pd.Series(raw_high)
        raw_low_s = pd.Series(raw_low)
        dc_upper = raw_high_s.rolling(donchian_period).max().values
        dc_lower = raw_low_s.rolling(donchian_period).min().values
        dc_bias = np.zeros(n, dtype=np.int8)
        current_bias = 0
        for i in range(n):
            if not np.isnan(dc_lower[i]) and raw_low[i] <= dc_lower[i]:
                current_bias = 1   # lower band touch → bull bias
            if not np.isnan(dc_upper[i]) and raw_high[i] >= dc_upper[i]:
                current_bias = -1  # upper band touch → bear bias
            dc_bias[i] = current_bias
        n_bull_dc = int(np.sum(dc_bias == 1))
        n_bear_dc = int(np.sum(dc_bias == -1))
        n_neut_dc = int(np.sum(dc_bias == 0))
        print(f"    Donchian bias (period={donchian_period}): bull={n_bull_dc} ({100*n_bull_dc/n:.1f}%), "
              f"bear={n_bear_dc} ({100*n_bear_dc/n:.1f}%), "
              f"neutral={n_neut_dc} ({100*n_neut_dc/n:.1f}%)")

    el = time_module.perf_counter() - t0
    print(f"    {len(piv_low_idx)} pivot lows, {len(piv_high_idx)} pivot highs")
    print(f"    Prep done in {el:.1f}s")

    return {
        "n": n, "ha_close": ha_close, "ha_open": ha_open,
        "ha_high": ha_high, "ha_low": ha_low,
        "raw_open": raw_open, "raw_high": raw_high,
        "raw_low": raw_low, "raw_close": raw_close,
        "timestamps": timestamps,
        "bull_lgc": bull_lgc, "bear_lgc": bear_lgc,
        "bull_lgcr": bull_lgcr, "bear_lgcr": bear_lgcr,
        "bull_lgc_line": bull_lgc_line, "bear_lgc_line": bear_lgc_line,
        "piv_low_idx": piv_low_idx, "piv_low_lvl": piv_low_lvl,
        "piv_high_idx": piv_high_idx, "piv_high_lvl": piv_high_lvl,
        "sp_max": sp_max, "sp_min": sp_min,
        "body_low": body_low, "body_high": body_high,
        "ao": ao,
        "mtf_lgcr_bias": mtf_lgcr_bias,
        "donchian_bias": dc_bias,
    }


def run_backtest(pre, rr_ratio=3, be_trigger_r=2.0, warmup=300,
                 enable_c1=True, enable_c2=True, enable_c3=True,
                 partial_tp_r=0.0, partial_tp_pct=50.0, quiet=False, pivot_len=2,
                 active_filters=None):
    """
    Run backtest with optional partial TP.

    partial_tp_r:   R-multiple at which to take partial profit (0 = disabled)
    partial_tp_pct: Percentage of position to close at partial TP (default 50%)

    PnL accounting with partial TP (50% example):
      - Full TP:     0.5 * partial_tp_r + 0.5 * rr_ratio
      - Partial + BE: 0.5 * partial_tp_r + 0.0
      - Straight SL: -1.0R (full loss, partial not triggered)
      - Partial + SL: 0.5 * partial_tp_r - 0.5 (SL on remaining half)
    """
    t0 = time_module.perf_counter()

    if active_filters is None:
        active_filters = set()
    filter_stats = {}

    cases_str = f"{'C1' if enable_c1 else ''}{'C2' if enable_c2 else ''}{'C3' if enable_c3 else ''}"
    be_label = f"{be_trigger_r}R BE" if be_trigger_r else "NO BE"
    partial_label = f" | Partial: {partial_tp_pct:.0f}%@{partial_tp_r}R" if partial_tp_r > 0 else ""
    filter_label = f" | Filters: {','.join(sorted(active_filters)).upper()}" if active_filters else ""

    if not quiet:
        print(f"\n{'='*60}")
        print(f"BACKTEST: {be_label} | RR={rr_ratio}{partial_label} | Cases: {cases_str}{filter_label}")
        print(f"Entry: next raw candle open | SL/TP hit: raw high/low")
        print(f"Warmup: {warmup} bars")
        print(f"{'='*60}")

    use_partial = partial_tp_r > 0
    partial_frac = partial_tp_pct / 100.0  # e.g. 0.5 for 50%
    remain_frac = 1.0 - partial_frac       # e.g. 0.5

    n = pre["n"]
    ha_close = pre["ha_close"]
    ha_open = pre["ha_open"]
    ha_high = pre["ha_high"]
    ha_low = pre["ha_low"]
    raw_open = pre["raw_open"]
    raw_high = pre["raw_high"]
    raw_low = pre["raw_low"]
    timestamps = pre["timestamps"]
    body_low = pre["body_low"]
    body_high = pre["body_high"]
    sp_max = pre["sp_max"]
    sp_min = pre["sp_min"]
    piv_low_idx = pre["piv_low_idx"]
    piv_low_lvl = pre["piv_low_lvl"]
    piv_high_idx = pre["piv_high_idx"]
    piv_high_lvl = pre["piv_high_lvl"]

    if not quiet:
        print("  Scanning signals (Numba JIT)...")
    t1 = time_module.perf_counter()
    sig_bar, sig_trig, sig_side, sig_case, sig_swept, ns = scan_all_signals(
        n, warmup, 5,
        ha_close, ha_open, ha_high, ha_low, body_low, body_high,
        pre["bull_lgc"], pre["bear_lgc"], pre["bull_lgcr"], pre["bear_lgcr"],
        pre["bull_lgc_line"], pre["bear_lgc_line"],
        piv_low_idx, piv_low_lvl, len(piv_low_idx),
        piv_high_idx, piv_high_lvl, len(piv_high_idx),
        sp_max, sp_min,
        enable_c1, enable_c2, enable_c3, pivot_len
    )
    t_scan = time_module.perf_counter() - t1
    if not quiet:
        print(f"    Found {ns} raw signals in {t_scan:.1f}s")

    trades = []
    active = []
    n_pl = len(piv_low_idx)
    n_ph = len(piv_high_idx)
    skipped_invalid = 0

    pending = []

    sig_by_bar = {}
    for s in range(ns):
        b = sig_bar[s]
        if b not in sig_by_bar:
            sig_by_bar[b] = []
        sig_by_bar[b].append(s)

    if not quiet:
        print("  Simulating trades...")
    for bar in range(warmup, n):
        # ── Execute pending entries ──
        new_pending = []
        for pend in pending:
            ci = pend["trigger"]
            is_bear = pend["is_bear"]
            side_str = pend["side_str"]
            case_str = pend["case_str"]
            swept = pend["swept"]
            signal_bar = pend["signal_bar"]

            valid_sl, sl, ha_entry, ha_risk = calc_sl_and_ha_risk(
                ha_close, ha_low, ha_high,
                piv_low_idx, piv_low_lvl, n_pl,
                piv_high_idx, piv_high_lvl, n_ph,
                ci, is_bear, signal_bar)
            if not valid_sl:
                skipped_invalid += 1
                continue

            entry = raw_open[bar]

            if not is_bear and sl >= entry:
                skipped_invalid += 1
                continue
            if is_bear and sl <= entry:
                skipped_invalid += 1
                continue

            if not is_bear:
                tp = ha_entry + rr_ratio * ha_risk
                tp1 = ha_entry + partial_tp_r * ha_risk if use_partial else 0.0
            else:
                tp = ha_entry - rr_ratio * ha_risk
                tp1 = ha_entry - partial_tp_r * ha_risk if use_partial else 0.0

            trade = {
                "side": side_str, "case": case_str,
                "entry": entry, "sl": sl, "tp": tp,
                "tp1": tp1,
                "risk": ha_risk,
                "ha_entry": ha_entry,
                "original_sl": sl,
                "entry_bar": bar, "entry_ts": timestamps[bar],
                "signal_bar": signal_bar, "signal_ts": timestamps[signal_bar],
                "trigger_bar": ci, "trigger_ts": timestamps[ci],
                "swept_val": swept,
                "max_r": 0.0, "be_active": False,
                "partial_filled": False,
                "partial_pnl_r": 0.0,
                "result": None, "pnl_r": None,
                "exit_bar": None, "exit_ts": None, "duration_bars": None,
            }
            active.append(trade)
        pending = new_pending

        # ── Check active trades ──
        to_close = []
        for t in active:
            c_hi = raw_high[bar]
            c_lo = raw_low[bar]
            risk = t["risk"]
            cur_entry = t["entry"]

            if t["side"] == "BULL":
                if risk > 0:
                    t["max_r"] = max(t["max_r"], (c_hi - cur_entry) / risk)

                # ── Partial TP check (before BE/SL/TP) ──
                if use_partial and not t["partial_filled"] and c_hi >= t["tp1"]:
                    t["partial_filled"] = True
                    t["partial_pnl_r"] = partial_frac * partial_tp_r

                # ── BE trigger ──
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = cur_entry

                # ── SL hit ──
                if c_lo <= t["sl"]:
                    if t["partial_filled"]:
                        # Remaining portion hits SL
                        if t["be_active"] and t["sl"] == cur_entry:
                            remaining_pnl = 0.0  # BE on remainder
                            r = "P+BE"
                        else:
                            remaining_pnl = -remain_frac  # SL on remainder
                            r = "P+SL"
                        p = t["partial_pnl_r"] + remaining_pnl
                    else:
                        r = "BE" if (t["be_active"] and t["sl"] == cur_entry) else "SL"
                        p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))

                # ── Full TP hit ──
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
                    t["max_r"] = max(t["max_r"], (cur_entry - c_lo) / risk)

                # ── Partial TP check ──
                if use_partial and not t["partial_filled"] and c_lo <= t["tp1"]:
                    t["partial_filled"] = True
                    t["partial_pnl_r"] = partial_frac * partial_tp_r

                # ── BE trigger ──
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = cur_entry

                # ── SL hit ──
                if c_hi >= t["sl"]:
                    if t["partial_filled"]:
                        if t["be_active"] and t["sl"] == cur_entry:
                            remaining_pnl = 0.0
                            r = "P+BE"
                        else:
                            remaining_pnl = -remain_frac
                            r = "P+SL"
                        p = t["partial_pnl_r"] + remaining_pnl
                    else:
                        r = "BE" if (t["be_active"] and t["sl"] == cur_entry) else "SL"
                        p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))

                # ── Full TP hit ──
                elif c_lo <= t["tp"]:
                    if t["partial_filled"]:
                        p = t["partial_pnl_r"] + remain_frac * rr_ratio
                        r = "P+TP"
                    else:
                        p = float(rr_ratio)
                        r = "TP"
                    to_close.append((t, r, p))

        for t, result, pnl_r in to_close:
            t["result"] = result
            t["pnl_r"] = pnl_r
            t["exit_bar"] = bar
            t["exit_ts"] = timestamps[bar]
            t["duration_bars"] = bar - t["entry_bar"]
            trades.append(t)
            active.remove(t)

        # ── Queue new signals ──
        if bar in sig_by_bar:
            for s in sig_by_bar[bar]:
                ci = sig_trig[s]
                is_bear = sig_side[s] == 0
                side_str = "BEAR" if is_bear else "BULL"
                case_val = sig_case[s]
                case_map = {1: "C1", 2: "C2", 3: "C3"}
                case_str = case_map.get(case_val, "?")

                if bar + 1 >= n:
                    continue

                # ── Apply filters ──
                if active_filters and not apply_filters(active_filters, pre, bar, is_bear, filter_stats):
                    continue

                pending.append({
                    "trigger": ci,
                    "is_bear": is_bear,
                    "side_str": side_str,
                    "case_str": case_str,
                    "swept": sig_swept[s],
                    "signal_bar": bar,
                })

    for t in active:
        t["result"] = "OPEN"
        t["pnl_r"] = 0
        trades.append(t)

    el = time_module.perf_counter() - t0
    if not quiet:
        if skipped_invalid > 0:
            print(f"    Skipped {skipped_invalid} invalid (SL wrong side of raw open)")
        if active_filters:
            for filt in sorted(active_filters):
                total = filter_stats.get(f"{filt}_total", 0)
                bull = filter_stats.get(f"{filt}_bull", 0)
                bear = filter_stats.get(f"{filt}_bear", 0)
                print(f"    {filt.upper()} filter blocked {total} signals ({bull} BULL, {bear} BEAR)")
        print(f"  ✅ Done in {el:.1f}s — {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: EQUITY CURVE
# ═══════════════════════════════════════════════════════════════════

def compute_equity_curve(trades, starting_capital, risk_pct, rr_ratio,
                         taker_fee=0.0, maker_fee=0.0, total_bars=None):
    """Compute compounded equity curve from a list of closed trades.

    taker_fee: % fee for market-order entries (e.g. 0.055 means 0.055%)
    maker_fee: % fee for limit-order exits (e.g. 0.01 means 0.01%)
    total_bars: total bars in the dataset, used for exposure-time calculation
    """
    closed = [t for t in trades if t["result"] not in ("OPEN", None)]
    if not closed:
        comm_stats = {"total_taker": 0.0, "total_maker": 0.0, "total": 0.0}
        return [], starting_capital, starting_capital, 0, 0, {}, comm_stats

    balance = starting_capital
    peak = starting_capital
    max_dd_pct = 0.0
    max_dd_usd = 0.0

    equity_points = [{"trade": 0, "balance": balance, "result": "START",
                       "time": str(closed[0].get("entry_ts", ""))[:19],
                       "taker_fee_usd": 0.0, "maker_fee_usd": 0.0, "total_fee_usd": 0.0,
                       "total_bars": total_bars}]
    monthly_pnl = {}
    total_taker_fees = 0.0
    total_maker_fees = 0.0

    for i, t in enumerate(closed):
        risk_usd = balance * (risk_pct / 100.0)
        pnl_r = t.get("pnl_r", 0) or 0
        pnl_usd = risk_usd * pnl_r

        # Compute round-trip fees based on notional position size
        entry_price = t.get("entry", 0) or 0
        ha_risk = t.get("risk", 0) or 0
        if ha_risk > 0 and entry_price > 0 and (taker_fee > 0 or maker_fee > 0):
            position_size_usd = risk_usd * entry_price / ha_risk
            fee_taker = position_size_usd * (taker_fee / 100.0)
            fee_maker = position_size_usd * (maker_fee / 100.0)
            total_fee = fee_taker + fee_maker
        else:
            fee_taker = 0.0
            fee_maker = 0.0
            total_fee = 0.0

        total_taker_fees += fee_taker
        total_maker_fees += fee_maker

        # Fees are deducted from the compounded equity
        pnl_usd -= total_fee
        balance += pnl_usd

        if balance > peak:
            peak = balance
        dd_usd = peak - balance
        dd_pct = dd_usd / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd

        exit_ts = str(t.get("exit_ts", ""))[:19]
        equity_points.append({
            "trade": i + 1,
            "balance": round(balance, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_r": round(pnl_r, 3),
            "risk_usd": round(risk_usd, 2),
            "result": t["result"],
            "side": t["side"],
            "case": t["case"],
            "time": exit_ts,
            "dd_pct": round(dd_pct, 2),
            "taker_fee_usd": round(fee_taker, 4),
            "maker_fee_usd": round(fee_maker, 4),
            "total_fee_usd": round(total_fee, 4),
            "total_bars": total_bars,
        })

        if len(exit_ts) >= 7:
            month_key = exit_ts[:7]
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0.0
            monthly_pnl[month_key] += pnl_usd

    comm_stats = {
        "total_taker": total_taker_fees,
        "total_maker": total_maker_fees,
        "total": total_taker_fees + total_maker_fees,
    }
    return equity_points, balance, peak, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats


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

    # ── Risk Metrics (compounded % daily basis) ──
    rm_pct = calc_risk_metrics_pct(equity_points)
    print(f"\n  ── Risk Metrics (compounded %) ─────────────")
    if rm_pct["sharpe_pct"] is not None:
        print(f"  📐 Sharpe  (%):   {rm_pct['sharpe_pct']:.2f}  {_sharpe_rating(rm_pct['sharpe_pct'])}")
        print(f"  📐 Sortino (%):   {rm_pct['sortino_pct']:.2f}  {_sortino_rating(rm_pct['sortino_pct'])}")
        print(f"  📐 Vol (%, ann):  {rm_pct['vol_pct']*100:.1f}%")
    else:
        print(f"  📐 (insufficient data to compute risk metrics)")

    results = [p["result"] for p in equity_points if p["result"] not in ("START", None)]
    max_consec_loss = 0
    max_consec_win = 0
    cur_loss = 0
    cur_win = 0
    for r in results:
        if r in ("SL",):
            cur_loss += 1
            cur_win = 0
            max_consec_loss = max(max_consec_loss, cur_loss)
        elif r in ("TP", "P+TP"):
            cur_win += 1
            cur_loss = 0
            max_consec_win = max(max_consec_win, cur_win)
        else:
            cur_loss = 0
            cur_win = 0

    print(f"\n  🔥 Max consecutive wins:   {max_consec_win}")
    print(f"  💀 Max consecutive losses: {max_consec_loss}")

    if monthly_pnl:
        print(f"\n  📅 MONTHLY P&L:")
        print(f"  {'Month':<10} {'P&L':>12} {'Cum':>12}")
        print("  " + "-" * 36)
        cum = 0
        max_abs = max(abs(v) for v in monthly_pnl.values()) if monthly_pnl else 1
        for month in sorted(monthly_pnl.keys()):
            pnl = monthly_pnl[month]
            cum += pnl
            bar = "█" * max(1, int(abs(pnl) / max_abs * 20)) if pnl != 0 else ""
            sign = "🟢" if pnl >= 0 else "🔴"
            print(f"  {month:<10} ${pnl:>+11,.2f} ${cum:>+11,.2f}  {sign} {bar}")

    if len(equity_points) > 2:
        balances = [p["balance"] for p in equity_points]
        n_pts = min(50, len(balances))
        step = max(1, len(balances) // n_pts)
        sampled = balances[::step]
        if sampled[-1] != balances[-1]:
            sampled.append(balances[-1])

        min_b = min(sampled)
        max_b = max(sampled)
        rng = max_b - min_b if max_b > min_b else 1
        height = 12

        print(f"\n  📈 EQUITY CURVE:")
        print(f"  ${max_b:>10,.0f} ┤")
        for row in range(height - 1, -1, -1):
            line = "  " + " " * 12 + "│"
            for val in sampled:
                bar_height = int((val - min_b) / rng * height)
                if bar_height == row:
                    line += "●"
                elif bar_height > row:
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


def export_equity_csv(equity_points, filename):
    pd.DataFrame(equity_points).to_csv(filename, index=False)
    print(f"  💾 {len(equity_points)} equity points → {filename}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: RESULTS + TRADE TABLE + CSV
# ═══════════════════════════════════════════════════════════════════

_DAYS_PER_YEAR = 365.25   # used for duration-in-years calculations
_ANNUALIZE_DAYS = 365     # crypto trades 24/7 — use 365 for annualizing daily returns

def fmt(v):
    if v is None:
        return "—"
    v = float(v)
    if abs(v) < 0.01:
        return f"{v:.8f}"
    elif abs(v) < 1:
        return f"{v:.6f}"
    elif abs(v) < 100:
        return f"{v:.4f}"
    else:
        return f"{v:.2f}"


def calc_max_dd(trades):
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


def _sharpe_rating(v):
    if v is None:
        return ""
    if v < 1:
        return "[POOR]"
    if v < 2:
        return "[OK]"
    if v < 3:
        return "[GOOD]"
    return "[EXCELLENT]"


def _sortino_rating(v):
    if v is None:
        return ""
    if v < 1.5:
        return "[POOR]"
    if v < 3:
        return "[OK]"
    if v < 5:
        return "[GOOD]"
    return "[EXCELLENT]"


def calc_risk_metrics_r(closed):
    """Compute Sharpe, Sortino, and annualized Volatility on a flat-R per-trade basis."""
    r_series = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
    if len(r_series) < 2:
        return {"sharpe_r": None, "sortino_r": None, "vol_r": None, "trades_per_year": None}

    # Determine duration in years from first entry to last exit
    first_ts = closed[0].get("entry_ts") or closed[0].get("exit_ts")
    last_ts = closed[-1].get("exit_ts") or closed[-1].get("entry_ts")
    num_years = 1.0
    if first_ts and last_ts:
        try:
            t0 = pd.to_datetime(str(first_ts))
            t1 = pd.to_datetime(str(last_ts))
            num_years = max((t1 - t0).days / _DAYS_PER_YEAR, 1 / _DAYS_PER_YEAR)
        except Exception:
            num_years = 1.0

    trades_per_year = len(r_series) / num_years
    mean_r = np.mean(r_series)
    std_r = np.std(r_series)

    sharpe_r = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0

    neg_r = [r for r in r_series if r < 0]
    downside_std = np.std(neg_r) if len(neg_r) >= 2 else 0.0
    sortino_r = (mean_r / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 999.0

    vol_r = std_r * np.sqrt(trades_per_year)

    return {
        "sharpe_r": sharpe_r,
        "sortino_r": sortino_r,
        "vol_r": vol_r,
        "trades_per_year": trades_per_year,
    }


def calc_bh_metrics(raw_close, timestamps):
    """Compute Buy & Hold metrics from raw close prices.

    Returns a dict with:
      return_pct  — total B&H return as a percentage
      max_dd_pct  — maximum peak-to-trough drawdown from close prices (%)
      sharpe      — annualized Sharpe from daily close returns (sqrt(365))
      sortino     — annualized Sortino from daily close returns
    """
    if raw_close is None or len(raw_close) < 2:
        return {"return_pct": None, "max_dd_pct": None, "sharpe": None, "sortino": None}

    bh_return_pct = (raw_close[-1] / raw_close[0] - 1) * 100

    # Max drawdown from close prices
    peak = raw_close[0]
    max_dd_pct = 0.0
    for c in raw_close:
        if c > peak:
            peak = c
        if peak > 0:
            dd = (peak - c) / peak * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

    # Daily returns via resampling
    try:
        ts_pd = pd.to_datetime(pd.Series(timestamps).astype(str), errors="coerce")
        close_series = pd.Series(raw_close, index=ts_pd)
        daily_close = close_series.resample("D").last().dropna()
        if len(daily_close) >= 2:
            daily_ret = daily_close.pct_change().dropna().values
            if len(daily_ret) >= 2:
                mean_d = np.mean(daily_ret)
                std_d = np.std(daily_ret)
                sharpe = (mean_d / std_d) * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0
                neg_d = daily_ret[daily_ret < 0]
                downside_d = np.std(neg_d) if len(neg_d) >= 2 else 0.0
                sortino = (mean_d / downside_d) * np.sqrt(_ANNUALIZE_DAYS) if downside_d > 0 else 999.0
            else:
                sharpe, sortino = None, None
        else:
            sharpe, sortino = None, None
    except Exception:
        sharpe, sortino = None, None

    return {
        "return_pct": bh_return_pct,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "sortino": sortino,
    }


def enrich_equity_with_bh(equity_points, raw_close, timestamps, initial_capital):
    """Add a 'bh_balance' column to each equity point by matching to the nearest bar close."""
    if not equity_points or raw_close is None or len(raw_close) < 2:
        return equity_points

    try:
        ts_pd = pd.to_datetime(pd.Series(timestamps).astype(str), errors="coerce")
        bh_values = initial_capital * raw_close / raw_close[0]
        bh_series = pd.Series(bh_values, index=ts_pd)
        bh_series = bh_series[~bh_series.index.isna()].sort_index()

        for pt in equity_points:
            pt_time_str = pt.get("time", "")
            if not pt_time_str:
                pt["bh_balance"] = None
                continue
            pt_time = pd.to_datetime(str(pt_time_str), errors="coerce")
            if pd.isna(pt_time):
                pt["bh_balance"] = None
                continue
            idx = bh_series.index.get_indexer([pt_time], method="nearest")[0]
            if 0 <= idx < len(bh_series):
                pt["bh_balance"] = round(float(bh_series.iloc[idx]), 2)
            else:
                pt["bh_balance"] = None
    except Exception:
        for pt in equity_points:
            pt["bh_balance"] = None

    return equity_points


def print_bh_comparison(bh_metrics, initial_capital, final_balance):
    """Print Buy & Hold benchmark section and strategy vs B&H comparison."""
    if bh_metrics.get("return_pct") is None:
        return

    bh_ret = bh_metrics["return_pct"]
    bh_dd = bh_metrics.get("max_dd_pct", 0) or 0
    bh_sharpe = bh_metrics.get("sharpe")
    bh_sortino = bh_metrics.get("sortino")

    strategy_return_pct = (final_balance / initial_capital - 1) * 100

    print(f"\n  ── Buy & Hold Benchmark ────────────────────────────────")
    print(f"  B&H Return:       {bh_ret:+.1f}%")
    print(f"  B&H Max DD:       {bh_dd:.1f}%")
    if bh_sharpe is not None:
        print(f"  B&H Sharpe:       {bh_sharpe:.2f}")
    if bh_sortino is not None:
        sortino_display = f"{bh_sortino:.2f}" if bh_sortino < 999 else "999.00"
        print(f"  B&H Sortino:      {sortino_display}")

    print(f"\n  ── Strategy vs B&H ─────────────────────────────────────")
    print(f"  Strategy Return:  {strategy_return_pct:+.1f}%")
    if abs(bh_ret) > 0.01:
        if bh_ret > 0:
            outperf = strategy_return_pct / bh_ret
            print(f"  Outperformance:   {outperf:.1f}×")
        else:
            diff = strategy_return_pct - bh_ret
            print(f"  Outperformance:   {diff:+.1f}pp (B&H was negative)")
    else:
        print(f"  Outperformance:   ∞ (B&H return ≈ 0)")


def calc_risk_metrics_pct(equity_points):
    """Compute Sharpe, Sortino, and annualized Volatility on a daily compounded-% basis."""
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

    mean_d = np.mean(daily_ret)
    std_d = np.std(daily_ret)

    sharpe_pct = (mean_d / std_d) * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0

    neg_d = daily_ret[daily_ret < 0]
    downside_d = np.std(neg_d) if len(neg_d) >= 2 else 0.0
    sortino_pct = (mean_d / downside_d) * np.sqrt(_ANNUALIZE_DAYS) if downside_d > 0 else 999.0

    vol_pct = std_d * np.sqrt(_ANNUALIZE_DAYS) if std_d > 0 else 0.0

    return {"sharpe_pct": sharpe_pct, "sortino_pct": sortino_pct, "vol_pct": vol_pct}


def print_results(trades, label, rr_ratio=3, partial_tp_r=0, partial_tp_pct=50):
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    if not trades:
        print("No trades.")
        return

    closed = [t for t in trades if t["result"] not in ("OPEN", None)]
    wins_full = [t for t in closed if t["result"] == "TP"]
    wins_partial = [t for t in closed if t["result"] == "P+TP"]
    losses = [t for t in closed if t["result"] == "SL"]
    bes = [t for t in closed if t["result"] == "BE"]
    p_be = [t for t in closed if t["result"] == "P+BE"]
    p_sl = [t for t in closed if t["result"] == "P+SL"]
    opens = [t for t in trades if t["result"] == "OPEN"]
    total_closed = len(closed)

    print(f"\n📊 TRADES: {len(trades)} total | {total_closed} closed | ⏳ {len(opens)} open")
    if partial_tp_r > 0:
        print(f"   ✅ TP: {len(wins_full)} | P+TP: {len(wins_partial)} | "
              f"❌ SL: {len(losses)} | P+SL: {len(p_sl)} | "
              f"🔄 BE: {len(bes)} | P+BE: {len(p_be)}")
        all_wins = len(wins_full) + len(wins_partial) + len(p_be) + len(p_sl)
        print(f"   Positive exits: {len(wins_full) + len(wins_partial) + len(p_be)}/{total_closed} "
              f"({(len(wins_full) + len(wins_partial) + len(p_be))/total_closed*100:.1f}%)")
    else:
        print(f"   ✅ {len(wins_full)} TP | ❌ {len(losses)} SL | 🔄 {len(bes)} BE")
    if total_closed > 0:
        pure_wins = len(wins_full) + len(wins_partial)
        print(f"   Win Rate (TP only): {pure_wins/total_closed*100:.1f}%")

    pnl = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
    max_dd = calc_max_dd(closed)
    rpt = pnl / total_closed if total_closed > 0 else 0

    print(f"\n💰 Net: {pnl:+.1f}R (${pnl*1000:+,.0f} at $1k/R)")
    print(f"📉 Max Drawdown: {max_dd:.1f}R")
    if max_dd > 0:
        print(f"📊 Return/DD ratio: {pnl/max_dd:.2f}x")
    print(f"📊 R per trade: {rpt:+.2f}R")

    gross_win = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
    gross_loss = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    print(f"📊 Profit Factor: {pf:.2f}")

    # ── Risk Metrics (flat R basis) ──
    rm_r = calc_risk_metrics_r(closed)
    print(f"\n── Risk Metrics (flat R) ──────────────────")
    if rm_r["sharpe_r"] is not None:
        tpy = rm_r["trades_per_year"]
        print(f"📐 Trades/year:   {tpy:.1f}")
        print(f"📐 Sharpe  (R):   {rm_r['sharpe_r']:.2f}  {_sharpe_rating(rm_r['sharpe_r'])}")
        print(f"📐 Sortino (R):   {rm_r['sortino_r']:.2f}  {_sortino_rating(rm_r['sortino_r'])}")
        print(f"📐 Vol (R, ann):  {rm_r['vol_r']:.2f}R")
    else:
        print(f"📐 (insufficient trades to compute risk metrics)")

    if partial_tp_r > 0:
        print(f"\n📊 PARTIAL TP BREAKDOWN:")
        for res_type in ["TP", "P+TP", "SL", "P+SL", "BE", "P+BE"]:
            group = [t for t in closed if t["result"] == res_type]
            if group:
                avg_r = np.mean([t["pnl_r"] for t in group if t["pnl_r"] is not None])
                print(f"   {res_type:<5}: {len(group):>3} trades | avg {avg_r:+.2f}R each | "
                      f"total {sum(t['pnl_r'] for t in group if t['pnl_r'] is not None):+.1f}R")

    cases = {'C1': 'C1 (LGCR)', 'C2': 'C2 (LG Line)', 'C3': 'C3 (Pivot)'}
    print(f"\n📊 BY CASE:")
    for c, lbl in cases.items():
        ct = [t for t in closed if t["case"] == c]
        if ct:
            net = sum(t["pnl_r"] for t in ct if t["pnl_r"] is not None)
            w = sum(1 for t in ct if t["pnl_r"] and t["pnl_r"] > 0)
            print(f"   {lbl}: {len(ct)} trades | {w}W | Net: {net:+.1f}R | "
                  f"R/trade: {net/len(ct):+.2f}R")

    print(f"\n📊 BY SIDE:")
    for s in ['BULL', 'BEAR']:
        st = [t for t in closed if t["side"] == s]
        if st:
            net = sum(t["pnl_r"] for t in st if t["pnl_r"] is not None)
            w = sum(1 for t in st if t["pnl_r"] and t["pnl_r"] > 0)
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


def print_trade_table(trades, label=""):
    print(f"\n{'='*200}")
    print(f"  TRADE TABLE: {label}")
    print(f"{'='*200}")
    print(f"{'#':>3} {'Side':<5} {'Case':<4} {'Res':<5} {'PnL':>7} "
          f"{'HA Entry':>14} {'Real Entry':>14} {'SL(HA piv)':>14} {'TP1':>14} {'TP2':>14} {'HA Risk':>10} "
          f"{'Swept':>14} {'MaxR':>5} "
          f"{'Dur':>4} {'Signal Time':<22} {'Exit Time':<22}")
    print("-" * 200)

    for i, t in enumerate(trades):
        sig_ts = str(t.get("signal_ts", t.get("trigger_ts", "")))[:19]
        exit_ts = str(t.get("exit_ts", ""))[:19]
        r = t["result"]
        m = {"TP": "✅", "P+TP": "✅", "SL": "❌", "P+SL": "⚠️",
             "BE": "🔄", "P+BE": "💰", "OPEN": "⏳"}.get(r, "?")
        sv = t.get("swept_val")
        sv_s = fmt(sv) if sv is not None and sv != 0 else ""
        ha_e = t.get("ha_entry", 0)
        dur = t.get("duration_bars")
        dur_s = str(dur) if dur is not None else ""
        tp1 = t.get("tp1", 0)
        tp1_s = fmt(tp1) if tp1 and tp1 > 0 else "—"

        print(f"{i+1:>3} {t['side']:<5} {t['case']:<4} {m} {r:<4} "
              f"{t.get('pnl_r',0):>+6.2f}R "
              f"{fmt(ha_e):>14} {fmt(t['entry']):>14} {fmt(t.get('original_sl', t['sl'])):>14} "
              f"{tp1_s:>14} {fmt(t['tp']):>14} {fmt(t['risk']):>10} "
              f"{sv_s:>14} {t.get('max_r',0):>5.1f} "
              f"{dur_s:>4} {sig_ts:<22} {exit_ts:<22}")

    print("-" * 200)
    w = sum(1 for t in trades if t.get("pnl_r") and t["pnl_r"] > 0)
    l = sum(1 for t in trades if t.get("pnl_r") and t["pnl_r"] < 0)
    b = sum(1 for t in trades if t.get("pnl_r") is not None and t["pnl_r"] == 0 and t["result"] != "OPEN")
    net = sum(t.get("pnl_r", 0) for t in trades if t.get("pnl_r") is not None)
    print(f"    Total: {len(trades)} | 💚 {w} positive | ❌ {l} negative | 🔄 {b} zero | Net: {net:+.1f}R (${net*1000:+,.0f})")


def export_csv(trades, filename):
    rows = []
    for i, t in enumerate(trades):
        rows.append({
            "#": i+1, "side": t["side"], "case": t["case"], "result": t["result"],
            "pnl_r": round(t.get("pnl_r", 0) or 0, 3),
            "ha_entry": round(t.get("ha_entry", 0), 8),
            "entry_raw_open": round(t["entry"], 8),
            "sl_ha_pivot": round(t.get("original_sl", t["sl"]), 8),
            "tp1": round(t.get("tp1", 0) or 0, 8),
            "tp2": round(t["tp"], 8),
            "ha_risk": round(t["risk"], 8),
            "swept_val": t.get("swept_val", ""),
            "max_r": round(t.get("max_r", 0), 2),
            "partial_filled": t.get("partial_filled", False),
            "duration_bars": t.get("duration_bars", ""),
            "trigger_time": str(t.get("trigger_ts", ""))[:19],
            "signal_time": str(t.get("signal_ts", t.get("trigger_ts", "")))[:19],
            "entry_time": str(t.get("entry_ts", ""))[:19],
            "exit_time": str(t.get("exit_ts", ""))[:19],
        })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  💾 {len(rows)} trades → {filename}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: CASE COMBINATION OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def run_case_optimizer(pre, rr_ratio, be_trigger_r, warmup, capital, risk_pct,
                       partial_tp_r=0, partial_tp_pct=50, pivot_len=2, active_filters=None):
    combos = [
        ("C1",      True,  False, False),
        ("C2",      False, True,  False),
        ("C3",      False, False, True),
        ("C1+C2",   True,  True,  False),
        ("C1+C3",   True,  False, True),
        ("C2+C3",   False, True,  True),
        ("C1+C2+C3", True, True,  True),
    ]

    partial_label = f" | Partial {partial_tp_pct:.0f}%@{partial_tp_r}R" if partial_tp_r > 0 else ""
    print(f"\n{'='*80}")
    print(f"  CASE COMBINATION OPTIMIZER")
    print(f"  RR={rr_ratio} | BE={be_trigger_r}R{partial_label} | Warmup={warmup}")
    print(f"{'='*80}")

    results = []

    for label, c1, c2, c3 in combos:
        trades = run_backtest(pre, rr_ratio=rr_ratio, be_trigger_r=be_trigger_r,
                              warmup=warmup, enable_c1=c1, enable_c2=c2, enable_c3=c3,
                              partial_tp_r=partial_tp_r, partial_tp_pct=partial_tp_pct,
                              quiet=True, pivot_len=pivot_len, active_filters=active_filters)

        closed = [t for t in trades if t["result"] not in ("OPEN", None)]
        total = len(closed)

        net_r = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
        wins = sum(1 for t in closed if t.get("pnl_r") and t["pnl_r"] > 0)
        losses_n = sum(1 for t in closed if t.get("pnl_r") and t["pnl_r"] < 0)
        wr = wins / total * 100 if total > 0 else 0
        rpt = net_r / total if total > 0 else 0
        max_dd = calc_max_dd(closed)
        ret_dd = net_r / max_dd if max_dd > 0 else 0
        gross_win = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
        gross_loss = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 999
        rm_r = calc_risk_metrics_r(closed)
        sharpe = rm_r["sharpe_r"] if rm_r["sharpe_r"] is not None else 0.0

        results.append({
            "cases": label,
            "trades": total,
            "W": wins,
            "L": losses_n,
            "WR%": round(wr, 1),
            "net_R": round(net_r, 1),
            "R/trade": round(rpt, 3),
            "max_DD": round(max_dd, 1),
            "Ret/DD": round(ret_dd, 2),
            "PF": round(pf, 2),
            "Sharpe": round(sharpe, 2),
        })

        marker = "⭐" if ret_dd >= 5 else "✅" if ret_dd >= 2 else "⚠️" if net_r > 0 else "❌"
        print(f"  {marker} {label:<10} | {total:>3} trades | {wr:>5.1f}% WR | "
              f"{net_r:>+6.1f}R | DD={max_dd:.1f}R | Ret/DD={ret_dd:.1f}x | PF={pf:.2f} | Sharpe={sharpe:.2f}")

    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"  RANKED BY RET/DD (minimum 5 trades)")
    print(f"{'='*80}")
    df_ranked = df[df["trades"] >= 5].sort_values("Ret/DD", ascending=False)
    print(df_ranked.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"  RANKED BY NET R")
    print(f"{'='*80}")
    df_net = df.sort_values("net_R", ascending=False)
    print(df_net.to_string(index=False))

    if len(df_ranked) > 0:
        best = df_ranked.iloc[0]
        print(f"\n  🏆 BEST BY RET/DD: {best['cases']} — "
              f"{best['net_R']:+.1f}R, {best['Ret/DD']:.1f}x Ret/DD, "
              f"{best['WR%']:.0f}% WR, PF={best['PF']:.2f}")

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: PARTIAL TP OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def run_partial_optimizer(pre, rr_ratio, be_trigger_r, warmup, capital, risk_pct,
                          enable_c1, enable_c2, enable_c3, pivot_len=2, active_filters=None):
    """Test multiple partial TP levels and compare."""
    cases_str = f"{'C1' if enable_c1 else ''}{'C2' if enable_c2 else ''}{'C3' if enable_c3 else ''}"

    configs = [
        ("No Partial",       0.0, 0),
        ("50% @ 1.0R",       1.0, 50),
        ("50% @ 1.5R",       1.5, 50),
        ("50% @ 2.0R",       2.0, 50),
        ("33% @ 1.0R",       1.0, 33),
        ("33% @ 1.5R",       1.5, 33),
        ("33% @ 2.0R",       2.0, 33),
    ]

    print(f"\n{'='*90}")
    print(f"  PARTIAL TP OPTIMIZER — {cases_str} | RR={rr_ratio} | BE={be_trigger_r}R")
    print(f"{'='*90}")

    results = []

    for label, pt_r, pt_pct in configs:
        trades = run_backtest(pre, rr_ratio=rr_ratio, be_trigger_r=be_trigger_r,
                              warmup=warmup, enable_c1=enable_c1, enable_c2=enable_c2,
                              enable_c3=enable_c3,
                              partial_tp_r=pt_r, partial_tp_pct=float(pt_pct),
                              quiet=True, pivot_len=pivot_len, active_filters=active_filters)

        closed = [t for t in trades if t["result"] not in ("OPEN", None)]
        total = len(closed)
        net_r = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
        wins = sum(1 for t in closed if t.get("pnl_r") and t["pnl_r"] > 0)
        wr = wins / total * 100 if total > 0 else 0
        rpt = net_r / total if total > 0 else 0
        max_dd = calc_max_dd(closed)
        ret_dd = net_r / max_dd if max_dd > 0 else 0
        gross_win = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
        gross_loss = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 999
        rm_r = calc_risk_metrics_r(closed)
        sharpe = rm_r["sharpe_r"] if rm_r["sharpe_r"] is not None else 0.0

        # Count partial-specific results
        p_tp = sum(1 for t in closed if t["result"] == "P+TP")
        p_be = sum(1 for t in closed if t["result"] == "P+BE")
        p_sl = sum(1 for t in closed if t["result"] == "P+SL")
        partials = sum(1 for t in closed if t.get("partial_filled"))

        results.append({
            "config": label,
            "trades": total,
            "W": wins,
            "WR%": round(wr, 1),
            "net_R": round(net_r, 1),
            "R/trade": round(rpt, 3),
            "max_DD": round(max_dd, 1),
            "Ret/DD": round(ret_dd, 2),
            "PF": round(pf, 2),
            "Sharpe": round(sharpe, 2),
            "partials": partials,
            "P+TP": p_tp,
            "P+BE": p_be,
            "P+SL": p_sl,
        })

        marker = "⭐" if ret_dd >= 5 else "✅" if ret_dd >= 2 else "⚠️" if net_r > 0 else "❌"
        partial_info = f"| {partials} partials" if partials > 0 else ""
        print(f"  {marker} {label:<16} | {total:>3} trades | {wr:>5.1f}% WR | "
              f"{net_r:>+6.1f}R | DD={max_dd:.1f}R | Ret/DD={ret_dd:.1f}x | PF={pf:.2f} | Sharpe={sharpe:.2f} {partial_info}")

    df = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print(f"  RANKED BY RET/DD")
    print(f"{'='*90}")
    df_ranked = df.sort_values("Ret/DD", ascending=False)
    print(df_ranked[["config", "trades", "W", "WR%", "net_R", "R/trade",
                     "max_DD", "Ret/DD", "PF", "Sharpe", "partials"]].to_string(index=False))

    if len(df_ranked) > 0:
        best = df_ranked.iloc[0]
        print(f"\n  🏆 BEST: {best['config']} — "
              f"{best['net_R']:+.1f}R, {best['Ret/DD']:.1f}x Ret/DD, PF={best['PF']:.2f}")

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 12: BAYESIAN OPTIMIZER (Optuna)
# ═══════════════════════════════════════════════════════════════════

def _create_optuna_storage(storage_str):
    """Parse a storage string and return an Optuna storage object (or None for in-memory).

    Supported formats:
      - "" / None          → in-memory (default, no persistence)
      - "journal:<path>"   → Optuna JournalStorage backed by a local file
      - "sqlite:///..."    → SQLite RDB storage (passed through as a URL string)
      - "postgresql://..."  → PostgreSQL RDB storage (passed through as a URL string)
    """
    if not storage_str:
        return None
    if storage_str.startswith("journal:"):
        path = storage_str[len("journal:"):]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
        except ImportError:
            try:
                # Older Optuna (<3.1) layout
                from optuna.storages import JournalStorage, JournalFileBackend  # type: ignore
            except ImportError:
                print("❌ JournalStorage not available. Upgrade optuna: pip install -U optuna")
                sys.exit(1)
        return JournalStorage(JournalFileBackend(path))
    # sqlite://, postgresql://, etc. — pass through as a URL string
    return storage_str


def _build_worker_cmd(args, worker_trials):
    """Build the subprocess command list for a single Bayesian optimization worker."""
    script = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, script]

    # Data-loading args
    cmd += ["--symbol", args.symbol, "--tf", args.tf]
    if args.start:
        cmd += ["--start", args.start]
    if args.end:
        cmd += ["--end", args.end]
    if not args.start and not args.end:
        cmd += ["--days", str(args.days)]
    cmd += ["--capital", str(args.capital)]
    cmd += ["--risk-pct", str(args.risk_pct)]
    # Pass explicit warmup so workers don't recompute (avoids edge-case drift)
    computed_warmup = args.warmup if args.warmup is not None else compute_warmup(args.tf)
    cmd += ["--warmup", str(computed_warmup)]
    cmd += ["--taker-fee", str(args.taker_fee)]
    cmd += ["--maker-fee", str(args.maker_fee)]
    if args.filters and args.filters.strip().lower() not in ("none", ""):
        cmd += ["--filters", args.filters]
    if getattr(args, 'mtf_lgcr', '').strip():
        cmd += ["--mtf-lgcr", args.mtf_lgcr.strip()]
    if args.force:
        cmd += ["--force"]

    # Optimizer args
    cmd += ["--optimize-bayesian"]
    cmd += ["--n-trials", str(worker_trials)]
    cmd += ["--objective", args.objective]
    cmd += ["--study-name", args.study_name]
    cmd += ["--storage", args.storage]
    # Force single-process in each worker; do not pass --plot
    cmd += ["--n-jobs", "1"]
    cmd += ["--worker"]
    return cmd


def _run_bayesian_parallel(args, pre_pv1, pre_pv2, warmup, capital, risk_pct,
                            sym_safe, timeframe, date_tag, taker_fee, maker_fee,
                            active_filters):
    """Coordinator: spawn *n_jobs* worker processes, wait, then run analysis once."""
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. Run: pip install optuna")
        sys.exit(1)

    n_jobs = args.n_jobs
    n_trials = args.n_trials
    storage = args.storage
    study_name = args.study_name

    # Ensure journal file directory exists before spawning workers
    _create_optuna_storage(storage)

    # Split trials across workers (first workers get the extra trial when not evenly divisible)
    base_trials = n_trials // n_jobs
    remainder = n_trials % n_jobs

    print(f"\n{'='*80}")
    print(f"  PARALLEL BAYESIAN OPTIMIZER — {n_trials} trials across {n_jobs} workers")
    print(f"  study: '{study_name}' | storage: {storage}")
    print(f"{'='*80}")

    procs = []
    for i in range(n_jobs):
        worker_trials = base_trials + (1 if i < remainder else 0)
        if worker_trials == 0:
            continue
        cmd = _build_worker_cmd(args, worker_trials)
        print(f"  ▶ Worker {i+1}/{n_jobs}: {worker_trials} trials")
        # Workers are silent (--worker flag); redirect their output to avoid pipe-buffer issues
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((i + 1, proc))

    # Wait for all workers
    try:
        for _, proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt received — terminating workers...")
        for _, proc in procs:
            proc.terminate()
        for _, proc in procs:
            proc.wait()
        raise

    failed = [idx for idx, proc in procs if proc.returncode != 0]
    if failed:
        print(f"\n⚠️  {len(failed)} worker(s) exited with non-zero status: {failed}")

    # Load the completed study and run the full analysis once
    print(f"\n  ✅ Workers finished. Loading study '{study_name}' for final analysis...")
    optuna_storage = _create_optuna_storage(storage)
    completed_study = optuna.load_study(study_name=study_name, storage=optuna_storage)

    run_bayesian_optimizer(
        pre_pv1=pre_pv1,
        pre_pv2=pre_pv2,
        warmup=warmup,
        capital=capital,
        risk_pct=risk_pct,
        n_trials=n_trials,
        objective_name=args.objective,
        sym_safe=sym_safe,
        timeframe=timeframe,
        date_tag=date_tag,
        plot=args.plot,
        taker_fee=taker_fee,
        maker_fee=maker_fee,
        active_filters=active_filters,
        study_name=study_name,
        storage=storage,
        _prebuilt_study=completed_study,
    )


def run_bayesian_optimizer(pre_pv1, pre_pv2, warmup, capital, risk_pct,
                           n_trials=200, objective_name="return_dd",
                           sym_safe="BTC_USDT", timeframe="5m",
                           date_tag="", plot=False,
                           taker_fee=0.0, maker_fee=0.0, active_filters=None,
                           study_name="goat_opt", storage="",
                           worker=False, _prebuilt_study=None):
    """Run Bayesian optimization with Optuna (TPE sampler).

    Parameters
    ----------
    study_name : str
        Name of the Optuna study (used with persistent storage).
    storage : str
        Storage specifier. "" = in-memory. "journal:<path>" = file-backed
        JournalStorage. Any Optuna RDB URL is also accepted.
    worker : bool
        When True, run optimization trials only and skip all reporting/plotting
        (used by parallel worker subprocesses).
    _prebuilt_study : optuna.Study or None
        When provided, skip the optimize() call and go straight to analysis
        (used by the parallel coordinator after workers complete).
    """
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pre_map = {1: pre_pv1, 2: pre_pv2}

    # --- If a pre-built study is supplied, skip optimization and go straight to analysis ---
    if _prebuilt_study is not None:
        study = _prebuilt_study
    else:
        if not worker:
            print(f"\n{'='*80}")
            print(f"  BAYESIAN OPTIMIZER (Optuna TPE) — {n_trials} trials | objective: {objective_name}")
            print(f"{'='*80}")

        def objective(trial):
            rr = trial.suggest_float("rr", 2.0, 6.0, step=0.5)
            be = trial.suggest_float("be", 1.0, 3.5, step=0.5)
            partial_r = trial.suggest_float("partial_r", 0.0, 2.5, step=0.5)
            partial_pct = trial.suggest_categorical("partial_pct", [25, 33, 50])
            pivot_len = trial.suggest_categorical("pivot_len", [1, 2])
            cases_str = trial.suggest_categorical("cases", ["1", "2", "3", "12", "13", "23", "123"])

            c1 = "1" in cases_str
            c2 = "2" in cases_str
            c3 = "3" in cases_str

            pre = pre_map[pivot_len]

            trades = run_backtest(pre, rr_ratio=rr, be_trigger_r=be,
                                  warmup=warmup, enable_c1=c1, enable_c2=c2, enable_c3=c3,
                                  partial_tp_r=partial_r, partial_tp_pct=float(partial_pct),
                                  quiet=True, pivot_len=pivot_len, active_filters=active_filters)

            closed = [t for t in trades if t["result"] not in ("OPEN", None)]
            if len(closed) < 5:
                # Too few trades → unreliable; large negative penalty discourages this region
                return -999.0

            pnl_r_list = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
            net_r = sum(pnl_r_list)

            # Account for fees by converting per-trade fee USD to R and subtracting
            if taker_fee > 0 or maker_fee > 0:
                adj_pnl_r_list = []
                sim_balance = capital
                for t in closed:
                    raw_r = t.get("pnl_r", 0) or 0
                    risk_usd = sim_balance * (risk_pct / 100.0)
                    entry_p = t.get("entry", 0) or 0
                    ha_r = t.get("risk", 0) or 0
                    if ha_r > 0 and entry_p > 0 and risk_usd > 0:
                        pos_usd = risk_usd * entry_p / ha_r
                        fee_r = pos_usd * (taker_fee + maker_fee) / 100.0 / risk_usd
                    else:
                        fee_r = 0.0
                    adj_r = raw_r - fee_r
                    adj_pnl_r_list.append(adj_r)
                    sim_balance += risk_usd * adj_r
                pnl_r_list = adj_pnl_r_list
                net_r = sum(pnl_r_list)

            # Store Sharpe and Sortino for later display
            rm_r = calc_risk_metrics_r(closed)
            trial.set_user_attr("sharpe_r", round(rm_r["sharpe_r"], 3) if rm_r["sharpe_r"] is not None else 0.0)
            trial.set_user_attr("sortino_r", round(rm_r["sortino_r"], 3) if rm_r["sortino_r"] is not None else 0.0)

            if objective_name == "net_r":
                return net_r
            elif objective_name in ("return_dd", "calmar"):
                _arr = np.array(pnl_r_list)
                _cumsum = _arr.cumsum()
                max_dd = abs(min(0.0, min(_cumsum - np.maximum.accumulate(_cumsum))))
                return net_r / max_dd if max_dd > 0 else 0.0
            elif objective_name == "sharpe":
                mean_r = np.mean(pnl_r_list)
                std_r = np.std(pnl_r_list)
                return mean_r / std_r if std_r > 0 else 0.0
            elif objective_name == "profit_factor":
                gross_win = sum(v for v in pnl_r_list if v > 0)
                gross_loss = abs(sum(v for v in pnl_r_list if v < 0))
                # Cap at 999 when no losses; avoids inf which can skew TPE sampler
                return gross_win / gross_loss if gross_loss > 0 else 999.0
            return 0.0

        def progress_callback(study, trial):
            if trial.number % 10 == 0:
                try:
                    best_val = study.best_value
                    best_p = study.best_params
                    print(f"  Trial {trial.number}/{n_trials} | Best so far: {best_val:.3f} | "
                          f"Params: {best_p}")
                except Exception:
                    print(f"  Trial {trial.number}/{n_trials} | (no completed trials yet)")

        optuna_storage = _create_optuna_storage(storage)
        use_persistent = optuna_storage is not None
        study = optuna.create_study(
            study_name=study_name if use_persistent else None,
            storage=optuna_storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=use_persistent,
        )
        callbacks = [] if worker else [progress_callback]
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

        if worker:
            # Worker mode: trials done, coordinator handles the rest
            return

    best = study.best_params
    best_val = study.best_value

    print(f"\n{'='*80}")
    print(f"  🏆 BEST PARAMETERS (objective={objective_name}={best_val:.4f})")
    print(f"{'='*80}")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # Top 10 trials
    trials_data = []
    for t in study.trials:
        if t.value is not None:
            row = {"trial": t.number, "value": t.value}
            row.update(t.params)
            row["sharpe_r"] = t.user_attrs.get("sharpe_r", 0.0)
            row["sortino_r"] = t.user_attrs.get("sortino_r", 0.0)
            trials_data.append(row)
    top10_df = (pd.DataFrame(trials_data)
                .sort_values("value", ascending=False)
                .head(10))
    print(f"\n  TOP 10 TRIALS:")
    print(top10_df.to_string(index=False))

    # CSV export
    trials_df = study.trials_dataframe()
    csv_name = f"optuna_{sym_safe}_{timeframe}_{objective_name}_{n_trials}trials{date_tag}.csv"
    trials_df.to_csv(csv_name, index=False)
    print(f"\n  💾 All trials → {csv_name}")

    # Visualizations
    try:
        from optuna.visualization import (
            plot_param_importances,
            plot_contour,
            plot_optimization_history,
        )
        fig = plot_param_importances(study)
        fig.write_image(f"optuna_{sym_safe}_{timeframe}_importance.png")
        fig = plot_contour(study)
        fig.write_image(f"optuna_{sym_safe}_{timeframe}_contour.png")
        fig = plot_optimization_history(study)
        fig.write_image(f"optuna_{sym_safe}_{timeframe}_history.png")
        print(f"  📊 Saved: optuna_*_importance.png, optuna_*_contour.png, optuna_*_history.png")
    except ImportError:
        print("  ⚠️  Install plotly + kaleido for visualization: pip install plotly kaleido")
    except Exception as e:
        print(f"  ⚠️  Visualization error: {e}")

    # Re-run best config with full output
    print(f"\n{'='*80}")
    print(f"  BEST CONFIG — FULL BACKTEST")
    print(f"{'='*80}")
    b_rr = best["rr"]
    b_be = best["be"]
    b_partial_r = best["partial_r"]
    b_partial_pct = float(best["partial_pct"])
    b_pivot_len = best["pivot_len"]
    b_cases_str = best.get("cases", "123")
    b_c1 = "1" in b_cases_str
    b_c2 = "2" in b_cases_str
    b_c3 = "3" in b_cases_str

    best_pre = pre_map[b_pivot_len]
    assert "n" in best_pre, f"Precompute object for pivot_len={b_pivot_len} is missing key 'n'"

    best_trades = run_backtest(
        best_pre,
        rr_ratio=b_rr, be_trigger_r=b_be, warmup=warmup,
        enable_c1=b_c1, enable_c2=b_c2, enable_c3=b_c3,
        partial_tp_r=b_partial_r, partial_tp_pct=b_partial_pct,
        quiet=False, pivot_len=b_pivot_len, active_filters=active_filters,
    )

    cases_label = f"{'C1' if b_c1 else ''}{'C2' if b_c2 else ''}{'C3' if b_c3 else ''}"
    partial_label = f" | Partial: {b_partial_pct:.0f}%@{b_partial_r}R" if b_partial_r > 0 else ""
    run_label = f"{b_be}R BE | RR={b_rr}{partial_label} | {cases_label} | pv{b_pivot_len}"
    run_label_full = f"{run_label} | ${capital:,.0f} @ {risk_pct}%"

    print_results(best_trades, run_label, rr_ratio=b_rr,
                  partial_tp_r=b_partial_r, partial_tp_pct=b_partial_pct)

    eq_pts, final_bal, peak_bal, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats = \
        compute_equity_curve(best_trades, capital, risk_pct, b_rr,
                             taker_fee=taker_fee, maker_fee=maker_fee,
                             total_bars=best_pre["n"])

    print_equity_curve(eq_pts, capital, final_bal, peak_bal,
                       max_dd_pct, max_dd_usd, monthly_pnl,
                       risk_pct, b_rr, run_label_full)

    if comm_stats["total"] > 0:
        print(f"\n  💸 Commissions (Bayesian best run):")
        print(f"    Taker fees:  ${comm_stats['total_taker']:,.2f}")
        print(f"    Maker fees:  ${comm_stats['total_maker']:,.2f}")
        print(f"    Total fees:  ${comm_stats['total']:,.2f}")

    # ── Buy & Hold benchmark ──
    bh_metrics = calc_bh_metrics(best_pre["raw_close"], best_pre["timestamps"])
    print_bh_comparison(bh_metrics, capital, final_bal)
    enrich_equity_with_bh(eq_pts, best_pre["raw_close"], best_pre["timestamps"], capital)

    # Export CSVs for best run
    be_tag = f"be{b_be}".replace(".", "")
    cases_tag = cases_label.lower()
    p_tag = f"_p{b_partial_r}".replace(".", "") if b_partial_r > 0 else ""
    pv_tag = f"_pv{b_pivot_len}"
    trades_csv = f"optuna_best_{sym_safe}_{timeframe}_{be_tag}_{cases_tag}{p_tag}{date_tag}{pv_tag}_trades.csv"
    equity_csv = f"optuna_best_{sym_safe}_{timeframe}_{be_tag}_{cases_tag}{p_tag}{date_tag}{pv_tag}_equity.csv"
    export_csv(best_trades, trades_csv)
    export_equity_csv(eq_pts, equity_csv)

    if plot:
        try:
            from goat_21_plot import load_data as load_plot_data, plot_all
            print("📈 Generating chart...")
            plot_trades_df, plot_equity_df = load_plot_data(trades_csv, equity_csv)
            plot_all(plot_trades_df, plot_equity_df, title=run_label_full, trades_file=trades_csv,
                     bh_metrics=bh_metrics)
        except ImportError as e:
            print(f"⚠️  Could not generate chart: {e}"
                  " — ensure goat_21_plot.py is present and its dependencies are installed.")


# ═══════════════════════════════════════════════════════════════════
# WARMUP + PARSE
# ═══════════════════════════════════════════════════════════════════

def compute_warmup(timeframe, override=None):
    if override is not None:
        return override
    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360,
        "8h": 480, "12h": 720, "1d": 1440,
    }
    mins = tf_minutes.get(timeframe, 5)
    warmup = max(50, int(1440 / mins))
    return warmup


def parse_cases(cases_str):
    cases_str = cases_str.strip().lower()
    if cases_str in ("all", "123"):
        return True, True, True
    c1 = "1" in cases_str
    c2 = "2" in cases_str
    c3 = "3" in cases_str
    if not (c1 or c2 or c3):
        print(f"⚠️  Invalid --cases '{cases_str}', using all cases")
        return True, True, True
    return c1, c2, c3


def parse_filters(filters_str):
    """Parse comma-separated filter string into a set of active filter names."""
    if filters_str.strip().lower() in ("none", ""):
        return set()
    filters = set()
    valid_filters = {"ao", "mtf_lgcr", "donchian"}  # Add new filter names here as they're implemented
    for f in filters_str.split(","):
        f = f.strip().lower()
        if f in valid_filters:
            filters.add(f)
        elif f and f != "none":
            print(f"⚠️  Unknown filter '{f}', ignoring. Valid filters: {', '.join(sorted(valid_filters))}")
    return filters


def apply_filters(active_filters, pre, bar, is_bear, filter_stats):
    """
    Check all active filters for a signal. Returns True if signal is ALLOWED, False if BLOCKED.
    Updates filter_stats dict with block counts.
    """
    for filt in active_filters:
        if filt == "ao":
            ao_val = pre["ao"][bar]
            if not is_bear and ao_val > 0:
                # Block LONG when AO is positive (bullish momentum — not counter-trend)
                filter_stats["ao_total"] = filter_stats.get("ao_total", 0) + 1
                filter_stats["ao_bull"] = filter_stats.get("ao_bull", 0) + 1
                return False
            if is_bear and ao_val < 0:
                # Block SHORT when AO is negative (bearish momentum — not counter-trend)
                filter_stats["ao_total"] = filter_stats.get("ao_total", 0) + 1
                filter_stats["ao_bear"] = filter_stats.get("ao_bear", 0) + 1
                return False
        elif filt == "mtf_lgcr":
            bias_arr = pre.get("mtf_lgcr_bias")
            if bias_arr is None:
                # Bias not precomputed (e.g. called without --mtf-lgcr); skip gracefully
                continue
            bias_val = int(bias_arr[bar])
            if bias_val == 1 and is_bear:
                # Bull HTF bias → allow LONGs only, block SHORT signals
                filter_stats["mtf_lgcr_total"] = filter_stats.get("mtf_lgcr_total", 0) + 1
                filter_stats["mtf_lgcr_bear"] = filter_stats.get("mtf_lgcr_bear", 0) + 1
                return False
            if bias_val == -1 and not is_bear:
                # Bear HTF bias → allow SHORTs only, block LONG signals
                filter_stats["mtf_lgcr_total"] = filter_stats.get("mtf_lgcr_total", 0) + 1
                filter_stats["mtf_lgcr_bull"] = filter_stats.get("mtf_lgcr_bull", 0) + 1
                return False
        elif filt == "donchian":
            dc_bias_arr = pre.get("donchian_bias")
            if dc_bias_arr is None:
                continue
            dc_val = int(dc_bias_arr[bar])
            if dc_val == 1 and is_bear:
                # Bull Donchian bias → block SHORT signals
                filter_stats["donchian_total"] = filter_stats.get("donchian_total", 0) + 1
                filter_stats["donchian_bear"] = filter_stats.get("donchian_bear", 0) + 1
                return False
            if dc_val == -1 and not is_bear:
                # Bear Donchian bias → block LONG signals
                filter_stats["donchian_total"] = filter_stats.get("donchian_total", 0) + 1
                filter_stats["donchian_bull"] = filter_stats.get("donchian_bull", 0) + 1
                return False
    return True


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GOATv2 — Configurable BE + Equity + Partial TP + Case Selection")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--rr", type=float, default=3.0, help="TP target in R (default 3.0)")
    parser.add_argument("--be", type=float, default=2.0, help="BE trigger in R (default 2.0)")
    parser.add_argument("--partial", type=float, default=0.0,
                        help="Partial TP at this R-multiple (0 = disabled, e.g. 1.5)")
    parser.add_argument("--partial-pct", type=float, default=50.0,
                        help="Percentage of position to close at partial TP (default 50)")
    parser.add_argument("--capital", type=float, default=50000)
    parser.add_argument("--risk-pct", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cases", type=str, default="123")
    parser.add_argument("--optimize-cases", action="store_true")
    parser.add_argument("--optimize-partial", action="store_true",
                        help="Test multiple partial TP levels and compare")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for backtest (YYYY-MM-DD). When set, data is fetched from "
                             "this date (plus warmup) instead of --days bars back.")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for backtest (YYYY-MM-DD). When set, data is trimmed to this date.")
    parser.add_argument("--plot", action="store_true",
                        help="Auto-generate chart after backtest")
    parser.add_argument("--pivot-len", type=int, default=2, choices=[1, 2],
                        help="Pivot detection length: 1 = 1+1+1 (3 candles), 2 = 2+1+2 (5 candles, default)")
    parser.add_argument("--optimize-bayesian", action="store_true",
                        help="Enable Bayesian optimization using Optuna (TPE sampler)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of Optuna trials (default: 200)")
    parser.add_argument("--objective", type=str, default="return_dd",
                        choices=["return_dd", "sharpe", "profit_factor", "net_r", "calmar"],
                        help="Objective function to maximize (default: return_dd)")
    parser.add_argument("--taker-fee", type=float, default=0.055,
                        help="Taker fee %% for market-order entries (default: 0.055)")
    parser.add_argument("--maker-fee", type=float, default=0.01,
                        help="Maker fee %% for limit-order exits (default: 0.01)")
    parser.add_argument("--filters", type=str, default="none",
                        help="Comma-separated signal filters: ao, mtf_lgcr, donchian, none (default: none). "
                             "AO: block LONG if AO>0, block SHORT if AO<0. "
                             "mtf_lgcr: block against stateful HTF HA-LGCR bias (requires --mtf-lgcr). "
                             "donchian: block against Donchian Channel touch-based bias (requires --donchian-period).")
    parser.add_argument("--mtf-lgcr", type=str, default="",
                        help="MTF LGCR filter pair as <LTF>/<HTF> (e.g. 5m/30m, 30m/4h, 4h/1d, 1d/1w). "
                             "Required when --filters includes mtf_lgcr. LTF must match --tf.")
    parser.add_argument("--donchian-period", type=int, default=200,
                        help="Donchian Channel lookback period for touch-based bias filter (default: 200)")
    parser.add_argument("--study-name", type=str, default="goat_opt",
                        help="Optuna study name (default: goat_opt). Used for persistent storage.")
    parser.add_argument("--storage", type=str, default="",
                        help="Optuna storage specifier. Default '' = in-memory (single process). "
                             "Use 'journal:<path>' for file-based shared storage (recommended for "
                             "--n-jobs > 1, e.g. --storage journal:optuna_journal.log). "
                             "SQLite and PostgreSQL URLs are also accepted.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel worker processes for --optimize-bayesian "
                             "(default: 1 = no parallelism). Conservative default for low-RAM machines. "
                             "Requires --storage when > 1.")
    # Internal flag used by parallel worker subprocesses — not intended for direct user invocation
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.tf
    days = args.days
    rr = args.rr
    be = args.be
    partial_r = args.partial
    partial_pct = args.partial_pct
    capital = args.capital
    risk_pct = args.risk_pct
    taker_fee = args.taker_fee
    maker_fee = args.maker_fee
    start_date = args.start
    end_date = args.end
    pivot_len = args.pivot_len
    sym_safe = symbol.split(":")[0].replace("/", "_")

    warmup = compute_warmup(timeframe, args.warmup)
    enable_c1, enable_c2, enable_c3 = parse_cases(args.cases)
    cases_str = f"{'C1' if enable_c1 else ''}{'C2' if enable_c2 else ''}{'C3' if enable_c3 else ''}"
    active_filters = parse_filters(args.filters)

    # ── MTF LGCR filter validation ──
    mtf_lgcr_pair = args.mtf_lgcr.strip()
    mtf_htf = None
    if "mtf_lgcr" in active_filters:
        if not mtf_lgcr_pair:
            print("❌ --mtf-lgcr is required when mtf_lgcr filter is active "
                  "(e.g. --mtf-lgcr 5m/30m).")
            sys.exit(1)
        parts = mtf_lgcr_pair.split("/")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            print(f"❌ Invalid --mtf-lgcr format '{mtf_lgcr_pair}'. "
                  f"Expected <LTF>/<HTF> (e.g. 5m/30m, 30m/4h, 4h/1d, 1d/1w).")
            sys.exit(1)
        mtf_ltf, mtf_htf = parts[0].strip(), parts[1].strip()
        if mtf_ltf != timeframe:
            print(f"❌ --mtf-lgcr LTF '{mtf_ltf}' must equal --tf '{timeframe}'.")
            sys.exit(1)
        print(f"MTF LGCR filter: {mtf_lgcr_pair} (stateful)")

    if active_filters:
        print(f"Active filters: {', '.join(sorted(active_filters)).upper()}")

    partial_label = f" | Partial: {partial_pct:.0f}%@{partial_r}R" if partial_r > 0 else ""
    filter_label = f" | Filters: {','.join(sorted(active_filters)).upper()}" if active_filters else ""
    mode = ("OPTIMIZER" if args.optimize_cases
            else "PARTIAL OPTIMIZER" if args.optimize_partial
            else "BAYESIAN OPTIMIZER" if args.optimize_bayesian
            else "BACKTEST")

    # Build a human-readable date-range label for header / filenames
    if start_date or end_date:
        date_range_label = f"{start_date or 'start'} → {end_date or 'now'}"
        date_tag = f"_{(start_date or '').replace('-', '')}_{(end_date or 'now').replace('-', '')}"
    else:
        date_range_label = f"{days}d"
        date_tag = f"_{days}d"

    print("=" * 60)
    print(f"GOATv2 — {be}R BE + RR={rr} + EQUITY CURVE")
    print(f"{symbol} — {timeframe} — {date_range_label}")
    print(f"BE: {be}R | RR: {rr}{partial_label}")
    print(f"Capital: ${capital:,.0f} | Risk: {risk_pct}%/trade")
    print(f"Fees: taker={taker_fee}% / maker={maker_fee}%")
    print(f"Cases: {cases_str} | Mode: {mode}{filter_label}")
    print(f"Warmup: {warmup} bars {'(auto)' if args.warmup is None else '(manual)'}")
    print(f"Pivot: {pivot_len}+1+{pivot_len} ({2*pivot_len+1} candles)")
    print(f"")
    print(f"  EXECUTION MODEL:")
    print(f"  1. HA signal on bar N → SL = HA pivot ({pivot_len}+1+{pivot_len})")
    print(f"  2. HA_risk = |HA_close[N] - SL|")
    if partial_r > 0:
        print(f"  3. TP1 = HA_close[N] ± {partial_r} × HA_risk → close {partial_pct:.0f}%")
        print(f"  4. TP2 = HA_close[N] ± {rr} × HA_risk → close remaining {100-partial_pct:.0f}%")
    else:
        print(f"  3. TP = HA_close[N] ± {rr} × HA_risk")
    print(f"  {'5' if partial_r > 0 else '4'}. Entry = raw open of bar N+1")
    print(f"  {'6' if partial_r > 0 else '5'}. SL/TP checked vs raw high/low")
    print(f"  {'7' if partial_r > 0 else '6'}. BE: when max favorable ≥ {be}R → SL moves to entry")
    print("=" * 60)

    df_raw = get_ohlcv(symbol, timeframe, days, force_download=args.force,
                       start_date=start_date, end_date=end_date)
    if df_raw is None or len(df_raw) == 0:
        print(f"\n❌ ERROR: No data returned for {symbol} {timeframe} {date_range_label}")
        exit(1)

    # Print actual date range used
    actual_start = df_raw['timestamp'].iloc[0].strftime('%Y-%m-%d') if 'timestamp' in df_raw.columns else '?'
    actual_end = df_raw['timestamp'].iloc[-1].strftime('%Y-%m-%d') if 'timestamp' in df_raw.columns else '?'
    print(f"  Data range: {actual_start} → {actual_end} ({len(df_raw)} bars)")

    pre = precompute_all(df_raw, pivot_len=pivot_len, mtf_lgcr_htf=mtf_htf,
                         donchian_period=args.donchian_period if "donchian" in active_filters else None)

    print("\n⚡ Numba JIT warmup (first run compiles, be patient)...")

    # ── OPTIMIZER MODES ──
    if args.optimize_cases:
        opt_df = run_case_optimizer(pre, rr, be, warmup, capital, risk_pct,
                                    partial_tp_r=partial_r, partial_tp_pct=partial_pct,
                                    pivot_len=pivot_len, active_filters=active_filters)
        csv_file = f"goat_optimize_cases_{sym_safe}_{timeframe}{date_tag}.csv"
        opt_df.to_csv(csv_file, index=False)
        print(f"\n  💾 Optimizer results → {csv_file}")
        exit(0)

    if args.optimize_partial:
        opt_df = run_partial_optimizer(pre, rr, be, warmup, capital, risk_pct,
                                       enable_c1, enable_c2, enable_c3,
                                       pivot_len=pivot_len, active_filters=active_filters)
        csv_file = f"goat_optimize_partial_{sym_safe}_{timeframe}{date_tag}.csv"
        opt_df.to_csv(csv_file, index=False)
        print(f"\n  💾 Partial optimizer results → {csv_file}")
        exit(0)

    if args.optimize_bayesian:
        pre_pv1 = precompute_all(df_raw, pivot_len=1, mtf_lgcr_htf=mtf_htf,
                                 donchian_period=args.donchian_period if "donchian" in active_filters else None)
        pre_pv2 = precompute_all(df_raw, pivot_len=2, mtf_lgcr_htf=mtf_htf,
                                 donchian_period=args.donchian_period if "donchian" in active_filters else None)

        if args.n_jobs > 1:
            if not args.storage:
                print("❌ --n-jobs > 1 requires --storage "
                      "(e.g. --storage journal:optuna_journal.log)")
                sys.exit(1)
            _run_bayesian_parallel(
                args=args,
                pre_pv1=pre_pv1,
                pre_pv2=pre_pv2,
                warmup=warmup,
                capital=capital,
                risk_pct=risk_pct,
                sym_safe=sym_safe,
                timeframe=timeframe,
                date_tag=date_tag,
                taker_fee=taker_fee,
                maker_fee=maker_fee,
                active_filters=active_filters,
            )
        else:
            run_bayesian_optimizer(
                pre_pv1=pre_pv1,
                pre_pv2=pre_pv2,
                warmup=warmup,
                capital=capital,
                risk_pct=risk_pct,
                n_trials=args.n_trials,
                objective_name=args.objective,
                sym_safe=sym_safe,
                timeframe=timeframe,
                date_tag=date_tag,
                plot=args.plot,
                taker_fee=taker_fee,
                maker_fee=maker_fee,
                active_filters=active_filters,
                study_name=args.study_name,
                storage=args.storage,
                worker=args.worker,
            )
        exit(0)

    # ── SINGLE BACKTEST MODE ──
    trades = run_backtest(pre, rr_ratio=rr, be_trigger_r=be, warmup=warmup,
                          enable_c1=enable_c1, enable_c2=enable_c2, enable_c3=enable_c3,
                          partial_tp_r=partial_r, partial_tp_pct=partial_pct,
                          pivot_len=pivot_len, active_filters=active_filters)

    run_label = f"{be}R BE | RR={rr}{partial_label} | {cases_str}{filter_label}"
    run_label_full = f"{run_label} | ${capital:,.0f} @ {risk_pct}%"

    print_results(trades, run_label,
                  rr_ratio=rr, partial_tp_r=partial_r, partial_tp_pct=partial_pct)
    print_trade_table(trades, run_label)

    eq_pts, final_bal, peak_bal, max_dd_pct, max_dd_usd, monthly_pnl, comm_stats = \
        compute_equity_curve(trades, capital, risk_pct, rr,
                             taker_fee=taker_fee, maker_fee=maker_fee,
                             total_bars=pre["n"])

    print_equity_curve(eq_pts, capital, final_bal, peak_bal,
                       max_dd_pct, max_dd_usd, monthly_pnl,
                       risk_pct, rr, run_label_full)

    if comm_stats["total"] > 0:
        print(f"\n  💸 Commissions:")
        print(f"    Taker fees:  ${comm_stats['total_taker']:,.2f}")
        print(f"    Maker fees:  ${comm_stats['total_maker']:,.2f}")
        print(f"    Total fees:  ${comm_stats['total']:,.2f}")

    # ── Buy & Hold benchmark ──
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
        print(f"    Fees paid: ${comm_stats['total']:,.2f} (taker: ${comm_stats['total_taker']:,.2f} | maker: ${comm_stats['total_maker']:,.2f})")
    print(f"\n  Compounding advantage: ${final_bal - capital - pnl_r*1000:+,.2f}")

    print(f"\n📁 Exporting CSVs...")
    be_tag = f"be{be}".replace(".", "")
    cases_tag = cases_str.lower()
    partial_tag = f"_p{partial_r}".replace(".", "") if partial_r > 0 else ""
    pv_tag = f"_pv{pivot_len}"
    trades_csv = f"goat_eq_{sym_safe}_{timeframe}_{be_tag}_{cases_tag}{partial_tag}{date_tag}{pv_tag}_trades.csv"
    equity_csv = f"goat_eq_{sym_safe}_{timeframe}_{be_tag}_{cases_tag}{partial_tag}{date_tag}{pv_tag}_equity.csv"
    export_csv(trades, trades_csv)
    export_equity_csv(eq_pts, equity_csv)

    if args.plot:
        try:
            from goat_21_plot import load_data as load_plot_data, plot_all
            print("📈 Generating chart...")
            plot_trades_df, plot_equity_df = load_plot_data(trades_csv, equity_csv)
            plot_all(plot_trades_df, plot_equity_df, title=run_label_full, trades_file=trades_csv,
                     bh_metrics=bh_metrics)
        except ImportError as e:
            print(f"⚠️  Could not generate chart: {e}"
                  " — ensure goat_21_plot.py is present and its dependencies are installed.")

    print(f"\n{'='*60}")
    print(f"  📋 EXECUTION MODEL:")
    print(f"     1. HA signal detected on bar N")
    print(f"     2. SL = nearest HA pivot ({pivot_len}+1+{pivot_len})")
    print(f"     3. HA_risk = |HA_close[N] - SL|  ← defines 1R")
    if partial_r > 0:
        print(f"     4. TP1 = HA_close[N] ± {partial_r} × HA_risk → close {partial_pct:.0f}%")
        print(f"     5. TP2 = HA_close[N] ± {rr} × HA_risk → close remaining")
    else:
        print(f"     4. TP = HA_close[N] ± {rr} × HA_risk  ← HA-based")
    print(f"     Entry = raw open of bar N+1 (market order)")
    print(f"     SL/TP checked against raw high/low")
    print(f"     BE: when max favorable ≥ {be}R → SL moves to entry")
    print(f"     Cases enabled: {cases_str}")
    print(f"{'='*60}")