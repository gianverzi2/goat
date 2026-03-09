"""
GOATv2 Numba Backtest — 0.5R BE + Ichimoku TK Cross Filter
Conversion (Tenkan) above Base (Kijun) = longs only, and vice versa.
Tests: same TF Ichimoku vs higher TF (M30) Ichimoku, entries always on M5.

Ichimoku settings from TradingView:
  Conversion Line Length: 20
  Base Line Length: 60
  Leading Span B Length: 120
  Lagging Span: 30

Usage:
  python3 goat_16_vbt_ichimoku.py --symbol BTC/USDT:USDT --tf 5m --days 180
"""

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
# SECTION 2: PIVOTS
# ═══════════════════════════════════════════════════════════════════

HA_PIVOT_LENGTH = 2

@njit
def precompute_pivots_numba(ha_low, ha_high, n):
    max_pivots = n
    pli = np.empty(max_pivots, dtype=np.int64)
    plv = np.empty(max_pivots, dtype=np.float64)
    phi = np.empty(max_pivots, dtype=np.int64)
    phv = np.empty(max_pivots, dtype=np.float64)
    nl = 0
    nh = 0
    for i in range(HA_PIVOT_LENGTH, n - HA_PIVOT_LENGTH):
        is_low = True
        for off in range(1, HA_PIVOT_LENGTH + 1):
            if not (ha_low[i] < ha_low[i - off] and ha_low[i] < ha_low[i + off]):
                is_low = False
                break
        if is_low:
            pli[nl] = i
            plv[nl] = ha_low[i]
            nl += 1
        is_high = True
        for off in range(1, HA_PIVOT_LENGTH + 1):
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
# SECTION 4: NUMBA-JIT CASE CHECKS
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

        last_k = -1
        last_swept = 0.0

        for k in range(prior_idx + 1, cur):
            l1_ok = line1_valid
            l2_ok = line2_valid
            if k > prior_idx + 1:
                if line1_valid:
                    if is_bear:
                        if sparse_max(sp_max, prior_idx + 1, k - 1) > line1:
                            l1_ok = False
                    else:
                        if sparse_min(sp_min, prior_idx + 1, k - 1) < line1:
                            l1_ok = False
                if line2_valid:
                    if is_bear:
                        if sparse_max(sp_max, prior_idx + 1, k - 1) > line2:
                            l2_ok = False
                    else:
                        if sparse_min(sp_min, prior_idx + 1, k - 1) < line2:
                            l2_ok = False

            if is_bear:
                wick = ha_high[k]
                sw2 = (wick >= line2 * 0.999999) and l2_ok and line2_valid
                sw1 = (wick >= line1 * 0.999999) and l1_ok and line1_valid
            else:
                wick = ha_low[k]
                sw2 = (wick <= line2 * 1.000001) and l2_ok and line2_valid
                sw1 = (wick <= line1 * 1.000001) and l1_ok and line1_valid

            if not (sw1 or sw2):
                continue

            # ── Skip bars with no actual wick (flat-top/bottom body) ──
            if is_bear:
                body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
                if wick <= body_top:
                    continue
            else:
                body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
                if wick >= body_bot:
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
                last_swept = line2
            else:
                last_swept = line1
            last_k = k

        if last_k >= 0:
            return True, last_swept

    return False, 0.0


@njit
def check_case2_jit(ha_close, ha_open, ha_high, ha_low,
                    body_low, body_high,
                    lgc_flags, lgc_lines, cur, is_bear,
                    sp_max, sp_min):
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

    last_k = -1
    for k in range(best_idx + 1, cur):
        if is_bear:
            wick = ha_high[k]
            if wick < line_level:
                continue
        else:
            wick = ha_low[k]
            if wick > line_level:
                continue

        # ── Skip bars with no actual wick ──
        if is_bear:
            body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
            if wick <= body_top:
                continue
        else:
            body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
            if wick >= body_bot:
                continue

        if k > best_idx + 1:
            if is_bear:
                if sparse_max(sp_max, best_idx + 1, k - 1) > line_level:
                    continue
            else:
                if sparse_min(sp_min, best_idx + 1, k - 1) < line_level:
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

        last_k = k

    if last_k >= 0:
        return True, line_level
    return False, 0.0


@njit
def check_case3_jit(ha_close, ha_open, ha_high, ha_low,
                    body_low, body_high,
                    piv_idx_arr, piv_lvl_arr, n_pivots, cur, is_bear,
                    sp_max, sp_min):
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

    last_k = -1
    for k in range(piv_idx + 1, cur):
        if is_bear:
            wick = ha_high[k]
            if wick < piv_level * 0.999999:
                continue
        else:
            wick = ha_low[k]
            if wick > piv_level * 1.000001:
                continue

        # ── Skip bars with no actual wick ──
        if is_bear:
            body_top = ha_open[k] if ha_open[k] > ha_close[k] else ha_close[k]
            if wick <= body_top:
                continue
        else:
            body_bot = ha_open[k] if ha_open[k] < ha_close[k] else ha_close[k]
            if wick >= body_bot:
                continue

        if k > piv_idx + 1:
            if is_bear:
                if sparse_max(sp_max, piv_idx + 1, k - 1) > piv_level:
                    continue
            else:
                if sparse_min(sp_min, piv_idx + 1, k - 1) < piv_level:
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

        last_k = k

    if last_k >= 0:
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
                     sp_max, sp_min):
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
                                         pi, pl, np_, ci, is_bear, sp_max, sp_min)
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
def calc_levels_jit(ha_close, piv_low_idx, piv_low_lvl, n_pl,
                    piv_high_idx, piv_high_lvl, n_ph,
                    trigger_idx, is_bear, rr_ratio):
    entry = ha_close[trigger_idx]
    if not is_bear:
        best_idx = -1
        for p in range(n_pl):
            if piv_low_idx[p] <= trigger_idx and piv_low_lvl[p] < entry:
                if best_idx < 0 or piv_low_idx[p] > piv_low_idx[best_idx]:
                    best_idx = p
        if best_idx < 0:
            return False, 0.0, 0.0, 0.0, 0.0
        sl = piv_low_lvl[best_idx]
        risk = abs(entry - sl)
        if risk == 0:
            return False, 0.0, 0.0, 0.0, 0.0
        tp = entry + rr_ratio * risk
    else:
        best_idx = -1
        for p in range(n_ph):
            if piv_high_idx[p] <= trigger_idx and piv_high_lvl[p] > entry:
                if best_idx < 0 or piv_high_idx[p] > piv_high_idx[best_idx]:
                    best_idx = p
        if best_idx < 0:
            return False, 0.0, 0.0, 0.0, 0.0
        sl = piv_high_lvl[best_idx]
        risk = abs(sl - entry)
        if risk == 0:
            return False, 0.0, 0.0, 0.0, 0.0
        tp = entry - rr_ratio * risk
    return True, entry, sl, tp, risk


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: ICHIMOKU CALCULATION
# ═══════════════════════════════════════════════════════════════════

def ichimoku_line(high, low, period):
    """
    Ichimoku line = (highest high + lowest low) / 2 over period.
    Same as TradingView's donchian midline.
    """
    n = len(high)
    line = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1: i + 1])
        ll = np.min(low[i - period + 1: i + 1])
        line[i] = (hh + ll) / 2.0
    return line


def calculate_ichimoku(high, low, conv_period=20, base_period=60):
    """
    Returns conversion line (tenkan) and base line (kijun).
    Settings: Conversion=20, Base=60 (from TradingView image).
    """
    tenkan = ichimoku_line(high, low, conv_period)
    kijun = ichimoku_line(high, low, base_period)
    return tenkan, kijun


def resample_to_htf(df_raw, htf="30m"):
    """
    Resample M5 data to higher timeframe for HTF Ichimoku.
    Returns HTF tenkan/kijun arrays mapped back to M5 bar indices.
    """
    df = df_raw.copy()
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp_dt')

    # Convert htf string to pandas frequency
    # "30m" → "30T", "1h" → "1h"
    if htf.endswith('m'):
        pd_freq = htf.replace('m', 'T')
    else:
        pd_freq = htf

    htf_ohlc = df.resample(pd_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # Calculate Ichimoku on HTF
    htf_high = htf_ohlc['high'].values.astype(np.float64)
    htf_low = htf_ohlc['low'].values.astype(np.float64)
    htf_tenkan, htf_kijun = calculate_ichimoku(htf_high, htf_low, 20, 60)

    # Map HTF values back to M5 bars using forward fill
    htf_tenkan_series = pd.Series(htf_tenkan, index=htf_ohlc.index, name='tenkan')
    htf_kijun_series = pd.Series(htf_kijun, index=htf_ohlc.index, name='kijun')

    m5_index = pd.to_datetime(df_raw['timestamp'])
    tenkan_m5 = htf_tenkan_series.reindex(m5_index, method='ffill').values.astype(np.float64)
    kijun_m5 = htf_kijun_series.reindex(m5_index, method='ffill').values.astype(np.float64)

    tenkan_m5 = np.where(np.isnan(tenkan_m5), 0.0, tenkan_m5)
    kijun_m5 = np.where(np.isnan(kijun_m5), 0.0, kijun_m5)

    return tenkan_m5, kijun_m5


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: PRE-COMPUTE + BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def precompute_all(df_raw):
    t0 = time_module.perf_counter()
    print("  Computing HA + patterns + pivots + sparse tables + Ichimoku...")

    ha = calculate_heikin_ashi(df_raw).reset_index(drop=True)
    n = len(ha)

    ha_close = ha['HA_Close'].values.astype(np.float64)
    ha_open = ha['HA_Open'].values.astype(np.float64)
    ha_high = ha['HA_High'].values.astype(np.float64)
    ha_low = ha['HA_Low'].values.astype(np.float64)
    raw_close = df_raw['close'].values.astype(np.float64)
    raw_high = df_raw['high'].values.astype(np.float64)
    raw_low = df_raw['low'].values.astype(np.float64)
    timestamps = ha['timestamp'].values

    (bull_lg, bear_lg, bull_lgc, bear_lgc,
     bull_lgc_line, bear_lgc_line, bull_lgcr, bear_lgcr) = \
        detect_all_patterns_numba(n, ha_open, ha_close, ha_high, ha_low)

    piv_low_idx, piv_low_lvl, piv_high_idx, piv_high_lvl = \
        precompute_pivots_numba(ha_low, ha_high, n)

    sp_max = build_sparse_table(ha_close, "max")
    sp_min = build_sparse_table(ha_close, "min")
    body_low = np.minimum(ha_open, ha_close)
    body_high = np.maximum(ha_open, ha_close)

    # Ichimoku on same TF (M5)
    tenkan_m5, kijun_m5 = calculate_ichimoku(raw_high, raw_low, 20, 60)

    # Ichimoku on HTF (M30) mapped to M5
    print("  Computing M30 Ichimoku (HTF)...")
    tenkan_htf, kijun_htf = resample_to_htf(df_raw, "30m")

    # Find where Ichimoku becomes valid
    m5_valid = np.argmax(~np.isnan(tenkan_m5) & ~np.isnan(kijun_m5))
    htf_valid_idx = np.argmax((tenkan_htf > 0) & (kijun_htf > 0))

    el = time_module.perf_counter() - t0
    print(f"    {len(piv_low_idx)} pivot lows, {len(piv_high_idx)} pivot highs")
    print(f"    M5 Ichimoku valid from bar {m5_valid}")
    print(f"    M30 Ichimoku valid from bar {htf_valid_idx}")
    print(f"    Prep done in {el:.1f}s")

    return {
        "n": n, "ha_close": ha_close, "ha_open": ha_open,
        "ha_high": ha_high, "ha_low": ha_low,
        "raw_close": raw_close, "raw_high": raw_high, "raw_low": raw_low,
        "timestamps": timestamps,
        "bull_lgc": bull_lgc, "bear_lgc": bear_lgc,
        "bull_lgcr": bull_lgcr, "bear_lgcr": bear_lgcr,
        "bull_lgc_line": bull_lgc_line, "bear_lgc_line": bear_lgc_line,
        "piv_low_idx": piv_low_idx, "piv_low_lvl": piv_low_lvl,
        "piv_high_idx": piv_high_idx, "piv_high_lvl": piv_high_lvl,
        "sp_max": sp_max, "sp_min": sp_min,
        "body_low": body_low, "body_high": body_high,
        "tenkan_m5": tenkan_m5, "kijun_m5": kijun_m5,
        "tenkan_htf": tenkan_htf, "kijun_htf": kijun_htf,
    }


def run_backtest(pre, rr_ratio=3, be_trigger_r=0.5,
                 ichi_filter="none", warmup=300):
    """
    ichi_filter: "none", "m5", "htf"
    """
    t0 = time_module.perf_counter()

    filter_labels = {"none": "NO FILTER", "m5": "ICHIMOKU M5", "htf": "ICHIMOKU M30→M5"}
    filt_label = filter_labels.get(ichi_filter, ichi_filter)
    print(f"\n{'='*60}")
    print(f"BACKTEST: {be_trigger_r}R BE | {filt_label}")
    print(f"{'='*60}")

    n = pre["n"]
    ha_close = pre["ha_close"]
    ha_open = pre["ha_open"]
    ha_high = pre["ha_high"]
    ha_low = pre["ha_low"]
    raw_close = pre["raw_close"]
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

    # Select Ichimoku lines
    if ichi_filter == "m5":
        tenkan = pre["tenkan_m5"]
        kijun = pre["kijun_m5"]
    elif ichi_filter == "htf":
        tenkan = pre["tenkan_htf"]
        kijun = pre["kijun_htf"]
    else:
        tenkan = None
        kijun = None

    print("  Scanning signals (Numba JIT)...")
    t1 = time_module.perf_counter()
    sig_bar, sig_trig, sig_side, sig_case, sig_swept, ns = scan_all_signals(
        n, warmup, 5,
        ha_close, ha_open, ha_high, ha_low, body_low, body_high,
        pre["bull_lgc"], pre["bear_lgc"], pre["bull_lgcr"], pre["bear_lgcr"],
        pre["bull_lgc_line"], pre["bear_lgc_line"],
        piv_low_idx, piv_low_lvl, len(piv_low_idx),
        piv_high_idx, piv_high_lvl, len(piv_high_idx),
        sp_max, sp_min
    )
    t_scan = time_module.perf_counter() - t1
    print(f"    Found {ns} raw signals in {t_scan:.1f}s")

    trades = []
    active = []
    n_pl = len(piv_low_idx)
    n_ph = len(piv_high_idx)
    filtered_count = 0
    filtered_bull = 0
    filtered_bear = 0

    sig_by_bar = {}
    for s in range(ns):
        b = sig_bar[s]
        if b not in sig_by_bar:
            sig_by_bar[b] = []
        sig_by_bar[b].append(s)

    print("  Simulating trades...")
    for bar in range(warmup, n):
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
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = cur_entry
                if c_lo <= t["sl"]:
                    r = "BE" if (t["be_active"] and t["sl"] == cur_entry) else "SL"
                    p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))
                elif c_hi >= t["tp"]:
                    to_close.append((t, "TP", float(rr_ratio)))
            else:
                if risk > 0:
                    t["max_r"] = max(t["max_r"], (cur_entry - c_lo) / risk)
                if be_trigger_r and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = cur_entry
                if c_hi >= t["sl"]:
                    r = "BE" if (t["be_active"] and t["sl"] == cur_entry) else "SL"
                    p = 0.0 if r == "BE" else -1.0
                    to_close.append((t, r, p))
                elif c_lo <= t["tp"]:
                    to_close.append((t, "TP", float(rr_ratio)))

        for t, result, pnl_r in to_close:
            t["result"] = result
            t["pnl_r"] = pnl_r
            t["exit_bar"] = bar
            t["exit_ts"] = timestamps[bar]
            t["duration_bars"] = bar - t["entry_bar"]
            trades.append(t)
            active.remove(t)

        # ── Open new trades ──
        if bar in sig_by_bar:
            for s in sig_by_bar[bar]:
                ci = sig_trig[s]
                is_bear = sig_side[s] == 0
                side_str = "BEAR" if is_bear else "BULL"
                case_val = sig_case[s]
                case_map = {1: "C1", 2: "C2", 3: "C3"}
                case_str = case_map.get(case_val, "?")

                # ── ICHIMOKU TK FILTER ──
                tk_state = "—"
                if tenkan is not None and kijun is not None:
                    t_val = tenkan[bar]
                    k_val = kijun[bar]
                    if np.isnan(t_val) or np.isnan(k_val) or t_val == 0 or k_val == 0:
                        filtered_count += 1
                        continue
                    # Tenkan > Kijun = bullish → longs only
                    # Tenkan < Kijun = bearish → shorts only
                    if t_val > k_val:
                        tk_state = "T>K"
                        if side_str == "BEAR":
                            filtered_count += 1
                            filtered_bear += 1
                            continue
                    elif t_val < k_val:
                        tk_state = "T<K"
                        if side_str == "BULL":
                            filtered_count += 1
                            filtered_bull += 1
                            continue
                    else:
                        # T == K, no clear direction, skip
                        filtered_count += 1
                        continue

                valid, entry, sl, tp, risk = calc_levels_jit(
                    ha_close, piv_low_idx, piv_low_lvl, n_pl,
                    piv_high_idx, piv_high_lvl, n_ph,
                    ci, is_bear, rr_ratio)
                if not valid:
                    continue

                # Store Ichimoku values for analysis
                t_val_store = 0.0
                k_val_store = 0.0
                if tenkan is not None:
                    tv = tenkan[bar]
                    kv = kijun[bar]
                    if not np.isnan(tv):
                        t_val_store = tv
                    if not np.isnan(kv):
                        k_val_store = kv

                trade = {
                    "side": side_str, "case": case_str,
                    "entry": entry, "sl": sl, "tp": tp, "risk": risk,
                    "original_sl": sl,
                    "entry_bar": bar, "entry_ts": timestamps[bar],
                    "trigger_bar": ci, "trigger_ts": timestamps[ci],
                    "swept_val": sig_swept[s],
                    "max_r": 0.0, "be_active": False,
                    "result": None, "pnl_r": None,
                    "exit_bar": None, "exit_ts": None, "duration_bars": None,
                    "tenkan": t_val_store,
                    "kijun": k_val_store,
                    "tk_state": tk_state,
                    "price_at_entry": raw_close[bar],
                }
                active.append(trade)

    for t in active:
        t["result"] = "OPEN"
        t["pnl_r"] = 0
        trades.append(t)

    el = time_module.perf_counter() - t0
    if ichi_filter != "none":
        print(f"    Ichimoku filter removed {filtered_count} signals "
              f"({filtered_bull} BULL when T<K, {filtered_bear} BEAR when T>K)")
    print(f"  ✅ Done in {el:.1f}s — {len(trades)} trades")
    return trades, filtered_count, filtered_bull, filtered_bear


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: RESULTS + TRADE TABLE + CSV
# ═══════════════════════════════════════════════════════════════════

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


def print_results(trades, label, rr_ratio=3):
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    if not trades:
        print("No trades.")
        return

    wins = [t for t in trades if t["result"] == "TP"]
    losses = [t for t in trades if t["result"] == "SL"]
    bes = [t for t in trades if t["result"] == "BE"]
    opens = [t for t in trades if t["result"] == "OPEN"]
    total_closed = len(wins) + len(losses) + len(bes)

    print(f"\n📊 TRADES: {len(trades)} total | ✅ {len(wins)} TP | ❌ {len(losses)} SL | "
          f"🔄 {len(bes)} BE | ⏳ {len(opens)} open")
    if total_closed > 0:
        print(f"   Win Rate: {len(wins)/total_closed*100:.1f}%")

    pnl = sum(t["pnl_r"] for t in trades if t["pnl_r"] is not None)
    max_dd = calc_max_dd(trades)
    print(f"\n💰 Net: {pnl:+.1f}R (${pnl*1000:+,.0f})")
    print(f"📉 Max Drawdown: {max_dd:.1f}R (${max_dd*1000:,.0f})")
    if max_dd > 0:
        print(f"📊 Return/DD ratio: {pnl/max_dd:.2f}x")

    cases = {'C1': 'C1 (LGCR)', 'C2': 'C2 (LG Line)', 'C3': 'C3 (Pivot)'}
    print(f"\n📊 BY CASE:")
    for c, lbl in cases.items():
        w = len([t for t in wins if t["case"] == c])
        l = len([t for t in losses if t["case"] == c])
        b = len([t for t in bes if t["case"] == c])
        tot = w + l + b
        if tot > 0:
            print(f"   {lbl}: {w}W/{l}L/{b}BE = {w/tot*100:.0f}% | Net: {w*rr_ratio - l:+.0f}R")

    print(f"\n📊 BY SIDE:")
    for s in ['BULL', 'BEAR']:
        w = len([t for t in wins if t["side"] == s])
        l = len([t for t in losses if t["side"] == s])
        b = len([t for t in bes if t["side"] == s])
        tot = w + l + b
        if tot > 0:
            print(f"   {s}: {w}W/{l}L/{b}BE = {w/tot*100:.0f}% | Net: {w*rr_ratio - l:+.0f}R")

    if losses:
        mrs = [t["max_r"] for t in losses]
        print(f"\n📐 LOSS MAX-R: ", end="")
        for th in [0.5, 1.0, 1.5, 2.0]:
            c = sum(1 for r in mrs if r >= th)
            print(f"≥{th}R:{c}/{len(mrs)}({c/len(mrs)*100:.0f}%) ", end="")
        print()

    closed_dur = [t for t in trades if t.get("duration_bars") and t["result"] != "OPEN"]
    if closed_dur:
        durs = [t["duration_bars"] for t in closed_dur]
        win_durs = [t["duration_bars"] for t in closed_dur if t["result"] == "TP"]
        loss_durs = [t["duration_bars"] for t in closed_dur if t["result"] == "SL"]
        print(f"\n⏱️  DURATION (bars): all avg={np.mean(durs):.0f} med={np.median(durs):.0f}", end="")
        if win_durs:
            print(f" | wins avg={np.mean(win_durs):.0f}", end="")
        if loss_durs:
            print(f" | losses avg={np.mean(loss_durs):.0f}", end="")
        print()


def print_trade_table(trades, label=""):
    print(f"\n{'='*170}")
    print(f"  TRADE TABLE: {label}")
    print(f"{'='*170}")
    print(f"{'#':>3} {'Side':<5} {'Case':<4} {'Res':<3} {'PnL':>6} "
          f"{'Entry':>14} {'SL':>14} {'TP':>14} {'Swept':>14} {'MaxR':>5} "
          f"{'Dur':>4} {'TK':>5} {'Trigger Time':<22} {'Exit Time':<22}")
    print("-" * 170)

    for i, t in enumerate(trades):
        trig_ts = str(t.get("trigger_ts", ""))[:19]
        exit_ts = str(t.get("exit_ts", ""))[:19]
        r = t["result"]
        m = "✅" if r == "TP" else "❌" if r == "SL" else "🔄" if r == "BE" else "⏳"
        sv = t.get("swept_val")
        sv_s = fmt(sv) if sv is not None and sv != 0 else ""
        tk = t.get("tk_state", "—")

        print(f"{i+1:>3} {t['side']:<5} {t['case']:<4} {m:<3} "
              f"{t.get('pnl_r',0):>+5.1f}R "
              f"{fmt(t['entry']):>14} {fmt(t.get('original_sl', t['sl'])):>14} "
              f"{fmt(t['tp']):>14} {sv_s:>14} {t.get('max_r',0):>5.1f} "
              f"{t.get('duration_bars',''):>4} {tk:>5} "
              f"{trig_ts:<22} {exit_ts:<22}")

    print("-" * 170)
    w = sum(1 for t in trades if t["result"] == "TP")
    l = sum(1 for t in trades if t["result"] == "SL")
    b = sum(1 for t in trades if t["result"] == "BE")
    net = sum(t.get("pnl_r", 0) for t in trades if t.get("pnl_r") is not None)
    print(f"    Total: {len(trades)} | ✅ {w} | ❌ {l} | 🔄 {b} | Net: {net:+.1f}R (${net*1000:+,.0f})")


def export_csv(trades, filename):
    rows = []
    for i, t in enumerate(trades):
        rows.append({
            "#": i+1, "side": t["side"], "case": t["case"], "result": t["result"],
            "pnl_r": t.get("pnl_r", 0),
            "entry": t["entry"], "sl": t.get("original_sl", t["sl"]), "tp": t["tp"],
            "risk": t["risk"], "swept_val": t.get("swept_val", ""),
            "max_r": round(t.get("max_r", 0), 2),
            "duration_bars": t.get("duration_bars", ""),
            "trigger_bar": t.get("trigger_bar", ""),
            "trigger_time": str(t.get("trigger_ts", ""))[:19],
            "entry_time": str(t.get("entry_ts", ""))[:19],
            "exit_time": str(t.get("exit_ts", ""))[:19],
            "tenkan": round(t.get("tenkan", 0), 2),
            "kijun": round(t.get("kijun", 0), 2),
            "tk_state": t.get("tk_state", ""),
            "price_at_entry": round(t.get("price_at_entry", 0), 2),
        })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  💾 {len(rows)} trades → {filename}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOATv2 — 0.5R BE + Ichimoku TK Filter")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--rr", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.tf
    days = args.days
    rr = args.rr
    sym_safe = symbol.replace("/", "_").replace(":", "_")

    print("=" * 60)
    print(f"GOATv2 — 0.5R BE + ICHIMOKU TK FILTER")
    print(f"{symbol} — {timeframe} — {days}d — RR {rr}")
    print(f"Ichimoku: Conv=20, Base=60, SpanB=120, Lag=30")
    print(f"Filter: Tenkan > Kijun = longs only, T < K = shorts only")
    print("=" * 60)

    df_raw = get_ohlcv(symbol, timeframe, days, force_download=args.force)

    pre = precompute_all(df_raw)

    print("\n⚡ Numba JIT warmup (first run compiles, be patient)...")

    # ── 1. No filter (reference) ──
    trades_ref, _, _, _ = run_backtest(
        pre, rr_ratio=rr, be_trigger_r=0.5, ichi_filter="none")
    print_results(trades_ref, "0.5R BE — NO FILTER (reference)", rr_ratio=rr)
    print_trade_table(trades_ref, "0.5R BE — NO FILTER")

    # ── 2. Ichimoku same TF ──
    trades_m5, filt_m5, fb_m5, fs_m5 = run_backtest(
        pre, rr_ratio=rr, be_trigger_r=0.5, ichi_filter="m5")
    print_results(trades_m5, f"0.5R BE + ICHIMOKU {timeframe.upper()}", rr_ratio=rr)
    print_trade_table(trades_m5, f"0.5R BE + ICHIMOKU {timeframe.upper()}")

    # ── 3. Ichimoku HTF (M30) ──
    trades_htf, filt_htf, fb_htf, fs_htf = run_backtest(
        pre, rr_ratio=rr, be_trigger_r=0.5, ichi_filter="htf")
    print_results(trades_htf, "0.5R BE + ICHIMOKU M30 (HTF)", rr_ratio=rr)
    print_trade_table(trades_htf, "0.5R BE + ICHIMOKU M30")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"  COMPARISON — {symbol} {timeframe} {days}d RR={rr} | 0.5R BE")
    print(f"{'='*100}")

    print(f"\n  {'Strategy':<35} {'W':>4} {'L':>4} {'BE':>4} {'Tot':>5} "
          f"{'WR%':>6} {'Net R':>8} {'USD':>14} {'MaxDD':>7} {'Ret/DD':>7} {'R/trade':>8}")
    print("  " + "-" * 100)

    strats = [
        ("No Filter (ref)", trades_ref),
        (f"Ichimoku {timeframe.upper()} (same TF)", trades_m5),
        ("Ichimoku M30 (HTF)", trades_htf),
    ]

    for name, tds in strats:
        w = len([t for t in tds if t["result"] == "TP"])
        l = len([t for t in tds if t["result"] == "SL"])
        b = len([t for t in tds if t["result"] == "BE"])
        tot = w + l + b
        wr = w / tot * 100 if tot > 0 else 0
        pnl = sum(t["pnl_r"] for t in tds if t["pnl_r"] is not None)
        dd = calc_max_dd(tds)
        ret_dd = pnl / dd if dd > 0 else 999
        rpt = pnl / tot if tot > 0 else 0
        print(f"  {name:<35} {w:>4} {l:>4} {b:>4} {tot:>5} "
              f"{wr:>5.1f}% {pnl:>+7.1f}R ${pnl*1000:>+12,.0f} "
              f"{dd:>6.1f}R {ret_dd:>6.1f}x {rpt:>+7.2f}R")

    # ── Filter impact details ──
    w0 = len([t for t in trades_ref if t["result"] == "TP"])
    l0 = len([t for t in trades_ref if t["result"] == "SL"])
    b0 = len([t for t in trades_ref if t["result"] == "BE"])
    pnl0 = sum(t["pnl_r"] for t in trades_ref if t["pnl_r"] is not None)
    dd0 = calc_max_dd(trades_ref)

    for name, tds, fc, fb, fs in [
        (f"Ichimoku {timeframe.upper()}", trades_m5, filt_m5, fb_m5, fs_m5),
        ("Ichimoku M30", trades_htf, filt_htf, fb_htf, fs_htf),
    ]:
        w = len([t for t in tds if t["result"] == "TP"])
        l = len([t for t in tds if t["result"] == "SL"])
        b = len([t for t in tds if t["result"] == "BE"])
        pnl = sum(t["pnl_r"] for t in tds if t["pnl_r"] is not None)
        dd = calc_max_dd(tds)

        print(f"\n  📊 {name} FILTER IMPACT:")
        print(f"     Signals filtered: {fc} ({fb} BULL when T<K, {fs} BEAR when T>K)")
        print(f"     Trades:  {w0+l0+b0} → {w+l+b} ({w+l+b-(w0+l0+b0):+d})")
        print(f"     Wins:    {w0} → {w} ({w-w0:+d})")
        print(f"     Losses:  {l0} → {l} ({l-l0:+d})")
        print(f"     BE:      {b0} → {b} ({b-b0:+d})")
        print(f"     Net R:   {pnl0:+.1f} → {pnl:+.1f} ({pnl-pnl0:+.1f}R)")
        print(f"     MaxDD:   {dd0:.1f}R → {dd:.1f}R ({dd-dd0:+.1f}R)")

    # ── Analyze what was filtered from baseline ──
    print(f"\n{'='*80}")
    print(f"  WHAT EACH FILTER REMOVES (from no-filter baseline)")
    print(f"{'='*80}")

    for filt_name, tenkan_arr, kijun_arr in [
        (f"Ichimoku {timeframe.upper()}", pre["tenkan_m5"], pre["kijun_m5"]),
        ("Ichimoku M30", pre["tenkan_htf"], pre["kijun_htf"]),
    ]:
        aligned = []
        against = []
        for t in trades_ref:
            if t["result"] == "OPEN":
                continue
            bar = t["entry_bar"]
            tv = tenkan_arr[bar]
            kv = kijun_arr[bar]
            if np.isnan(tv) or np.isnan(kv) or tv == 0 or kv == 0:
                against.append(t)
                continue
            if (t["side"] == "BULL" and tv > kv) or (t["side"] == "BEAR" and tv < kv):
                aligned.append(t)
            else:
                against.append(t)

        print(f"\n  {filt_name}:")
        for label, group in [("KEPT (aligned T/K)", aligned),
                              ("REMOVED (against T/K)", against)]:
            if not group:
                print(f"     {label}: 0 trades")
                continue
            w = len([t for t in group if t["result"] == "TP"])
            l = len([t for t in group if t["result"] == "SL"])
            b = len([t for t in group if t["result"] == "BE"])
            tot = w + l + b
            wr = w / tot * 100 if tot > 0 else 0
            pnl = sum(t["pnl_r"] for t in group if t["pnl_r"] is not None)
            print(f"     {label}:")
            print(f"       {tot} trades | ✅ {w} TP | ❌ {l} SL | 🔄 {b} BE")
            print(f"       WR: {wr:.1f}% | Net: {pnl:+.1f}R (${pnl*1000:+,.0f}) | R/trade: {pnl/tot:+.2f}R")

    # ── Export ──
    print(f"\n📁 Exporting CSVs...")
    export_csv(trades_ref, f"goat_ichi_{sym_safe}_{timeframe}_05be_nofilt.csv")
    export_csv(trades_m5, f"goat_ichi_{sym_safe}_{timeframe}_05be_ichi_m5.csv")
    export_csv(trades_htf, f"goat_ichi_{sym_safe}_{timeframe}_05be_ichi_m30.csv")

    print(f"\n{'='*60}")
    print(f"  💡 Open {symbol} {timeframe} on TradingView")
    print(f"  💡 Add Ichimoku (20,60,120,30) to validate TK crosses")
    print(f"{'='*60}")