"""
GOATv2 M30 Backtest using VectorBT
Downloads BTC 30m data from Bybit, replays your exact GOAT logic bar-by-bar,
then simulates trades WITH and WITHOUT 1.0R BE trigger.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_ohlcv(symbol="BTC/USDT:USDT", timeframe="30m", days_back=180):
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    exchange.load_markets()

    tf_ms = 30 * 60 * 1000  # 30 min in milliseconds
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (days_back * 86400 * 1000)
    limit = 1000
    chunk_ms = limit * tf_ms  # how far one request covers

    all_candles = []

    print(f"Downloading {symbol} {timeframe} data, last {days_back} days...")
    print(f"  From: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)}")
    print(f"  To:   {datetime.fromtimestamp(now_ms/1000, tz=timezone.utc)}")

    cursor = start_ms

    while cursor < now_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except Exception as e:
            print(f"  ⚠️ Error at cursor {cursor}: {e}, retrying...")
            import time
            time.sleep(2)
            continue

        if not candles:
            # No data at this point, jump forward one chunk
            cursor += chunk_ms
            continue

        all_candles.extend(candles)
        last_ts = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
        first_ts = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
        print(f"  Fetched {len(candles)} candles ({first_ts} → {last_ts}), "
              f"total: {len(all_candles)}")

        # Move cursor past the last candle we got
        cursor = candles[-1][0] + tf_ms

        if len(candles) < limit:
            # Less than full page = we've reached the end of available data
            # But keep going in case there's a gap
            cursor = candles[-1][0] + tf_ms

    # Deduplicate and sort
    df = pd.DataFrame(all_candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    print(f"\n✅ Downloaded {len(df)} candles")
    print(f"   From: {df['timestamp'].iloc[0]}")
    print(f"   To:   {df['timestamp'].iloc[-1]}")
    print(f"   Span: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} days")
    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: CORE GOAT LOGIC (copied from your modules)
# ═══════════════════════════════════════════════════════════════════

# ─── Heikin Ashi ─────────────────────────────────────────────────

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    ha_df['HA_Open'] = np.nan
    if len(ha_df) > 0:
        ha_df.loc[0, 'HA_Open'] = (ha_df.loc[0, 'open'] + ha_df.loc[0, 'close']) / 2
        for i in range(1, len(ha_df)):
            ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2
        for i in range(len(ha_df)):
            ha_df.loc[i, 'HA_High'] = max(ha_df.loc[i, 'high'],
                                            max(ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close']))
            ha_df.loc[i, 'HA_Low'] = min(ha_df.loc[i, 'low'],
                                           min(ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close']))
    return ha_df


# ─── LG / LGC / LGCR Detection ──────────────────────────────────

def detect_lg(df):
    lg_levels = []
    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        if prev['HA_High'] < cur['HA_Close']:
            lg_levels.append({'type': 'bullish', 'index': i, 'price': cur['HA_Low']})
        if prev['HA_Low'] > cur['HA_Close']:
            lg_levels.append({'type': 'bearish', 'index': i, 'price': cur['HA_High']})
    return lg_levels


def detect_lgc(df):
    lgc_levels = []
    for i in range(2, len(df)):
        two_back = df.iloc[i - 2]
        one_back = df.iloc[i - 1]
        cur = df.iloc[i]
        if (two_back['HA_High'] < cur['HA_Low'] and
                one_back['HA_Close'] > one_back['HA_Open']):
            lgc_levels.append({'type': 'bullish', 'index': i - 1,
                               'line_level': two_back['HA_High'], 'is_untested': True})
        if (two_back['HA_Low'] > cur['HA_High'] and
                one_back['HA_Close'] < one_back['HA_Open']):
            lgc_levels.append({'type': 'bearish', 'index': i - 1,
                               'line_level': two_back['HA_Low'], 'is_untested': True})
    return lgc_levels


def detect_lgcr(df, lg_levels):
    lgcr_levels = []
    ref_bull = None
    ref_bear = None
    bull_flag = False
    bear_flag = False
    for i in range(len(df)):
        cur = df.iloc[i]
        is_bull_lg = any(lg['type'] == 'bullish' and lg['index'] == i for lg in lg_levels)
        is_bear_lg = any(lg['type'] == 'bearish' and lg['index'] == i for lg in lg_levels)
        if is_bull_lg:
            ref_bear = cur['HA_Low']
            bear_flag = False
        if is_bear_lg:
            ref_bull = cur['HA_High']
            bull_flag = False
        if (is_bull_lg and ref_bull is not None and
                cur['HA_Close'] > ref_bull and not bull_flag):
            lgcr_levels.append({'type': 'bullish', 'index': i, 'price': cur['HA_Low']})
            bull_flag = True
        if (is_bear_lg and ref_bear is not None and
                cur['HA_Close'] < ref_bear and not bear_flag):
            lgcr_levels.append({'type': 'bearish', 'index': i, 'price': cur['HA_High']})
            bear_flag = True
    return lgcr_levels


def detect_patterns(df):
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


# ─── Pivot Detection ────────────────────────────────────────────

HA_PIVOT_LENGTH = 2

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


# ─── Direction Config ────────────────────────────────────────────

def compare(value, threshold, op):
    if op == "gt":   return value > threshold
    elif op == "lt": return value < threshold
    elif op == "gte": return value >= threshold
    elif op == "lte": return value <= threshold
    return False


def sweep_reaches(wick_value, line_value, tolerance_factor):
    if tolerance_factor < 1:
        return wick_value >= line_value * tolerance_factor
    else:
        return wick_value <= line_value * tolerance_factor


def body_intersects_level(df, idx, level):
    body_low = min(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    body_high = max(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    return body_low <= level <= body_high


def get_direction_config(side):
    if side == "BEAR":
        return {
            "side": "BEAR",
            "lgc_col": "bearish_LGC", "lgcr_col": "bearish_LGCR",
            "lgc_line_col": "bearish_LGC_line",
            "lgcr_line1_col": "HA_Close", "lgcr_line2_col": "HA_High",
            "sweep_col": "HA_High", "sweep_level_col": "HA_Low",
            "line_above_price": True,
            "invalidate_op": "gt", "tolerance_factor": 0.999999,
            "lgcr_swept_label_1": "LGCR_HA_high_swept",
            "lgcr_swept_label_2": "LGCR_HA_close_swept",
            "pivot_swept_label": "pivot_high_swept",
        }
    else:
        return {
            "side": "BULL",
            "lgc_col": "bullish_LGC", "lgcr_col": "bullish_LGCR",
            "lgc_line_col": "bullish_LGC_line",
            "lgcr_line1_col": "HA_Close", "lgcr_line2_col": "HA_Low",
            "sweep_col": "HA_Low", "sweep_level_col": "HA_High",
            "line_above_price": False,
            "invalidate_op": "lt", "tolerance_factor": 1.000001,
            "lgcr_swept_label_1": "LGCR_HA_low_swept",
            "lgcr_swept_label_2": "LGCR_HA_close_swept",
            "pivot_swept_label": "pivot_low_swept",
        }


# ─── LGCR Selectors ─────────────────────────────────────────────

def select_prior_lgcrs_bear(df, cur):
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bearish_LGCR']:
            if df.loc[i, 'HA_Close'] > trigger_close or df.loc[i, 'HA_High'] > trigger_close:
                prior.append(i)
    if not prior:
        return []
    return sorted(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_High'] - trigger_close)
    ))


def select_prior_lgcrs_bull(df, cur):
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bullish_LGCR']:
            if df.loc[i, 'HA_Close'] < trigger_close or df.loc[i, 'HA_Low'] < trigger_close:
                prior.append(i)
    if not prior:
        return []
    return sorted(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_Low'] - trigger_close)
    ))


# ─── Projection-first: candidate sweep bar finder ───────────────

def _find_sweep_candidates(df, cur, body_low, body_high, dcfg):
    """
    Projection-first sweep candidate finder.

    Scans backward from cur-1 and returns bar indices k where:
      - The projection (HA_Low for BEAR, HA_High for BULL) falls inside
        the trigger bar body [body_low, body_high].
      - A real wick exists at bar k.
      - No bar body between k+1 and cur-1 intersects the projection level.
      - Sweep bar k has not been re-swept by a subsequent bar before the trigger.

    Returns list of candidate bar indices (most recent first).
    """
    side = dcfg["side"]
    candidates = []

    for k in range(cur - 1, -1, -1):
        proj = df.loc[k, dcfg["sweep_level_col"]]

        if not (body_low <= proj <= body_high):
            continue

        wick = df.loc[k, dcfg["sweep_col"]]
        ha_open_k = df.loc[k, 'HA_Open']
        ha_close_k = df.loc[k, 'HA_Close']
        body_top_k = max(ha_open_k, ha_close_k)
        body_bot_k = min(ha_open_k, ha_close_k)

        if side == "BEAR" and wick <= body_top_k:
            continue
        if side == "BULL" and wick >= body_bot_k:
            continue

        fwd_block = False
        for j in range(k + 1, cur):
            if body_intersects_level(df, j, proj):
                fwd_block = True
                break
        if fwd_block:
            continue

        reswept = False
        for j in range(k + 1, cur):
            if side == "BEAR" and df.loc[j, 'HA_High'] > df.loc[k, 'HA_High']:
                reswept = True
                break
            if side == "BULL" and df.loc[j, 'HA_Low'] < df.loc[k, 'HA_Low']:
                reswept = True
                break
        if reswept:
            continue

        candidates.append(k)

    return candidates


# ─── Case 1: LGCR Sweep (given sweep bar k) ─────────────────────

def _check_c1_for_sweep(df, cur, k, dcfg, all_lgcrs):
    """
    Case 1 (projection-first): check if sweep bar k swept a valid prior LGCR line.

    Checks:
      1. Line validity from LGCR bar to k (no close through)
      1.5. First wick counts: earlier wick between source and k consumes the line
      2. k's wick reaches the LGCR line
      3. k's close does NOT break through the line (wick-only)
      4. First sweep check: k is the first bar after prior_idx to reach the swept level
      5. Highest/lowest wick check: k has most extreme wick in [prior_idx+1, cur-1]
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']

    all_priors = [i for i in all_lgcrs if i < k]
    if not all_priors:
        return False, None, None, None, None

    for prior_idx in all_priors:
        line1 = df.loc[prior_idx, dcfg["lgcr_line1_col"]]
        line2 = df.loc[prior_idx, dcfg["lgcr_line2_col"]]

        # ── 1. Line validity from LGCR bar to k (no close through) ──
        line1_valid = True
        line2_valid = True
        for j in range(prior_idx + 1, k):
            c = df.loc[j, 'HA_Close']
            if line1_valid and compare(c, line1, dcfg["invalidate_op"]):
                line1_valid = False
            if line2_valid and compare(c, line2, dcfg["invalidate_op"]):
                line2_valid = False
            if not line1_valid and not line2_valid:
                break

        if not line1_valid and not line2_valid:
            continue

        # ── 1.5. First wick counts: earlier wick consumes the line ──
        for j in range(prior_idx + 1, k):
            j_wick = df.loc[j, dcfg["sweep_col"]]
            if line2_valid and sweep_reaches(j_wick, line2, dcfg["tolerance_factor"]):
                line2_valid = False
            if line1_valid and sweep_reaches(j_wick, line1, dcfg["tolerance_factor"]):
                line1_valid = False
            if not line1_valid and not line2_valid:
                break
        if not line1_valid and not line2_valid:
            continue

        # ── 2. k's wick reaches a valid line ──
        swept_line2 = line2_valid and sweep_reaches(wick, line2, dcfg["tolerance_factor"])
        swept_line1 = line1_valid and sweep_reaches(wick, line1, dcfg["tolerance_factor"])
        if not (swept_line1 or swept_line2):
            continue

        # ── 3. Wick-only (close must NOT break through the swept line) ──
        swept_ref = line2 if swept_line2 else line1
        if side == "BEAR" and sweep_close > swept_ref:
            continue
        if side == "BULL" and sweep_close < swept_ref:
            continue

        # ── 4. First sweep check: k must be the first bar after prior_idx to reach swept level ──
        first_sweep_ok = True
        for j in range(prior_idx + 1, k):
            if sweep_reaches(df.loc[j, dcfg["sweep_col"]], swept_ref, dcfg["tolerance_factor"]):
                first_sweep_ok = False
                break
        if not first_sweep_ok:
            continue

        # ── 5. Highest/lowest wick check: k must have the most extreme wick in [prior_idx+1, cur-1] ──
        k_wick = df.loc[k, dcfg["sweep_col"]]
        extreme_ok = True
        for j in range(prior_idx + 1, cur):
            if j == k:
                continue
            j_wick = df.loc[j, dcfg["sweep_col"]]
            if side == "BEAR" and j_wick > k_wick:
                extreme_ok = False
                break
            if side == "BULL" and j_wick < k_wick:
                extreme_ok = False
                break
        if not extreme_ok:
            continue

        swept_label = dcfg["lgcr_swept_label_1"] if swept_line2 else dcfg["lgcr_swept_label_2"]
        swept_value = line2 if swept_line2 else line1
        return True, "LGCR", swept_label, swept_value, prior_idx

    return False, None, None, None, None


# ─── Case 2: LG Line Sweep (given sweep bar k) ──────────────────

def _check_c2_for_sweep(df, cur, k, dcfg):
    """
    Case 2 (projection-first): check if sweep bar k swept a valid prior LG line.

    Checks:
      1. Line validity from LGC bar to k (no close through)
      1.5. First wick counts: earlier wick between source and k consumes the line
      2. k's wick reaches the LG line
      3. k's close does NOT break through the line (wick-only)
      4. First sweep check: k is the first bar after lgc_idx to reach the line level
      5. Highest/lowest wick check: k has most extreme wick in [lgc_idx+1, cur-1]
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']
    cur_price = df.loc[cur, 'HA_Close']

    candidates_lg = []
    for i in range(k - 1, -1, -1):
        if df.loc[i, dcfg["lgc_col"]]:
            line_level = df.loc[i, dcfg["lgc_line_col"]]
            if pd.isna(line_level):
                continue
            if dcfg["line_above_price"] and line_level >= cur_price:
                candidates_lg.append((i, line_level))
            elif not dcfg["line_above_price"] and line_level <= cur_price:
                candidates_lg.append((i, line_level))

    if not candidates_lg:
        return False, None, None, None, None

    candidates_lg.sort(key=lambda x: abs(x[1] - cur_price))

    for lgc_idx, line_level in candidates_lg:
        # ── 1. Line validity from LGC bar to k (no close through) ──
        line_valid = True
        for j in range(lgc_idx + 1, k):
            if compare(df.loc[j, 'HA_Close'], line_level, dcfg["invalidate_op"]):
                line_valid = False
                break
        if not line_valid:
            continue

        # ── 1.5. First wick counts: earlier wick consumes the line ──
        line_consumed = False
        for j in range(lgc_idx + 1, k):
            if sweep_reaches(df.loc[j, dcfg["sweep_col"]], line_level, dcfg["tolerance_factor"]):
                line_consumed = True
                break
        if line_consumed:
            continue

        # ── 2. k's wick reaches the line ──
        if not sweep_reaches(wick, line_level, dcfg["tolerance_factor"]):
            continue

        # ── 3. Wick-only (close must NOT break through the line) ──
        if side == "BEAR" and sweep_close > line_level:
            continue
        if side == "BULL" and sweep_close < line_level:
            continue

        # ── 4. First sweep check: k must be the first bar after lgc_idx to reach line_level ──
        first_sweep_ok = True
        for j in range(lgc_idx + 1, k):
            if sweep_reaches(df.loc[j, dcfg["sweep_col"]], line_level, dcfg["tolerance_factor"]):
                first_sweep_ok = False
                break
        if not first_sweep_ok:
            continue

        # ── 5. Highest/lowest wick check: k must have the most extreme wick in [lgc_idx+1, cur-1] ──
        k_wick = df.loc[k, dcfg["sweep_col"]]
        extreme_ok = True
        for j in range(lgc_idx + 1, cur):
            if j == k:
                continue
            j_wick = df.loc[j, dcfg["sweep_col"]]
            if side == "BEAR" and j_wick > k_wick:
                extreme_ok = False
                break
            if side == "BULL" and j_wick < k_wick:
                extreme_ok = False
                break
        if not extreme_ok:
            continue

        return True, "LG_LINE", "LG_line_swept", line_level, lgc_idx

    return False, None, None, None, None


# ─── Case 3: Pivot Sweep (given sweep bar k) ────────────────────

def _check_c3_for_sweep(df, cur, k, dcfg):
    """
    Case 3 (projection-first): check if sweep bar k swept a valid prior pivot.

    Checks:
      1. Pivot validity from pivot bar to k (no close through)
      1.5. Sweep-of-sweep chain: pivot consumed if intermediate pivot already swept it
      2. k's wick reaches the pivot level
      3. k's close does NOT break through the pivot (wick-only)
      4. First sweep check: k is the first bar after pivot_idx to reach pivot level
      5. Highest/lowest wick check: k has most extreme wick in [pivot_idx+1, cur-1]
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']
    cur_price = df.loc[cur, 'HA_Close']

    if side == "BEAR":
        pivots = find_ha_pivot_highs(df, 0, k)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl >= cur_price]
    else:
        pivots = find_ha_pivot_lows(df, 0, k)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl <= cur_price]

    if not pivot_candidates:
        return False, None, None, None, None

    pivot_candidates.sort(key=lambda x: abs(x[1] - cur_price))

    for pivot_idx, pivot_level in pivot_candidates:
        # ── 1. Pivot validity from pivot bar to k (no close through) ──
        pivot_valid = True
        for j in range(pivot_idx + 1, k):
            if compare(df.loc[j, 'HA_Close'], pivot_level, dcfg["invalidate_op"]):
                pivot_valid = False
                break
        if not pivot_valid:
            continue

        # ── 1.5. Sweep-of-sweep chain: consumed if intermediate pivot already swept it ──
        pivot_consumed = False
        for j_idx, _ in pivots:
            if j_idx <= pivot_idx:
                continue
            if j_idx >= k:
                break
            if sweep_reaches(df.loc[j_idx, dcfg["sweep_col"]], pivot_level, dcfg["tolerance_factor"]):
                pivot_consumed = True
                break
        if pivot_consumed:
            continue

        # ── 2. k's wick reaches the pivot level ──
        if not sweep_reaches(wick, pivot_level, dcfg["tolerance_factor"]):
            continue

        # ── 3. Wick-only (close must NOT break through the pivot) ──
        if side == "BEAR" and sweep_close > pivot_level:
            continue
        if side == "BULL" and sweep_close < pivot_level:
            continue

        # ── 4. First sweep check: k must be the first bar after pivot_idx to reach pivot level ──
        first_sweep_ok = True
        for j in range(pivot_idx + 1, k):
            if sweep_reaches(df.loc[j, dcfg["sweep_col"]], pivot_level, dcfg["tolerance_factor"]):
                first_sweep_ok = False
                break
        if not first_sweep_ok:
            continue

        # ── 5. Highest/lowest wick check: k must have the most extreme wick in [pivot_idx+1, cur-1] ──
        k_wick = df.loc[k, dcfg["sweep_col"]]
        extreme_ok = True
        for j in range(pivot_idx + 1, cur):
            if j == k:
                continue
            j_wick = df.loc[j, dcfg["sweep_col"]]
            if side == "BEAR" and j_wick > k_wick:
                extreme_ok = False
                break
            if side == "BULL" and j_wick < k_wick:
                extreme_ok = False
                break
        if not extreme_ok:
            continue

        return True, "PIVOT", dcfg["pivot_swept_label"], pivot_level, pivot_idx

    return False, None, None, None, None


# ─── Unified GOAT Check ─────────────────────────────────────────

def check_goat(df, side):
    """
    Unified GOAT check using projection-first sweep detection.

    Returns: (triggered, case_label, swept_label, swept_value, source_bar_idx)
    """
    n = len(df)
    if n < 5:
        return False, None, None, None, None

    cur = n - 1
    dcfg = get_direction_config(side)

    has_lgc = df.loc[cur, dcfg["lgc_col"]]
    has_lgcr = df.loc[cur, dcfg["lgcr_col"]]
    if not (has_lgc and has_lgcr):
        return False, None, None, None, None

    body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
    body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])

    selector = select_prior_lgcrs_bear if side == "BEAR" else select_prior_lgcrs_bull
    all_lgcrs = selector(df, cur)

    sweep_candidates = _find_sweep_candidates(df, cur, body_low, body_high, dcfg)
    if not sweep_candidates:
        return False, None, None, None, None

    for k in sweep_candidates:
        triggered, case_label, swept_label, swept_value, source_bar_idx = \
            _check_c1_for_sweep(df, cur, k, dcfg, all_lgcrs)
        if triggered:
            return True, case_label, swept_label, swept_value, source_bar_idx

        triggered, case_label, swept_label, swept_value, source_bar_idx = \
            _check_c2_for_sweep(df, cur, k, dcfg)
        if triggered:
            return True, case_label, swept_label, swept_value, source_bar_idx

        triggered, case_label, swept_label, swept_value, source_bar_idx = \
            _check_c3_for_sweep(df, cur, k, dcfg)
        if triggered:
            return True, case_label, swept_label, swept_value, source_bar_idx

    return False, None, None, None, None


# ─── Trade Level Calculation ─────────────────────────────────────

def calculate_trade_levels(ha_df, trigger_idx, side, rr_ratio=3, signal_bar=None):
    search_end = (signal_bar + 1) if signal_bar is not None else (trigger_idx + 1)
    validate_end = signal_bar if signal_bar is not None else trigger_idx
    entry = ha_df.loc[trigger_idx, 'HA_Close']
    if side == "BULL":
        pivots = find_ha_pivot_lows(ha_df, 0, search_end)
        sl_cands = [(idx, lvl) for idx, lvl in pivots if lvl < entry]
        if not sl_cands:
            return None
        sl_cands.sort(key=lambda x: x[0], reverse=True)
        sl = None
        for piv_idx, piv_lvl in sl_cands:
            valid = True
            for j in range(piv_idx + 1, validate_end + 1):
                if ha_df.loc[j, 'HA_Low'] < piv_lvl:
                    valid = False
                    break
            if valid:
                sl = piv_lvl
                break
        if sl is None:
            return None
        risk = abs(entry - sl)
        tp = entry + rr_ratio * risk
    else:
        pivots = find_ha_pivot_highs(ha_df, 0, search_end)
        sl_cands = [(idx, lvl) for idx, lvl in pivots if lvl > entry]
        if not sl_cands:
            return None
        sl_cands.sort(key=lambda x: x[0], reverse=True)
        sl = None
        for piv_idx, piv_lvl in sl_cands:
            valid = True
            for j in range(piv_idx + 1, validate_end + 1):
                if ha_df.loc[j, 'HA_High'] > piv_lvl:
                    valid = False
                    break
            if valid:
                sl = piv_lvl
                break
        if sl is None:
            return None
        risk = abs(sl - entry)
        tp = entry - rr_ratio * risk
    return {"entry": entry, "sl": sl, "tp": tp, "risk": risk}


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df_raw, rr_ratio=3, be_trigger_r=None, warmup=300, analysis_window=200):
    """
    Replay GOATv2 bar-by-bar on historical data.
    be_trigger_r: if set (e.g. 1.0), move SL to entry when trade reaches this R in profit.
    """
    # Compute HA on full dataset
    ha_full = calculate_heikin_ashi(df_raw)
    ha_full = detect_patterns(ha_full)

    trades = []
    active = []  # list of dicts with trade info
    lookback_bars = 5

    start_bar = max(warmup, analysis_window)

    print(f"\n{'='*60}")
    print(f"BACKTEST: {'NO BE' if be_trigger_r is None else f'{be_trigger_r}R BE TRIGGER'}")
    print(f"{'='*60}")
    print(f"Total bars: {len(ha_full)}, starting scan at bar {start_bar}")

    for bar in range(start_bar, len(ha_full)):
        # ── Check active trades against this bar's regular candle data ──
        trades_to_close = []
        for t in active:
            candle_high = df_raw.loc[bar, 'high']
            candle_low = df_raw.loc[bar, 'low']
            risk = t["risk"]

            if t["side"] == "BULL":
                if risk > 0:
                    r_reached = (candle_high - t["entry"]) / risk
                    t["max_r"] = max(t["max_r"], r_reached)

                # Check BE trigger
                if be_trigger_r is not None and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = t["entry"]  # Move SL to breakeven

                if candle_low <= t["sl"]:
                    pnl_r = 0.0 if t["be_active"] and t["sl"] == t["entry"] else -1.0
                    trades_to_close.append((t, "BE" if pnl_r == 0 else "SL", pnl_r))
                elif candle_high >= t["tp"]:
                    trades_to_close.append((t, "TP", rr_ratio))
            else:  # BEAR
                if risk > 0:
                    r_reached = (t["entry"] - candle_low) / risk
                    t["max_r"] = max(t["max_r"], r_reached)

                if be_trigger_r is not None and not t["be_active"] and t["max_r"] >= be_trigger_r:
                    t["be_active"] = True
                    t["sl"] = t["entry"]

                if candle_high >= t["sl"]:
                    pnl_r = 0.0 if t["be_active"] and t["sl"] == t["entry"] else -1.0
                    trades_to_close.append((t, "BE" if pnl_r == 0 else "SL", pnl_r))
                elif candle_low <= t["tp"]:
                    trades_to_close.append((t, "TP", rr_ratio))

        for t, result, pnl_r in trades_to_close:
            t["result"] = result
            t["pnl_r"] = pnl_r
            t["exit_bar"] = bar
            t["exit_ts"] = ha_full.loc[bar, 'timestamp']
            trades.append(t)
            active.remove(t)

        # ── Scan for new signals ──
        window_start = max(bar - lookback_bars, 0)
        for check_idx in range(bar, window_start - 1, -1):
            is_bear = (ha_full.loc[check_idx, "bearish_LGC"] and
                       ha_full.loc[check_idx, "bearish_LGCR"])
            is_bull = (ha_full.loc[check_idx, "bullish_LGC"] and
                       ha_full.loc[check_idx, "bullish_LGCR"])
            if not (is_bear or is_bull):
                continue

            sub_df = ha_full.iloc[:check_idx + 1].reset_index(drop=True)

            for side, is_active in [("BEAR", is_bear), ("BULL", is_bull)]:
                if not is_active:
                    continue

                triggered, case_label, swept_label, swept_val, source_bar_idx = check_goat(sub_df, side)
                if not triggered:
                    continue

                # Check if we already have a trade from this signal
                sig_key = (check_idx, side)
                already = any(t.get("sig_key") == sig_key for t in active)
                already2 = any(t.get("sig_key") == sig_key for t in trades)
                if already or already2:
                    continue

                levels = calculate_trade_levels(ha_full, check_idx, side, rr_ratio, signal_bar=bar)
                if not levels:
                    continue

                active.append({
                    "sig_key": sig_key,
                    "side": side,
                    "case": case_label,
                    "entry": levels["entry"],
                    "sl": levels["sl"],
                    "tp": levels["tp"],
                    "risk": levels["risk"],
                    "entry_bar": bar,
                    "entry_ts": ha_full.loc[bar, 'timestamp'],
                    "max_r": 0.0,
                    "be_active": False,
                    "result": None,
                    "pnl_r": None,
                    "exit_bar": None,
                    "exit_ts": None,
                })

        # Progress
        if bar % 200 == 0:
            print(f"  Bar {bar}/{len(ha_full)-1}, signals so far: {len(trades) + len(active)}, "
                  f"active: {len(active)}")

    # Close remaining active trades as "OPEN"
    for t in active:
        t["result"] = "OPEN"
        t["pnl_r"] = 0
        trades.append(t)

    return trades


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: RESULTS
# ═══════════════════════════════════════════════════════════════════

def print_results(trades, label, rr_ratio=3):
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")

    if not trades:
        print("No trades generated.")
        return

    wins = [t for t in trades if t["result"] == "TP"]
    losses = [t for t in trades if t["result"] == "SL"]
    bes = [t for t in trades if t["result"] == "BE"]
    opens = [t for t in trades if t["result"] == "OPEN"]

    total_closed = len(wins) + len(losses) + len(bes)
    total_pnl_r = sum(t["pnl_r"] for t in trades if t["pnl_r"] is not None)

    print(f"Total signals: {len(trades)}")
    print(f"  ✅ Wins (TP):    {len(wins)}")
    print(f"  ❌ Losses (SL):  {len(losses)}")
    print(f"  🔄 Breakeven:    {len(bes)}")
    print(f"  ⏳ Still open:   {len(opens)}")
    print(f"")

    if total_closed > 0:
        wr = len(wins) / total_closed * 100
        print(f"Win Rate: {wr:.1f}% ({len(wins)}/{total_closed})")

    print(f"")
    print(f"Net PnL: {total_pnl_r:+.1f}R")
    print(f"  Wins:   +{len(wins) * rr_ratio:.1f}R")
    print(f"  Losses: -{len(losses) * 1:.1f}R")
    if bes:
        print(f"  BE:      {len(bes)} × 0R")

    print(f"\n💰 If 1R = $1,000:")
    print(f"  Net PnL: ${total_pnl_r * 1000:+,.0f}")
    print(f"  Wins:    ${len(wins) * rr_ratio * 1000:+,}")
    print(f"  Losses:  ${len(losses) * -1000:+,}")

    # Max R distribution for losses
    if losses:
        max_rs = [t["max_r"] for t in losses]
        print(f"\n📐 Loss Max-R distribution:")
        for threshold in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            count = sum(1 for r in max_rs if r >= threshold)
            print(f"  ≥ {threshold}R: {count}/{len(losses)} ({count/len(losses)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GOATv2 M30 BACKTEST — BTC/USDT:USDT Perpetual")
    print("=" * 60)

    # Download data
    df_raw = download_ohlcv(symbol="BTC/USDT:USDT", timeframe="30m", days_back=180)

    # Run WITHOUT BE
    trades_no_be = run_backtest(df_raw, rr_ratio=3, be_trigger_r=None)
    print_results(trades_no_be, "WITHOUT Breakeven", rr_ratio=3)

    # Run WITH 1.0R BE
    trades_with_be = run_backtest(df_raw, rr_ratio=3, be_trigger_r=1.0)
    print_results(trades_with_be, "WITH 1.0R Breakeven", rr_ratio=3)

    # Comparison
    pnl_no = sum(t["pnl_r"] for t in trades_no_be if t["pnl_r"] is not None)
    pnl_be = sum(t["pnl_r"] for t in trades_with_be if t["pnl_r"] is not None)
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"Without BE: {pnl_no:+.1f}R  (${pnl_no*1000:+,.0f})")
    print(f"With 1.0R BE: {pnl_be:+.1f}R  (${pnl_be*1000:+,.0f})")
    print(f"Improvement: {pnl_be - pnl_no:+.1f}R  (${(pnl_be-pnl_no)*1000:+,.0f})")
    print(f"{'='*60}")