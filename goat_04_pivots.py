"""
04 — Pivot detection: HA pivots (for GOAT Case 3), regular pivots (for SL/TP),
     and trade level calculation.
"""

HA_PIVOT_LENGTH = 2


# ─── HA Pivot Detection (for GOAT Case 3) ───────────────────────

def find_ha_pivot_lows(df, start_idx, end_idx):
    """Find Heikin-Ashi pivot lows within [start_idx, end_idx)."""
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
    """Find Heikin-Ashi pivot highs within [start_idx, end_idx)."""
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


# ─── Regular Candle Pivots (for SL/TP) ──────────────────────────

def find_pivot_low(df_regular, before_idx, entry_price, pivot_length):
    """Find the nearest regular-candle pivot low below entry price."""
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
    """Find the nearest regular-candle pivot high above entry price."""
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


# ─── Trade Level Calculation ─────────────────────────────────────

def calculate_trade_levels(ha_df, trigger_idx, side, rr_ratio, signal_bar=None, sweep_source_bar=None):
    """Calculate entry / SL / TP from HA pivots around the trigger bar.

    signal_bar: if provided, used as validate_end (the bar up to which we check
                that no bar broke through the pivot). Falls back to trigger_idx when None.
    sweep_source_bar: if provided, search pivots only from sweep_source_bar to
                      trigger_idx (inclusive). This constrains the SL range to the
                      sweep source → trigger bar range. Falls back to 0 when None.
    """
    search_start = sweep_source_bar if sweep_source_bar is not None else 0
    search_end = trigger_idx + 1
    validate_end = signal_bar if signal_bar is not None else trigger_idx
    entry = ha_df.loc[trigger_idx, 'HA_Close']

    if side == "BULL":
        pivots = find_ha_pivot_lows(ha_df, search_start, search_end)
        sl_candidates = [(idx, lvl) for idx, lvl in pivots if lvl < entry]
        if not sl_candidates:
            return None
        sl_candidates.sort(key=lambda x: x[0], reverse=True)
        valid_pivots = []
        for piv_idx, piv_lvl in sl_candidates:
            valid = True
            for j in range(piv_idx + 1, validate_end + 1):
                if ha_df.loc[j, 'HA_Low'] < piv_lvl:
                    valid = False
                    break
            if valid:
                valid_pivots.append((piv_idx, piv_lvl))
        if not valid_pivots:
            return None
        pivot_idx, sl = min(valid_pivots, key=lambda x: x[1])
        risk = abs(entry - sl)
        tp = entry + rr_ratio * risk
    else:
        pivots = find_ha_pivot_highs(ha_df, search_start, search_end)
        sh_candidates = [(idx, lvl) for idx, lvl in pivots if lvl > entry]
        if not sh_candidates:
            return None
        sh_candidates.sort(key=lambda x: x[0], reverse=True)
        valid_pivots = []
        for piv_idx, piv_lvl in sh_candidates:
            valid = True
            for j in range(piv_idx + 1, validate_end + 1):
                if ha_df.loc[j, 'HA_High'] > piv_lvl:
                    valid = False
                    break
            if valid:
                valid_pivots.append((piv_idx, piv_lvl))
        if not valid_pivots:
            return None
        pivot_idx, sl = max(valid_pivots, key=lambda x: x[1])
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