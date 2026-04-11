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
    """Calculate entry / SL / TP from HA data around the trigger bar.

    When sweep_source_bar is provided (C1/C2/C3), SL is the nearest 2+1+2 HA
    pivot high/low in [sweep_source_bar, trigger_idx] on the correct side of
    entry price.  Falls back to simple min(HA_Low)/max(HA_High) if no 2+1+2
    pivot is found in that range.

    When sweep_source_bar is NOT provided, searches the last 50 bars for a
    2+1+2 pivot low/high without the strict "no subsequent bar broke through"
    validation that caused most failures on valid setups.

    signal_bar: kept for backward compatibility with existing callers; not used
                in the new implementation.
    sweep_source_bar: if provided, defines the start of the SL search range.
    """
    entry = ha_df.loc[trigger_idx, 'HA_Close']

    if sweep_source_bar is not None:
        # Primary: nearest 2+1+2 HA pivot in [sweep_source_bar, trigger_idx].
        # find_ha_pivot_* end_idx is exclusive, so pass trigger_idx + 1.
        if side == "BULL":
            pivots = find_ha_pivot_lows(ha_df, sweep_source_bar, trigger_idx + 1)
            candidates = [(idx, lvl) for idx, lvl in pivots if lvl < entry]
            if candidates:
                # Lowest pivot low below entry in range
                pivot_idx, sl = min(candidates, key=lambda x: x[1])
            else:
                # Fallback: simple min(HA_Low) in the range
                range_slice = ha_df.loc[sweep_source_bar:trigger_idx]
                pivot_idx = int(range_slice['HA_Low'].idxmin())
                sl = ha_df.loc[pivot_idx, 'HA_Low']
                if sl >= entry:
                    return None
        else:
            pivots = find_ha_pivot_highs(ha_df, sweep_source_bar, trigger_idx + 1)
            candidates = [(idx, lvl) for idx, lvl in pivots if lvl > entry]
            if candidates:
                # Highest pivot high above entry in range
                pivot_idx, sl = max(candidates, key=lambda x: x[1])
            else:
                # Fallback: simple max(HA_High) in the range
                range_slice = ha_df.loc[sweep_source_bar:trigger_idx]
                pivot_idx = int(range_slice['HA_High'].idxmax())
                sl = ha_df.loc[pivot_idx, 'HA_High']
                if sl <= entry:
                    return None
    else:
        # Fallback: search last 50 bars for a pivot using 2+1+2 pattern.
        # No "no bar broke through" validation — that was causing valid setups to fail.
        lookback = 50
        search_start = max(0, trigger_idx - lookback)
        search_end = trigger_idx + 1
        if side == "BULL":
            pivots = find_ha_pivot_lows(ha_df, search_start, search_end)
            sl_candidates = [(idx, lvl) for idx, lvl in pivots if lvl < entry]
            if not sl_candidates:
                return None
            pivot_idx, sl = min(sl_candidates, key=lambda x: x[1])
        else:
            pivots = find_ha_pivot_highs(ha_df, search_start, search_end)
            sh_candidates = [(idx, lvl) for idx, lvl in pivots if lvl > entry]
            if not sh_candidates:
                return None
            pivot_idx, sl = max(sh_candidates, key=lambda x: x[1])

    risk = abs(entry - sl)
    tp = entry + rr_ratio * risk if side == "BULL" else entry - rr_ratio * risk

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