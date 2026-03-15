"""
goat_live/signals.py
Fetches 1m OHLCV from Bybit, computes Heikin-Ashi, runs detect_patterns()
and check_goat(), and returns a signal dict (or None if no signal).

Signal dict:
{
    "side":        "BULL" | "BEAR",
    "case_label":  str,          # e.g. "Case 1 (LGCR sweep)"
    "swept_label": str | None,
    "swept_value": float | None,
    "trigger_idx": int,          # integer position in the df
    "bar_ts":      int,          # Unix ms timestamp of the trigger bar
}

Returns None if no signal.
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heikin-Ashi calculation (standalone copy so we don't import the 94 KB file)
# ---------------------------------------------------------------------------

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Heikin-Ashi columns and append them to a copy of df.
    Input df must have columns: open, high, low, close.
    Adds: HA_Open, HA_Close, HA_High, HA_Low.
    """
    ha = df.copy()
    o = ha["open"].values.astype(np.float64)
    h = ha["high"].values.astype(np.float64)
    lo = ha["low"].values.astype(np.float64)
    c = ha["close"].values.astype(np.float64)

    ha_close = (o + h + lo + c) / 4.0
    ha_open = np.empty(len(ha))
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_body_max = np.maximum(ha_open, ha_close)
    ha_body_min = np.minimum(ha_open, ha_close)
    ha_high = np.maximum(h, ha_body_max)
    ha_low = np.minimum(lo, ha_body_min)

    ha["HA_Close"] = ha_close
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha_high
    ha["HA_Low"] = ha_low
    return ha


# ---------------------------------------------------------------------------
# Pattern detection wrapper (handles missing function_LG_major_functions)
# ---------------------------------------------------------------------------

def _import_detect_patterns():
    """
    Try to import detect_patterns from goat_03_patterns.
    Falls back to detect_all_patterns_numba from goat_20_vbt_equity if unavailable.
    Returns a callable or raises ImportError.
    """
    try:
        from goat_03_patterns import detect_patterns  # type: ignore[import]
        logger.debug("Using detect_patterns from goat_03_patterns.")
        return detect_patterns
    except ImportError as primary_err:
        logger.warning(
            "Could not import goat_03_patterns (%s). "
            "Trying fallback: detect_all_patterns_numba from goat_20_vbt_equity.",
            primary_err,
        )
        try:
            from goat_20_vbt_equity import detect_all_patterns_numba  # type: ignore[import]
            logger.info("Using detect_all_patterns_numba fallback.")
            return detect_all_patterns_numba
        except ImportError as fallback_err:
            raise ImportError(
                "Neither goat_03_patterns nor goat_20_vbt_equity pattern detection "
                f"could be imported.\nPrimary: {primary_err}\nFallback: {fallback_err}"
            ) from fallback_err


_detect_patterns_fn = None  # lazy init


def _get_detect_patterns():
    global _detect_patterns_fn
    if _detect_patterns_fn is None:
        _detect_patterns_fn = _import_detect_patterns()
    return _detect_patterns_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_closed_candles(exchange_obj, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch `limit` 1m candles from Bybit via the ccxt exchange object and
    drop the last (still-forming) candle so we only process closed bars.

    Retries up to 3 times with exponential backoff (5s, 10s, 15s) on rate
    limit errors before raising.

    Returns a DataFrame with columns: timestamp (ms int), open, high, low, close, volume.
    Index is a RangeIndex; timestamp column is kept as-is for state tracking.
    """
    _retry_delays = (5, 10, 15)
    last_exc = None
    for attempt, delay in enumerate((*_retry_delays, None), start=1):
        try:
            raw = exchange_obj.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            break
        except Exception as exc:
            exc_str = str(exc)
            if "RateLimitExceeded" in type(exc).__name__ or "10006" in exc_str:
                last_exc = exc
                if delay is not None:
                    logger.warning(
                        "fetch_ohlcv rate limit hit (attempt %d/%d) — retrying in %ds",
                        attempt, len(_retry_delays) + 1, delay,
                    )
                    time.sleep(delay)
                    continue
            raise
    else:
        raise last_exc or RuntimeError(
            "fetch_ohlcv failed after 3 rate-limit retries"
        )

    if not raw:
        raise RuntimeError("fetch_ohlcv returned empty list")

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.astype(
        {"timestamp": "int64", "open": float, "high": float, "low": float, "close": float, "volume": float}
    )

    # Drop the last row — it's the candle currently forming
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def get_signal(exchange_obj, cfg: dict, last_processed_ts: Optional[int] = None) -> Optional[dict]:
    """
    Main signal function called each iteration of the polling loop.

    Steps:
      1. Fetch closed 1m candles.
      2. Skip if we've already processed this bar (same timestamp as last_processed_ts).
      3. Compute Heikin-Ashi.
      4. Run detect_patterns().
      5. Call check_goat() for BULL and BEAR on the last bar.
      6. Return signal dict if any fires, else None.

    Returns:
        A signal dict (keys: side, case_label, swept_label, swept_value,
        trigger_idx, bar_ts, ha_df) if a signal fires, or None if no signal
        or the bar has already been processed.
    """
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    warmup_bars = cfg["warmup_bars"]
    rr_ratio = cfg["rr_ratio"]

    # --- 1. Fetch candles ---
    df = fetch_closed_candles(exchange_obj, symbol, timeframe, warmup_bars + 1)

    last_bar_ts = int(df["timestamp"].iloc[-1])
    logger.debug("Last closed bar ts=%d, last_processed_ts=%s", last_bar_ts, last_processed_ts)

    # --- 2. Dedup guard ---
    if last_processed_ts is not None and last_bar_ts <= last_processed_ts:
        logger.debug("Bar ts=%d already processed, skipping.", last_bar_ts)
        return None

    # --- 3. Heikin-Ashi ---
    ha_df = calculate_heikin_ashi(df)

    # --- 4. Pattern detection ---
    detect_fn = _get_detect_patterns()
    ha_df = detect_fn(ha_df)

    # --- 5. check_goat on the last bar ---
    from goat_05_cases import check_goat  # type: ignore[import]

    trigger_idx = len(ha_df) - 1

    for side in ("BULL", "BEAR"):
        triggered, case_label, swept_label, swept_value = check_goat(ha_df, side, symbol)
        if triggered:
            logger.info(
                "Signal: %s %s | %s | swept=%s @ %s",
                side, symbol, case_label, swept_label, swept_value,
            )
            return {
                "side": side,
                "case_label": case_label,
                "swept_label": swept_label,
                "swept_value": swept_value,
                "trigger_idx": trigger_idx,
                "bar_ts": last_bar_ts,
                "ha_df": ha_df,          # pass df downstream for level calc
            }

    logger.debug("No signal on bar ts=%d", last_bar_ts)
    return None
