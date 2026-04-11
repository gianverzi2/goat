"""
goat_live/risk.py
Position sizing and trade level helpers.

Fixed notional: $20 per trade.
qty = notional_usd / entry_price  (rounded down to lot step)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calc_qty(notional_usd: float, entry_price: float, exchange_obj) -> float:
    """
    Compute contract quantity for a fixed notional trade.

    qty = floor(notional_usd / entry_price, lot_step)

    Args:
        notional_usd:  Fixed dollar notional (e.g. 20.0).
        entry_price:   Expected fill price (HA_Close of the trigger bar).
        exchange_obj:  BybitExchange instance (for lot-step rounding).

    Returns:
        Rounded-down contract quantity (float).
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")

    raw_qty = notional_usd / entry_price
    qty = exchange_obj.round_qty(raw_qty)

    logger.debug(
        "qty calc: notional=%.2f entry=%.6f raw_qty=%.6f rounded_qty=%.6f",
        notional_usd, entry_price, raw_qty, qty,
    )

    if qty <= 0:
        raise ValueError(
            f"Calculated qty is zero after rounding. "
            f"notional={notional_usd} entry={entry_price} raw={raw_qty}. "
            f"The position size is too small for this symbol's lot step."
        )
    return qty


def get_trade_levels(ha_df, trigger_idx: int, side: str, rr_ratio: float, signal_bar: int = None, sweep_source_bar: int = None) -> Optional[dict]:
    """
    Delegate to goat_04_pivots.calculate_trade_levels() to compute
    entry, SL and TP prices from Heikin-Ashi pivot analysis.

    Args:
        ha_df:            Heikin-Ashi DataFrame with pattern columns.
        trigger_idx:      Integer row index of the trigger bar.
        side:             "BULL" or "BEAR".
        rr_ratio:         Risk-reward ratio (e.g. 3.0).
        signal_bar:       Kept for backward compatibility with existing callers; not
                          used in the new SL calculation logic.
        sweep_source_bar: Integer row index of the sweep source bar (e.g. the swept
                          pivot bar for C3, the LGCR bar for C1, the LGC bar for C2).
                          When provided, SL = nearest 2+1+2 HA pivot high/low in
                          [sweep_source_bar, trigger_idx] on the correct side of entry,
                          falling back to simple max/min if no pivot is found.
                          Defaults to None.

    Returns:
        Dict with keys: entry, sl, tp, risk, rr, HA_entry, HA_sl, HA_tp, ...
        Returns None if no suitable SL level found.
    """
    from goat_04_pivots import calculate_trade_levels  # type: ignore[import]

    levels = calculate_trade_levels(
        ha_df, trigger_idx, side, rr_ratio,
        signal_bar=signal_bar,
        sweep_source_bar=sweep_source_bar,
    )
    if levels is None:
        logger.warning(
            "calculate_trade_levels returned None for side=%s idx=%d — no pivot found.",
            side, trigger_idx,
        )
        return None

    logger.info(
        "Trade levels: entry=%.6f sl=%.6f tp=%.6f risk=%.6f rr=%.1f",
        levels["entry"], levels["sl"], levels["tp"], levels["risk"], levels["rr"],
    )
    return levels
