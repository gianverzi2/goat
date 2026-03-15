"""
goat_live/run.py
Main entry point for the GOAT live trading bot.

Run from the repo root:
    python -m goat_live.run

The bot polls Bybit every ~10 seconds, waits for a new closed 1m candle,
evaluates the GOAT signal (C1/C2/C3 for BULL and BEAR), and if a signal
fires it opens a market order with SL and TP.

Safety features:
- DRY_RUN mode (default OFF): logs everything without placing orders.
- 1 open position at a time: checks exchange before entering.
- Reduce-only SL/TP orders.
- State persistence: avoids double-trading after restart.
- Graceful shutdown on SIGINT / SIGTERM.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Resolve repo root so we can import the existing GOAT modules
# ---------------------------------------------------------------------------
import os

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Local imports (after path fix)
# ---------------------------------------------------------------------------
from goat_live.config import load_config, setup_logging
from goat_live.exchange import BybitExchange
from goat_live.risk import calc_qty, get_trade_levels
from goat_live.signals import get_signal
from goat_live.state import (
    clear_open_trade,
    load_state,
    save_state,
    set_open_trade,
    update_last_ts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, _frame):
    global _shutdown_requested
    logger.info("Signal %s received — requesting graceful shutdown.", signum)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Trade helpers
# ---------------------------------------------------------------------------

def _entry_and_sl_tp_sides(signal_side: str):
    """Return (entry_order_side, sl_order_side, tp_order_side) for ccxt."""
    if signal_side == "BULL":
        return "buy", "sell", "sell"
    else:  # BEAR
        return "sell", "buy", "buy"


def _execute_trade(exchange: BybitExchange, cfg: dict, signal: dict) -> Optional[dict]:
    """
    Execute a GOAT trade:
      1. Calculate trade levels (entry/SL/TP).
      2. Calculate qty from $notional.
      3. Place market entry order.
      4. Place SL and TP reduce-only orders.
      5. Return a trade record dict (for state persistence).
    """
    ha_df = signal["ha_df"]
    trigger_idx = signal["trigger_idx"]
    side = signal["side"]

    # --- Trade levels ---
    levels = get_trade_levels(ha_df, trigger_idx, side, cfg["rr_ratio"])
    if levels is None:
        logger.warning("No trade levels available — skipping trade.")
        return None

    entry_price = levels["entry"]
    sl_price = levels["sl"]
    tp_price = levels["tp"]

    # --- Qty ---
    qty = calc_qty(cfg["notional_usd"], entry_price, exchange)

    entry_side, sl_side, tp_side = _entry_and_sl_tp_sides(side)

    logger.info(
        "Executing trade: %s %s | qty=%.6f | entry=%.6f | sl=%.6f | tp=%.6f | rr=%.1f",
        side, cfg["symbol"], qty, entry_price, sl_price, tp_price, levels["rr"],
    )

    # --- Place orders ---
    entry_order = exchange.place_market_order(entry_side, qty)
    sl_order = exchange.place_stop_loss(sl_side, qty, sl_price)
    tp_order = exchange.place_take_profit(tp_side, qty, tp_price)

    trade_record = {
        "side": side,
        "case_label": signal.get("case_label"),
        "entry": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "qty": qty,
        "entry_order_id": entry_order.get("id") if entry_order else None,
        "sl_order_id": sl_order.get("id") if sl_order else None,
        "tp_order_id": tp_order.get("id") if tp_order else None,
        "bar_ts": signal["bar_ts"],
    }
    return trade_record


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    setup_logging(cfg)

    logger.info(
        "GOAT Live Bot starting | symbol=%s tf=%s notional=$%.0f rr=%.1f leverage=%dx dry_run=%s",
        cfg["symbol"],
        cfg["timeframe"],
        cfg["notional_usd"],
        cfg["rr_ratio"],
        cfg["leverage"],
        cfg["dry_run"],
    )

    if cfg["dry_run"]:
        logger.warning(
            "*** DRY_RUN mode is ON — no real orders will be placed. ***"
        )

    if not cfg["api_key"] or not cfg["api_secret"]:
        if not cfg["dry_run"]:
            logger.error(
                "BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env when DRY_RUN=false. Exiting."
            )
            sys.exit(1)
        else:
            logger.warning("API credentials not set — OK since DRY_RUN=true.")

    # --- Connect to exchange ---
    exchange = BybitExchange(cfg)

    # --- Load persisted state (before any position-modifying operations) ---
    state = load_state()
    last_processed_ts: Optional[int] = state.get("last_processed_ts")
    logger.info("Loaded state: last_processed_ts=%s open_trade=%s",
                last_processed_ts, state.get("open_trade"))

    # --- Setup leverage only when no position is currently open ---
    existing_pos = exchange.get_open_position()
    if existing_pos is None:
        exchange.set_position_mode(hedge=cfg["hedge_mode"])
        exchange.set_leverage(cfg["leverage"])
    else:
        logger.warning(
            "Open position detected on startup (size=%s) — skipping leverage/mode change "
            "to avoid affecting existing margin.",
            existing_pos.get("contracts"),
        )

    try:
        while not _shutdown_requested:
            # --- 1. Check for open position ---
            open_position = exchange.get_open_position()
            if open_position is not None:
                logger.debug(
                    "Open position detected (size=%s) — skipping signal check.",
                    open_position.get("contracts"),
                )
                time.sleep(cfg["poll_interval_sec"])
                continue

            # Clear any stale open_trade in state if position is now flat
            if state.get("open_trade") is not None:
                logger.info("Position closed externally — clearing state open_trade.")
                state = clear_open_trade(state)
                save_state(state)

            # --- 2. Get signal ---
            try:
                signal = get_signal(exchange, cfg, last_processed_ts)
            except Exception as exc:
                logger.error("get_signal error: %s", exc, exc_info=True)
                logger.info("Backing off for 30s after API error.")
                time.sleep(30)
                continue

            if signal is None:
                time.sleep(cfg["poll_interval_sec"])
                continue

            bar_ts = signal["bar_ts"]

            # --- 3. Execute trade ---
            try:
                trade_record = _execute_trade(exchange, cfg, signal)
            except Exception as exc:
                logger.error("_execute_trade error: %s", exc, exc_info=True)
                # Still update last_processed_ts so we don't retry this bar
                state = update_last_ts(state, bar_ts)
                last_processed_ts = bar_ts
                save_state(state)
                time.sleep(cfg["poll_interval_sec"])
                continue

            # --- 4. Update state ---
            state = update_last_ts(state, bar_ts)
            last_processed_ts = bar_ts
            if trade_record is not None:
                state = set_open_trade(state, trade_record)
            save_state(state)

            time.sleep(cfg["poll_interval_sec"])

    finally:
        logger.info("Shutting down — cancelling open orders and closing exchange.")
        exchange.cancel_all_orders()
        exchange.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
