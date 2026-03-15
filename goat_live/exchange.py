"""
goat_live/exchange.py
ccxt Bybit wrapper: connect, place orders, cancel orders, query positions.
"""

import logging
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)


class BybitExchange:
    """Thin wrapper around ccxt.bybit for GOAT live bot operations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.dry_run = cfg["dry_run"]
        self.hedge_mode = cfg.get("hedge_mode", False)

        self.exchange = ccxt.bybit(
            {
                "apiKey": cfg["api_key"],
                "secret": cfg["api_secret"],
                "enableRateLimit": True,
                "rateLimit": 200,
                "options": {
                    "defaultType": "linear",  # linear perpetuals (USDT-margined)
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                },
            }
        )

        # Load markets so we can inspect contract specs
        self.exchange.load_markets()
        logger.info("Markets loaded. Connected to Bybit (mainnet).")

        # Cache market info for the trading symbol
        if self.symbol not in self.exchange.markets:
            raise ValueError(
                f"Symbol {self.symbol!r} not found on Bybit. "
                f"Check GOAT_SYMBOL in your .env."
            )
        self.market = self.exchange.markets[self.symbol]
        logger.info(
            "Market info for %s: contractSize=%s, precision=%s",
            self.symbol,
            self.market.get("contractSize"),
            self.market.get("precision"),
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def set_leverage(self, leverage: int) -> None:
        """Set leverage on both long and short sides."""
        if self.dry_run:
            logger.info("[DRY_RUN] Would set leverage to %sx", leverage)
            return
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            logger.info("Leverage set to %sx for %s", leverage, self.symbol)
        except Exception as exc:
            logger.warning("set_leverage failed (may already be set): %s", exc)

    def set_position_mode(self, hedge: bool = False) -> None:
        """Set one-way (hedge=False) or hedge position mode."""
        if self.dry_run:
            logger.info("[DRY_RUN] Would set position mode: hedge=%s", hedge)
            return
        try:
            self.exchange.set_position_mode(hedge, self.symbol)
            logger.info("Position mode set: hedge=%s", hedge)
        except Exception as exc:
            # Bybit raises if mode is already set correctly
            logger.debug("set_position_mode: %s (may already be correct)", exc)

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_position(self) -> Optional[dict]:
        """
        Return the open position dict for self.symbol if any, else None.
        Considers a position open when abs(contracts) > 0.
        In hedge mode, checks both long (positionIdx=1) and short (positionIdx=2) sides.
        """
        try:
            if self.hedge_mode:
                for position_idx in (1, 2):
                    positions = self.exchange.fetch_positions(
                        [self.symbol], params={"positionIdx": position_idx}
                    )
                    for pos in positions:
                        contracts = float(pos.get("contracts") or 0)
                        if abs(contracts) > 0:
                            return pos
            else:
                positions = self.exchange.fetch_positions([self.symbol])
                for pos in positions:
                    contracts = float(pos.get("contracts") or 0)
                    if abs(contracts) > 0:
                        return pos
        except Exception as exc:
            logger.error("fetch_positions error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Quantity helpers
    # ------------------------------------------------------------------

    def round_qty(self, raw_qty: float) -> float:
        """Round qty down to the exchange's lot step for this symbol."""
        precision = self.market.get("precision", {})
        amount_precision = precision.get("amount")
        if amount_precision is None:
            return raw_qty

        # Determine the step size from the precision value.
        # ccxt may express it as an integer (number of decimal places) or
        # as a float (lot step, e.g. 0.1, 1.0, 0.01).
        if isinstance(amount_precision, int):
            step = 10 ** (-amount_precision)
        else:
            step = float(amount_precision)

        if step <= 0:
            return raw_qty

        import math
        return math.floor(raw_qty / step) * step

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(self, side: str, qty: float) -> Optional[dict]:
        """
        Place a market order.
        side: 'buy' (long entry) or 'sell' (short entry)
        qty:  contract quantity (already rounded)
        Returns the order dict or None on failure / dry-run.
        """
        if self.dry_run:
            logger.info(
                "[DRY_RUN] Market %s %s @ market for %s",
                side.upper(), qty, self.symbol
            )
            return None

        try:
            params = {}
            if self.hedge_mode:
                params["positionIdx"] = 1 if side == "buy" else 2
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params,
            )
            logger.info(
                "Market %s order placed: id=%s qty=%s",
                side.upper(), order.get("id"), qty
            )
            return order
        except Exception as exc:
            logger.error("place_market_order failed: %s", exc)
            raise

    def place_stop_loss(self, side: str, qty: float, sl_price: float) -> Optional[dict]:
        """
        Place a reduce-only stop-loss market order.
        side: 'sell' for long position SL, 'buy' for short position SL

        Uses triggerPrice + triggerDirection (required by Bybit for all conditional orders).
        SL for LONG (side='sell'): price must drop → triggerDirection='below'.
        SL for SHORT (side='buy'): price must rise → triggerDirection='above'.
        """
        if self.dry_run:
            logger.info(
                "[DRY_RUN] Stop-loss %s %s @ %s for %s",
                side.upper(), qty, sl_price, self.symbol
            )
            return None

        # For a long SL the price needs to drop; for a short SL it needs to rise.
        trigger_direction = "below" if side == "sell" else "above"
        try:
            params = {
                "triggerPrice": sl_price,
                "triggerBy": "LastPrice",
                "triggerDirection": trigger_direction,
                "reduceOnly": True,
            }
            if self.hedge_mode:
                params["positionIdx"] = 1 if side == "sell" else 2
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params,
            )
            logger.info(
                "Stop-loss order placed: id=%s side=%s qty=%s sl=%.6f triggerDirection=%s",
                order.get("id"), side.upper(), qty, sl_price, trigger_direction
            )
            return order
        except Exception as exc:
            logger.error("place_stop_loss failed: %s", exc)
            raise

    def place_take_profit(self, side: str, qty: float, tp_price: float) -> Optional[dict]:
        """
        Place a reduce-only take-profit limit order (PostOnly for maker rebate).
        side: 'sell' for long position TP, 'buy' for short position TP

        Uses type='limit' + price + triggerPrice + triggerDirection (required by Bybit).
        TP for LONG (side='sell'): price must rise → triggerDirection='above'.
        TP for SHORT (side='buy'): price must drop → triggerDirection='below'.
        """
        if self.dry_run:
            logger.info(
                "[DRY_RUN] Take-profit %s %s @ %s for %s",
                side.upper(), qty, tp_price, self.symbol
            )
            return None

        # For a long TP the price needs to rise; for a short TP it needs to drop.
        trigger_direction = "above" if side == "sell" else "below"
        try:
            params = {
                "triggerPrice": tp_price,
                "triggerBy": "LastPrice",
                "triggerDirection": trigger_direction,
                "timeInForce": "PostOnly",
                "reduceOnly": True,
            }
            if self.hedge_mode:
                params["positionIdx"] = 1 if side == "sell" else 2
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side=side,
                amount=qty,
                price=tp_price,
                params=params,
            )
            logger.info(
                "Take-profit order placed: id=%s side=%s qty=%s tp=%.6f triggerDirection=%s",
                order.get("id"), side.upper(), qty, tp_price, trigger_direction
            )
            return order
        except Exception as exc:
            logger.error("place_take_profit failed: %s", exc)
            raise

    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the symbol (used on position close / restart)."""
        if self.dry_run:
            logger.info("[DRY_RUN] Would cancel all open orders for %s", self.symbol)
            return
        try:
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("All open orders cancelled for %s", self.symbol)
        except Exception as exc:
            logger.warning("cancel_all_orders: %s", exc)

    def close(self) -> None:
        """Release exchange resources."""
        try:
            self.exchange.close()
        except Exception:
            pass
