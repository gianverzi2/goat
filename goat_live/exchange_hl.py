"""
goat_live/exchange_hl.py
ccxt Hyperliquid wrapper: connect, place orders, cancel orders, query positions.

Same public interface as BybitExchange so the bot can switch between exchanges
via config without changing any logic in run.py or signals.py.
"""

import logging
import math
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)


class HyperliquidExchange:
    """Thin wrapper around ccxt.hyperliquid for GOAT live bot operations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.dry_run = cfg["dry_run"]
        self.hedge_mode = False  # Hyperliquid does not use hedge mode

        options = {
            "defaultType": "swap",
        }

        exchange_params = {
            "walletAddress": cfg.get("hl_wallet_address", ""),
            "privateKey": cfg.get("hl_private_key", ""),
            "enableRateLimit": True,
            "options": options,
        }

        # Testnet support
        if cfg.get("testnet", False):
            exchange_params["sandbox"] = True
            logger.info("Hyperliquid TESTNET mode enabled.")

        self.exchange = ccxt.hyperliquid(exchange_params)

        # Load markets so we can inspect contract specs
        self.exchange.load_markets()
        net_label = "testnet" if cfg.get("testnet") else "mainnet"
        logger.info("Markets loaded. Connected to Hyperliquid (%s).", net_label)

        # Cache market info for the trading symbol
        if self.symbol not in self.exchange.markets:
            raise ValueError(
                f"Symbol {self.symbol!r} not found on Hyperliquid. "
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
        """Set leverage for the symbol."""
        if self.dry_run:
            logger.info("[DRY_RUN] Would set leverage to %sx", leverage)
            return
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            logger.info("Leverage set to %sx for %s", leverage, self.symbol)
        except Exception as exc:
            logger.warning("set_leverage failed (may already be set): %s", exc)

    def set_position_mode(self, hedge: bool = False) -> None:
        """Hyperliquid uses one-way mode only. This is a no-op for compatibility."""
        logger.debug("set_position_mode called (no-op on Hyperliquid)")

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_position(self) -> Optional[dict]:
        """
        Return the open position dict for self.symbol if any, else None.
        Considers a position open when abs(contracts) > 0.
        """
        try:
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
        if isinstance(amount_precision, int):
            step = 10 ** (-amount_precision)
        else:
            step = float(amount_precision)

        if step <= 0:
            return raw_qty

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
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
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

        Uses stopLossPrice via Hyperliquid's trigger order mechanism.
        """
        if self.dry_run:
            logger.info(
                "[DRY_RUN] Stop-loss %s %s @ %s for %s",
                side.upper(), qty, sl_price, self.symbol
            )
            return None

        try:
            # Hyperliquid trigger direction: 'above' or 'below'
            trigger_direction = "below" if side == "sell" else "above"
            params = {
                "triggerPrice": sl_price,
                "triggerType": "lastPrice",
                "triggerDirection": trigger_direction,
                "reduceOnly": True,
            }
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="stop",
                side=side,
                amount=qty,
                price=sl_price,
                params=params,
            )
            logger.info(
                "Stop-loss order placed: id=%s side=%s qty=%s sl=%.6f",
                order.get("id"), side.upper(), qty, sl_price
            )
            return order
        except Exception as exc:
            logger.error("place_stop_loss failed: %s", exc)
            raise

    def place_take_profit(self, side: str, qty: float, tp_price: float) -> Optional[dict]:
        """
        Place a reduce-only take-profit limit order.
        side: 'sell' for long position TP, 'buy' for short position TP

        Uses limit order with trigger on Hyperliquid.
        """
        if self.dry_run:
            logger.info(
                "[DRY_RUN] Take-profit %s %s @ %s for %s",
                side.upper(), qty, tp_price, self.symbol
            )
            return None

        try:
            # TP for long: price must rise → above; TP for short: price must drop → below
            trigger_direction = "above" if side == "sell" else "below"
            params = {
                "triggerPrice": tp_price,
                "triggerType": "lastPrice",
                "triggerDirection": trigger_direction,
                "reduceOnly": True,
            }
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side=side,
                amount=qty,
                price=tp_price,
                params=params,
            )
            logger.info(
                "Take-profit order placed: id=%s side=%s qty=%s tp=%.6f",
                order.get("id"), side.upper(), qty, tp_price
            )
            return order
        except Exception as exc:
            logger.error("place_take_profit failed: %s", exc)
            raise

    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the symbol."""
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
