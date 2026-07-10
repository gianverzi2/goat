"""
09 — Hyperliquid direct execution module.
Places bracket orders (entry + SL + TP) directly when signals fire,
bypassing the Discord webhook round-trip.

Requires environment variables:
  HL_WALLET_ADDRESS  — Hyperliquid wallet address
  HL_PRIVATE_KEY     — Hyperliquid wallet private key
  HL_USE_TESTNET     — "true" for testnet, "false" for mainnet (default: true)

Trading config is passed via the scanner's CFG dict (keys prefixed with "exec_").
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded SDK singletons (initialized on first trade)
# ---------------------------------------------------------------------------

_exchange = None
_info = None
_initialized = False


def _init_sdk():
    """Initialize the Hyperliquid SDK (exchange + info) once."""
    global _exchange, _info, _initialized
    if _initialized:
        return

    try:
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
    except ImportError as e:
        logger.error(
            "hyperliquid-python SDK not installed. "
            "Install with: pip install hyperliquid-python\n%s", e
        )
        raise

    use_testnet = os.environ.get("HL_USE_TESTNET", "true").lower() in ("1", "true", "yes")
    base_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL

    wallet_address = os.environ.get("HL_WALLET_ADDRESS", "")
    private_key = os.environ.get("HL_PRIVATE_KEY", "")

    if not wallet_address or not private_key:
        logger.warning(
            "HL_WALLET_ADDRESS / HL_PRIVATE_KEY not set — execution disabled."
        )
        _initialized = True
        return

    _info = Info(base_url, skip_ws=True)
    _exchange = Exchange(private_key, base_url, account_address=wallet_address)
    _initialized = True

    env_label = "TESTNET" if use_testnet else "MAINNET"
    logger.info("Hyperliquid SDK initialized (%s) for wallet %s…%s",
                env_label, wallet_address[:6], wallet_address[-4:])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_account_value() -> float:
    """Return total account value in USD."""
    _init_sdk()
    if _info is None:
        return 0.0
    wallet = os.environ.get("HL_WALLET_ADDRESS", "")
    user_state = _info.user_state(wallet)
    return float(user_state.get("marginSummary", {}).get("accountValue", 0))


def _get_open_positions_count() -> int:
    """Return number of currently open positions. Returns 999 on error to block trading."""
    _init_sdk()
    if _info is None:
        return 999
    wallet = os.environ.get("HL_WALLET_ADDRESS", "")
    try:
        user_state = _info.user_state(wallet)
        positions = user_state.get("assetPositions", [])
        return sum(
            1 for pos in positions
            if float(pos.get("position", {}).get("szi", 0)) != 0
        )
    except Exception as e:
        logger.error("Failed to fetch positions: %s", e)
        return 999  # Block trading on API error (fail-safe)


def _get_mid_price(coin: str) -> Optional[float]:
    """Get current mid/mark price for a coin."""
    _init_sdk()
    if _info is None:
        return None
    try:
        all_mids = _info.all_mids()
        if coin in all_mids:
            return float(all_mids[coin])
        return None
    except Exception as e:
        logger.error("Failed to get mid price for %s: %s", coin, e)
        return None


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------

async def execute_signal(signal: dict, cfg: dict) -> bool:
    """
    Execute a trade on Hyperliquid directly from a scanner signal.

    Args:
        signal: dict with keys:
            - coin: str (e.g. "GOAT" — base asset, not the full pair)
            - is_buy: bool
            - entry: float
            - sl: float
            - tp: float
        cfg: scanner CFG dict with exec_ prefixed trading params:
            - exec_enabled: bool (default False)
            - exec_risk_percent: float (default 0.01 = 1%)
            - exec_max_positions: int (default 3)
            - exec_max_retries: int (default 3)

    Returns:
        True if order placed successfully, False otherwise.
    """
    if not cfg.get("exec_enabled", False):
        logger.debug("Execution disabled (exec_enabled=False), skipping.")
        return False

    _init_sdk()
    if _exchange is None:
        logger.warning("Hyperliquid SDK not initialized (missing credentials). Skipping execution.")
        return False

    coin = signal["coin"]
    is_buy = signal["is_buy"]
    entry_px = signal["entry"]
    sl_px = signal["sl"]
    tp_px = signal["tp"]

    risk_percent = cfg.get("exec_risk_percent", 0.01)
    max_positions = cfg.get("exec_max_positions", 3)
    max_retries = cfg.get("exec_max_retries", 3)

    logger.info("Executing trade: %s %s | Entry=%.6f SL=%.6f TP=%.6f",
                "BUY" if is_buy else "SELL", coin, entry_px, sl_px, tp_px)

    # 1. Check open positions
    open_count = _get_open_positions_count()
    if open_count >= max_positions:
        logger.warning("Max positions reached (%d/%d). Skipping.", open_count, max_positions)
        return False

    # 2. Get account value and calculate size
    account_value = _get_account_value()
    if account_value <= 100:
        logger.warning("Account value too low ($%.2f). Skipping.", account_value)
        return False

    risk_amount = account_value * risk_percent
    risk_per_unit = abs(entry_px - sl_px)
    if risk_per_unit == 0:
        logger.error("SL and Entry are the same price. Skipping.")
        return False

    size = round(risk_amount / risk_per_unit, 4)
    if size < 0.001:
        logger.warning("Calculated size too small (%.6f). Skipping.", size)
        return False

    logger.info("Account=$%.2f | Risk=$%.2f (%.1f%%) | Size=%.4f",
                account_value, risk_amount, risk_percent * 100, size)

    # 3. Place bracket order with retries
    for attempt in range(1, max_retries + 1):
        mid = _get_mid_price(coin)
        # Offset entry by ~8bps to ensure post-only fill (configurable via exec_entry_offset_bps)
        offset_bps = cfg.get("exec_entry_offset_bps", 8) / 10000
        offset = (mid or entry_px) * offset_bps
        limit_entry_px = round(
            entry_px - offset if is_buy else entry_px + offset, 6
        )

        logger.info("Attempt %d/%d | Limit Entry @ %.6f | Mid: %s",
                    attempt, max_retries, limit_entry_px, mid)

        orders = [
            {
                "coin": coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": str(limit_entry_px),
                "order_type": {"limit": {"tif": "Alo"}},  # Add Liquidity Only (post-only, maker rebate)
                "reduce_only": False,
            },
            {
                "coin": coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": str(tp_px),
                "order_type": {"limit": {"tif": "Alo"}},  # Post-only TP for maker rebate
                "reduce_only": True,
            },
            {
                "coin": coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": str(sl_px),
                "order_type": {
                    "trigger": {
                        "triggerPx": str(sl_px),
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            },
        ]

        try:
            result = _exchange.bulk_orders(orders, grouping="normalTpsl")
            if isinstance(result, dict) and result.get("status") == "ok":
                logger.info("✅ Bracket order placed for %s (%s)",
                            coin, "LONG" if is_buy else "SHORT")
                return True
            else:
                logger.warning("Order rejected (attempt %d): %s", attempt, result)
        except Exception as e:
            logger.error("Order placement error (attempt %d): %s", attempt, e)

        await asyncio.sleep(1.5)

    logger.error("Failed to place order for %s after %d attempts.", coin, max_retries)
    return False
