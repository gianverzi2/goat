"""
price_feed.py — Price poller via ccxt (Bybit for USDT, Hyperliquid for USDC).
"""
import logging

import ccxt.async_support as ccxt

_exchange: ccxt.Exchange | None = None
_exchange_hl: ccxt.Exchange | None = None


def _get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bybit({"enableRateLimit": True})
    return _exchange


def _get_exchange_hl() -> ccxt.Exchange:
    global _exchange_hl
    if _exchange_hl is None:
        _exchange_hl = ccxt.hyperliquid({"enableRateLimit": True})
    return _exchange_hl


async def poll_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current mid price for each symbol.

    Routes USDC symbols to Hyperliquid and USDT symbols to Bybit.
    Returns {symbol: price} dict. Symbols not found are omitted.
    """
    prices: dict[str, float] = {}
    for symbol in symbols:
        # Route based on quote currency: USDC → Hyperliquid, USDT → Bybit
        # Parse "BASE/QUOTE:SETTLE" format to extract the quote currency
        parts = symbol.split("/")
        quote = parts[1].split(":")[0] if len(parts) > 1 else ""
        if quote == "USDC":
            exchange = _get_exchange_hl()
        else:
            exchange = _get_exchange()
        try:
            ticker = await exchange.fetch_ticker(symbol)
            bid = ticker.get("bid") or 0
            ask = ticker.get("ask") or 0
            last = ticker.get("last") or 0
            if bid is not None and ask is not None:
                prices[symbol] = (bid + ask) / 2
            elif last:
                prices[symbol] = last
        except Exception as exc:
            logging.warning("price_feed: could not fetch %s — %s", symbol, exc)
    return prices


async def close_exchange():
    """Close the shared ccxt sessions (call on app shutdown)."""
    global _exchange, _exchange_hl
    if _exchange is not None:
        await _exchange.close()
        _exchange = None
    if _exchange_hl is not None:
        await _exchange_hl.close()
        _exchange_hl = None
