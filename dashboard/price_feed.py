"""
price_feed.py — Bybit price poller via ccxt (public API, no key needed).
"""
import asyncio
import logging

import ccxt.async_support as ccxt

_exchange: ccxt.Exchange | None = None


def _get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bybit({"enableRateLimit": True})
    return _exchange


async def poll_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current mid price for each symbol from Bybit.

    Returns {symbol: price} dict. Symbols not found are omitted.
    """
    exchange = _get_exchange()
    prices: dict[str, float] = {}
    for symbol in symbols:
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
    """Close the shared ccxt session (call on app shutdown)."""
    global _exchange
    if _exchange is not None:
        await _exchange.close()
        _exchange = None
