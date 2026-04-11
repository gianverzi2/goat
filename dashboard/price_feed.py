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
    """Fetch current mid price for each symbol using batch ticker API.

    Routes USDC symbols to Hyperliquid and USDT symbols to Bybit.
    Returns {symbol: price} dict. Symbols not found are omitted.
    """
    prices: dict[str, float] = {}

    # Split symbols by exchange
    bybit_symbols = []
    hl_symbols = []
    for symbol in symbols:
        parts = symbol.split("/")
        quote = parts[1].split(":")[0] if len(parts) > 1 else ""
        if quote == "USDC":
            hl_symbols.append(symbol)
        else:
            bybit_symbols.append(symbol)

    # Batch fetch from Bybit
    if bybit_symbols:
        exchange = _get_exchange()
        try:
            tickers = await exchange.fetch_tickers(bybit_symbols)
            for sym, ticker in tickers.items():
                _extract_price(prices, sym, ticker)
        except Exception as exc:
            logging.warning("price_feed: Bybit batch fetch failed — %s", exc)

    # Batch fetch from Hyperliquid
    if hl_symbols:
        exchange_hl = _get_exchange_hl()
        try:
            tickers = await exchange_hl.fetch_tickers(hl_symbols)
            for sym, ticker in tickers.items():
                _extract_price(prices, sym, ticker)
        except Exception as exc:
            logging.warning("price_feed: Hyperliquid batch fetch failed — %s", exc)

    return prices


def _extract_price(prices: dict, symbol: str, ticker: dict):
    """Extract mid or last price from a ticker dict."""
    bid = ticker.get("bid") or 0
    ask = ticker.get("ask") or 0
    last = ticker.get("last") or 0
    if bid and ask:
        prices[symbol] = (bid + ask) / 2
    elif last:
        prices[symbol] = last


async def close_exchange():
    """Close the shared ccxt sessions (call on app shutdown)."""
    global _exchange, _exchange_hl
    if _exchange is not None:
        await _exchange.close()
        _exchange = None
    if _exchange_hl is not None:
        await _exchange_hl.close()
        _exchange_hl = None
