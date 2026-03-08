"""
02 — Data fetching: retry-aware OHLCV fetcher.
"""

import asyncio
import logging
import ccxt.async_support as ccxt


async def fetch_with_retry(exchange, symbol, timeframe, fetch_limit,
                           max_retries=3, rate_limit_delay=0.2):
    """Fetch OHLCV data with retry logic for rate limits and transient errors."""
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(rate_limit_delay)
            return await exchange.fetch_ohlcv(
                symbol, timeframe, limit=fetch_limit,
                params={"category": "linear"}
            )
        except ccxt.RateLimitExceeded:
            wait_time = (attempt + 1) * 5
            logging.warning(f"Rate limited on {symbol}. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Retry {attempt+1}/{max_retries} for {symbol}: {e}")
            await asyncio.sleep(2 * (attempt + 1))
    return None