"""GOATv2 Scanner — Hyperliquid 30m Timeframe"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
import logging
import traceback
import ccxt.async_support as ccxt

# ─── Updated imports: from split modules instead of function_goat_common ───
from goat_01_utilis import load_symbols
from goat_06_trades import (
    active_trades, trade_results, check_active_trades
)
from goat_08_analyze import analyze_symbol

# ─── 30m Config ─────────────────────────────────────────────────
CFG = {
    "exchange_name": "Hyperliquid",
    "timeframe": "30m",
    "tf_minutes": 30,
    "symbols_file": "perp_symbols.csv",
    "max_symbols": 0,
    "fetch_limit": 200,
    "post_close_delay_sec": 2,
    "max_retries": 3,
    "rate_limit_delay": 0.2,
    "rr_ratio": 3,
    "pivot_length": 2,
    "reload_every_n_cycles": 60,  # ~30 hours
    "allow_forming_on_provisional": True,  # allow forming bar on provisional runs
    "lookback_bars": 5,           # override default (3) for GOAT scan window
    "webhook_url": "https://discord.com/api/webhooks/1490295319475261502/lx42UrPOxNr4ilev5Y_eyiEPczOjxMTezSSsFIMYqMvr3PsI27pLEz_oQ4RunYZG96y-",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_scan_plan(now: datetime, tf_minutes: int, post_close_delay_sec: int):
    """Return a list of (run_at, run_type) for the current bar:
       - provisional: ~15m before close
       - final: a few seconds after close
    """
    base = now.replace(second=0, microsecond=0)
    minutes_to_add = tf_minutes - (now.minute % tf_minutes)
    if minutes_to_add == tf_minutes and now.second == 0:
        minutes_to_add = tf_minutes
    next_close = base + timedelta(minutes=minutes_to_add)

    provisional_at = next_close - timedelta(minutes=15)
    final_at = next_close + timedelta(seconds=post_close_delay_sec)

    plan = []
    if provisional_at > now:
        plan.append((provisional_at, "provisional"))
    if final_at > now:
        plan.append((final_at, "final"))
    return sorted(plan, key=lambda x: x[0])


async def run_scan_cycle(exchange, symbols, run_type):
    df_regular_map = {}

    if active_trades:
        logging.info(f"[TRADE MONITOR] Checking {len(active_trades)} active trade(s)...")
        await check_active_trades(exchange, CFG)

    for symbol in symbols:
        try:
            await analyze_symbol(exchange, symbol, df_regular_map, CFG, run_type=run_type)
        except Exception as e:
            logging.error(f"Error scanning {symbol}: {e}")
        await asyncio.sleep(0.2)


async def main():
    symbols = load_symbols(CFG["symbols_file"], CFG["max_symbols"])
    logging.info(f"Starting GOATv2 {CFG['exchange_name']} {CFG['timeframe']} scanner for {len(symbols)} symbols")
    logging.info(f"Symbols: {symbols}")

    exchange = ccxt.hyperliquid({
        "walletAddress": os.environ.get("HYPERLIQUID_API_KEY"),
        "secret": os.environ.get("HYPERLIQUID_SECRET"),
        "enableRateLimit": True,
    })
    try:
        await exchange.load_markets()

        cycle_count = 0
        while True:
            cycle_count += 1

            if cycle_count % CFG["reload_every_n_cycles"] == 0:
                symbols = load_symbols(CFG["symbols_file"], CFG["max_symbols"])
                logging.info(f"Reloaded {len(symbols)} symbols")

            now = datetime.now(timezone.utc)
            plan = compute_scan_plan(now, CFG["tf_minutes"], CFG["post_close_delay_sec"])

            # If nothing is scheduled (shouldn't happen), sleep a bit
            if not plan:
                await asyncio.sleep(5)
                continue

            for run_at, run_type in plan:
                now = datetime.now(timezone.utc)
                sleep_s = (run_at - now).total_seconds()
                if sleep_s > 0:
                    logging.info(f"Next {run_type} scan at {run_at.strftime('%H:%M:%S')} UTC "
                                 f"(sleep {sleep_s:.1f}s) | Symbols: {len(symbols)} | "
                                 f"Active trades: {len(active_trades)} | "
                                 f"W/L: {trade_results['wins']}/{trade_results['losses']}")
                    await asyncio.sleep(sleep_s)

                try:
                    await run_scan_cycle(exchange, symbols, run_type)
                except Exception as e:
                    logging.error(f"{run_type} cycle error: {e}")
                    traceback.print_exc()

    finally:
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
