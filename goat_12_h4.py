"""GOATv2 Scanner — 4h Timeframe"""

import asyncio
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

# ─── 4h Config ──────────────────────────────────────────────────
CFG = {
    "exchange_name": "Bybit",
    "timeframe": "4h",
    "tf_minutes": 240,
    "symbols_file": "bybit_perps_symbols.csv",
    "max_symbols": 0,
    "fetch_limit": 200,
    "post_close_delay_sec": 5,
    "max_retries": 3,
    "rate_limit_delay": 0.2,
    "rr_ratio": 3,
    "pivot_length": 2,
    "reload_every_n_cycles": 42,          # ~7 days (42 × 4h)
    "enable_provisional": True,
    "provisional_offset_sec": 900,        # 15m before close
    "allow_forming_on_provisional": True,
    "enable_preliminary": True,
    "preliminary_interval_min": 120,      # every 2h
    "lookback_bars": 5,
    "webhook_url": "https://discord.com/api/webhooks/1472619891168247950/GUBY7b3-5xAzRgJZyDOe5engz2qeiYjJbZ4pNMtR_DmLDBgC7C5zAcG96I9_LGtzbTmZ",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_scan_plan(now: datetime, tf_minutes: int, post_close_delay_sec: int):
    """
    Return a list of (run_at, run_type) for the current 4h bar:
      - preliminary: every 2h within the bar (if enabled)
      - provisional: 15m before close (if enabled)
      - final: a few seconds after close
    """
    # ── Find next bar close (4h bars align to 00:00, 04:00, 08:00, …) ──
    base = now.replace(minute=0, second=0, microsecond=0)
    hours_to_add = tf_minutes // 60 - (now.hour % (tf_minutes // 60))
    if hours_to_add == 0 and now.minute == 0 and now.second == 0:
        hours_to_add = tf_minutes // 60
    if hours_to_add == tf_minutes // 60 and now.minute == 0 and now.second == 0:
        pass  # exactly on the boundary
    next_close = base + timedelta(hours=hours_to_add)

    # Current bar opened tf_minutes ago from next_close
    bar_open = next_close - timedelta(minutes=tf_minutes)

    plan = []

    # ── Preliminary runs: every 2h within the bar ────────────────
    if CFG.get("enable_preliminary", False):
        interval = CFG.get("preliminary_interval_min", 120)
        t = bar_open + timedelta(minutes=interval)
        while t < next_close - timedelta(minutes=30):  # stop 30m before close
            if t > now:
                plan.append((t, "provisional"))  # treated as provisional
            t += timedelta(minutes=interval)

    # ── Provisional: 15m before close ────────────────────────────
    if CFG.get("enable_provisional", False):
        offset = CFG.get("provisional_offset_sec", 900)
        provisional_at = next_close - timedelta(seconds=offset)
        if provisional_at > now:
            plan.append((provisional_at, "provisional"))

    # ── Final: after close ───────────────────────────────────────
    final_at = next_close + timedelta(seconds=post_close_delay_sec)
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

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "rateLimit": 200,
        "options": {"defaultType": "swap"},
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

            if not plan:
                await asyncio.sleep(10)
                continue

            for run_at, run_type in plan:
                now = datetime.now(timezone.utc)
                sleep_s = (run_at - now).total_seconds()
                if sleep_s > 0:
                    logging.info(
                        f"Next {run_type} scan at {run_at.strftime('%H:%M:%S')} UTC "
                        f"(sleep {sleep_s:.1f}s) | Symbols: {len(symbols)} | "
                        f"Active trades: {len(active_trades)} | "
                        f"W/L: {trade_results['wins']}/{trade_results['losses']}"
                    )
                    await asyncio.sleep(max(1, sleep_s))

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