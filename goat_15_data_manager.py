"""
GOATv2 Data Manager — Download, store, reload OHLCV data.
Stores as parquet files (fast, compact) in ./data/ folder.

Usage:
  python3 goat_15_data_manager.py --symbol BTC/USDT:USDT --tf 30m --days 180
  python3 goat_15_data_manager.py --symbol BTC/USDT:USDT --tf 5m --days 180
  python3 goat_15_data_manager.py --symbols BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT --tf 5m --days 90
  python3 goat_15_data_manager.py --list

  # From code
  from goat_15_data_manager import get_ohlcv
  df = get_ohlcv("BTC/USDT:USDT", "30m", days=180)
  df = get_ohlcv("BTC/USDT:USDT", "5m", days=720, start_date="2024-03-01", end_date="2026-03-01")
"""

import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time as time_module
import argparse

DATA_DIR = "./data"


def _symbol_to_filename(symbol, timeframe, days_back):
    """Convert symbol to safe filename."""
    safe = symbol.replace("/", "_").replace(":", "_")
    return f"{safe}_{timeframe}_{days_back}d.parquet"


def _get_tf_ms(timeframe):
    """Convert timeframe string to milliseconds."""
    unit = timeframe[-1]
    val = int(timeframe[:-1])
    if unit == "m":
        return val * 60 * 1000
    elif unit == "h":
        return val * 3600 * 1000
    elif unit == "d":
        return val * 86400 * 1000
    raise ValueError(f"Unknown timeframe: {timeframe}")


def download_ohlcv(symbol, timeframe="30m", days_back=180, exchange_id="bybit"):
    """Download OHLCV data from exchange."""
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }) if exchange_id == "bybit" else getattr(ccxt, exchange_id)({"enableRateLimit": True})

    exchange.load_markets()

    tf_ms = _get_tf_ms(timeframe)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (days_back * 86400 * 1000)
    limit = 1000
    chunk_ms = limit * tf_ms

    all_candles = []
    print(f"  📥 Downloading {symbol} {timeframe} ({days_back}d)...")

    cursor = start_ms
    while cursor < now_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except Exception as e:
            print(f"    ⚠️ Error: {e}, retrying...")
            time_module.sleep(2)
            continue
        if not candles:
            cursor += chunk_ms
            continue
        all_candles.extend(candles)
        cursor = candles[-1][0] + tf_ms
        if len(all_candles) % 5000 < limit:
            print(f"    ...{len(all_candles)} candles so far")

    if not all_candles:
        print(f"  ❌ No candles returned for {symbol} {timeframe} {days_back}d")
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    print(f"  ✅ {symbol} {timeframe}: {len(df)} candles "
          f"({df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


def save_ohlcv(df, symbol, timeframe, days_back):
    """Save DataFrame to parquet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = _symbol_to_filename(symbol, timeframe, days_back)
    path = os.path.join(DATA_DIR, fname)
    df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  💾 Saved → {path} ({size_mb:.1f} MB)")
    return path


def load_ohlcv(symbol, timeframe, days_back):
    """Load DataFrame from parquet. Returns None if not found."""
    fname = _symbol_to_filename(symbol, timeframe, days_back)
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"  📂 Loaded {path}: {len(df)} candles")
    return df


def get_ohlcv(symbol, timeframe, days, force_download=False, exchange_id="bybit",
              start_date=None, end_date=None):
    """
    Main entry point: load from cache or download.
    - Returns cached parquet if fresh (< 24h old)
    - Re-downloads if stale or missing
    - --force always re-downloads
    - start_date / end_date (YYYY-MM-DD strings): when provided, calculate days_back to cover
      the full range (plus 30-day warmup), download/cache using that days value, then filter
      the returned dataframe to exactly start_date → end_date.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Resolve days_back when date range is given ──
    days_back = days
    if start_date is not None or end_date is not None:
        now_utc = datetime.now(timezone.utc)
        WARMUP_DAYS = 30

        if end_date is not None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end_dt = now_utc

        # days_back is measured from *now*, so we need (now → start) + warmup buffer
        if start_date is not None:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_to_start = (now_utc - start_dt).days
            days_back = days_to_start + WARMUP_DAYS
        else:
            # only end_date given: days back from end + shift to now
            days_from_end_to_now = (now_utc - end_dt).days
            days_back = days + days_from_end_to_now

    sym_safe = symbol.replace("/", "_").replace(":", "_")
    parquet_path = os.path.join(DATA_DIR, f"{sym_safe}_{timeframe}_{days_back}d.parquet")

    # ── Check cache first ──
    if not force_download and os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            if 'timestamp' in df.columns and len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                last_ts = df['timestamp'].iloc[-1]
                now_utc = pd.Timestamp.now(tz='UTC')
                age_hours = (now_utc - last_ts).total_seconds() / 3600
                if age_hours < 24:
                    print(f"  📂 Loaded from cache: {parquet_path}")
                    print(f"     {len(df)} candles | last: {str(last_ts)[:19]} | age: {age_hours:.1f}h")
                    df = _filter_date_range(df, start_date, end_date)
                    return df
                else:
                    print(f"  ⏰ Cache stale ({age_hours:.0f}h old), re-downloading...")
            else:
                print(f"  📂 Loaded from cache: {parquet_path}")
                df = _filter_date_range(df, start_date, end_date)
                return df
        except Exception as e:
            print(f"  ⚠️ Cache read failed ({e}), re-downloading...")

    # ── Download fresh data ──
    df = download_ohlcv(symbol, timeframe, days_back=days_back, exchange_id=exchange_id)

    if df is None or len(df) == 0:
        print(f"  ❌ Download failed for {symbol} {timeframe} {days_back}d")
        return None

    # ── Save to cache ──
    save_ohlcv(df, symbol, timeframe, days_back)

    df = _filter_date_range(df, start_date, end_date)
    return df


def _filter_date_range(df, start_date, end_date):
    """Filter dataframe to [start_date, end_date] inclusive. Both are optional YYYY-MM-DD strings."""
    if start_date is None and end_date is None:
        return df
    if 'timestamp' not in df.columns:
        return df
    ts = df['timestamp']
    if start_date is not None:
        start_dt = pd.Timestamp(start_date, tz='UTC')
        df = df[ts >= start_dt]
    if end_date is not None:
        # include the full end day
        end_dt = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
        df = df[df['timestamp'] < end_dt]
    return df.reset_index(drop=True)


def download_batch(symbols, timeframe="30m", days_back=180, force=False, exchange_id="bybit"):
    """Download multiple symbols at once."""
    print(f"\n{'='*60}")
    print(f"  BATCH DOWNLOAD: {len(symbols)} symbols × {timeframe} × {days_back}d")
    print(f"{'='*60}")

    results = {}
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}")
        try:
            df = get_ohlcv(symbol, timeframe, days_back, force_download=force, exchange_id=exchange_id)
            results[symbol] = df
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[symbol] = None

    ok = sum(1 for v in results.values() if v is not None)
    print(f"\n✅ Done: {ok}/{len(symbols)} symbols downloaded")
    return results


def list_cached():
    """List all cached data files."""
    if not os.path.exists(DATA_DIR):
        print("No data directory found.")
        return []
    files = sorted(os.listdir(DATA_DIR))
    if not files:
        print("No cached data files.")
        return []
    print(f"\n📁 Cached data ({DATA_DIR}/):")
    total_mb = 0
    for f in files:
        path = os.path.join(DATA_DIR, f)
        mb = os.path.getsize(path) / (1024 * 1024)
        total_mb += mb
        try:
            df = pd.read_parquet(path)
            rows = len(df)
            first = df['timestamp'].iloc[0].strftime('%Y-%m-%d') if 'timestamp' in df.columns else '?'
            last = df['timestamp'].iloc[-1].strftime('%Y-%m-%d') if 'timestamp' in df.columns else '?'
            print(f"  {f:<55} {rows:>6} bars | {first} → {last} | {mb:.1f} MB")
        except:
            print(f"  {f:<55} {mb:.1f} MB")
    print(f"  {'─'*55}")
    print(f"  Total: {len(files)} files, {total_mb:.1f} MB")
    return files


# ═══════════════════════════════════════════════════════════════════
# PRESET SYMBOL LISTS
# ═══════════════════════════════════════════════════════════════════

TOP_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "DOT/USDT:USDT", "MATIC/USDT:USDT", "NEAR/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "FIL/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "IMX/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT",
    "SEI/USDT:USDT", "TIA/USDT:USDT", "JTO/USDT:USDT", "PYTH/USDT:USDT",
    "WLD/USDT:USDT", "STRK/USDT:USDT", "MANTA/USDT:USDT", "ORDI/USDT:USDT",
    "1000PEPE/USDT:USDT", "1000BONK/USDT:USDT", "WIF/USDT:USDT",
    "GALA/USDT:USDT", "SAND/USDT:USDT", "MANA/USDT:USDT", "AXS/USDT:USDT",
    "FTM/USDT:USDT", "RUNE/USDT:USDT", "AAVE/USDT:USDT", "MKR/USDT:USDT",
    "CRV/USDT:USDT", "LDO/USDT:USDT", "SNX/USDT:USDT", "COMP/USDT:USDT",
    "SUSHI/USDT:USDT", "DYDX/USDT:USDT", "GMX/USDT:USDT", "PENDLE/USDT:USDT",
    "STX/USDT:USDT", "ALGO/USDT:USDT", "HBAR/USDT:USDT",
]

TEST_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
]


# ═══���═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOATv2 Data Manager")
    parser.add_argument("--symbol", type=str, help="Single symbol to download")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--preset", type=str, choices=["top", "test"], help="Use preset symbol list")
    parser.add_argument("--tf", type=str, default="30m", help="Timeframe (default: 30m)")
    parser.add_argument("--days", type=int, default=180, help="Days back (default: 180)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--list", action="store_true", help="List cached files")
    args = parser.parse_args()

    if args.list:
        list_cached()
    elif args.symbol:
        get_ohlcv(args.symbol, args.tf, args.days, force_download=args.force)
    elif args.symbols:
        syms = [s.strip() for s in args.symbols.split(",")]
        download_batch(syms, args.tf, args.days, force=args.force)
    elif args.preset:
        syms = TOP_SYMBOLS if args.preset == "top" else TEST_SYMBOLS
        download_batch(syms, args.tf, args.days, force=args.force)
    else:
        print("Usage examples:")
        print("  python3 goat_15_data_manager.py --symbol BTC/USDT:USDT --tf 30m --days 180")
        print("  python3 goat_15_data_manager.py --preset top --tf 5m --days 90")
        print("  python3 goat_15_data_manager.py --preset test --tf 30m --days 180")
        print("  python3 goat_15_data_manager.py --list")
        print("  python3 goat_15_data_manager.py --symbols BTC/USDT:USDT,ETH/USDT:USDT --tf 5m --days 90 --force")