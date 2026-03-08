"""
01 — Shared utilities: formatting, body intersect check, symbol loading.
"""

import csv
import logging


def fmt(price):
    """Format price with appropriate decimal places."""
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


def body_intersects_level(df, idx, level):
    """Check if a bar's HA body (Open↔Close) contains the given level."""
    body_low = min(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    body_high = max(df.loc[idx, 'HA_Open'], df.loc[idx, 'HA_Close'])
    return body_low <= level <= body_high


def load_symbols(filepath, max_symbols=0):
    """Load trading symbols from a CSV file."""
    symbols = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if row[0] == "symbol":
                    continue
                symbols.append(row[0])
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        logging.info(f"Loaded {len(symbols)} symbols from {filepath}")
    except FileNotFoundError:
        logging.error(f"Symbols file not found: {filepath}. Run fetch_bybit_perps.py first.")
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    return symbols