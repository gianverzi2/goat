"""
goat_live/run_multi.py
Launch one goat_live.run worker per symbol loaded from a CSV file.

This keeps the existing single-symbol engine unchanged while supporting
multi-symbol operation from one command.
"""

import csv
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from goat_live.config import load_config, setup_logging

logger = logging.getLogger(__name__)
_children = []
_shutdown_requested = False


def _handle_signal(signum, _frame):
    global _shutdown_requested
    logger.info("Signal %s received — stopping all workers.", signum)
    _shutdown_requested = True
    for proc in _children:
        if proc.poll() is None:
            proc.terminate()


def _safe_symbol_slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()


def load_symbols(path: Path, max_symbols: int = 0) -> list[str]:
    symbols = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            symbol = (row[0] or "").strip()
            if not symbol or symbol.startswith("#") or symbol == "symbol":
                continue
            # Keep only unified symbols like BTC/USDC:USDC
            if "/" not in symbol or ":" not in symbol:
                skipped += 1
                continue
            symbols.append(symbol)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    if skipped:
        logger.info("Skipped %s non-symbol rows from %s", skipped, path)
    return symbols


def main():
    cfg = load_config()
    setup_logging(cfg)

    symbols_file = Path(cfg.get("symbols_file", "crypto_perp_symbols.csv"))
    if not symbols_file.is_absolute():
        symbols_file = Path.cwd() / symbols_file

    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")

    symbols = load_symbols(symbols_file, int(cfg.get("max_symbols", 0)))
    if not symbols:
        raise RuntimeError(f"No symbols loaded from {symbols_file}")

    logger.info(
        "Starting multi-symbol workers: %s symbols from %s",
        len(symbols),
        symbols_file,
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    state_dir = Path(__file__).resolve().parent
    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"

    for symbol in symbols:
        if _shutdown_requested:
            break
        child_env = base_env.copy()
        child_env["GOAT_SYMBOL"] = symbol
        child_env["GOAT_STATE_FILE"] = str(
            state_dir / f"state_{_safe_symbol_slug(symbol)}.json"
        )
        proc = subprocess.Popen([sys.executable, "-m", "goat_live.run"], env=child_env)
        _children.append(proc)
        logger.info("Started worker pid=%s symbol=%s", proc.pid, symbol)

    try:
        while _children:
            alive = []
            for proc in _children:
                rc = proc.poll()
                if rc is None:
                    alive.append(proc)
                else:
                    logger.warning("Worker pid=%s exited with code=%s", proc.pid, rc)
            _children[:] = alive
            if _shutdown_requested:
                break
            if _children:
                signal.pause()
    finally:
        for proc in _children:
            if proc.poll() is None:
                proc.terminate()
        for proc in _children:
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        logger.info("All workers stopped.")


if __name__ == "__main__":
    main()
