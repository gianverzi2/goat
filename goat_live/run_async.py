"""
goat_live/run_async.py
Single-process async multi-symbol runner.

Architecture:
  - ONE shared async ccxt connection for all OHLCV fetches (non-blocking).
  - ONE clearinghouseState call per cycle to get all open positions at once.
  - Per-symbol state persisted to a single JSON file.
  - Symbols batched (BATCH_SIZE=10) to avoid overwhelming the exchange.
  - Timing follows the same scan-plan model as the hl_10_m5 scanner.
  - For live order placement, lazy per-symbol HyperliquidExchange objects are
    created on first signal (reuses all existing order/SL/TP logic).

Run from the repo root with venv active:
    python -m goat_live.run_async
"""

import asyncio
import csv
import json
import logging
import math
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt.async_support as ccxt_async

from goat_live.config import load_config, setup_logging
from goat_live.risk import calc_qty, get_trade_levels
from goat_live.signals import calculate_heikin_ashi, _get_detect_patterns, compute_ao

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
POST_CLOSE_DELAY_SEC = 2

_shutdown_requested = False

# {symbol: {"last_processed_ts": int|None, "open_trade": dict|None}}
_symbol_state: dict = {}
_STATE_PATH: Optional[Path] = None

# Lazy per-symbol sync exchange objects (only created when live-trading a signal)
_symbol_exchanges: dict = {}


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, _frame):
    global _shutdown_requested
    logger.info("Signal %s received — requesting graceful shutdown.", signum)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

def _load_symbols(path: Path, max_symbols: int = 0) -> list:
    symbols = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            sym = (row[0] or "").strip()
            if not sym or sym.startswith("#") or sym == "symbol":
                continue
            if "/" not in sym or ":" not in sym:
                continue
            symbols.append(sym)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    logger.info("Loaded %d symbols from %s", len(symbols), path)
    return symbols


# ---------------------------------------------------------------------------
# State (single JSON, keyed by symbol)
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Could not read state file (%s) — starting fresh.", exc)
    return {}


def _save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to save state: %s", exc)


def _sym_state(symbol: str) -> dict:
    return _symbol_state.get(symbol, {"last_processed_ts": None, "open_trade": None})


def _set_sym_state(symbol: str, last_processed_ts, open_trade):
    _symbol_state[symbol] = {
        "last_processed_ts": last_processed_ts,
        "open_trade": open_trade,
    }
    _save_state(_symbol_state, _STATE_PATH)


# ---------------------------------------------------------------------------
# Scan timing (same model as hl_10_m5.py scanner)
# ---------------------------------------------------------------------------

def _compute_scan_plan(now: datetime, tf_minutes: int) -> list:
    now_floor = now.replace(second=0, microsecond=0)
    minutes_to_next = tf_minutes - (now_floor.minute % tf_minutes)
    if minutes_to_next == 0:
        minutes_to_next = tf_minutes
    next_close = now_floor + timedelta(minutes=minutes_to_next)
    final_at = next_close + timedelta(seconds=POST_CLOSE_DELAY_SEC)
    if final_at > now:
        return [(final_at, "final")]
    return []


# ---------------------------------------------------------------------------
# Async OHLCV fetch
# ---------------------------------------------------------------------------

async def _fetch_candles(exchange, symbol: str, timeframe: str, limit: int):
    for attempt in range(3):
        try:
            await asyncio.sleep(0.15)
            raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            break
        except Exception as exc:
            if attempt == 2:
                raise
            wait = (attempt + 1) * 5
            logger.warning("fetch_ohlcv retry %d/3 [%s]: %s — wait %ds", attempt + 1, symbol, exc, wait)
            await asyncio.sleep(wait)

    if not raw:
        raise RuntimeError(f"fetch_ohlcv empty for {symbol}")

    import pandas as pd
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.astype({"timestamp": "int64", "open": float, "high": float,
                    "low": float, "close": float, "volume": float})
    return df.iloc[:-1].reset_index(drop=True)   # drop forming candle


# ---------------------------------------------------------------------------
# Position query — one call returns ALL open positions
# ---------------------------------------------------------------------------

async def _fetch_all_positions(exchange, wallet_address: str) -> dict:
    """
    Returns {COIN: {"contracts": float, "side": "long"|"short"}} for all
    symbols with abs(size) > 0.  One API call for all symbols — efficient.
    """
    request = {"type": "clearinghouseState", "user": wallet_address.lower()}
    try:
        info_method = (
            getattr(exchange, "public_post_info", None)
            or getattr(exchange, "publicPostInfo", None)
        )
        if info_method is None:
            return {}
        response = await info_method(request)
        out = {}
        for item in (response.get("assetPositions") or []):
            entry = item.get("position") or {}
            coin = str(entry.get("coin") or "").upper()
            contracts = float(entry.get("szi") or 0.0)
            if abs(contracts) > 0:
                out[coin] = {
                    "contracts": abs(contracts),
                    "side": "long" if contracts > 0 else "short",
                    "info": item,
                }
        return out
    except Exception as exc:
        logger.error("fetch_all_positions error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Round qty using market precision
# ---------------------------------------------------------------------------

def _round_qty(raw_qty: float, market: dict) -> float:
    precision = (market.get("precision") or {})
    amount_precision = precision.get("amount")
    if amount_precision is None:
        return raw_qty
    step = 10 ** (-amount_precision) if isinstance(amount_precision, int) else float(amount_precision)
    if step <= 0:
        return raw_qty
    return math.floor(raw_qty / step) * step


# ---------------------------------------------------------------------------
# Lazy per-symbol sync exchange (for live order placement only)
# ---------------------------------------------------------------------------

def _get_symbol_exchange(symbol: str, cfg: dict):
    if symbol not in _symbol_exchanges:
        from goat_live.exchange_hl import HyperliquidExchange
        sym_cfg = {**cfg, "symbol": symbol}
        _symbol_exchanges[symbol] = HyperliquidExchange(sym_cfg)
        logger.info("Created exchange object for %s", symbol)
    return _symbol_exchanges[symbol]


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def _execute_trade_sync(symbol: str, signal: dict, cfg: dict) -> Optional[dict]:
    """Run in asyncio.to_thread — uses sync HyperliquidExchange."""
    ex = _get_symbol_exchange(symbol, cfg)

    ha_df = signal["ha_df"]
    trigger_idx = signal["trigger_idx"]
    side = signal["side"]

    levels = get_trade_levels(
        ha_df, trigger_idx, side, cfg["rr_ratio"],
        signal_bar=trigger_idx,
        sweep_source_bar=signal.get("sweep_source_bar"),
    )
    if levels is None:
        logger.warning("[%s] No trade levels — skipping.", symbol)
        return None

    entry_price = levels["entry"]
    market = ex.exchange.markets.get(symbol, {})

    class _QtyProxy:
        def round_qty(self, raw):
            return _round_qty(raw, market)

    qty = calc_qty(cfg["notional_usd"], entry_price, _QtyProxy())

    entry_side = "buy" if side == "BULL" else "sell"
    sl_side    = "sell" if side == "BULL" else "buy"
    tp_side    = "sell" if side == "BULL" else "buy"

    logger.info(
        "[%s] Executing %s | qty=%.6f | entry=%.6f | sl=%.6f | tp=%.6f",
        symbol, side, qty, entry_price, levels["sl"], levels["tp"],
    )

    entry_order = ex.place_market_order(entry_side, qty)
    sl_order    = ex.place_stop_loss(sl_side, qty, levels["sl"])
    tp_order    = ex.place_take_profit(tp_side, qty, levels["tp"])

    return {
        "side": side,
        "case_label": signal.get("case_label"),
        "entry": entry_price,
        "sl": levels["sl"],
        "tp": levels["tp"],
        "qty": qty,
        "entry_order_id": entry_order.get("id") if entry_order else None,
        "sl_order_id":    sl_order.get("id")    if sl_order    else None,
        "tp_order_id":    tp_order.get("id")    if tp_order    else None,
        "bar_ts": signal["bar_ts"],
    }


# ---------------------------------------------------------------------------
# Per-symbol check
# ---------------------------------------------------------------------------

async def _check_symbol(exchange, symbol: str, cfg: dict, open_positions: dict):
    state = _sym_state(symbol)
    last_ts = state.get("last_processed_ts")

    # Derive coin name for position lookup (e.g. BTC/USDC:USDC → BTC)
    coin = symbol.split("/")[0].upper()

    # Skip if exchange says position is open
    if coin in open_positions:
        logger.debug("[%s] Position open on exchange — skipping.", symbol)
        _set_sym_state(symbol, last_ts, state.get("open_trade"))
        return

    # Clear stale open_trade if position is now flat
    if state.get("open_trade") is not None:
        logger.info("[%s] Position closed externally — clearing state.", symbol)
        _set_sym_state(symbol, last_ts, None)
        state = _sym_state(symbol)

    # Fetch candles
    try:
        df = await _fetch_candles(exchange, symbol, cfg["timeframe"], cfg["warmup_bars"] + 1)
    except Exception as exc:
        logger.error("[%s] candle fetch error: %s", symbol, exc)
        return

    last_bar_ts = int(df["timestamp"].iloc[-1])
    if last_ts is not None and last_bar_ts <= last_ts:
        logger.debug("[%s] Bar %d already processed.", symbol, last_bar_ts)
        return

    # HA + patterns
    ha_df = calculate_heikin_ashi(df)
    detect_fn = _get_detect_patterns()
    ha_df = detect_fn(ha_df)

    from goat_05_cases import check_goat

    trigger_idx = len(ha_df) - 1
    signal = None

    for side in ("BULL", "BEAR"):
        triggered, case_label, swept_label, swept_value, source_bar_idx = check_goat(
            ha_df, side, symbol
        )
        if not triggered:
            continue

        if cfg.get("ao_filter", False):
            ao = compute_ao(df)
            if side == "BULL" and ao > 0:
                logger.info("[%s] AO filter blocked LONG", symbol)
                continue
            if side == "BEAR" and ao < 0:
                logger.info("[%s] AO filter blocked SHORT", symbol)
                continue

        signal = {
            "side": side,
            "case_label": case_label,
            "swept_label": swept_label,
            "swept_value": swept_value,
            "trigger_idx": trigger_idx,
            "bar_ts": last_bar_ts,
            "ha_df": ha_df,
            "sweep_source_bar": source_bar_idx,
        }
        logger.info("🎯 Signal: %s %s | %s | swept=%s @ %s",
                    side, symbol, case_label, swept_label, swept_value)
        break

    # Always advance last_processed_ts
    open_trade = state.get("open_trade")

    if signal is None:
        _set_sym_state(symbol, last_bar_ts, open_trade)
        return

    if cfg["dry_run"]:
        logger.info("[DRY_RUN] Would open %s trade on %s", signal["side"], symbol)
        _set_sym_state(symbol, last_bar_ts, open_trade)
        return

    # Live: execute trade in thread to avoid blocking the event loop
    try:
        trade_record = await asyncio.to_thread(_execute_trade_sync, symbol, signal, cfg)
        _set_sym_state(symbol, last_bar_ts, trade_record)
    except Exception as exc:
        logger.error("[%s] trade execution error: %s", symbol, exc)
        _set_sym_state(symbol, last_bar_ts, open_trade)


# ---------------------------------------------------------------------------
# Scan cycle
# ---------------------------------------------------------------------------

async def _run_scan_cycle(exchange, symbols: list, cfg: dict, open_positions: dict):
    total = len(symbols)
    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        logger.info("Scanning batch %d–%d of %d symbols…", i + 1, min(i + BATCH_SIZE, total), total)
        await asyncio.gather(*[
            _check_symbol(exchange, sym, cfg, open_positions)
            for sym in batch
        ])
        await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    global _symbol_state, _STATE_PATH

    cfg = load_config()
    setup_logging(cfg)

    logger.info(
        "GOAT Async Bot starting | exchange=%s testnet=%s tf=%s notional=$%.0f "
        "rr=%.1f leverage=%dx dry_run=%s",
        cfg["exchange"], cfg["testnet"], cfg["timeframe"],
        cfg["notional_usd"], cfg["rr_ratio"], cfg["leverage"], cfg["dry_run"],
    )
    if cfg["dry_run"]:
        logger.warning("*** DRY_RUN mode is ON — no real orders will be placed. ***")

    # Resolve symbols file
    symbols_file = Path(cfg.get("symbols_file", "crypto_perp_symbols.csv"))
    if not symbols_file.is_absolute():
        symbols_file = Path.cwd() / symbols_file
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")

    symbols = _load_symbols(symbols_file, int(cfg.get("max_symbols", 0)))
    if not symbols:
        raise RuntimeError(f"No symbols loaded from {symbols_file}")

    # State file
    state_dir = Path(__file__).resolve().parent
    _STATE_PATH = Path(os.getenv("GOAT_STATE_FILE", str(state_dir / "state_multi.json")))
    _symbol_state = _load_state(_STATE_PATH)
    logger.info("State loaded from %s (%d symbols tracked)", _STATE_PATH, len(_symbol_state))

    # Wallet address for position queries
    wallet = (cfg.get("hl_wallet_address", "") or "").strip()

    # Build single async exchange connection
    exchange_params = {
        "walletAddress": wallet,
        "privateKey": (cfg.get("hl_private_key", "") or "").strip(),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if cfg.get("testnet", False):
        exchange_params["sandbox"] = True

    exchange = ccxt_async.hyperliquid(exchange_params)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        await exchange.load_markets()
        logger.info("Markets loaded. %d symbols scheduled.", len(symbols))

        tf_minutes = int(cfg["timeframe"].replace("m", "").replace("h", "")) * (
            60 if "h" in cfg["timeframe"] else 1
        )

        while not _shutdown_requested:
            now = datetime.now(timezone.utc)
            plan = _compute_scan_plan(now, tf_minutes)

            if not plan:
                await asyncio.sleep(1)
                continue

            for run_at, run_type in plan:
                if _shutdown_requested:
                    break
                now = datetime.now(timezone.utc)
                sleep_s = (run_at - now).total_seconds()
                if sleep_s > 0:
                    logger.info(
                        "Next %s scan at %s UTC (sleep %.1fs) | symbols=%d",
                        run_type, run_at.strftime("%H:%M:%S"), sleep_s, len(symbols),
                    )
                    await asyncio.sleep(max(1, sleep_s))

                if _shutdown_requested:
                    break

                # Fetch all open positions once per cycle
                open_positions = await _fetch_all_positions(exchange, wallet)
                logger.debug("Open positions: %s", list(open_positions.keys()))

                try:
                    await _run_scan_cycle(exchange, symbols, cfg, open_positions)
                except Exception as exc:
                    logger.error("%s cycle error: %s", run_type, exc, exc_info=True)

    finally:
        logger.info("Shutting down async runner.")
        await exchange.close()
        # Close any lazy sync exchanges
        for sym, ex in _symbol_exchanges.items():
            try:
                ex.close()
            except Exception:
                pass
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
