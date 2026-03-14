"""
goat_live/state.py
Persist bot state to a JSON file so it survives restarts without double-trading.

State schema:
{
    "last_processed_ts": <int ms> | null,   // last closed bar we acted on
    "open_trade": {                          // present while a position is live
        "side":       "BULL" | "BEAR",
        "case_label": str,
        "entry":      float,
        "sl":         float,
        "tp":         float,
        "qty":        float,
        "entry_order_id": str | null,
        "sl_order_id":    str | null,
        "tp_order_id":    str | null,
        "bar_ts":     int,
    } | null
}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "state.json"


def _default_state() -> dict:
    return {"last_processed_ts": None, "open_trade": None}


def load_state(path: Optional[Path] = None) -> dict:
    """Load state from JSON file. Returns default state if file missing or corrupt."""
    fpath = path or _DEFAULT_STATE_PATH
    if not fpath.exists():
        logger.info("No state file at %s — starting fresh.", fpath)
        return _default_state()
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("State loaded from %s", fpath)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read state file (%s) — starting fresh.", exc)
        return _default_state()


def save_state(state: dict, path: Optional[Path] = None) -> None:
    """Persist state dict to JSON file (atomic write via temp file)."""
    fpath = path or _DEFAULT_STATE_PATH
    tmp = fpath.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(fpath)
        logger.debug("State saved to %s", fpath)
    except OSError as exc:
        logger.error("Failed to save state: %s", exc)


def update_last_ts(state: dict, ts: int) -> dict:
    """Return a new state dict with last_processed_ts updated."""
    return {**state, "last_processed_ts": ts}


def set_open_trade(state: dict, trade: dict) -> dict:
    """Return a new state dict recording an open trade."""
    return {**state, "open_trade": trade}


def clear_open_trade(state: dict) -> dict:
    """Return a new state dict with open_trade cleared."""
    return {**state, "open_trade": None}
