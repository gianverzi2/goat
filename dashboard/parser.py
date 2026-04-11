"""
parser.py — Discord message parser for GOAT trade signals.
"""
import logging
import re
from datetime import datetime, timezone


# ── Message type detection ──────────────────────────────────────────────────

def detect_message_type(text: str) -> str:
    """Return 'open', 'close', or 'unknown'."""
    if "Trade Closed:" in text:
        return "close"
    if "HA Entry:" in text:
        return "open"
    return "unknown"


# ── Open message parser ─────────────────────────────────────────────────────

_OPEN_TRADE_ID = re.compile(r"🆔\s*(GOATv2_\S+)")
# Use specific character classes (no alternation that creates backtracking paths)
# Bounded quantifiers prevent polynomial backtracking (ReDoS)
_OPEN_SYMBOL   = re.compile(r"(?:(?:Bybit|Hyperliquid)[ \t]+[A-Za-z0-9]{1,10}[ \t]+)?([A-Z0-9]{1,20}/[A-Z0-9]{1,10}:[A-Z0-9]{1,10})")
_OPEN_SYMBOL2  = re.compile(r"([A-Za-z0-9]{1,20}/USD[CT](?::[A-Za-z0-9]{1,10})?)")
_OPEN_SIDE     = re.compile(r"\b(BULL|BEAR)\b")
_OPEN_CASE     = re.compile(r"📊[ \t]*GOATv2_(?:BULL|BEAR)_([A-Z0-9_]+)")
_OPEN_TIME     = re.compile(r"⏰[ \t]*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)")
_OPEN_SWEPT    = re.compile(r"🔖[^:\n]{0,40}:[ \t]*([\d.]+)")
_OPEN_ENTRY    = re.compile(r"HA Entry:[ \t]*([\d.]+)")
_OPEN_SL       = re.compile(r"HA SL:[ \t]*([\d.]+)")
_OPEN_TP       = re.compile(r"HA TP\([^)\n]{0,20}\):[ \t]*([\d.]+)")
_OPEN_RISK     = re.compile(r"HA Risk:[ \t]*([\d.]+)")
_OPEN_TF       = re.compile(r"(?:Bybit|Hyperliquid)[ \t]+(\w{1,5})[ \t]+[A-Z]")


def parse_open_message(text: str, tf: str) -> dict | None:
    """Parse a GOAT open trade message. Returns a dict or None."""
    if detect_message_type(text) != "open":
        return None

    trade_id_m = _OPEN_TRADE_ID.search(text)
    if not trade_id_m:
        return None
    trade_id = trade_id_m.group(1)

    # Exchange: detect from full text (message starts with a separator line)
    exchange = "HL" if "Hyperliquid" in text else "Bybit"

    # Symbol: search full text (e.g. "Bybit 5m XAUT/USDT:USDT" or "Hyperliquid 5m BTC/USDC:USDC")
    symbol = None
    sym_m = _OPEN_SYMBOL.search(text)
    if sym_m:
        symbol = sym_m.group(1)       # e.g. XAUT/USDT:USDT or BTC/USDC:USDC (keep full)
    if not symbol:
        sym_m2 = _OPEN_SYMBOL2.search(text)
        if sym_m2:
            symbol = sym_m2.group(1)  # keep full, don't split on colon

    # Side: from trade_id or body
    side_m = _OPEN_SIDE.search(trade_id)
    if not side_m:
        side_m = _OPEN_SIDE.search(text)
    side = side_m.group(1) if side_m else "BULL"

    # Case type
    case_m = _OPEN_CASE.search(text)
    case_type = case_m.group(1) if case_m else "PIVOT"
    # Normalise: PIVOT / LGCR / LG_LINE
    if case_type not in ("PIVOT", "LGCR", "LG_LINE"):
        case_type = "PIVOT"

    # Timestamp
    time_m = _OPEN_TIME.search(text)
    if time_m:
        opened_at = time_m.group(1)
    else:
        logging.warning("parse_open_message: no timestamp found in message, using current UTC time")
        opened_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Swept price
    swept_m = _OPEN_SWEPT.search(text)
    swept = float(swept_m.group(1)) if swept_m else None

    # Entry / SL / TP / Risk
    entry_m = _OPEN_ENTRY.search(text)
    sl_m    = _OPEN_SL.search(text)
    tp_m    = _OPEN_TP.search(text)
    risk_m  = _OPEN_RISK.search(text)

    if not (entry_m and sl_m and tp_m):
        return None

    # TF: prefer webhook param, but extract from message text as fallback validation
    tf_m = _OPEN_TF.search(text)
    if tf_m:
        raw_tf = tf_m.group(1).lower()
        # Normalize: "30m" → "m30", "5m" → "m5", "4h"→"h4", "d"→"d"
        tf_map = {"5m": "m5", "30m": "m30", "4h": "h4", "h4": "h4", "d": "d", "1d": "d",
                  "m5": "m5", "m30": "m30"}
        parsed_tf = tf_map.get(raw_tf, raw_tf)
        # Use parsed TF if the webhook tf param seems wrong or default
        if parsed_tf and parsed_tf != tf:
            logging.info("parse_open_message: TF from message '%s' differs from webhook '%s', using message TF", parsed_tf, tf)
            tf = parsed_tf

    return {
        "trade_id":  trade_id,
        "symbol":    symbol,
        "side":      side,
        "tf":        tf,
        "case_type": case_type,
        "exchange":  exchange,
        "entry":     float(entry_m.group(1)),
        "sl":        float(sl_m.group(1)),
        "tp":        float(tp_m.group(1)),
        "risk":      float(risk_m.group(1)) if risk_m else None,
        "swept":     swept,
        "status":    "active",
        "opened_at": opened_at,
    }


# ── Close message parser ────────────────────────────────────────────────────

_CLOSE_TRADE_ID = re.compile(r"🆔\s*(GOATv2_\S+)")
_CLOSE_RESULT   = re.compile(r"Trade Closed:\s*(TP|SL|BE)")
_CLOSE_HIT      = re.compile(r"Hit:\s*([\d.]+)")
_CLOSE_PNL      = re.compile(r"PnL:\s*([+-]?[\d.]+)")


def parse_close_message(text: str) -> dict | None:
    """Parse a GOAT close trade message. Returns a dict or None."""
    if detect_message_type(text) != "close":
        return None

    trade_id_m = _CLOSE_TRADE_ID.search(text)
    if not trade_id_m:
        return None

    result_m = _CLOSE_RESULT.search(text)
    raw_result = result_m.group(1) if result_m else "SL"
    status_map = {"TP": "tp", "SL": "sl", "BE": "be_hit"}
    status = status_map.get(raw_result, "sl")

    hit_m = _CLOSE_HIT.search(text)
    pnl_m = _CLOSE_PNL.search(text)

    return {
        "trade_id":    trade_id_m.group(1),
        "status":      status,
        "close_price": float(hit_m.group(1)) if hit_m else None,
        "pnl":         float(pnl_m.group(1)) if pnl_m else None,
        "closed_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
