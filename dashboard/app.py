"""
app.py — FastAPI backend for GOAT trade dashboard.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from db import close_trade, get_active_symbols, get_trades, init_db, insert_trade, update_max_r
from parser import detect_message_type, parse_close_message, parse_open_message
from price_feed import close_exchange, poll_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    PRICE_POLL_INTERVAL = int(os.getenv("PRICE_POLL_INTERVAL", "5"))
except ValueError:
    raise ValueError(
        "PRICE_POLL_INTERVAL environment variable must be an integer number of seconds"
    )

try:
    APP_PORT = int(os.getenv("PORT", "8080"))
except ValueError:
    raise ValueError("PORT environment variable must be an integer")

APP_HOST = os.getenv("HOST", "0.0.0.0")

# In-memory price cache: {symbol: price}
_prices: dict[str, float] = {}


async def _price_poller():
    """Background task: poll Bybit every PRICE_POLL_INTERVAL seconds."""
    while True:
        try:
            symbols = await get_active_symbols()
            if symbols:
                fresh = await poll_prices(symbols)
                _prices.update(fresh)

                # Update max_r for active trades
                trades = await get_trades({"status": "active"})
                for trade in trades:
                    sym = trade["symbol"]
                    price = _prices.get(sym)
                    if price is None:
                        continue
                    entry = trade["entry"]
                    sl = trade["sl"]
                    risk = abs(entry - sl)
                    if risk <= 0:
                        continue
                    if trade["side"] == "BULL":
                        current_r = (price - entry) / risk
                    else:
                        current_r = (entry - price) / risk
                    if current_r > trade.get("max_r", 0):
                        await update_max_r(trade["trade_id"], current_r)

                    # Check SL/TP hit using the freshest max_r value
                    tp = trade["tp"]
                    be_level = trade.get("be_level", 1.0)
                    max_r_val = max(trade.get("max_r", 0), current_r)

                    # Effective SL: moves to entry once BE threshold is reached
                    effective_sl = entry if max_r_val >= be_level else sl

                    status = None
                    pnl = None

                    if trade["side"] == "BULL":
                        if price <= effective_sl:
                            status = "be_hit" if effective_sl == entry else "sl"
                            pnl = 0.0 if status == "be_hit" else -1.0
                        elif price >= tp:
                            status = "tp"
                            pnl = (tp - entry) / risk
                    else:  # BEAR
                        if price >= effective_sl:
                            status = "be_hit" if effective_sl == entry else "sl"
                            pnl = 0.0 if status == "be_hit" else -1.0
                        elif price <= tp:
                            status = "tp"
                            pnl = (entry - tp) / risk

                    if status is not None:
                        closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                        # Use target level as close_price, not the live mid-price
                        if status == "tp":
                            close_price = tp
                        elif status == "be_hit":
                            close_price = entry
                        else:
                            close_price = sl
                        logger.info(
                            "Price poller closed %s: %s at %.8f (live=%.8f)",
                            trade["trade_id"], status, close_price, price,
                        )
                        await close_trade(trade["trade_id"], status, close_price, pnl, closed_at)
        except Exception as exc:
            logger.error("Price poller error: %s", exc)
        await asyncio.sleep(PRICE_POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(_price_poller())
    yield
    task.cancel()
    await close_exchange()


app = FastAPI(title="GOAT Dashboard", lifespan=lifespan)

# Serve static files
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.post("/webhook")
async def webhook(request: Request, tf: str = Query(default="m30")):
    """Receive raw Discord message text and persist the trade."""
    body = await request.body()
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return JSONResponse({"status": "ignored", "reason": "empty body"}, status_code=200)

    msg_type = detect_message_type(text)
    logger.info("webhook tf=%s type=%s length=%d", tf, msg_type, len(text))

    if msg_type == "open":
        trade = parse_open_message(text, tf)
        if trade is None:
            return JSONResponse({"status": "error", "reason": "parse failed"}, status_code=422)
        await insert_trade(trade)
        return JSONResponse({"status": "ok", "action": "inserted", "trade_id": trade["trade_id"]})

    if msg_type == "close":
        close_data = parse_close_message(text)
        if close_data is None:
            return JSONResponse({"status": "error", "reason": "parse failed"}, status_code=422)
        await close_trade(
            close_data["trade_id"],
            close_data["status"],
            close_data.get("close_price"),
            close_data.get("pnl"),
            close_data["closed_at"],
        )
        return JSONResponse({"status": "ok", "action": "closed", "trade_id": close_data["trade_id"]})

    return JSONResponse({"status": "ignored", "reason": "unknown message type"})


@app.get("/api/trades")
async def api_trades(
    tf: str | None = Query(default=None),
    symbol: str | None = Query(default=None),
    status: str | None = Query(default=None),
    side: str | None = Query(default=None),
    case_type: str | None = Query(default=None),
    exchange: str | None = Query(default=None),
    from_date: str | None = Query(default=None, alias="from"),
    to_date: str | None = Query(default=None, alias="to"),
):
    filters = {
        k: v for k, v in {
            "tf": tf,
            "symbol": symbol,
            "status": status,
            "side": side,
            "case_type": case_type,
            "exchange": exchange,
            "from": from_date,
            "to": to_date,
        }.items()
        if v is not None
    }
    trades = await get_trades(filters)
    return JSONResponse(trades)


@app.get("/api/prices")
async def api_prices():
    return JSONResponse(_prices)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
