"""
06 — Trade management: ID generation, active trade tracking, trade monitoring.
"""

import logging
import pandas as pd

from goat_01_utilis import fmt
from goat_02_data import fetch_with_retry
from goat_07_discord import send_discord_notification

# ─── Shared State ───────────────────────────────────────────────
alerted_goats = set()
alerted_provisional = set()
bull_goat_count = 0
bear_goat_count = 0
active_trades = {}
trade_counter = 0
trade_results = {
    "wins": 0, "losses": 0,
    "LGCR": {"wins": 0, "losses": 0},
    "LG_LINE": {"wins": 0, "losses": 0},
    "PIVOT": {"wins": 0, "losses": 0},
}


# ─── Trade ID ───────────────────────────────────────────────────

def generate_trade_id(symbol, side, timestamp):
    """Generate a unique trade identifier."""
    global trade_counter
    trade_counter += 1
    sym_short = symbol.replace("/USDT:USDT", "").replace("/", "")
    ts_short = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
    return f"GOATv2_{side}_{sym_short}_{ts_short}_{trade_counter:04d}"


# ─── Trade Monitoring ────────────────────────────────────────────

async def check_active_trades(exchange, cfg):
    """Check all active trades against latest candle data for TP/SL hits."""
    if not active_trades:
        return

    symbols_to_check = set(t["symbol"] for t in active_trades.values())
    latest_candles = {}

    for symbol in symbols_to_check:
        try:
            ohlcv = await fetch_with_retry(
                exchange, symbol, cfg["timeframe"],
                cfg["fetch_limit"], cfg["max_retries"], cfg["rate_limit_delay"]
            )
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df[["open", "high", "low", "close", "volume"]] = (
                    df[["open", "high", "low", "close", "volume"]].astype(float)
                )
                latest_candles[symbol] = df
        except Exception as e:
            logging.error(f"Error fetching candles for trade check on {symbol}: {e}")

    trades_to_close = []

    for trade_id, trade in active_trades.items():
        symbol = trade["symbol"]
        if symbol not in latest_candles:
            continue

        df = latest_candles[symbol]
        entry_ts = trade["entry_timestamp"]
        new_candles = df[df["timestamp"] > entry_ts]
        if new_candles.empty:
            continue

        side = trade["side"]
        entry = trade["entry"]
        sl = trade["sl"]
        tp = trade["tp"]
        risk = trade["risk"]
        current_close = df.iloc[-1]["close"]

        hit_tp = False
        hit_sl = False
        hit_ts = None
        hit_price = None
        max_favorable = 0.0

        for _, candle in new_candles.iterrows():
            if side == "BULL":
                if risk > 0:
                    max_favorable = max(max_favorable, (candle["high"] - entry) / risk)
                if candle["low"] <= sl:
                    hit_sl, hit_ts, hit_price = True, candle["timestamp"], sl
                    break
                if candle["high"] >= tp:
                    hit_tp, hit_ts, hit_price = True, candle["timestamp"], tp
                    break
            else:
                if risk > 0:
                    max_favorable = max(max_favorable, (entry - candle["low"]) / risk)
                if candle["high"] >= sl:
                    hit_sl, hit_ts, hit_price = True, candle["timestamp"], sl
                    break
                if candle["low"] <= tp:
                    hit_tp, hit_ts, hit_price = True, candle["timestamp"], tp
                    break

        if hit_tp or hit_sl:
            result = "TP ✅" if hit_tp else "SL ❌"
            pnl = trade["risk"] * cfg["rr_ratio"] if hit_tp else -trade["risk"]
            pnl_sign = "+" if hit_tp else ""

            if hit_tp:
                trade_results["wins"] += 1
            else:
                trade_results["losses"] += 1

            case_label = trade.get("case_label", "UNKNOWN")
            if case_label in trade_results:
                if hit_tp:
                    trade_results[case_label]["wins"] += 1
                else:
                    trade_results[case_label]["losses"] += 1
            else:
                logging.warning(f"[TRADE CLOSED] Unknown case_label '{case_label}' for {trade_id}; per-case stats not updated")

            total = trade_results["wins"] + trade_results["losses"]
            win_rate = (trade_results["wins"] / total * 100) if total > 0 else 0

            swept_label = trade.get("swept_label")
            swept_value = trade.get("swept_value")

            emoji = "🎯" if hit_tp else "🛑"
            msg = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{emoji} **Trade Closed: {result}**\n"
                f"🆔 `{trade_id}`\n"
                f"📊 {trade['symbol']} {side} {cfg['timeframe']}\n"
                f"📊 GOATv2_{side}_{case_label}\n"
            )
            if swept_label and swept_value is not None:
                msg += f"🔖 {swept_label}: {fmt(swept_value)}\n"
            msg += f"💰 Entry: {fmt(entry)} | Hit: {fmt(hit_price)} | PnL: {pnl_sign}{fmt(pnl)}\n"
            if hit_sl:
                msg += f"📐 Max R before reversal: {max_favorable:.1f}R\n"
            msg += (
                f"\n📈 **Stats:** {trade_results['wins']}W / {trade_results['losses']}L "
                f"(Win Rate: {win_rate:.1f}%)\n"
            )
            case_parts = []
            for c in ["LGCR", "LG_LINE", "PIVOT"]:
                cw = trade_results[c]["wins"]
                cl = trade_results[c]["losses"]
                ct = cw + cl
                if ct > 0:
                    cr = cw / ct * 100
                    case_parts.append(f"{c}: {cw}W/{cl}L ({cr:.1f}%)")
            if case_parts:
                msg += f"📊 {' | '.join(case_parts)}"

            logging.info(msg.replace("**", "").replace("`", ""))
            await send_discord_notification(msg, cfg["webhook_url"])
            trades_to_close.append(trade_id)
        else:
            if side == "BULL":
                unrealized = current_close - entry
                dist_to_tp = tp - current_close
                dist_to_sl = current_close - sl
            else:
                unrealized = entry - current_close
                dist_to_tp = current_close - tp
                dist_to_sl = sl - current_close
            current_r = unrealized / risk if risk > 0 else 0

            logging.info(
                f"[TRADE OPEN] {trade_id} | {symbol} {side} {cfg['timeframe']} | "
                f"Entry={fmt(entry)} Current={fmt(current_close)} | "
                f"PnL={unrealized:+.6f} ({current_r:+.1f}R) | "
                f"MaxR={max_favorable:.1f}R | "
                f"Dist→TP={fmt(dist_to_tp)} Dist→SL={fmt(dist_to_sl)}"
            )

    for trade_id in trades_to_close:
        del active_trades[trade_id]