"""
08 — Main analysis orchestrator: analyze_symbol().
Imports all modules and runs the full GOAT scan per symbol.
"""

import time
import logging
import pandas as pd

from function_LG_major_functions import calculate_heikin_ashi

from goat_01_utilis import fmt
from goat_02_data import fetch_with_retry
from goat_03_patterns import detect_patterns
from goat_04_pivots import calculate_trade_levels
from goat_05_cases import check_goat
from goat_06_trades import (
    alerted_goats, alerted_provisional,
    bull_goat_count, bear_goat_count,
    active_trades, trade_results,
    generate_trade_id, check_active_trades,
)
from goat_07_discord import send_discord_notification


# ── HA stabilization: fetch extra bars so the recursive HA_Open
#    seed is far enough back that the tail values don't shift
#    when the window slides forward between scans.
HA_WARMUP_BARS = 300  # extra bars fetched but trimmed after HA calc


async def analyze_symbol(exchange, symbol, df_regular_map, cfg, run_type="final"):
    """Full GOAT analysis for a single symbol."""
    t0 = time.perf_counter()

    analysis_bars = cfg["fetch_limit"]  # bars we actually want (e.g. 200)
    total_fetch = analysis_bars + HA_WARMUP_BARS  # fetch more for HA warmup

    ohlcv = await fetch_with_retry(
        exchange, symbol, cfg["timeframe"],
        total_fetch, cfg["max_retries"], cfg["rate_limit_delay"]
    )
    if not ohlcv:
        logging.info(f"{symbol}: no OHLCV fetched")
        return False
    if len(ohlcv) < 6:
        logging.info(f"{symbol}: only {len(ohlcv)} bars fetched; need >=6")
        return False

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]].astype(float)
    )

    last_candle_open = df.iloc[-1]["timestamp"]
    last_candle_close = last_candle_open + pd.Timedelta(minutes=cfg["tf_minutes"])
    forming = last_candle_close > pd.Timestamp.now(tz="UTC")

    drop_forming = (run_type == "final") or (not cfg.get("allow_forming_on_provisional", False))
    if forming and drop_forming:
        logging.debug(f"{symbol}: dropping forming candle {last_candle_open} (run_type={run_type})")
        df = df.iloc[:-1]

    if len(df) < 5:
        logging.info(f"{symbol}: not enough closed bars after forming-filter (len={len(df)})")
        return False

    # ── Compute HA on FULL dataset (including warmup bars) ───────
    ha_full = calculate_heikin_ashi(df)
    if ha_full.empty or len(ha_full) < 5:
        logging.info(f"{symbol}: HA DF empty or <5 rows")
        return False

    # ── Trim warmup bars: keep only the last `analysis_bars` ────
    #    This ensures HA_Open values are stabilized and won't shift
    #    when the window slides forward on the next scan.
    if len(ha_full) > analysis_bars:
        ha_df = ha_full.iloc[-analysis_bars:].copy().reset_index(drop=True)
        # Also trim the regular df to match
        df_regular = df.iloc[-analysis_bars:].copy().reset_index(drop=True)
    else:
        ha_df = ha_full.reset_index(drop=True)
        df_regular = df.copy().reset_index(drop=True)

    df_regular_map[symbol] = df_regular

    ha_df = detect_patterns(ha_df)

    # ─── Debug tables ────────────────────────────────────────────
    tail = ha_df.tail(5).copy()
    debug_cols = ["timestamp", "HA_Open", "HA_High", "HA_Low", "HA_Close",
                  "bullish_LGC", "bearish_LGC", "bullish_LGCR", "bearish_LGCR"]
    logging.info(f"[DBG TABLE] {symbol}\n{tail[debug_cols].to_string(index=False)}")

    recent = ha_df.tail(10)
    logging.info(
        f"[FLAGS] {symbol} last10 "
        f"LGC(bull/bear)={recent['bullish_LGC'].sum()}/{recent['bearish_LGC'].sum()} "
        f"LGCR(bull/bear)={recent['bullish_LGCR'].sum()}/{recent['bearish_LGCR'].sum()}"
    )

    # ─── Globals ─────────────────────────────────────────────────
    global bull_goat_count, bear_goat_count

    last_idx = len(ha_df) - 1
    LOOKBACK_BARS = cfg.get("lookback_bars", 3)
    goat_found = False

    # ─── Diagnostic: lookback window ─────────────────────────────
    lookback_start = max(last_idx - LOOKBACK_BARS, 0)
    bear_both = sum(1 for i in range(lookback_start, last_idx + 1)
                    if ha_df.loc[i, 'bearish_LGC'] and ha_df.loc[i, 'bearish_LGCR'])
    bull_both = sum(1 for i in range(lookback_start, last_idx + 1)
                    if ha_df.loc[i, 'bullish_LGC'] and ha_df.loc[i, 'bullish_LGCR'])
    logging.info(
        f"[DIAG_LOOKBACK] {symbol} bars {lookback_start}-{last_idx}: "
        f"BEAR_BOTH={bear_both} BULL_BOTH={bull_both}"
    )

    # ─── Main loop: check each bar in lookback window ────────────
    for check_idx in range(last_idx, max(last_idx - LOOKBACK_BARS, 0) - 1, -1):
        is_bear = ha_df.loc[check_idx, "bearish_LGC"] and ha_df.loc[check_idx, "bearish_LGCR"]
        is_bull = ha_df.loc[check_idx, "bullish_LGC"] and ha_df.loc[check_idx, "bullish_LGCR"]

        if not (is_bear or is_bull):
            continue

        logging.info(
            f"[CANDIDATE] {symbol} bar {check_idx} {ha_df.loc[check_idx,'timestamp']} "
            f"is_bear={is_bear} is_bull={is_bull}"
        )

        sub_df = ha_df.iloc[:check_idx + 1].reset_index(drop=True)

        # ─── Unified GOAT check ─────────────────────────────────
        goat_bear, bear_case, bear_swept_label, bear_swept_value, bear_source_bar = (
            check_goat(sub_df, "BEAR", symbol=symbol) if is_bear else (False, None, None, None, None)
        )
        goat_bull, bull_case, bull_swept_label, bull_swept_value, bull_source_bar = (
            check_goat(sub_df, "BULL", symbol=symbol) if is_bull else (False, None, None, None, None)
        )

        triggered = goat_bear or goat_bull

        if not triggered:
            logging.info(
                f"[DIAG_ALL_CASES_FAILED] {symbol} bar {check_idx}: "
                f"gate passed but all cases failed"
            )
            continue

        ts_str = ha_df.loc[check_idx, "timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        side = "BEAR" if goat_bear else "BULL"
        case_label = bear_case if goat_bear else bull_case
        swept_label = bear_swept_label if goat_bear else bull_swept_label
        swept_value = bear_swept_value if goat_bear else bull_swept_value
        source_bar = bear_source_bar if goat_bear else bull_source_bar
        alert_key = (symbol, ts_str, side)

        # ─── Provisional alerts ──────────────────────────────────
        if run_type == "provisional":
            if alert_key in alerted_goats or alert_key in alerted_provisional:
                goat_found = True
                continue

            alerted_provisional.add(alert_key)
            msg = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🟠 **PROVISIONAL — {cfg['exchange_name']} {cfg['timeframe']} {symbol}**\n"
                f"⏰ {ts_str} (forming bar)\n"
                f"📊 GOATv2_{side}_{case_label}\n"
            )
            if swept_label and swept_value is not None:
                msg += f"🔖 {swept_label}: {fmt(swept_value)}\n"
            logging.info(msg.replace("**", "").replace("`", ""))
            await send_discord_notification(msg, cfg["webhook_url"])
            goat_found = True
            continue

        # ─── Final alerts ────────────────────────────────────────
        if alert_key in alerted_goats:
            goat_found = True
            continue

        alerted_goats.add(alert_key)
        alerted_provisional.add(alert_key)

        trade = calculate_trade_levels(ha_df, check_idx, side, cfg["rr_ratio"], signal_bar=last_idx, sweep_source_bar=source_bar)

        if side == "BULL":
            bull_goat_count += 1
        else:
            bear_goat_count += 1

        trade_id = generate_trade_id(symbol, side, ts_str)

        emoji = "🟢" if side == "BULL" else "🔴"
        msg = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} **{cfg['exchange_name']} {cfg['timeframe']} {symbol}**\n"
            f"🆔 `{trade_id}`\n"
            f"⏰ {ts_str}\n"
            f"📊 GOATv2_{side}_{case_label}\n"
        )
        if swept_label and swept_value is not None:
            msg += f"🔖 {swept_label}: {fmt(swept_value)}\n"

        if trade:
            msg += (
                f"\n💰 HA Entry: {fmt(trade['HA_entry'])} | "
                f"HA SL: {fmt(trade['HA_sl'])} | "
                f"HA TP({cfg['rr_ratio']}R): {fmt(trade['HA_tp'])} | "
                f"HA Risk: {fmt(trade['HA_risk'])}\n"
            )
            active_trades[trade_id] = {
                "symbol": symbol,
                "side": side,
                "entry": trade["HA_entry"],
                "sl": trade["HA_sl"],
                "tp": trade["HA_tp"],
                "risk": trade["HA_risk"],
                "entry_timestamp": ha_df.loc[check_idx, "timestamp"],
                "trade_id": trade_id,
            }
            logging.info(
                f"[TRADE REGISTERED] {trade_id} | {symbol} {side} {cfg['timeframe']} | "
                f"Entry={fmt(trade['entry'])} SL={fmt(trade['sl'])} TP={fmt(trade['tp'])}"
            )
        else:
            msg += "\n⚠️ Could not find pivot for SL calculation\n"

        msg += (
            f"\n🟢 Bull GOATv2 signals: {bull_goat_count}\n"
            f"🔴 Bear GOATv2 signals: {bear_goat_count}"
        )
        if active_trades:
            msg += f"\n📋 Active trades: {len(active_trades)}"

        logging.info(msg.replace("**", "").replace("`", ""))
        await send_discord_notification(msg, cfg["webhook_url"])
        goat_found = True

    if not goat_found:
        logging.info(f"{symbol}: no GOATv2 in last {LOOKBACK_BARS} bars")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logging.info(f"[SCAN TIME] {symbol} run_type={run_type} took {elapsed_ms:.1f} ms")
    return goat_found