"""
Export GOATv2 backtest trades to CSV for visual validation on TradingView.
Run after goat_14_vbt_vec.py finishes, or paste this at the end of that file.
"""

import pandas as pd
import json


def export_trades_csv(trades, filename="goat_bt_trades.csv"):
    """Export trades to CSV with all fields needed for visual validation."""
    rows = []
    for i, t in enumerate(trades):
        entry_ts = t.get("entry_ts")
        exit_ts = t.get("exit_ts")
        
        row = {
            "#": i + 1,
            "side": t["side"],
            "case": t["case"],
            "result": t["result"],
            "pnl_r": t.get("pnl_r", 0),
            "entry": t["entry"],
            "sl": t.get("original_sl", t["sl"]),
            "tp": t["tp"],
            "risk": t["risk"],
            "max_r": round(t.get("max_r", 0), 2),
            "entry_bar": t["entry_bar"],
            "exit_bar": t.get("exit_bar", ""),
            "duration_bars": t.get("duration_bars", ""),
            "entry_time": str(entry_ts)[:19] if entry_ts is not None else "",
            "exit_time": str(exit_ts)[:19] if exit_ts is not None else "",
            "filled_count": t.get("filled_count", 1),
            "avg_entry": round(t.get("avg_entry", t["entry"]), 2),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\n✅ Exported {len(rows)} trades to {filename}")
    return df


def print_trade_table(trades, label=""):
    """Print a formatted trade table to terminal."""
    if label:
        print(f"\n{'='*120}")
        print(f"  {label}")
        print(f"{'='*120}")
    
    print(f"\n{'#':>3} {'Side':<5} {'Case':<8} {'Result':<4} {'PnL':>6} "
          f"{'Entry':>12} {'SL':>12} {'TP':>12} {'MaxR':>5} "
          f"{'Dur':>4} {'Entry Time':<20} {'Exit Time':<20}")
    print("-" * 120)
    
    for i, t in enumerate(trades):
        entry_ts = str(t.get("entry_ts", ""))[:16]
        exit_ts = str(t.get("exit_ts", ""))[:16]
        result = t["result"]
        
        # Color indicators
        if result == "TP":
            marker = "✅"
        elif result == "SL":
            marker = "❌"
        elif result == "BE":
            marker = "🔄"
        else:
            marker = "⏳"
        
        print(f"{i+1:>3} {t['side']:<5} {t['case']:<8} {marker:<4} "
              f"{t.get('pnl_r', 0):>+5.1f}R "
              f"{t['entry']:>12.2f} {t.get('original_sl', t['sl']):>12.2f} "
              f"{t['tp']:>12.2f} {t.get('max_r', 0):>5.1f} "
              f"{t.get('duration_bars', ''):>4} {entry_ts:<20} {exit_ts:<20}")
    
    # Summary
    wins = sum(1 for t in trades if t["result"] == "TP")
    losses = sum(1 for t in trades if t["result"] == "SL")
    bes = sum(1 for t in trades if t["result"] == "BE")
    net = sum(t.get("pnl_r", 0) for t in trades if t.get("pnl_r") is not None)
    print("-" * 120)
    print(f"    Total: {len(trades)} | ✅ {wins} | ❌ {losses} | 🔄 {bes} | Net: {net:+.1f}R")


def print_trades_for_tradingview(trades):
    """Print trade entry/exit times + prices for quick TradingView lookup."""
    print(f"\n{'='*80}")
    print(f"  TRADINGVIEW LOOKUP — Jump to these timestamps")
    print(f"{'='*80}")
    
    for i, t in enumerate(trades):
        entry_ts = str(t.get("entry_ts", ""))[:16]
        exit_ts = str(t.get("exit_ts", ""))[:16]
        result = t["result"]
        marker = "✅" if result == "TP" else "❌" if result == "SL" else "🔄" if result == "BE" else "⏳"
        
        print(f"\n  Trade #{i+1}: {t['side']} {t['case']} → {marker} {result} ({t.get('pnl_r', 0):+.1f}R)")
        print(f"    📅 Entry: {entry_ts}  |  Price: {t['entry']:.2f}")
        print(f"    📅 Exit:  {exit_ts}  |  SL: {t.get('original_sl', t['sl']):.2f}  |  TP: {t['tp']:.2f}")
        print(f"    📐 Max R: {t.get('max_r', 0):.1f}  |  Duration: {t.get('duration_bars', '?')} bars")


# ═══════════════════════════════════════════════════════════════
# If running standalone, load from the backtest
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Import and run the backtest to get trades
    from goat_14_vbt_vec import download_ohlcv, run_backtest
    
    print("=" * 60)
    print("GOATv2 TRADE EXPORT — BTC/USDT:USDT M30")
    print("=" * 60)
    
    df_raw = download_ohlcv(symbol="BTC/USDT:USDT", timeframe="30m", days_back=180)
    
    # Run baseline (No BE, No DCA)
    trades = run_backtest(df_raw, rr_ratio=3, be_trigger_r=None, use_dca=False)
    
    # Print trade table
    print_trade_table(trades, "BASELINE TRADES — No BE, No DCA")
    
    # Print TradingView lookup
    print_trades_for_tradingview(trades)
    
    # Export to CSV
    export_trades_csv(trades, "goat_bt_trades_baseline.csv")
    
    print(f"\n💡 Open TradingView on BTCUSDT.P M30 and check each timestamp above.")
    print(f"   Verify: entry level, SL pivot, sweep bar, and trigger bar.")