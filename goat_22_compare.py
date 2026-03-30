"""
GOATv2 — Multi-Coin Comparison Runner
Launches goat_20_vbt_equity.py logic for multiple coins, collects stats,
ranks by Return/DD, and suggests capital allocation.

Usage:
  python3 goat_22_compare.py --coins LTC,LINK,SOL,BTC,ETH --tf 5m --days 720
  python3 goat_22_compare.py --coins LTC,LINK,SOL --tf 5m --days 720 --partial 1.5
  python3 goat_22_compare.py --coins LTC,LINK,SOL --tf 5m --days 720 --partial 1.5 --partial-pct 75
  python3 goat_22_compare.py --coins LTC,LINK,SOL --tf 5m --days 180 --rr 4 --be 1.5
  python3 goat_22_compare.py --coins LTC,LINK,SOL --tf 5m --days 720 --cases 13
  python3 goat_22_compare.py --coins-file coins.txt --tf 5m --days 720
  python3 goat_22_compare.py --coins LTC,LINK,SOL --tf 5m --days 720 --filters ao
"""

import pandas as pd
import numpy as np
import time as time_module
import argparse
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# Import backtest engine components
from goat_20_vbt_equity import (
    precompute_all, run_backtest, calc_max_dd,
    compute_warmup, parse_cases, parse_filters, compute_equity_curve,
    export_csv, export_equity_csv,
)
from goat_15_data_manager import get_ohlcv


def run_single_coin(symbol, timeframe, days, rr, be, warmup,
                    enable_c1, enable_c2, enable_c3,
                    partial_tp_r, partial_tp_pct, force,
                    start_date=None, end_date=None, active_filters=None,
                    donchian_period=20):
    """Run backtest for a single coin, return stats dict."""
    t0 = time_module.perf_counter()

    df_raw = get_ohlcv(symbol, timeframe, days, force_download=force,
                       start_date=start_date, end_date=end_date)
    if df_raw is None or len(df_raw) == 0:
        return None

    pre = precompute_all(df_raw, donchian_period=donchian_period)
    trades = run_backtest(
        pre, rr_ratio=rr, be_trigger_r=be, warmup=warmup,
        enable_c1=enable_c1, enable_c2=enable_c2, enable_c3=enable_c3,
        partial_tp_r=partial_tp_r, partial_tp_pct=partial_tp_pct,
        quiet=True, active_filters=active_filters,
    )

    closed = [t for t in trades if t["result"] not in ("OPEN", None)]
    total = len(closed)
    elapsed = time_module.perf_counter() - t0

    if total == 0:
        return {"symbol": symbol, "trades": 0, "elapsed": elapsed}

    # ── Core stats ──
    wins = sum(1 for t in closed if t.get("pnl_r") and t["pnl_r"] > 0.001)
    losses = sum(1 for t in closed if t.get("pnl_r") and t["pnl_r"] < -0.001)
    bes = total - wins - losses
    wr = wins / total * 100

    net_r = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
    rpt = net_r / total
    max_dd_r = calc_max_dd(closed)
    ret_dd = net_r / max_dd_r if max_dd_r > 0 else 999

    gross_win = sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] > 0)
    gross_loss = abs(sum(t["pnl_r"] for t in closed if t["pnl_r"] and t["pnl_r"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    # ── Monthly stats ──
    exit_months = {}
    for t in closed:
        ts = str(t.get("exit_ts", ""))[:7]
        if len(ts) >= 7:
            if ts not in exit_months:
                exit_months[ts] = 0.0
            exit_months[ts] += t.get("pnl_r", 0) or 0
    green_months = sum(1 for v in exit_months.values() if v > 0)
    total_months = len(exit_months)
    monthly_wr = green_months / total_months * 100 if total_months > 0 else 0
    r_per_month = net_r / total_months if total_months > 0 else 0

    # ── Streaks ──
    max_cw = 0
    max_cl = 0
    cw = 0
    cl = 0
    for t in closed:
        pnl = t.get("pnl_r", 0) or 0
        if pnl > 0.001:
            cw += 1; cl = 0; max_cw = max(max_cw, cw)
        elif pnl < -0.001:
            cl += 1; cw = 0; max_cl = max(max_cl, cl)
        else:
            cw = 0; cl = 0

    # ── By case ──
    case_stats = {}
    for c in ["C1", "C2", "C3"]:
        ct = [t for t in closed if t["case"] == c]
        if ct:
            cn = len(ct)
            cw_n = sum(1 for t in ct if t.get("pnl_r") and t["pnl_r"] > 0.001)
            c_net = sum(t["pnl_r"] for t in ct if t["pnl_r"] is not None)
            case_stats[c] = {"n": cn, "w": cw_n, "wr": cw_n/cn*100, "net": c_net}

    # ── Duration ──
    durs = [t["duration_bars"] for t in closed if t.get("duration_bars")]
    avg_dur = np.mean(durs) if durs else 0
    med_dur = np.median(durs) if durs else 0

    # ── Compounded equity ──
    eq_pts, final_bal, peak_bal, max_dd_pct, max_dd_usd, _, _ = \
        compute_equity_curve(closed, 50000, 2.0, rr)
    comp_return = (final_bal - 50000) / 50000 * 100

    # ── Rolling R stability (last 50 trades avg vs overall) ──
    if total >= 50:
        last_50_r = np.mean([t["pnl_r"] for t in closed[-50:] if t["pnl_r"] is not None])
    else:
        last_50_r = rpt
    stability = last_50_r / rpt if rpt != 0 else 1.0

    return {
        "symbol": symbol,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "bes": bes,
        "wr": round(wr, 1),
        "net_r": round(net_r, 1),
        "rpt": round(rpt, 3),
        "max_dd_r": round(max_dd_r, 1),
        "ret_dd": round(ret_dd, 1),
        "pf": round(pf, 2),
        "monthly_wr": round(monthly_wr, 0),
        "green_months": green_months,
        "total_months": total_months,
        "r_per_month": round(r_per_month, 1),
        "max_win_streak": max_cw,
        "max_loss_streak": max_cl,
        "case_stats": case_stats,
        "avg_dur": round(avg_dur, 0),
        "med_dur": round(med_dur, 0),
        "comp_return": round(comp_return, 1),
        "comp_dd": round(max_dd_pct, 1),
        "stability": round(stability, 2),
        "elapsed": round(elapsed, 1),
        "trades_list": trades,  # keep for CSV export
    }


def grade(val, thresholds, reverse=False):
    """Return emoji grade based on thresholds [weak, ok, good]."""
    if reverse:  # lower is better (DD, loss streak)
        if val <= thresholds[2]:
            return "🟢"
        elif val <= thresholds[1]:
            return "🟡"
        elif val <= thresholds[0]:
            return "🟠"
        return "🔴"
    else:  # higher is better
        if val >= thresholds[2]:
            return "🟢"
        elif val >= thresholds[1]:
            return "🟡"
        elif val >= thresholds[0]:
            return "🟠"
        return "🔴"


def print_comparison(results, rr, be, partial_r, partial_pct, timeframe, days,
                     start_date=None, end_date=None):
    valid = [r for r in results if r and r["trades"] > 0]
    failed = [r for r in results if r is None or r["trades"] == 0]

    if not valid:
        print("\n❌ No valid results to compare.")
        return

    # Sort by Return/DD (primary metric)
    valid.sort(key=lambda x: x["ret_dd"], reverse=True)

    partial_label = f" | Partial {partial_pct:.0f}%@{partial_r}R" if partial_r > 0 else ""

    if start_date or end_date:
        date_range_label = f"{start_date or 'start'} → {end_date or 'now'}"
    else:
        date_range_label = f"{days}d"

    print(f"\n{'═'*120}")
    print(f"  GOATv2 MULTI-COIN COMPARISON")
    print(f"  {timeframe} | {date_range_label} | RR={rr} | BE={be}R{partial_label}")
    print(f"  Ranked by Return/DD (risk-adjusted performance)")
    print(f"{'═'*120}")

    # ── Main Ranking Table ──
    print(f"\n{'Rank':<5} {'Coin':<18} {'Trades':>6} {'WR%':>6} {'Net R':>7} {'R/trade':>8} "
          f"{'MaxDD':>6} {'Ret/DD':>7} {'PF':>6} {'Mo WR':>7} {'R/mo':>6} "
          f"{'MaxLS':>6} {'Stab':>5} {'Grade'}")
    print("─" * 120)

    for i, r in enumerate(valid):
        # Composite grade
        g_retdd = grade(r["ret_dd"], [2, 5, 15])
        g_rpt = grade(r["rpt"], [0, 0.15, 0.30])
        g_dd = grade(r["max_dd_r"], [15, 10, 5], reverse=True)
        g_mwr = grade(r["monthly_wr"], [40, 60, 75])
        g_stab = grade(r["stability"], [0.3, 0.6, 0.8])
        g_ls = grade(r["max_loss_streak"], [10, 7, 5], reverse=True)

        overall = f"{g_retdd}{g_rpt}{g_dd}{g_mwr}{g_stab}{g_ls}"

        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"

        print(f"{medal:<5} {r['symbol']:<18} {r['trades']:>6} {r['wr']:>5.1f}% "
              f"{r['net_r']:>+6.1f}R {r['rpt']:>+7.3f}R "
              f"{r['max_dd_r']:>5.1f}R {r['ret_dd']:>6.1f}x {r['pf']:>5.2f} "
              f"{r['monthly_wr']:>4.0f}%({r['green_months']}/{r['total_months']}) "
              f"{r['r_per_month']:>+5.1f}R "
              f"{r['max_loss_streak']:>5} {r['stability']:>5.2f} {overall}")

    # ── Compounded Results ──
    print(f"\n{'═'*90}")
    print(f"  COMPOUNDED RESULTS ($50k start, 2% risk)")
    print(f"{'═'*90}")
    print(f"{'Rank':<5} {'Coin':<18} {'Final $':>12} {'Return':>8} {'MaxDD%':>8} {'CompRet/DD':>10}")
    print("─" * 65)

    for i, r in enumerate(valid):
        final = 50000 * (1 + r["comp_return"] / 100)
        comp_ret_dd = r["comp_return"] / r["comp_dd"] if r["comp_dd"] > 0 else 999
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        print(f"{medal:<5} {r['symbol']:<18} ${final:>10,.0f} {r['comp_return']:>+6.1f}% "
              f"{r['comp_dd']:>6.1f}% {comp_ret_dd:>9.1f}x")

    # ── Case Breakdown ──
    print(f"\n{'═'*90}")
    print(f"  CASE BREAKDOWN")
    print(f"{'═'*90}")
    print(f"{'Coin':<18} {'C1 trades':>10} {'C1 WR':>7} {'C1 net':>8} "
          f"{'C2 trades':>10} {'C2 WR':>7} {'C2 net':>8} "
          f"{'C3 trades':>10} {'C3 WR':>7} {'C3 net':>8}")
    print("─" * 105)

    for r in valid:
        cs = r["case_stats"]
        c1 = cs.get("C1", {"n": 0, "wr": 0, "net": 0})
        c2 = cs.get("C2", {"n": 0, "wr": 0, "net": 0})
        c3 = cs.get("C3", {"n": 0, "wr": 0, "net": 0})
        print(f"{r['symbol']:<18} {c1['n']:>10} {c1['wr']:>5.0f}% {c1['net']:>+6.0f}R "
              f"{c2['n']:>10} {c2['wr']:>5.0f}% {c2['net']:>+6.0f}R "
              f"{c3['n']:>10} {c3['wr']:>5.0f}% {c3['net']:>+6.0f}R")

    # ── Capital Allocation ──
    print(f"\n{'═'*70}")
    print(f"  SUGGESTED CAPITAL ALLOCATION (by Ret/DD weight)")
    print(f"{'═'*70}")

    # Only allocate to coins with positive Ret/DD
    alloc_pool = [r for r in valid if r["ret_dd"] > 0 and r["net_r"] > 0]
    if alloc_pool:
        total_retdd = sum(r["ret_dd"] for r in alloc_pool)
        print(f"{'Coin':<18} {'Ret/DD':>8} {'Weight':>8} {'$50k alloc':>12} {'$100k alloc':>12}")
        print("─" * 62)

        for r in alloc_pool:
            w = r["ret_dd"] / total_retdd
            print(f"{r['symbol']:<18} {r['ret_dd']:>7.1f}x {w:>7.1f}% "
                  f"${50000*w:>10,.0f} ${100000*w:>10,.0f}")

        print("─" * 62)
        print(f"{'TOTAL':<18} {total_retdd:>7.1f}x {'100%':>8} "
              f"${'50,000':>10} ${'100,000':>10}")
    else:
        print("  No coins with positive return to allocate.")

    # ── Grade Legend ──
    print(f"\n{'═'*70}")
    print(f"  GRADE LEGEND (6 dots: Ret/DD, R/trade, DD, Monthly WR, Stability, Loss Streak)")
    print(f"  🟢 = Good  🟡 = OK  🟠 = Weak  🔴 = Bad")
    print(f"{'═'*70}")

    # ── Failed coins ──
    if failed:
        print(f"\n  ⚠️  Failed/no data: {[r['symbol'] if r else '?' for r in failed]}")

    return valid


def export_comparison_csv(valid, filename):
    rows = []
    for i, r in enumerate(valid):
        rows.append({
            "rank": i + 1,
            "symbol": r["symbol"],
            "trades": r["trades"],
            "wins": r["wins"],
            "losses": r["losses"],
            "bes": r["bes"],
            "wr_pct": r["wr"],
            "net_r": r["net_r"],
            "r_per_trade": r["rpt"],
            "max_dd_r": r["max_dd_r"],
            "ret_dd": r["ret_dd"],
            "profit_factor": r["pf"],
            "monthly_wr_pct": r["monthly_wr"],
            "green_months": r["green_months"],
            "total_months": r["total_months"],
            "r_per_month": r["r_per_month"],
            "max_win_streak": r["max_win_streak"],
            "max_loss_streak": r["max_loss_streak"],
            "comp_return_pct": r["comp_return"],
            "comp_dd_pct": r["comp_dd"],
            "stability": r["stability"],
            "avg_duration": r["avg_dur"],
            "med_duration": r["med_dur"],
            "elapsed_sec": r["elapsed"],
        })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"\n  💾 Comparison → {filename}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GOATv2 — Multi-Coin Comparison Runner")
    parser.add_argument("--coins", type=str, default=None,
                        help="Comma-separated coin list: LTC,LINK,SOL,BTC,ETH")
    parser.add_argument("--coins-file", type=str, default=None,
                        help="File with one coin per line")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--rr", type=float, default=3.0)
    parser.add_argument("--be", type=float, default=2.0)
    parser.add_argument("--partial", type=float, default=0.0)
    parser.add_argument("--partial-pct", type=float, default=50.0)
    parser.add_argument("--cases", type=str, default="123")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-trades", action="store_true",
                        help="Also export individual trade CSVs per coin")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--filters", type=str, default="none",
                        help="Comma-separated signal filters: ao, mtf_lgcr, donchian, none (default: none). "
                             "AO: block LONG if AO>0, block SHORT if AO<0. "
                             "Note: mtf_lgcr requires goat_20_vbt_equity.py with --mtf-lgcr.")
    parser.add_argument("--donchian-period", type=int, default=20,
                        help="Donchian Channel lookback period for touch-based bias filter (default: 20)")
    args = parser.parse_args()

    # ── Parse coin list ──
    coins = []
    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    elif args.coins_file:
        if os.path.exists(args.coins_file):
            with open(args.coins_file) as f:
                coins = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
        else:
            print(f"❌ File not found: {args.coins_file}")
            sys.exit(1)
    else:
        print("❌ Provide --coins or --coins-file")
        print("   Example: --coins LTC,LINK,SOL,BTC,ETH")
        print("   Example: --coins-file coins.txt")
        sys.exit(1)

    # Build full symbol names
    symbols = []
    for c in coins:
        if "/" in c:
            symbols.append(c)  # Already full: SOL/USDT:USDT
        elif c.endswith("USDT"):
            symbols.append(f"{c[:-4]}/USDT:USDT")  # SOLUSDT → SOL/USDT:USDT
        else:
            symbols.append(f"{c}/USDT:USDT")  # SOL → SOL/USDT:USDT

    warmup = compute_warmup(args.tf, args.warmup)
    enable_c1, enable_c2, enable_c3 = parse_cases(args.cases)
    cases_str = f"{'C1' if enable_c1 else ''}{'C2' if enable_c2 else ''}{'C3' if enable_c3 else ''}"
    active_filters = parse_filters(args.filters)
    partial_label = f" | Partial {args.partial_pct:.0f}%@{args.partial}R" if args.partial > 0 else ""
    filter_label = f" | Filters: {','.join(sorted(active_filters)).upper()}" if active_filters else ""

    if args.start or args.end:
        date_range_label = f"{args.start or 'start'} → {args.end or 'now'}"
    else:
        date_range_label = f"{args.days}d"

    print("=" * 70)
    print(f"  GOATv2 MULTI-COIN COMPARISON")
    print(f"  {len(symbols)} coins | {args.tf} | {date_range_label} | RR={args.rr} | BE={args.be}R")
    print(f"  Cases: {cases_str}{partial_label}{filter_label}")
    print(f"  Warmup: {warmup} bars")
    print(f"  Coins: {', '.join(symbols)}")
    print("=" * 70)

    print("\n⚡ Numba JIT warmup on first coin (be patient)...\n")

    results = []
    total_t0 = time_module.perf_counter()

    for i, symbol in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {symbol}...", end="", flush=True)
        try:
            r = run_single_coin(
                symbol, args.tf, args.days, args.rr, args.be, warmup,
                enable_c1, enable_c2, enable_c3,
                args.partial, args.partial_pct, args.force,
                start_date=args.start, end_date=args.end,
                active_filters=active_filters,
                donchian_period=args.donchian_period,
            )
            if r and r["trades"] > 0:
                print(f" ✅ {r['trades']} trades, {r['net_r']:+.1f}R, "
                      f"Ret/DD={r['ret_dd']:.1f}x ({r['elapsed']:.1f}s)")
            else:
                print(f" ⚠️  No trades")
            results.append(r)
        except Exception as e:
            print(f" ❌ Error: {e}")
            results.append(None)

    total_elapsed = time_module.perf_counter() - total_t0
    print(f"\n  Total time: {total_elapsed:.1f}s")

    # ── Print comparison ──
    valid = print_comparison(
        results, args.rr, args.be, args.partial, args.partial_pct,
        args.tf, args.days, start_date=args.start, end_date=args.end,
    )

    # ── Build date tag for filenames ──
    if args.start or args.end:
        date_tag = f"_{(args.start or '').replace('-', '')}_{(args.end or 'now').replace('-', '')}"
    else:
        date_tag = f"_{args.days}d"

    # ── Export comparison CSV ──
    if valid:
        be_tag = f"be{args.be}".replace(".", "")
        partial_tag = f"_p{args.partial}".replace(".", "") if args.partial > 0 else ""
        csv_file = f"goat_compare_{args.tf}{date_tag}_{be_tag}_{cases_str.lower()}{partial_tag}.csv"
        export_comparison_csv(valid, csv_file)

    # ── Export individual trade CSVs ──
    if args.export_trades and valid:
        print(f"\n  📁 Exporting individual trade CSVs...")
        for r in valid:
            if r.get("trades_list"):
                sym_safe = r["symbol"].split(":")[0].replace("/", "_")
                be_tag = f"be{args.be}".replace(".", "")
                partial_tag = f"_p{args.partial}".replace(".", "") if args.partial > 0 else ""
                fname = f"goat_eq_{sym_safe}_{args.tf}_{be_tag}_{cases_str.lower()}{partial_tag}{date_tag}_trades.csv"
                export_csv(r["trades_list"], fname)

    print(f"\n{'═'*70}")
    print(f"  ✅ COMPARISON COMPLETE")
    print(f"  Primary ranking metric: Return/DD (flat R)")
    print(f"  Always compare risk-adjusted, not raw return")
    print(f"{'═'*70}")