"""
Inspect GOATv2 log — memory-efficient version.
Usage: python3 goat_09_inspect_log.py bot_goat_10_m5.log
"""

import re
import sys
from collections import defaultdict, Counter
from datetime import datetime

def inspect_log(logfile):
    # ─── Counters ───
    gate_passed = 0
    gate_failed = 0

    case_entered = defaultdict(int)
    case_triggered = defaultdict(int)
    case_failed = defaultdict(int)

    c1_no_prior_lgcr = 0
    c1_no_sweep = 0
    c1_first_sweep_failed = 0
    c2_no_lg_line = 0
    c2_no_sweep = 0
    c2_first_sweep_failed = 0
    c3_no_pivots = 0
    c3_no_sweep = 0

    all_cases_failed = 0
    availability = defaultdict(list)
    symbol_gates = defaultdict(int)
    symbol_triggers = defaultdict(lambda: defaultdict(int))

    # ─── Trade tracking ───
    trade_case_map = {}       # trade_id -> {case, side}
    trade_registered = {}     # trade_id -> timestamp string
    trade_closed = {}         # trade_id -> {result, timestamp}
    trade_max_r = {}          # trade_id -> float

    # Rolling window: keep last 20 lines to link nearby context
    window = []
    WINDOW_SIZE = 20

    print(f"Parsing {logfile}...")
    line_count = 0

    with open(logfile, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 500000 == 0:
                print(f"  ...processed {line_count} lines")

            line = line.rstrip('\n')
            window.append(line)
            if len(window) > WINDOW_SIZE:
                window.pop(0)

            # ─── GATE ───
            if 'GATE PASSED' in line:
                gate_passed += 1
                m = re.search(r'\] (\S+) bar', line)
                if m:
                    symbol_gates[m.group(1)] += 1
            elif 'GATE FAILED' in line:
                gate_failed += 1

            # ─── AVAILABILITY ───
            elif 'AVAILABILITY' in line:
                m = re.search(r'prior_LGCR=(.*?)$', line)
                if m:
                    val = m.group(1).strip()
                    availability['prior_LGCR_found' if val != 'NONE' else 'prior_LGCR_none'].append(1)
                m = re.search(r'LG_line_candidates=(\d+)', line)
                if m:
                    availability['lg_line_candidates'].append(int(m.group(1)))
                m = re.search(r'pivot_candidates=(\d+)', line)
                if m:
                    availability['pivot_candidates'].append(int(m.group(1)))

            # ─── CASE 1 ───
            elif 'DIAG_' in line and '_C1]' in line and 'no prior' in line:
                c1_no_prior_lgcr += 1
                case_entered['C1'] += 1
                case_failed['C1'] += 1
            elif 'DIAG_' in line and '_C1]' in line and 'prior LGCR at bar' in line:
                case_entered['C1'] += 1
            elif 'C1_NO_SWEEP' in line:
                if 'no sweep candidate found' in line:
                    c1_no_sweep += 1
                elif 'first sweep candidate failed' in line:
                    c1_first_sweep_failed += 1
                case_failed['C1'] += 1
            elif 'GOATv2_' in line and '_C1]' in line and 'Trigger' in line:
                case_triggered['C1'] += 1

            # ─── CASE 2 ───
            elif 'DIAG_' in line and '_C2]' in line and 'no ' in line and 'valid line' in line:
                c2_no_lg_line += 1
                case_entered['C2'] += 1
                case_failed['C2'] += 1
            elif 'DIAG_' in line and '_C2]' in line and 'best LG line' in line:
                case_entered['C2'] += 1
            elif 'C2_NO_SWEEP' in line:
                if 'no sweep candidate found' in line:
                    c2_no_sweep += 1
                elif 'first sweep candidate failed' in line:
                    c2_first_sweep_failed += 1
                case_failed['C2'] += 1
            elif 'GOATv2_' in line and '_C2]' in line and 'Trigger' in line:
                case_triggered['C2'] += 1

            # ─── CASE 3 ───
            elif 'DIAG_' in line and '_C3]' in line and 'no pivot' in line:
                c3_no_pivots += 1
                case_entered['C3'] += 1
                case_failed['C3'] += 1
            elif 'DIAG_' in line and '_C3]' in line and 'best pivot' in line:
                case_entered['C3'] += 1
            elif 'C3_NO_SWEEP' in line:
                c3_no_sweep += 1
                case_failed['C3'] += 1
            elif 'GOATv2_' in line and '_C3]' in line and 'Trigger' in line:
                case_triggered['C3'] += 1

            # ─── ALL CASES FAILED ───
            elif 'ALL_CASES_FAILED' in line:
                all_cases_failed += 1

            # ─── FINAL TRIGGER LINE ───
            elif 'GOATv2_' in line and 'TRIGGERED' in line:
                m = re.search(r'GOATv2_(\w+)\] (\S+):', line)
                if m:
                    sym = m.group(2)
                    if 'Case 1' in line:
                        symbol_triggers[sym]['C1'] += 1
                    elif 'Case 2' in line:
                        symbol_triggers[sym]['C2'] += 1
                    elif 'Case 3' in line:
                        symbol_triggers[sym]['C3'] += 1

            # ─── SIGNAL CASE TYPE: 📊 GOATv2_BULL_PIVOT etc ───
            if '📊 GOATv2_' in line:
                case_match = re.search(r'GOATv2_(BULL|BEAR)_(LGCR|LG_LINE|PIVOT)', line)
                if case_match:
                    # Look in window for trade ID
                    for wline in window:
                        id_match = re.search(r'(GOATv2_(?:BULL|BEAR)_\w+_\d+_\w+_\d+)', wline)
                        if id_match:
                            trade_case_map[id_match.group(1)] = {
                                'case': case_match.group(2),
                                'side': case_match.group(1)
                            }
                            break

            # ─── TRADE REGISTERED ───
            if 'TRADE REGISTERED' in line:
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                id_match = re.search(r'(GOATv2_(?:BULL|BEAR)_\w+_\d+_\w+_\d+)', line)
                if ts_match and id_match:
                    trade_registered[id_match.group(1)] = ts_match.group(1)

            # ─── TRADE CLOSED ───
            if 'Trade Closed' in line:
                result = 'TP' if 'TP' in line else 'SL' if 'SL' in line else 'UNKNOWN'
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                # Look in window for trade ID
                for wline in window:
                    id_match = re.search(r'(GOATv2_(?:BULL|BEAR)_\w+_\d+_\w+_\d+)', wline)
                    if id_match:
                        trade_closed[id_match.group(1)] = {
                            'result': result,
                            'timestamp': ts_match.group(1) if ts_match else None
                        }
                        break

            # ─── MAX R ───
            if 'Max R before reversal' in line:
                r_match = re.search(r'([\d.]+)R', line)
                # Look in window for trade ID
                for wline in window:
                    id_match = re.search(r'(GOATv2_(?:BULL|BEAR)_\w+_\d+_\w+_\d+)', wline)
                    if id_match:
                        if r_match:
                            trade_max_r[id_match.group(1)] = float(r_match.group(1))
                        break

    print(f"  Done. {line_count} lines parsed.\n")

    # ─── Build complete trade records ───
    all_trade_ids = set(list(trade_registered.keys()) + list(trade_closed.keys()))
    all_trades = []

    for tid in all_trade_ids:
        case_info = trade_case_map.get(tid, {})
        reg_ts = trade_registered.get(tid)
        close_info = trade_closed.get(tid, {})
        close_ts = close_info.get('timestamp')
        result = close_info.get('result', 'OPEN')
        max_r = trade_max_r.get(tid, None)

        duration_min = None
        if reg_ts and close_ts:
            try:
                t1 = datetime.strptime(reg_ts, '%Y-%m-%d %H:%M:%S')
                t2 = datetime.strptime(close_ts, '%Y-%m-%d %H:%M:%S')
                duration_min = (t2 - t1).total_seconds() / 60
            except:
                pass

        all_trades.append({
            'id': tid,
            'case': case_info.get('case', 'UNKNOWN'),
            'side': case_info.get('side', 'UNKNOWN'),
            'result': result,
            'max_r': max_r,
            'duration_min': duration_min,
        })

    # ═════════════════════════════════════════════════════════════
    # PRINT REPORT
    # ═════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  GOATv2 LOG INSPECTION REPORT")
    print("=" * 70)

    print(f"\n📊 GATE STATS:")
    print(f"  Gate passed (LGC+LGCR both true):  {gate_passed}")
    print(f"  Gate failed (missing LGC or LGCR):  {gate_failed}")

    print(f"\n📊 AVAILABILITY (when gate passed):")
    lgcr_found = len(availability.get('prior_LGCR_found', []))
    lgcr_none = len(availability.get('prior_LGCR_none', []))
    lg_counts = availability.get('lg_line_candidates', [])
    pv_counts = availability.get('pivot_candidates', [])
    print(f"  Prior LGCR found:    {lgcr_found}  |  Prior LGCR none: {lgcr_none}")
    if lg_counts:
        print(f"  LG line candidates:  min={min(lg_counts)} max={max(lg_counts)} avg={sum(lg_counts)/len(lg_counts):.1f}")
    if pv_counts:
        print(f"  Pivot candidates:    min={min(pv_counts)} max={max(pv_counts)} avg={sum(pv_counts)/len(pv_counts):.1f}")

    print(f"\n📊 CASE 1 — LGCR SWEEP:")
    print(f"  Entered:             {case_entered.get('C1', 0)}")
    print(f"  No prior LGCR:       {c1_no_prior_lgcr}")
    print(f"  No sweep found:      {c1_no_sweep}")
    print(f"  First sweep failed:  {c1_first_sweep_failed}")
    print(f"  ✅ TRIGGERED:         {case_triggered.get('C1', 0)}")

    print(f"\n📊 CASE 2 — LG LINE SWEEP:")
    print(f"  Entered:             {case_entered.get('C2', 0)}")
    print(f"  No LG line found:    {c2_no_lg_line}")
    print(f"  No sweep found:      {c2_no_sweep}")
    print(f"  First sweep failed:  {c2_first_sweep_failed}")
    print(f"  ✅ TRIGGERED:         {case_triggered.get('C2', 0)}")

    print(f"\n📊 CASE 3 — PIVOT SWEEP:")
    print(f"  Entered:             {case_entered.get('C3', 0)}")
    print(f"  No pivots found:     {c3_no_pivots}")
    print(f"  No sweep passed:     {c3_no_sweep}")
    print(f"  ✅ TRIGGERED:         {case_triggered.get('C3', 0)}")

    total_triggered = sum(case_triggered.values())
    print(f"\n📊 SUMMARY:")
    print(f"  Total triggers:      {total_triggered}")
    print(f"  All 3 cases failed:  {all_cases_failed}")
    if gate_passed + gate_failed > 0:
        print(f"  Gate pass rate:      {gate_passed}/{gate_passed+gate_failed} "
              f"({100*gate_passed/(gate_passed+gate_failed):.1f}%)")

    # ═════════════════════════════════════════════════════════════
    # TRADE RESULTS
    # ═════════════════════════════════════════════════════════════
    wins = [t for t in all_trades if t['result'] == 'TP']
    losses = [t for t in all_trades if t['result'] == 'SL']

    print(f"\n{'='*70}")
    print(f"  TRADE RESULTS")
    print(f"{'='*70}")
    print(f"\n📈 OVERALL:")
    print(f"  Total trades:  {len(all_trades)}")
    print(f"  ✅ Wins (TP):   {len(wins)}")
    print(f"  ❌ Losses (SL): {len(losses)}")
    still_open = [t for t in all_trades if t['result'] == 'OPEN']
    if still_open:
        print(f"  ⏳ Still open:  {len(still_open)}")
    if len(wins) + len(losses) > 0:
        wr = len(wins) / (len(wins) + len(losses)) * 100
        print(f"  Win rate:      {wr:.1f}%")
        net_r = len(wins) * 3 - len(losses)
        print(f"  Net PnL:       {net_r:+.0f}R")

    case_names = {'LGCR': 'C1 (LGCR Sweep)', 'LG_LINE': 'C2 (LG Line)', 'PIVOT': 'C3 (Pivot)'}

    print(f"\n📊 WINS BY CASE:")
    win_cases = Counter(t['case'] for t in wins)
    for case, label in case_names.items():
        print(f"  {label}: {win_cases.get(case, 0)}")
    if win_cases.get('UNKNOWN', 0) > 0:
        print(f"  UNKNOWN: {win_cases.get('UNKNOWN', 0)}")

    print(f"\n📊 LOSSES BY CASE:")
    loss_cases = Counter(t['case'] for t in losses)
    for case, label in case_names.items():
        print(f"  {label}: {loss_cases.get(case, 0)}")
    if loss_cases.get('UNKNOWN', 0) > 0:
        print(f"  UNKNOWN: {loss_cases.get('UNKNOWN', 0)}")

    print(f"\n📊 WIN RATE BY CASE:")
    for case, label in case_names.items():
        w = win_cases.get(case, 0)
        l = loss_cases.get(case, 0)
        t = w + l
        if t > 0:
            wr = w / t * 100
            net = w * 3 - l
            print(f"  {label}: {w}W / {l}L = {wr:.1f}% | Net: {net:+.0f}R")
        else:
            print(f"  {label}: no closed trades")

    print(f"\n📊 WIN RATE BY SIDE:")
    for side in ['BULL', 'BEAR']:
        w = len([t for t in wins if t['side'] == side])
        l = len([t for t in losses if t['side'] == side])
        t = w + l
        if t > 0:
            wr = w / t * 100
            net = w * 3 - l
            print(f"  {side}: {w}W / {l}L = {wr:.1f}% | Net: {net:+.0f}R")

    # ═════════════════════════════════════════════════════════════
    # TRADE DURATION
    # ═════════════════════════════════════════════════════════════
    trades_with_duration = [t for t in all_trades if t['duration_min'] is not None and t['duration_min'] > 0]

    if trades_with_duration:
        print(f"\n{'='*70}")
        print(f"  TRADE DURATION")
        print(f"{'='*70}")

        def duration_stats(durations, label):
            if not durations:
                print(f"\n⏱️  {label}: no data")
                return
            avg = sum(durations) / len(durations)
            durations_sorted = sorted(durations)
            median = durations_sorted[len(durations_sorted) // 2]
            mn = min(durations)
            mx = max(durations)
            print(f"\n⏱️  {label} ({len(durations)} trades):")
            print(f"  Average:  {avg:.0f} min ({avg/60:.1f} hrs)")
            print(f"  Median:   {median:.0f} min ({median/60:.1f} hrs)")
            print(f"  Fastest:  {mn:.0f} min ({mn/60:.1f} hrs)")
            print(f"  Slowest:  {mx:.0f} min ({mx/60:.1f} hrs)")
            buckets = [
                ("< 15 min", 0, 15),
                ("15-30 min", 15, 30),
                ("30-60 min", 30, 60),
                ("1-2 hrs", 60, 120),
                ("2-4 hrs", 120, 240),
                ("4-8 hrs", 240, 480),
                ("8-24 hrs", 480, 1440),
                ("1-3 days", 1440, 4320),
                ("> 3 days", 4320, 999999),
            ]
            print(f"  Distribution:")
            for label_b, lo, hi in buckets:
                count = sum(1 for d in durations if lo <= d < hi)
                if count > 0:
                    pct = count / len(durations) * 100
                    bar = '█' * int(pct / 2)
                    print(f"    {label_b:>12}: {count:4d} ({pct:5.1f}%) {bar}")

        all_dur = [t['duration_min'] for t in trades_with_duration]
        win_dur = [t['duration_min'] for t in trades_with_duration if t['result'] == 'TP']
        loss_dur = [t['duration_min'] for t in trades_with_duration if t['result'] == 'SL']

        duration_stats(all_dur, "ALL TRADES")
        duration_stats(win_dur, "WINS ONLY")
        duration_stats(loss_dur, "LOSSES ONLY")

        print(f"\n⏱️  DURATION BY CASE:")
        for case, label in case_names.items():
            case_dur = [t['duration_min'] for t in trades_with_duration if t['case'] == case]
            if case_dur:
                avg = sum(case_dur) / len(case_dur)
                print(f"  {label}: avg {avg:.0f} min ({avg/60:.1f} hrs) | {len(case_dur)} trades")

    # ═════════════════════════════════════════════════════════════
    # BREAKEVEN TRIGGER ANALYSIS
    # ═════════════════════════════════════════════════════════════
    losses_with_r = [t for t in losses if t['max_r'] is not None]

    if losses_with_r:
        print(f"\n{'='*70}")
        print(f"  BREAKEVEN TRIGGER ANALYSIS ({len(losses_with_r)} losses with Max R data)")
        print(f"{'='*70}")

        max_rs = [t['max_r'] for t in losses_with_r]

        print(f"\n📐 If we moved SL to breakeven at X R:")
        for trigger in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            saved = sum(1 for r in max_rs if r >= trigger)
            not_saved = len(max_rs) - saved
            pct = saved / len(max_rs) * 100
            current_net = len(wins) * 3 - len(losses)
            new_net = len(wins) * 3 - not_saved
            improvement = new_net - current_net
            print(f"  {trigger}R trigger: {saved}/{len(max_rs)} saved ({pct:.0f}%) | "
                  f"Net: {current_net:+.0f}R → {new_net:+.0f}R ({improvement:+.0f}R)")

        print(f"\n📐 MAX R BY CASE (losses only):")
        for case, label in case_names.items():
            case_rs = [t['max_r'] for t in losses_with_r if t['case'] == case]
            if case_rs:
                avg_r = sum(case_rs) / len(case_rs)
                above_1 = sum(1 for r in case_rs if r >= 1.0)
                print(f"  {label}: avg {avg_r:.1f}R | ≥1.0R: {above_1}/{len(case_rs)} ({above_1/len(case_rs)*100:.0f}%)")

    # ═════════════════════════════════════════════════════════════
    # TRIGGERS BY SYMBOL
    # ═════════════════════════════════════════════════════════════
    if symbol_triggers:
        print(f"\n📊 TRIGGERS BY SYMBOL:")
        for sym, cases in sorted(symbol_triggers.items()):
            parts = [f"{c}={n}" for c, n in sorted(cases.items())]
            print(f"  {sym}: {', '.join(parts)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 goat_09_inspect_log.py <logfile>")
        sys.exit(1)
    inspect_log(sys.argv[1])