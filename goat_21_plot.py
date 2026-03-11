"""
GOATv2 — Plot equity curve, drawdown, monthly P&L from backtest CSVs.
Reads the CSV files exported by goat_20_vbt_equity.py.
Supports partial TP results (P+TP, P+BE, P+SL).

Usage:
  python3 goat_21_plot.py --trades goat_eq_BTC_USDT_USDT_5m_be20_trades.csv --equity goat_eq_BTC_USDT_USDT_5m_be20_equity.csv
  python3 goat_21_plot.py --trades goat_eq_SOL_USDT_5m_be20_c1c2c3_p15_trades.csv --equity goat_eq_SOL_USDT_5m_be20_c1c2c3_p15_equity.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import argparse
import os


# ─── Result classification helpers ───────────────────────────────

ALL_CLOSED_RESULTS = ['TP', 'SL', 'BE', 'P+TP', 'P+BE', 'P+SL']
WIN_RESULTS  = ['TP', 'P+TP']               # full target hit
LOSS_RESULTS = ['SL']                        # full loss
BE_RESULTS   = ['BE']                        # flat zero
PARTIAL_POSITIVE = ['P+BE', 'P+SL']         # partial filled, net usually positive

_DAYS_PER_YEAR = 365.25   # used for duration-in-years calculations


def is_closed(result):
    return result in ALL_CLOSED_RESULTS


def classify_pnl(pnl_r):
    """Classify by actual PnL: positive / negative / zero."""
    if pd.isna(pnl_r):
        return 'zero'
    if pnl_r > 0.001:
        return 'positive'
    elif pnl_r < -0.001:
        return 'negative'
    return 'zero'


def load_data(trades_file, equity_file):
    trades = pd.read_csv(trades_file)
    equity = pd.read_csv(equity_file)

    for col in ['trigger_time', 'signal_time', 'entry_time', 'exit_time']:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors='coerce')

    if 'time' in equity.columns:
        equity['time'] = pd.to_datetime(equity['time'], errors='coerce')

    return trades, equity


def calc_sortino(r_values):
    if len(r_values) == 0:
        return 0.0
    mean_r = np.mean(r_values)
    downside = np.minimum(r_values, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return 999.0
    return mean_r / downside_std


def calc_sharpe_r(r_values, trades_per_year=None):
    """Annualized Sharpe on flat-R series. If trades_per_year is None, returns un-annualized."""
    if len(r_values) < 2:
        return 0.0
    mean_r = np.mean(r_values)
    std_r = np.std(r_values)
    if std_r == 0:
        return 0.0
    ratio = mean_r / std_r
    if trades_per_year is not None and trades_per_year > 0:
        return ratio * np.sqrt(trades_per_year)
    return ratio


def calc_sortino_r(r_values, trades_per_year=None):
    """Annualized Sortino on flat-R series."""
    if len(r_values) < 2:
        return 0.0
    mean_r = np.mean(r_values)
    neg_r = r_values[r_values < 0]
    if len(neg_r) < 2:
        return 999.0
    downside_std = np.std(neg_r)
    if downside_std == 0:
        return 999.0
    ratio = mean_r / downside_std
    if trades_per_year is not None and trades_per_year > 0:
        return ratio * np.sqrt(trades_per_year)
    return ratio


def _sharpe_grade(v):
    if v < 1:
        return "[POOR]"
    if v < 2:
        return "[OK]"
    if v < 3:
        return "[GOOD]"
    return "[EXCELLENT]"


def _sortino_grade(v):
    if v < 1.5:
        return "[POOR]"
    if v < 3:
        return "[OK]"
    if v < 5:
        return "[GOOD]"
    return "[EXCELLENT]"


def calc_monthly_wr(trades):
    closed = trades[trades['result'].isin(ALL_CLOSED_RESULTS)].copy()
    if len(closed) == 0:
        return 0.0, 0, 0

    time_col = 'exit_time' if 'exit_time' in closed.columns else 'entry_time'
    closed = closed.dropna(subset=[time_col]).copy()
    closed['month'] = closed[time_col].dt.to_period('M')
    monthly = closed.groupby('month')['pnl_r'].sum()

    green = (monthly > 0).sum()
    total = len(monthly)
    pct = green / total * 100 if total > 0 else 0
    return pct, green, total


def calc_recovery_factor(net_r, max_dd_r):
    if max_dd_r == 0:
        return 999.0
    return net_r / max_dd_r


def plot_all(trades, equity, title="GOATv2 Backtest", save=False, out_dir=".", trades_file="",
             bh_metrics=None):
    fig = plt.figure(figsize=(20, 28), facecolor='#1a1a2e')
    fig.suptitle(title, fontsize=18, fontweight='bold', color='white', y=0.98)

    gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.07, right=0.95, top=0.95, bottom=0.03)

    dark_bg = '#1a1a2e'
    panel_bg = '#16213e'
    grid_color = '#2a2a4a'
    text_color = '#e0e0e0'
    green = '#00e676'
    red = '#ff5252'
    orange = '#ffab40'
    cyan = '#18ffff'
    gold = '#ffd740'
    purple = '#bb86fc'

    # ─────────────────────────────────────────────────────────
    # 1. EQUITY CURVE (top, full width)
    # ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(panel_bg)

    eq_valid = equity.dropna(subset=['time', 'balance'])
    if len(eq_valid) > 1:
        ax1.plot(eq_valid['time'], eq_valid['balance'],
                 color=cyan, linewidth=1.5, alpha=0.9)
        ax1.fill_between(eq_valid['time'], eq_valid['balance'].min() * 0.95,
                         eq_valid['balance'], alpha=0.1, color=cyan)

        # Classify equity points by pnl_r
        if 'pnl_r' in eq_valid.columns:
            pos_eq = eq_valid[eq_valid['pnl_r'] > 0.001]
            neg_eq = eq_valid[eq_valid['pnl_r'] < -0.001]
            zero_eq = eq_valid[(eq_valid['pnl_r'].abs() <= 0.001) & (eq_valid['result'] != 'START')]
        else:
            # Fallback to result strings
            pos_eq = eq_valid[eq_valid['result'].isin(WIN_RESULTS + PARTIAL_POSITIVE)]
            neg_eq = eq_valid[eq_valid['result'].isin(LOSS_RESULTS)]
            zero_eq = eq_valid[eq_valid['result'].isin(BE_RESULTS)]

        if len(pos_eq) > 0:
            ax1.scatter(pos_eq['time'], pos_eq['balance'], color=green,
                       s=15, zorder=5, alpha=0.7, label=f'Win ({len(pos_eq)})')
        if len(neg_eq) > 0:
            ax1.scatter(neg_eq['time'], neg_eq['balance'], color=red,
                       s=15, zorder=5, alpha=0.7, label=f'Loss ({len(neg_eq)})')
        if len(zero_eq) > 0:
            ax1.scatter(zero_eq['time'], zero_eq['balance'], color=orange,
                       s=10, zorder=5, alpha=0.5, label=f'BE ({len(zero_eq)})')

        start_bal = eq_valid['balance'].iloc[0]
        end_bal = eq_valid['balance'].iloc[-1]
        ret_pct = (end_bal - start_bal) / start_bal * 100
        ax1.axhline(y=start_bal, color='white', linestyle='--', alpha=0.3, linewidth=0.8)

        # ── Buy & Hold overlay ──
        if 'bh_balance' in eq_valid.columns:
            bh_valid = eq_valid.dropna(subset=['bh_balance'])
            if len(bh_valid) > 1:
                ax1.plot(bh_valid['time'], bh_valid['bh_balance'],
                         color=orange, linewidth=1.2, alpha=0.85,
                         linestyle='--', label='Buy & Hold')

        ax1.set_title(f'Equity Curve — ${start_bal:,.0f}→${end_bal:,.0f} ({ret_pct:+,.1f}%)',
                      color=text_color, fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9, facecolor=panel_bg,
                   edgecolor=grid_color, labelcolor=text_color)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.tick_params(colors=text_color, labelsize=9)
    ax1.grid(True, alpha=0.2, color=grid_color)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color(grid_color)
    ax1.spines['left'].set_color(grid_color)

    # ─────────────────────────────────────────────────────────
    # 2. DRAWDOWN % (compounded)
    # ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(panel_bg)

    if 'dd_pct' in eq_valid.columns and len(eq_valid) > 1:
        dd = eq_valid['dd_pct'].fillna(0)
        ax2.fill_between(eq_valid['time'], 0, -dd, color=red, alpha=0.4)
        ax2.plot(eq_valid['time'], -dd, color=red, linewidth=0.8, alpha=0.8)

        max_dd = dd.max()
        max_dd_idx = dd.idxmax()
        if max_dd_idx in eq_valid.index:
            max_dd_time = eq_valid.loc[max_dd_idx, 'time']
            ax2.annotate(f'Max DD: -{max_dd:.1f}%',
                        xy=(max_dd_time, -max_dd),
                        xytext=(max_dd_time, -max_dd * 0.5),
                        fontsize=9, color=red, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=red, lw=1.5))

        ax2.set_title(f'Drawdown (compounded) — Max: {max_dd:.1f}%',
                      color=text_color, fontsize=13, fontweight='bold')
    else:
        ax2.set_title('Drawdown', color=text_color, fontsize=13)

    ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    ax2.tick_params(colors=text_color, labelsize=9)
    ax2.grid(True, alpha=0.2, color=grid_color)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color(grid_color)
    ax2.spines['left'].set_color(grid_color)

    # ─────────────────────────��───────────────────────────────
    # 3. CUMULATIVE R (flat)
    # ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(panel_bg)

    closed = trades[trades['result'].isin(ALL_CLOSED_RESULTS)].copy()
    if len(closed) > 0:
        time_col = 'exit_time' if 'exit_time' in closed.columns else 'entry_time'
        closed = closed.dropna(subset=[time_col])
        closed = closed.sort_values(time_col)
        closed['cum_r'] = closed['pnl_r'].cumsum()

        ax3.plot(closed[time_col], closed['cum_r'],
                 color=cyan, linewidth=1.5)
        ax3.fill_between(closed[time_col], 0, closed['cum_r'],
                         where=closed['cum_r'] >= 0, color=green, alpha=0.15)
        ax3.fill_between(closed[time_col], 0, closed['cum_r'],
                         where=closed['cum_r'] < 0, color=red, alpha=0.15)
        ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)

        final_r = closed['cum_r'].iloc[-1]
        ax3.set_title(f'Cumulative R (flat) — Net: {final_r:+.1f}R',
                      color=text_color, fontsize=12, fontweight='bold')
    else:
        ax3.set_title('Cumulative R', color=text_color, fontsize=12)

    ax3.set_ylabel('R', color=text_color, fontsize=10)
    ax3.tick_params(colors=text_color, labelsize=8)
    ax3.grid(True, alpha=0.2, color=grid_color)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color(grid_color)
    ax3.spines['left'].set_color(grid_color)

    # ─────────────────────────────────────────────────────────
    # 4. TRADE R DISTRIBUTION
    # ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(panel_bg)

    if len(closed) > 0:
        # Round to 2 decimals for grouping
        closed_dist = closed.copy()
        closed_dist['pnl_r_round'] = closed_dist['pnl_r'].round(2)
        unique_r = sorted(closed_dist['pnl_r_round'].unique())
        counts = [len(closed_dist[closed_dist['pnl_r_round'] == r]) for r in unique_r]
        bar_colors = [green if r > 0.001 else red if r < -0.001 else orange for r in unique_r]

        # Auto bar width
        if len(unique_r) > 1:
            min_gap = min(abs(unique_r[i+1] - unique_r[i]) for i in range(len(unique_r)-1))
            bar_width = max(0.02, min(0.3, min_gap * 0.8))
        else:
            bar_width = 0.3

        ax4.bar(unique_r, counts, width=bar_width, color=bar_colors, alpha=0.8,
                edgecolor='white', linewidth=0.3)

        for r, c in zip(unique_r, counts):
            ax4.text(r, c + 0.5, str(c), ha='center', va='bottom',
                    color=text_color, fontsize=10, fontweight='bold')

        ax4.set_title(f'Trade Distribution — {len(closed)} trades',
                      color=text_color, fontsize=12, fontweight='bold')
        ax4.set_xlabel('P&L (R)', color=text_color, fontsize=10)
        ax4.set_ylabel('Count', color=text_color, fontsize=10)

    ax4.tick_params(colors=text_color, labelsize=9)
    ax4.grid(True, alpha=0.2, color=grid_color, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_color(grid_color)
    ax4.spines['left'].set_color(grid_color)

    # ─────────────────────────────────────────────────────────
    # 5. MONTHLY P&L BAR CHART (flat R)
    # ─────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(panel_bg)

    if len(closed) > 0:
        time_col = 'exit_time' if 'exit_time' in closed.columns else 'entry_time'
        closed_m = closed.dropna(subset=[time_col]).copy()
        closed_m['month'] = closed_m[time_col].dt.to_period('M')
        monthly = closed_m.groupby('month')['pnl_r'].sum()

        months = [str(m) for m in monthly.index]
        values = monthly.values
        bar_colors_m = [green if v >= 0 else red for v in values]

        ax5.bar(range(len(months)), values, color=bar_colors_m, alpha=0.8,
                edgecolor='white', linewidth=0.3)

        for i, (m, v) in enumerate(zip(months, values)):
            ax5.text(i, v + (0.5 if v >= 0 else -1.5),
                    f'{v:+.0f}R', ha='center', va='bottom' if v >= 0 else 'top',
                    color=text_color, fontsize=8, fontweight='bold')

        ax5.set_xticks(range(len(months)))
        ax5.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
        ax5.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)

        total_r = values.sum()
        win_months = sum(1 for v in values if v > 0)
        ax5.set_title(f'Monthly P&L (flat R) — {win_months}/{len(months)} green months | Total: {total_r:+.0f}R',
                      color=text_color, fontsize=12, fontweight='bold')
        ax5.set_ylabel('R', color=text_color, fontsize=10)

    ax5.tick_params(colors=text_color, labelsize=8)
    ax5.grid(True, alpha=0.2, color=grid_color, axis='y')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_color(grid_color)
    ax5.spines['left'].set_color(grid_color)

    # ─────────────────────────────────────────────────────────
    # 6. ROLLING R/TRADE (edge stability)
    # ─────────────────────────────────────────────────────────
    ax_roll = fig.add_subplot(gs[4, :])
    ax_roll.set_facecolor(panel_bg)

    if len(closed) > 0:
        time_col = 'exit_time' if 'exit_time' in closed.columns else 'entry_time'
        closed_roll = closed.dropna(subset=[time_col]).sort_values(time_col).copy()

        window = max(10, min(50, len(closed_roll) // 4))

        if len(closed_roll) >= window + 5:
            rolling_r = closed_roll['pnl_r'].rolling(window=window).mean()
            ax_roll.plot(closed_roll[time_col], rolling_r,
                        color=gold, linewidth=1.5, alpha=0.9)
            ax_roll.fill_between(closed_roll[time_col], 0, rolling_r,
                                where=rolling_r >= 0, color=green, alpha=0.15)
            ax_roll.fill_between(closed_roll[time_col], 0, rolling_r,
                                where=rolling_r < 0, color=red, alpha=0.15)
            ax_roll.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)

            avg_r = closed_roll['pnl_r'].mean()
            ax_roll.axhline(y=avg_r, color=cyan, linewidth=1, alpha=0.5, linestyle='--')
            ax_roll.text(closed_roll[time_col].iloc[-1], avg_r,
                        f'  avg {avg_r:+.2f}R', color=cyan, fontsize=9, va='bottom')

            ax_roll.set_title(f'Rolling {window}-Trade R/trade (edge stability)',
                             color=text_color, fontsize=12, fontweight='bold')
        else:
            ax_roll.text(0.5, 0.5, f'Need more trades for rolling chart\n({len(closed_roll)} trades, need {window + 5}+)',
                        transform=ax_roll.transAxes, ha='center', va='center',
                        color=text_color, fontsize=12, alpha=0.5)
            ax_roll.set_title(f'Rolling R/trade (need ≥{window + 5} trades, have {len(closed_roll)})',
                             color=text_color, fontsize=12, fontweight='bold')
    else:
        ax_roll.set_title('Rolling R/trade', color=text_color, fontsize=12)

    ax_roll.set_ylabel('R/trade', color=text_color, fontsize=10)
    ax_roll.tick_params(colors=text_color, labelsize=8)
    ax_roll.grid(True, alpha=0.2, color=grid_color)
    ax_roll.spines['top'].set_visible(False)
    ax_roll.spines['right'].set_visible(False)
    ax_roll.spines['bottom'].set_color(grid_color)
    ax_roll.spines['left'].set_color(grid_color)

    # ─────────────────────────────────────────────────────────
    # 7. STATS: By Case & Side (bottom left)
    # ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[5, 0])
    ax6.set_facecolor(panel_bg)
    ax6.axis('off')

    if len(closed) > 0:
        cases = closed.groupby('case').agg(
            trades=('pnl_r', 'count'),
            wins=('pnl_r', lambda x: (x > 0.001).sum()),
            losses=('pnl_r', lambda x: (x < -0.001).sum()),
            bes=('pnl_r', lambda x: (x.abs() <= 0.001).sum()),
            net_r=('pnl_r', 'sum')
        ).reset_index()

        table_text = "CASE     TRADES   W    L   BE    WR%    NET R\n"
        table_text += "─" * 52 + "\n"
        for _, row in cases.iterrows():
            wr = row['wins'] / row['trades'] * 100 if row['trades'] > 0 else 0
            table_text += f"{row['case']:<9}{row['trades']:>5}  {int(row['wins']):>4} {int(row['losses']):>4} {int(row['bes']):>4}  {wr:>5.1f}%  {row['net_r']:>+6.0f}R\n"

        sides = closed.groupby('side').agg(
            trades=('pnl_r', 'count'),
            wins=('pnl_r', lambda x: (x > 0.001).sum()),
            losses=('pnl_r', lambda x: (x < -0.001).sum()),
            bes=('pnl_r', lambda x: (x.abs() <= 0.001).sum()),
            net_r=('pnl_r', 'sum')
        ).reset_index()

        table_text += "\nSIDE     TRADES   W    L   BE    WR%    NET R\n"
        table_text += "─" * 52 + "\n"
        for _, row in sides.iterrows():
            wr = row['wins'] / row['trades'] * 100 if row['trades'] > 0 else 0
            table_text += f"{row['side']:<9}{row['trades']:>5}  {int(row['wins']):>4} {int(row['losses']):>4} {int(row['bes']):>4}  {wr:>5.1f}%  {row['net_r']:>+6.0f}R\n"

        # ── Result type breakdown (if partial TP is in use) ──
        result_types = closed['result'].unique()
        has_partial = any(r.startswith('P+') for r in result_types)
        if has_partial:
            table_text += "\nRESULT   COUNT   AVG R   TOTAL R\n"
            table_text += "─" * 40 + "\n"
            for rt in sorted(result_types):
                group = closed[closed['result'] == rt]
                avg_r = group['pnl_r'].mean()
                tot_r = group['pnl_r'].sum()
                table_text += f"{rt:<9}{len(group):>5}  {avg_r:>+6.2f}R  {tot_r:>+6.1f}R\n"

        ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes,
                fontfamily='monospace', fontsize=10, color=text_color,
                verticalalignment='top')
        ax6.set_title('By Case & Side', color=text_color, fontsize=12, fontweight='bold')

    # ─────────────��───────────────────────────────────────────
    # 8. KEY METRICS (bottom right)
    # ─────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[5, 1])
    ax7.set_facecolor(panel_bg)
    ax7.axis('off')

    if len(closed) > 0:
        total = len(closed)
        # Classify by pnl_r, not result string
        wins_count = (closed['pnl_r'] > 0.001).sum()
        losses_count = (closed['pnl_r'] < -0.001).sum()
        bes_count = (closed['pnl_r'].abs() <= 0.001).sum()
        wr = wins_count / total * 100
        net_r = closed['pnl_r'].sum()
        rpt = net_r / total

        # Profit factor from actual PnL values
        gross_win = closed.loc[closed['pnl_r'] > 0, 'pnl_r'].sum()
        gross_loss = abs(closed.loc[closed['pnl_r'] < 0, 'pnl_r'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        # Max DD in R
        cum = closed['pnl_r'].cumsum()
        peak_r = cum.cummax()
        dd_r = (peak_r - cum).max()

        # Return/DD
        ret_dd = net_r / dd_r if dd_r > 0 else 999

        # Sortino
        r_values = closed['pnl_r'].values

        # Compute trades per year for annualization
        time_col = 'exit_time' if 'exit_time' in closed.columns else 'entry_time'
        _td = closed.dropna(subset=[time_col])
        _trades_per_year = None
        if len(_td) >= 2:
            _t0 = _td[time_col].min()
            _t1 = _td[time_col].max()
            _num_years = max((_t1 - _t0).days / _DAYS_PER_YEAR, 1 / _DAYS_PER_YEAR)
            _trades_per_year = len(_td) / _num_years

        sortino = calc_sortino(r_values)
        sharpe = calc_sharpe_r(r_values, _trades_per_year)
        sortino_ann = calc_sortino_r(r_values, _trades_per_year)

        # Monthly win rate
        monthly_wr_pct, green_months, total_months = calc_monthly_wr(trades)

        # Recovery factor
        recovery = calc_recovery_factor(net_r, dd_r)

        # R per month
        r_per_month = net_r / total_months if total_months > 0 else 0

        # Consecutive streaks (by pnl_r sign)
        max_cw = 0
        max_cl = 0
        cw = 0
        cl = 0
        for pnl in closed['pnl_r'].values:
            if pnl > 0.001:
                cw += 1
                cl = 0
                max_cw = max(max_cw, cw)
            elif pnl < -0.001:
                cl += 1
                cw = 0
                max_cl = max(max_cl, cl)
            else:
                cw = 0
                cl = 0

        # Duration
        if 'duration_bars' in closed.columns:
            avg_dur = closed['duration_bars'].mean()
            med_dur = closed['duration_bars'].median()
        else:
            avg_dur = 0
            med_dur = 0

        # Grades
        pf_grade = "[GOOD]" if pf >= 1.5 else "[OK]" if pf >= 1.2 else "[WEAK]"
        rpt_grade = "[GOOD]" if rpt >= 0.3 else "[OK]" if rpt >= 0.1 else "[WEAK]"
        ret_dd_grade = "[GOOD]" if ret_dd >= 5 else "[OK]" if ret_dd >= 2 else "[WEAK]"
        monthly_grade = "[GOOD]" if monthly_wr_pct >= 70 else "[OK]" if monthly_wr_pct >= 50 else "[WEAK]"
        recovery_grade = "[GOOD]" if recovery >= 10 else "[OK]" if recovery >= 5 else "[WEAK]"
        rpm_grade = "[GOOD]" if r_per_month >= 10 else "[OK]" if r_per_month >= 5 else "[WEAK]"
        sharpe_grade = _sharpe_grade(sharpe)
        sortino_grade = _sortino_grade(sortino_ann)

        # Check if partial TP is in use
        has_partial = any(r.startswith('P+') for r in closed['result'].unique())
        partial_info = ""
        if has_partial:
            p_count = closed['result'].str.startswith('P+').sum()
            partial_info = f"  Partial fills:     {p_count}/{total}\n"

        metrics = (
            f"  KEY METRICS\n"
            f"  {'─' * 40}\n"
            f"  Trades:            {total}\n"
            f"  Wins / Losses:     {wins_count} / {losses_count} / {bes_count} BE\n"
            f"  Win Rate:          {wr:.1f}%\n"
            f"  Net R:             {net_r:+.1f}R\n"
            f"{partial_info}"
            f"  {'─' * 40}\n"
            f"  R/trade:           {rpt:+.3f}R  {rpt_grade}\n"
            f"  Profit Factor:     {pf:.2f}  {pf_grade}\n"
            f"  Max DD:            {dd_r:.1f}R\n"
            f"  Return/DD:         {ret_dd:.2f}x  {ret_dd_grade}\n"
            f"  {'─' * 40}\n"
            f"  Sharpe (R, ann):   {sharpe:.2f}  {sharpe_grade}\n"
            f"  Sortino (R, ann):  {sortino_ann:.2f}  {sortino_grade}\n"
            f"  {'─' * 40}\n"
            f"  R/month:           {r_per_month:+.1f}R  {rpm_grade}\n"
            f"  Monthly WR:        {monthly_wr_pct:.0f}% ({green_months}/{total_months})  {monthly_grade}\n"
            f"  Recovery Factor:   {recovery:.1f}x  {recovery_grade}\n"
            f"  {'─' * 40}\n"
            f"  Max Win Streak:    {max_cw}\n"
            f"  Max Loss Streak:   {max_cl}\n"
            f"  Avg Duration:      {avg_dur:.0f} bars\n"
            f"  Med Duration:      {med_dur:.0f} bars\n"
        )

        # ── Buy & Hold section ──
        if bh_metrics and bh_metrics.get("return_pct") is not None:
            bh_ret = bh_metrics["return_pct"]
            bh_dd = bh_metrics.get("max_dd_pct") or 0
            bh_sh = bh_metrics.get("sharpe")
            bh_so = bh_metrics.get("sortino")

            # Equity-based strategy return (compounded)
            eq_bals = equity.dropna(subset=['balance'])['balance']
            if len(eq_bals) >= 2:
                strat_ret = (eq_bals.iloc[-1] / eq_bals.iloc[0] - 1) * 100
            else:
                strat_ret = 0.0

            if abs(bh_ret) > 0.01:
                if bh_ret > 0:
                    outperf_str = f"{strat_ret / bh_ret:.1f}×"
                else:
                    diff = strat_ret - bh_ret
                    outperf_str = f"{diff:+.1f}pp"
            else:
                outperf_str = "∞"

            bh_sh_str = f"{bh_sh:.2f}" if bh_sh is not None else "—"
            bh_so_str = f"{min(bh_so, 999):.2f}" if bh_so is not None else "—"

            metrics += (
                f"  {'─' * 40}\n"
                f"  B&H Return:        {bh_ret:+.1f}%\n"
                f"  B&H Max DD:        {bh_dd:.1f}%\n"
                f"  B&H Sharpe:        {bh_sh_str}\n"
                f"  B&H Sortino:       {bh_so_str}\n"
                f"  Strategy Return:   {strat_ret:+.1f}%\n"
                f"  Outperformance:    {outperf_str}\n"
            )

        ax7.text(0.05, 0.95, metrics, transform=ax7.transAxes,
                fontfamily='monospace', fontsize=9.5, color=text_color,
                verticalalignment='top')
        ax7.set_title('Key Metrics', color=text_color, fontsize=12, fontweight='bold')

    # ─────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────
    if trades_file:
        base = os.path.splitext(os.path.basename(trades_file))[0]
        tag = base.replace("goat_eq_", "").replace("_trades", "")
        out_file = os.path.join(out_dir, f"goat_chart_{tag}.png")
    else:
        out_file = os.path.join(out_dir, "goat_chart.png")

    plt.savefig(out_file, dpi=150, facecolor=dark_bg, bbox_inches='tight')
    print(f"  💾 Chart saved → {out_file}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOATv2 — Plot backtest results")
    parser.add_argument("--trades", required=True, help="Trades CSV file")
    parser.add_argument("--equity", required=True, help="Equity CSV file")
    parser.add_argument("--title", default=None, help="Chart title")
    parser.add_argument("--save", action="store_true", help="Save with auto-name")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    if args.title is None:
        base = os.path.splitext(os.path.basename(args.trades))[0]
        tag = base.replace("goat_eq_", "").replace("_trades", "")
        args.title = f"GOATv2 — {tag}"

    print(f"📊 Loading data...")
    trades, equity = load_data(args.trades, args.equity)
    print(f"  {len(trades)} trades, {len(equity)} equity points")

    print(f"📈 Plotting...")
    plot_all(trades, equity, title=args.title, save=args.save, out_dir=args.out_dir, trades_file=args.trades)

    print(f"✅ Done!")