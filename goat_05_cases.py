"""
05 — GOAT Case 1, 2, 3 logic + unified check_goat().
All sweep logic lives here. Works for both bull and bear via direction config.

Rules for all cases:
  1. Line must be valid at trigger time (no close through it from source to trigger)
  2. Wick reaches the line (sweep) — but close must NOT break through (wick-only)
  3. Only the FIRST bar to sweep a level is valid (once swept, level is consumed)
  4. Projection (sweep bar's HA_Low for bear / HA_High for bull) inside trigger body
  5. No body intersection between sweep bar and trigger bar at the sweep level
     (projection can ONLY touch the trigger bar's body, no other)
  Lines are validated independently.
"""

import logging
import pandas as pd

from goat_00_config import (
    get_direction_config, compare, sweep_reaches, line_invalidated
)
from goat_01_utilis import fmt, body_intersects_level
from goat_04_pivots import find_ha_pivot_lows, find_ha_pivot_highs


# ─── LGCR Selectors (single — backward compat) ──────────────────

def _select_prior_lgcr_bear(df, cur):
    """Find the best (closest) prior bearish LGCR bar."""
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bearish_LGCR']:
            if df.loc[i, 'HA_Close'] > trigger_close or df.loc[i, 'HA_High'] > trigger_close:
                prior.append(i)
    if not prior:
        return None
    return min(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_High'] - trigger_close)
    ))


def _select_prior_lgcr_bull(df, cur):
    """Find the best (closest) prior bullish LGCR bar."""
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bullish_LGCR']:
            if df.loc[i, 'HA_Close'] < trigger_close or df.loc[i, 'HA_Low'] < trigger_close:
                prior.append(i)
    if not prior:
        return None
    return min(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_Low'] - trigger_close)
    ))


# ─── LGCR Selectors (multi — try all candidates) ────────────────

def _select_prior_lgcrs_bear(df, cur):
    """Find ALL prior bearish LGCR bars, sorted by closeness to trigger."""
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bearish_LGCR']:
            if df.loc[i, 'HA_Close'] > trigger_close or df.loc[i, 'HA_High'] > trigger_close:
                prior.append(i)
    if not prior:
        return []
    return sorted(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_High'] - trigger_close)
    ))


def _select_prior_lgcrs_bull(df, cur):
    """Find ALL prior bullish LGCR bars, sorted by closeness to trigger."""
    trigger_close = df.loc[cur, 'HA_Close']
    prior = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, 'bullish_LGCR']:
            if df.loc[i, 'HA_Close'] < trigger_close or df.loc[i, 'HA_Low'] < trigger_close:
                prior.append(i)
    if not prior:
        return []
    return sorted(prior, key=lambda i: min(
        abs(df.loc[i, 'HA_Close'] - trigger_close),
        abs(df.loc[i, 'HA_Low'] - trigger_close)
    ))


# ─── Case 1: LGCR Sweep ─────────────────────────────────────────

def _check_case1_lgcr_sweep(df, cur, symbol, dcfg):
    """
    Case 1: Prior LGCR sweep.
    Tries ALL prior LGCRs (closest first). For each, validates each line
    independently from LGCR bar to TRIGGER bar. If both lines are broken
    at trigger time, skip that LGCR entirely.

    Only the FIRST bar to sweep each line is valid (first-sweep rule).

    Checks per sweep candidate:
      1. Wick reaches a line that is still valid at trigger time
      2. Close must NOT break through the swept line (wick-only sweep)
      3. Line valid before sweep (no close through it from LGCR to sweep)
      4. Projection inside trigger body
      5. No body intersection between sweep and trigger at sweep level (forward only)
    """
    side = dcfg["side"]
    all_priors = dcfg["lgcr_selector_multi"](df, cur)

    if not all_priors:
        logging.info(
            f"[DIAG_{side}_C1] {symbol} bar {cur}: no prior {dcfg['lgcr_col']} found — "
            f"Case 1 skipped"
        )
        return False, None, None, None

    logging.info(
        f"[DIAG_{side}_C1] {symbol} bar {cur}: found {len(all_priors)} prior LGCR candidates "
        f"(bars: {all_priors[:5]}{'...' if len(all_priors) > 5 else ''})"
    )

    for prior_idx in all_priors:
        line1 = df.loc[prior_idx, dcfg["lgcr_line1_col"]]  # HA_Close
        line2 = df.loc[prior_idx, dcfg["lgcr_line2_col"]]  # HA_High (bear) or HA_Low (bull)

        # ── Validate each line independently: LGCR bar → trigger bar ──
        line1_valid_at_trigger = True
        line2_valid_at_trigger = True
        for j in range(prior_idx + 1, cur):
            c = df.loc[j, 'HA_Close']
            if line1_valid_at_trigger and compare(c, line1, dcfg["invalidate_op"]):
                line1_valid_at_trigger = False
            if line2_valid_at_trigger and compare(c, line2, dcfg["invalidate_op"]):
                line2_valid_at_trigger = False
            if not line1_valid_at_trigger and not line2_valid_at_trigger:
                break

        # If BOTH lines are broken at trigger time, skip this LGCR entirely
        if not line1_valid_at_trigger and not line2_valid_at_trigger:
            logging.info(
                f"[DIAG_{side}_C1] {symbol} bar {cur}: prior LGCR at bar {prior_idx} "
                f"({df.loc[prior_idx,'timestamp']}): BOTH lines invalidated before trigger — skipped"
            )
            continue

        logging.info(
            f"[DIAG_{side}_C1] {symbol} bar {cur}: trying prior LGCR at bar {prior_idx} "
            f"({df.loc[prior_idx,'timestamp']}), "
            f"lines=[{dcfg['lgcr_line1_col']}={fmt(line1)} ({'valid' if line1_valid_at_trigger else 'BROKEN'}), "
            f"{dcfg['lgcr_line2_col']}={fmt(line2)} ({'valid' if line2_valid_at_trigger else 'BROKEN'})], "
            f"bars_between={cur - prior_idx - 1}"
        )

        # ── Track which lines have already been swept (first-sweep rule) ──
        line1_already_swept = False
        line2_already_swept = False

        for k in range(prior_idx + 1, cur):
            # ── 1. Per-line validity BEFORE sweep bar ──
            line1_valid_at_k = line1_valid_at_trigger
            line2_valid_at_k = line2_valid_at_trigger
            if line1_valid_at_trigger:
                for j in range(prior_idx + 1, k):
                    if compare(df.loc[j, 'HA_Close'], line1, dcfg["invalidate_op"]):
                        line1_valid_at_k = False
                        break
            if line2_valid_at_trigger:
                for j in range(prior_idx + 1, k):
                    if compare(df.loc[j, 'HA_Close'], line2, dcfg["invalidate_op"]):
                        line2_valid_at_k = False
                        break

            # ── 2. Wick reaches a line that is valid at BOTH sweep time AND trigger time ──
            wick = df.loc[k, dcfg["sweep_col"]]
            swept_line2 = (sweep_reaches(wick, line2, dcfg["tolerance_factor"])
                           and line2_valid_at_k and line2_valid_at_trigger
                           and not line2_already_swept)
            swept_line1 = (sweep_reaches(wick, line1, dcfg["tolerance_factor"])
                           and line1_valid_at_k and line1_valid_at_trigger
                           and not line1_already_swept)
            if not (swept_line1 or swept_line2):
                continue

            # ── Mark lines as swept (first-sweep rule) ──
            # Once a line is touched, it's consumed regardless of outcome
            if swept_line2:
                line2_already_swept = True
            if swept_line1:
                line1_already_swept = True

            # ── 3. Close must NOT break through the swept line (wick-only) ──
            sweep_close = df.loc[k, 'HA_Close']
            swept_ref = line2 if swept_line2 else line1
            if side == "BEAR" and sweep_close > swept_ref:
                logging.info(
                    f"[DIAG_{side}_C1_PROJ] {symbol} bar {cur}: sweep_bar={k}, "
                    f"close={fmt(sweep_close)} > swept_ref={fmt(swept_ref)} — "
                    f"CLOSED THROUGH (not a sweep), skipped"
                )
                continue
            if side == "BULL" and sweep_close < swept_ref:
                logging.info(
                    f"[DIAG_{side}_C1_PROJ] {symbol} bar {cur}: sweep_bar={k}, "
                    f"close={fmt(sweep_close)} < swept_ref={fmt(swept_ref)} — "
                    f"CLOSED THROUGH (not a sweep), skipped"
                )
                continue

            sweep_level = df.loc[k, dcfg["sweep_level_col"]]

            # ── 4. Projection inside trigger body ──
            body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
            if not (body_low <= sweep_level <= body_high):
                logging.info(
                    f"[DIAG_{side}_C1_PROJ] {symbol} bar {cur}: sweep_bar={k}, "
                    f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                    f"proj_inside=False"
                )
                continue

            # ── 5. No body intersection between sweep and trigger (forward only) ──
            fwd_body_fail = False
            fwd_block_bar = None
            for j in range(k + 1, cur):
                if body_intersects_level(df, j, sweep_level):
                    fwd_body_fail = True
                    fwd_block_bar = j
                    break
            if fwd_body_fail:
                logging.info(
                    f"[DIAG_{side}_C1_PROJ] {symbol} bar {cur}: sweep_bar={k}, "
                    f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                    f"proj_inside=True, fwd_body_clear=False (blocked at bar {fwd_block_bar})"
                )
                continue

            logging.info(
                f"[DIAG_{side}_C1_PROJ] {symbol} bar {cur}: sweep_bar={k}, "
                f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                f"proj_inside=True, fwd_body_clear=True ✅"
            )

            # ── Valid — first sweep wins ──
            swept_label = dcfg["lgcr_swept_label_1"] if swept_line2 else dcfg["lgcr_swept_label_2"]
            swept_value = line2 if swept_line2 else line1

            logging.info(
                f"[GOATv2_{side}_C1] Trigger {cur} ({df.loc[cur,'timestamp']}), "
                f"LGCR prior {prior_idx} ({df.loc[prior_idx,'timestamp']}), "
                f"sweep bar {k} ({df.loc[k,'timestamp']}), "
                f"lines=[{dcfg['lgcr_line1_col']}={fmt(line1)}, {dcfg['lgcr_line2_col']}={fmt(line2)}], "
                f"line1_valid={swept_line1}, line2_valid={swept_line2}, "
                f"{swept_label}={fmt(swept_value)}"
            )
            return True, "LGCR", swept_label, swept_value

        # If we exhausted all sweep bars for this LGCR, try the next LGCR
        logging.info(
            f"[DIAG_{side}_C1] {symbol} bar {cur}: LGCR at bar {prior_idx} — "
            f"no valid first-sweep found"
        )

    logging.info(f"[GOATv2_{side}_C1] {symbol}: Case 1 (LGCR sweep) did not trigger")
    return False, None, None, None


# ─── Case 2: LG Line Sweep ──────────────────────────────────────

def _check_case2_lg_line_sweep(df, cur, symbol, dcfg):
    """
    Case 2: LG line sweep.
    Line must be valid at trigger time (no close through it from LGC bar to trigger).
    Only the FIRST bar to sweep the line is valid (first-sweep rule).

    Checks per sweep candidate:
      1. Line valid at trigger time
      2. Line valid before sweep
      3. Wick reaches the line
      4. Close must NOT break through the line (wick-only sweep)
      5. Projection inside trigger body
      6. No body intersection between sweep and trigger at sweep level (forward only)
    """
    side = dcfg["side"]
    cur_price = df.loc[cur, 'HA_Close']

    # Find LGC bars with a valid line on the correct side of price
    candidates_lg = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, dcfg["lgc_col"]]:
            line_level = df.loc[i, dcfg["lgc_line_col"]]
            if pd.isna(line_level):
                continue
            if dcfg["line_above_price"] and line_level >= cur_price:
                candidates_lg.append((i, line_level))
            elif not dcfg["line_above_price"] and line_level <= cur_price:
                candidates_lg.append((i, line_level))

    if not candidates_lg:
        logging.info(
            f"[DIAG_{side}_C2] {symbol} bar {cur}: no {dcfg['lgc_col']} with valid line — "
            f"Case 2 skipped"
        )
        return False, None, None, None

    lgc_idx, line_level = min(candidates_lg, key=lambda x: abs(x[1] - cur_price))

    # ── Validate line from LGC bar to trigger bar ──
    line_valid_at_trigger = True
    for j in range(lgc_idx + 1, cur):
        if line_invalidated(df.loc[j, 'HA_Close'], line_level, dcfg):
            line_valid_at_trigger = False
            break

    if not line_valid_at_trigger:
        logging.info(
            f"[DIAG_{side}_C2] {symbol} bar {cur}: best LG line at bar {lgc_idx} "
            f"({df.loc[lgc_idx,'timestamp']}), line_level={fmt(line_level)} — "
            f"INVALIDATED before trigger, skipped"
        )
        return False, None, None, None

    logging.info(
        f"[DIAG_{side}_C2] {symbol} bar {cur}: best LG line candidate at bar {lgc_idx} "
        f"({df.loc[lgc_idx,'timestamp']}), line_level={fmt(line_level)}, "
        f"total_candidates={len(candidates_lg)}, line_valid_at_trigger=True"
    )

    for k in range(lgc_idx + 1, cur):
        wick = df.loc[k, dcfg["sweep_col"]]
        # Bear: wick must reach UP to line; Bull: wick must reach DOWN to line
        if dcfg["side"] == "BEAR" and wick < line_level:
            continue
        if dcfg["side"] == "BULL" and wick > line_level:
            continue

        # ══ FIRST SWEEP FOUND — this is the only valid sweep bar ══
        # From here, this bar either passes all checks or the level is consumed

        # ── 1. Close must NOT break through the line (wick-only) ──
        sweep_close = df.loc[k, 'HA_Close']
        if side == "BEAR" and sweep_close > line_level:
            logging.info(
                f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"close={fmt(sweep_close)} > line={fmt(line_level)} — "
                f"CLOSED THROUGH (not a sweep), level consumed"
            )
            break  # level consumed — no more sweeps
        if side == "BULL" and sweep_close < line_level:
            logging.info(
                f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"close={fmt(sweep_close)} < line={fmt(line_level)} — "
                f"CLOSED THROUGH (not a sweep), level consumed"
            )
            break  # level consumed — no more sweeps

        # ── 2. Line validity before sweep ──
        line_valid = True
        for j in range(lgc_idx + 1, k):
            if line_invalidated(df.loc[j, 'HA_Close'], line_level, dcfg):
                line_valid = False
                break
        if not line_valid:
            logging.info(
                f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"line invalidated before sweep, level consumed"
            )
            break  # level consumed

        sweep_level = df.loc[k, dcfg["sweep_level_col"]]

        # ── 3. Projection inside trigger body ──
        body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        if not (body_low <= sweep_level <= body_high):
            logging.info(
                f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                f"proj_inside=False, level consumed"
            )
            break  # first sweep doesn't fit — level consumed

        # ── 4. No body intersection between sweep and trigger (forward only) ──
        fwd_body_fail = False
        fwd_block_bar = None
        for j in range(k + 1, cur):
            if body_intersects_level(df, j, sweep_level):
                fwd_body_fail = True
                fwd_block_bar = j
                break
        if fwd_body_fail:
            logging.info(
                f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                f"proj_inside=True, fwd_body_clear=False (blocked at bar {fwd_block_bar}), level consumed"
            )
            break  # first sweep blocked — level consumed

        logging.info(
            f"[DIAG_{side}_C2_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
            f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
            f"proj_inside=True, fwd_body_clear=True ✅"
        )

        logging.info(
            f"[GOATv2_{side}_C2] Trigger {cur} ({df.loc[cur,'timestamp']}), "
            f"LG line from {lgc_idx} ({df.loc[lgc_idx,'timestamp']}), "
            f"sweep bar {k} ({df.loc[k,'timestamp']}), LG_line_swept={fmt(line_level)}"
        )
        return True, "LG_LINE", "LG_line_swept", line_level

    logging.info(f"[GOATv2_{side}_C2] {symbol}: Case 2 (LG line) did not trigger")
    return False, None, None, None


# ─── Case 3: Pivot Sweep ────────────────────────────────────────

def _check_case3_pivot_sweep(df, cur, symbol, dcfg):
    """
    Case 3: Pivot sweep.
    Only the FIRST bar to sweep the pivot is valid (first-sweep rule).

    Checks per sweep candidate:
      1. Pivot valid before sweep (no close through it)
      2. Wick reaches the pivot
      3. Close must NOT break through the pivot (wick-only sweep)
      4. Projection inside trigger body
      5. No body intersection between sweep and trigger at sweep level (forward only)
    """
    side = dcfg["side"]
    cur_price = df.loc[cur, 'HA_Close']

    # Find pivots on the correct side
    if dcfg["side"] == "BEAR":
        pivots = find_ha_pivot_highs(df, 0, cur)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl >= cur_price]
    else:
        pivots = find_ha_pivot_lows(df, 0, cur)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl <= cur_price]

    if not pivot_candidates:
        logging.info(
            f"[DIAG_{side}_C3] {symbol} bar {cur}: no pivot candidates, "
            f"total_pivots_found={len(pivots)} — Case 3 skipped"
        )
        return False, None, None, None

    pivot_idx, pivot_level = min(pivot_candidates, key=lambda x: abs(x[1] - cur_price))

    logging.info(
        f"[DIAG_{side}_C3] {symbol} bar {cur}: best pivot at bar {pivot_idx} "
        f"({df.loc[pivot_idx,'timestamp']}), pivot_level={fmt(pivot_level)}, "
        f"total_pivot_candidates={len(pivot_candidates)}"
    )

    for k in range(pivot_idx + 1, cur):
        wick = df.loc[k, dcfg["sweep_col"]]
        if not sweep_reaches(wick, pivot_level, dcfg["tolerance_factor"]):
            continue

        # ══ FIRST SWEEP FOUND — this is the only valid sweep bar ══

        # ── 1. Close must NOT break through the pivot (wick-only) ──
        sweep_close = df.loc[k, 'HA_Close']
        if side == "BEAR" and sweep_close > pivot_level:
            logging.info(
                f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"close={fmt(sweep_close)} > pivot={fmt(pivot_level)} — "
                f"CLOSED THROUGH (not a sweep), level consumed"
            )
            break  # level consumed
        if side == "BULL" and sweep_close < pivot_level:
            logging.info(
                f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"close={fmt(sweep_close)} < pivot={fmt(pivot_level)} — "
                f"CLOSED THROUGH (not a sweep), level consumed"
            )
            break  # level consumed

        # ── 2. Pivot validity before sweep ──
        pivot_valid = True
        for j in range(pivot_idx + 1, k):
            if line_invalidated(df.loc[j, 'HA_Close'], pivot_level, dcfg):
                pivot_valid = False
                break
        if not pivot_valid:
            logging.info(
                f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"pivot invalidated before sweep, level consumed"
            )
            break  # level consumed

        sweep_level = df.loc[k, dcfg["sweep_level_col"]]

        # ── 3. Projection inside trigger body ──
        body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
        if not (body_low <= sweep_level <= body_high):
            logging.info(
                f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                f"proj_inside=False, level consumed"
            )
            break  # first sweep doesn't fit — level consumed

        # ── 4. No body intersection between sweep and trigger (forward only) ──
        fwd_body_fail = False
        fwd_block_bar = None
        for j in range(k + 1, cur):
            if body_intersects_level(df, j, sweep_level):
                fwd_body_fail = True
                fwd_block_bar = j
                break
        if fwd_body_fail:
            logging.info(
                f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
                f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
                f"proj_inside=True, fwd_body_clear=False (blocked at bar {fwd_block_bar}), level consumed"
            )
            break  # first sweep blocked — level consumed

        logging.info(
            f"[DIAG_{side}_C3_PROJ] {symbol} bar {cur}: sweep_bar={k} (FIRST SWEEP), "
            f"sweep_level={fmt(sweep_level)}, trigger_body=[{fmt(body_low)}, {fmt(body_high)}], "
            f"proj_inside=True, fwd_body_clear=True ✅"
        )

        logging.info(
            f"[GOATv2_{side}_C3] Trigger {cur} ({df.loc[cur,'timestamp']}), "
            f"pivot from {pivot_idx} ({df.loc[pivot_idx,'timestamp']}), "
            f"sweep bar {k} ({df.loc[k,'timestamp']}), "
            f"{dcfg['pivot_swept_label']}={fmt(pivot_level)}"
        )
        return True, "PIVOT", dcfg["pivot_swept_label"], pivot_level

    logging.info(f"[DIAG_{side}_C3_NO_SWEEP] {symbol}: no sweep candidate passed after pivot")
    return False, None, None, None


# ─── Unified GOAT Check ─────────────────────────────────────────

def check_goat(df, side, symbol="?"):
    """
    Unified GOAT check for both BULL and BEAR.
    Tries Case 1 → Case 2 → Case 3 in priority order.
    """
    n = len(df)
    if n < 5:
        return False, None, None, None

    cur = n - 1
    dcfg = get_direction_config(
        side,
        _select_prior_lgcr_bear, _select_prior_lgcr_bull,
        _select_prior_lgcrs_bear, _select_prior_lgcrs_bull,
    )

    has_lgc = df.loc[cur, dcfg["lgc_col"]]
    has_lgcr = df.loc[cur, dcfg["lgcr_col"]]

    if not (has_lgc and has_lgcr):
        if has_lgc or has_lgcr:
            logging.info(
                f"[DIAG_{side}_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
                f"{dcfg['lgc_col']}={has_lgc}, {dcfg['lgcr_col']}={has_lgcr} — GATE FAILED"
            )
        return False, None, None, None

    logging.info(
        f"[DIAG_{side}_GATE] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
        f"GATE PASSED ✅, entering case checks"
    )

    # ─── DIAGNOSTIC: dump what each case has to work with ────────
    prior_lgcr = dcfg["lgcr_selector"](df, cur)
    all_lgcrs = dcfg["lgcr_selector_multi"](df, cur)
    logging.info(f"[DIAG_{side}_AVAILABILITY] {symbol} bar {cur}: "
                 f"prior_LGCR={'bar '+str(prior_lgcr) if prior_lgcr is not None else 'NONE'}, "
                 f"total_LGCR_candidates={len(all_lgcrs)}")

    cur_price = df.loc[cur, 'HA_Close']
    lgc_candidates = []
    for i in range(cur - 1, -1, -1):
        if df.loc[i, dcfg["lgc_col"]]:
            ll = df.loc[i, dcfg["lgc_line_col"]]
            if pd.isna(ll):
                continue
            if dcfg["line_above_price"] and ll >= cur_price:
                lgc_candidates.append(i)
            elif not dcfg["line_above_price"] and ll <= cur_price:
                lgc_candidates.append(i)
    logging.info(f"[DIAG_{side}_AVAILABILITY] {symbol} bar {cur}: "
                 f"LG_line_candidates={len(lgc_candidates)} "
                 f"(bars: {lgc_candidates[:5]}{'...' if len(lgc_candidates) > 5 else ''})")

    if dcfg["side"] == "BEAR":
        pvts = find_ha_pivot_highs(df, 0, cur)
        pvt_count = sum(1 for _, lvl in pvts if lvl >= cur_price)
    else:
        pvts = find_ha_pivot_lows(df, 0, cur)
        pvt_count = sum(1 for _, lvl in pvts if lvl <= cur_price)
    logging.info(f"[DIAG_{side}_AVAILABILITY] {symbol} bar {cur}: "
                 f"pivot_candidates={pvt_count}")

    # ─── Try each case in priority order ─────────────────────────
    cases = [
        (_check_case1_lgcr_sweep, "Case 1 (LGCR sweep)"),
        (_check_case2_lg_line_sweep, "Case 2 (LG line)"),
        (_check_case3_pivot_sweep, "Case 3 (Pivot)"),
    ]

    for case_fn, case_name in cases:
        triggered, case_label, swept_label, swept_value = case_fn(df, cur, symbol, dcfg)
        if triggered:
            logging.info(f"[GOATv2_{side}] {symbol}: ✅ {case_name} TRIGGERED")
            return True, case_label, swept_label, swept_value
        logging.info(f"[GOATv2_{side}] {symbol}: ❌ {case_name} did not trigger")

    logging.info(
        f"[DIAG_ALL_CASES_FAILED] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
        f"LGC+LGCR gate passed but ALL 3 cases failed"
    )
    return False, None, None, None