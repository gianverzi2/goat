"""
05 — GOAT Case 1, 2, 3 logic + unified check_goat().
All sweep logic lives here. Works for both bull and bear via direction config.

"Projection-first" sweep detection flow:
  1. Gate: trigger bar (cur) must have both LGC + LGCR flags.
  2. Find ALL candidate sweep bars k (scanning backward from cur-1):
       BEAR: HA_Low[k] falls inside the trigger body [body_low(cur), body_high(cur)]
       BULL: HA_High[k] falls inside the trigger body
     Additional filters: real wick exists, no body intersection at projection level
     between k+1 and cur-1.
  3. For each candidate sweep bar k, check (C1 → C2 → C3 priority):
       C1 (LGCR sweep): k's wick reached a prior valid LGCR line
       C2 (LG line sweep): k's wick reached a prior valid LG line
       C3 (Pivot sweep): k is itself a 2+1+2 pivot AND its wick reached a prior valid pivot
  4. First match wins.

Validation rules applied per candidate:
  - Wick-only: sweep bar's close must NOT break through the swept level
  - Level validity: no close through the level between source bar and k
  - First wick counts (C1/C2): earlier wick between source and k consumes the line
  - Sweep-of-sweep chain (C3): prior pivot consumed if intermediate pivot already swept it
  - Real wick: HA_High > body top (BEAR) or HA_Low < body bottom (BULL)
  - 2+1+2 pivot check on sweep bar (Case 3 only)
  - No body intersection between sweep bar and trigger at projection level
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


# ─── Projection-first: candidate sweep bar finder ───────────────

def _find_sweep_candidates(df, cur, body_low, body_high, dcfg, symbol):
    """
    Projection-first sweep candidate finder.

    Scans backward from cur-1 and returns bar indices k where:
      - The projection (HA_Low for BEAR, HA_High for BULL) falls inside
        the trigger bar body [body_low, body_high].
      - A real wick exists at bar k:
          BEAR: HA_High[k] > max(HA_Open[k], HA_Close[k])
          BULL: HA_Low[k]  < min(HA_Open[k], HA_Close[k])
      - No bar body between k+1 and cur-1 intersects the projection level.

    Returns list of candidate bar indices (most recent first).
    """
    side = dcfg["side"]
    candidates = []

    for k in range(cur - 1, -1, -1):
        proj = df.loc[k, dcfg["sweep_level_col"]]

        # Inclusive bounds: touching the edge of the trigger body still qualifies.
        # This matches the original projection-inside-body check semantics.
        if not (body_low <= proj <= body_high):
            continue

        wick = df.loc[k, dcfg["sweep_col"]]
        ha_open_k = df.loc[k, 'HA_Open']
        ha_close_k = df.loc[k, 'HA_Close']
        body_top_k = max(ha_open_k, ha_close_k)
        body_bot_k = min(ha_open_k, ha_close_k)

        if side == "BEAR" and wick <= body_top_k:
            logging.info(
                f"[DIAG_{side}_SWEEP_CAND] {symbol} bar {cur}: k={k} "
                f"({df.loc[k,'timestamp']}): proj={fmt(proj)} in body, "
                f"HA_High={fmt(wick)} <= body_top={fmt(body_top_k)} — no real wick, skipped"
            )
            continue
        if side == "BULL" and wick >= body_bot_k:
            logging.info(
                f"[DIAG_{side}_SWEEP_CAND] {symbol} bar {cur}: k={k} "
                f"({df.loc[k,'timestamp']}): proj={fmt(proj)} in body, "
                f"HA_Low={fmt(wick)} >= body_bot={fmt(body_bot_k)} — no real wick, skipped"
            )
            continue

        fwd_block_bar = None
        for j in range(k + 1, cur):
            if body_intersects_level(df, j, proj):
                fwd_block_bar = j
                break

        if fwd_block_bar is not None:
            logging.info(
                f"[DIAG_{side}_SWEEP_CAND] {symbol} bar {cur}: k={k} "
                f"({df.loc[k,'timestamp']}): proj={fmt(proj)} in body, wick ok, "
                f"fwd body blocked at bar {fwd_block_bar} — skipped"
            )
            continue

        candidates.append(k)
        logging.info(
            f"[DIAG_{side}_SWEEP_CAND] {symbol} bar {cur}: k={k} "
            f"({df.loc[k,'timestamp']}): proj={fmt(proj)}, "
            f"wick={fmt(wick)} ✅ sweep candidate"
        )

    logging.info(
        f"[DIAG_{side}_SWEEP_CANDS] {symbol} bar {cur}: "
        f"found {len(candidates)} sweep candidates"
        + (f" (bars: {candidates[:10]}{'...' if len(candidates) > 10 else ''})"
           if candidates else "")
    )
    return candidates


# ─── Case 1: LGCR Sweep (given sweep bar k) ─────────────────────

def _check_c1_for_sweep(df, cur, k, symbol, dcfg, all_lgcrs):
    """
    Case 1 (projection-first): check if sweep bar k's wick swept a valid prior LGCR line.

    Given the already-validated sweep bar k (projection inside trigger body, real wick,
    no forward body intersection), scan all prior LGCR bars and check:
      1. LGCR line not invalidated from LGCR bar to k (no close through)
      1.5. First wick counts: if an earlier bar already wicked the line, it's consumed
      2. k's wick reaches the LGCR line
      3. k's close does NOT break through the line (wick-only)
    First matching LGCR wins (closest-first order from selector).

    all_lgcrs: precomputed list from lgcr_selector_multi (passed in to avoid re-calling
               the selector for each sweep candidate).
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']

    # Filter to LGCR bars that occurred before the sweep bar k
    # (selector result is already sorted closest-first relative to trigger price)
    all_priors = [i for i in all_lgcrs if i < k]

    if not all_priors:
        logging.info(
            f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
            f"no prior {dcfg['lgcr_col']} before sweep bar — C1 skip"
        )
        return False, None, None, None

    logging.info(
        f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
        f"found {len(all_priors)} prior LGCR candidates before sweep bar"
        f" (bars: {all_priors[:5]}{'...' if len(all_priors) > 5 else ''})"
    )

    for prior_idx in all_priors:
        line1 = df.loc[prior_idx, dcfg["lgcr_line1_col"]]  # HA_Close
        line2 = df.loc[prior_idx, dcfg["lgcr_line2_col"]]  # HA_High (bear) / HA_Low (bull)

        # ── 1. Line validity from LGCR bar to k (no close through) ──
        line1_valid = True
        line2_valid = True
        for j in range(prior_idx + 1, k):
            c = df.loc[j, 'HA_Close']
            if line1_valid and compare(c, line1, dcfg["invalidate_op"]):
                line1_valid = False
            if line2_valid and compare(c, line2, dcfg["invalidate_op"]):
                line2_valid = False
            if not line1_valid and not line2_valid:
                break

        if not line1_valid and not line2_valid:
            logging.info(
                f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                f"LGCR at bar {prior_idx} ({df.loc[prior_idx,'timestamp']}): "
                f"BOTH lines invalidated before sweep bar — skipped"
            )
            continue

        # ── 1.5. First wick counts: earlier wick consumes the line ──
        if line1_valid or line2_valid:
            for j in range(prior_idx + 1, k):
                j_wick = df.loc[j, dcfg["sweep_col"]]
                if line2_valid and sweep_reaches(j_wick, line2, dcfg["tolerance_factor"]):
                    line2_valid = False
                    logging.info(
                        f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                        f"LGCR at bar {prior_idx}: line2={fmt(line2)} consumed by prior wick "
                        f"at bar {j} ({df.loc[j,'timestamp']}) — skipped"
                    )
                if line1_valid and sweep_reaches(j_wick, line1, dcfg["tolerance_factor"]):
                    line1_valid = False
                    logging.info(
                        f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                        f"LGCR at bar {prior_idx}: line1={fmt(line1)} consumed by prior wick "
                        f"at bar {j} ({df.loc[j,'timestamp']}) — skipped"
                    )
                if not line1_valid and not line2_valid:
                    break
        if not line1_valid and not line2_valid:
            continue

        # ── 2. k's wick reaches a valid line ──
        swept_line2 = line2_valid and sweep_reaches(wick, line2, dcfg["tolerance_factor"])
        swept_line1 = line1_valid and sweep_reaches(wick, line1, dcfg["tolerance_factor"])

        if not (swept_line1 or swept_line2):
            logging.info(
                f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                f"LGCR at bar {prior_idx}: wick={fmt(wick)} doesn't reach any valid line "
                f"(line1={fmt(line1)} {'valid' if line1_valid else 'BROKEN'}, "
                f"line2={fmt(line2)} {'valid' if line2_valid else 'BROKEN'}) — skipped"
            )
            continue

        # ── 3. Wick-only (close must NOT break through the swept line) ──
        swept_ref = line2 if swept_line2 else line1
        if side == "BEAR" and sweep_close > swept_ref:
            logging.info(
                f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                f"LGCR at bar {prior_idx}: CLOSED THROUGH line={fmt(swept_ref)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue
        if side == "BULL" and sweep_close < swept_ref:
            logging.info(
                f"[DIAG_{side}_C1] {symbol} bar {cur}: k={k}: "
                f"LGCR at bar {prior_idx}: CLOSED THROUGH line={fmt(swept_ref)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue

        # ✅ Valid
        swept_label = dcfg["lgcr_swept_label_1"] if swept_line2 else dcfg["lgcr_swept_label_2"]
        swept_value = line2 if swept_line2 else line1
        logging.info(
            f"[GOATv2_{side}_C1] Trigger {cur} ({df.loc[cur,'timestamp']}), "
            f"LGCR prior {prior_idx} ({df.loc[prior_idx,'timestamp']}), "
            f"sweep bar {k} ({df.loc[k,'timestamp']}), "
            f"lines=[{dcfg['lgcr_line1_col']}={fmt(line1)}, {dcfg['lgcr_line2_col']}={fmt(line2)}], "
            f"swept_line1={swept_line1}, swept_line2={swept_line2}, "
            f"{swept_label}={fmt(swept_value)} ✅"
        )
        return True, "LGCR", swept_label, swept_value

    logging.info(f"[GOATv2_{side}_C1] {symbol}: sweep_bar={k}: Case 1 (LGCR sweep) did not trigger")
    return False, None, None, None




# ─── Case 2: LG Line Sweep (given sweep bar k) ──────────────────

def _check_c2_for_sweep(df, cur, k, symbol, dcfg):
    """
    Case 2 (projection-first): check if sweep bar k's wick swept a valid prior LG line.

    Given the already-validated sweep bar k, scan all prior LGC bars (before k) and check:
      1. LG line not invalidated from LGC bar to k (no close through)
      1.5. First wick counts: if an earlier bar already wicked the line, it's consumed
      2. k's wick reaches the LG line
      3. k's close does NOT break through the line (wick-only)
    First matching LG line wins (closest-to-current-price order).
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']
    cur_price = df.loc[cur, 'HA_Close']

    # Find all LGC bars before k with a valid line on the correct side of trigger price
    candidates_lg = []
    for i in range(k - 1, -1, -1):
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
            f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
            f"no {dcfg['lgc_col']} with valid line before sweep bar — C2 skip"
        )
        return False, None, None, None

    # Sort closest to current price first
    candidates_lg.sort(key=lambda x: abs(x[1] - cur_price))

    logging.info(
        f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
        f"found {len(candidates_lg)} LG line candidates before sweep bar"
    )

    for lgc_idx, line_level in candidates_lg:
        # ── 1. Line validity from LGC bar to k (no close through) ──
        line_valid = True
        for j in range(lgc_idx + 1, k):
            if line_invalidated(df.loc[j, 'HA_Close'], line_level, dcfg):
                line_valid = False
                break

        if not line_valid:
            logging.info(
                f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
                f"LG line from bar {lgc_idx} ({df.loc[lgc_idx,'timestamp']}), "
                f"line_level={fmt(line_level)} — INVALIDATED before sweep bar — skipped"
            )
            continue

        # ── 1.5. First wick counts: earlier wick consumes the line ──
        line_consumed = False
        for j in range(lgc_idx + 1, k):
            if sweep_reaches(df.loc[j, dcfg["sweep_col"]], line_level, dcfg["tolerance_factor"]):
                line_consumed = True
                logging.info(
                    f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
                    f"LG line from bar {lgc_idx}: line={fmt(line_level)} consumed by prior wick "
                    f"at bar {j} ({df.loc[j,'timestamp']}) — skipped"
                )
                break
        if line_consumed:
            continue

        # ── 2. k's wick reaches the line ──
        if not sweep_reaches(wick, line_level, dcfg["tolerance_factor"]):
            logging.info(
                f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
                f"LG line from bar {lgc_idx}: wick={fmt(wick)} doesn't reach "
                f"line={fmt(line_level)} — skipped"
            )
            continue

        # ── 3. Wick-only (close must NOT break through the line) ──
        if side == "BEAR" and sweep_close > line_level:
            logging.info(
                f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
                f"LG line from bar {lgc_idx}: CLOSED THROUGH line={fmt(line_level)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue
        if side == "BULL" and sweep_close < line_level:
            logging.info(
                f"[DIAG_{side}_C2] {symbol} bar {cur}: k={k}: "
                f"LG line from bar {lgc_idx}: CLOSED THROUGH line={fmt(line_level)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue

        # ✅ Valid
        logging.info(
            f"[GOATv2_{side}_C2] Trigger {cur} ({df.loc[cur,'timestamp']}), "
            f"LG line from {lgc_idx} ({df.loc[lgc_idx,'timestamp']}), "
            f"sweep bar {k} ({df.loc[k,'timestamp']}), "
            f"LG_line_swept={fmt(line_level)} ✅"
        )
        return True, "LG_LINE", "LG_line_swept", line_level

    logging.info(f"[GOATv2_{side}_C2] {symbol}: sweep_bar={k}: Case 2 (LG line) did not trigger")
    return False, None, None, None


# ─── Case 3: Pivot Sweep (given sweep bar k) ────────────────────

def _check_c3_for_sweep(df, cur, k, symbol, dcfg):
    """
    Case 3 (projection-first): check if sweep bar k's wick swept a valid prior pivot.

    Given the already-validated sweep bar k (any HA candle with a wick — wick is
    guaranteed by the caller _find_sweep_candidates):
      1. Find any prior 2+1+2 pivot (before k) on the correct side of trigger price
      1.5. Pivot not consumed by sweep-of-sweep chain (no intermediate pivot already swept it)
      2. Pivot not invalidated from pivot bar to k (no close through)
      3. k's wick reaches the pivot level
      4. k's close does NOT break through the pivot (wick-only)
    First matching pivot wins.
    """
    side = dcfg["side"]
    wick = df.loc[k, dcfg["sweep_col"]]
    sweep_close = df.loc[k, 'HA_Close']
    cur_price = df.loc[cur, 'HA_Close']

    # ── Find prior pivots before k on the correct side of trigger price ──
    if side == "BEAR":
        pivots = find_ha_pivot_highs(df, 0, k)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl >= cur_price]
    else:
        pivots = find_ha_pivot_lows(df, 0, k)
        pivot_candidates = [(idx, lvl) for idx, lvl in pivots if lvl <= cur_price]

    if not pivot_candidates:
        logging.info(
            f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k}: "
            f"no prior pivot candidates before sweep bar "
            f"(total_pivots_before_k={len(pivots)}) — C3 skip"
        )
        return False, None, None, None

    logging.info(
        f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k} ({df.loc[k,'timestamp']}): "
        f"checking {len(pivot_candidates)} prior pivot candidates"
    )

    for pivot_idx, pivot_level in pivot_candidates:
        # ── 1. Pivot validity from pivot bar to k (no close through) ──
        pivot_valid = True
        for j in range(pivot_idx + 1, k):
            if line_invalidated(df.loc[j, 'HA_Close'], pivot_level, dcfg):
                pivot_valid = False
                break

        if not pivot_valid:
            logging.info(
                f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k}: "
                f"pivot at bar {pivot_idx} ({df.loc[pivot_idx,'timestamp']}), "
                f"level={fmt(pivot_level)}: INVALIDATED before sweep bar — skipped"
            )
            continue

        # ── 1.5. Sweep-of-sweep chain: consumed if an intermediate pivot already swept it ──
        pivot_consumed = False
        for j_idx, _ in pivots:
            if j_idx <= pivot_idx:
                continue
            if j_idx >= k:
                break
            if sweep_reaches(df.loc[j_idx, dcfg["sweep_col"]], pivot_level, dcfg["tolerance_factor"]):
                pivot_consumed = True
                logging.info(
                    f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k}: "
                    f"pivot at bar {pivot_idx}: level={fmt(pivot_level)} consumed by intermediate "
                    f"pivot at bar {j_idx} ({df.loc[j_idx,'timestamp']}) — skipped"
                )
                break
        if pivot_consumed:
            continue

        # ── 2. k's wick reaches the pivot level ──
        if not sweep_reaches(wick, pivot_level, dcfg["tolerance_factor"]):
            continue

        # ── 3. Wick-only (close must NOT break through the pivot) ──
        if side == "BEAR" and sweep_close > pivot_level:
            logging.info(
                f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k}: "
                f"pivot at bar {pivot_idx}: CLOSED THROUGH pivot={fmt(pivot_level)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue
        if side == "BULL" and sweep_close < pivot_level:
            logging.info(
                f"[DIAG_{side}_C3] {symbol} bar {cur}: k={k}: "
                f"pivot at bar {pivot_idx}: CLOSED THROUGH pivot={fmt(pivot_level)}, "
                f"close={fmt(sweep_close)} — skipped"
            )
            continue

        # ✅ Valid
        logging.info(
            f"[GOATv2_{side}_C3] Trigger {cur} ({df.loc[cur,'timestamp']}), "
            f"pivot from {pivot_idx} ({df.loc[pivot_idx,'timestamp']}), "
            f"sweep bar {k} ({df.loc[k,'timestamp']}), "
            f"{dcfg['pivot_swept_label']}={fmt(pivot_level)} ✅"
        )
        return True, "PIVOT", dcfg["pivot_swept_label"], pivot_level

    logging.info(f"[GOATv2_{side}_C3] {symbol}: sweep_bar={k}: Case 3 (Pivot) did not trigger")
    return False, None, None, None


# ─── Unified GOAT Check ─────────────────────────────────────────

def check_goat(df, side, symbol="?"):
    """
    Unified GOAT check for both BULL and BEAR.

    Uses "projection-first" sweep detection:
      1. Gate: confirm trigger bar (cur) has both LGC + LGCR flags.
      2. Find ALL candidate sweep bars k scanning backward from cur-1:
           BEAR: HA_Low[k] falls inside trigger body [body_low, body_high]
           BULL: HA_High[k] falls inside trigger body
         Additional filters: real wick exists, no body intersection forward k+1…cur-1.
      3. For each candidate sweep bar, try Case 1 → Case 2 → Case 3.
      4. First match wins.

    Returns: (triggered, case_label, swept_label, swept_value)
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

    # ── 1. Gate: LGC + LGCR on trigger bar ──────────────────────
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

    # ── 2. Trigger body ──────────────────────────────────────────
    body_low = min(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])
    body_high = max(df.loc[cur, 'HA_Open'], df.loc[cur, 'HA_Close'])

    # ── DIAGNOSTIC: availability info ───────────────────────────
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

    # ── 3. Find candidate sweep bars (projection-first) ──────────
    sweep_candidates = _find_sweep_candidates(df, cur, body_low, body_high, dcfg, symbol)

    if not sweep_candidates:
        logging.info(
            f"[DIAG_ALL_CASES_FAILED] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
            f"LGC+LGCR gate passed but no sweep candidates found (projection check)"
        )
        return False, None, None, None

    # ── 4. For each candidate, try C1 → C2 → C3 ─────────────────
    # Case 1 needs the precomputed LGCR list (avoids re-calling selector per candidate).
    case_fns = [
        (lambda df, cur, k, sym, dcfg: _check_c1_for_sweep(df, cur, k, sym, dcfg, all_lgcrs),
         "Case 1 (LGCR sweep)"),
        (_check_c2_for_sweep, "Case 2 (LG line)"),
        (_check_c3_for_sweep, "Case 3 (Pivot)"),
    ]

    for k in sweep_candidates:
        logging.info(
            f"[DIAG_{side}_SWEEP_BAR] {symbol} bar {cur}: trying sweep_bar={k} "
            f"({df.loc[k,'timestamp']}), "
            f"wick={fmt(df.loc[k, dcfg['sweep_col']])}, "
            f"proj={fmt(df.loc[k, dcfg['sweep_level_col']])}"
        )
        for case_fn, case_name in case_fns:
            triggered, case_label, swept_label, swept_value = case_fn(df, cur, k, symbol, dcfg)
            if triggered:
                logging.info(
                    f"[GOATv2_{side}] {symbol}: ✅ sweep_bar={k}, {case_name} TRIGGERED"
                )
                return True, case_label, swept_label, swept_value
            logging.info(
                f"[GOATv2_{side}] {symbol}: sweep_bar={k}, ❌ {case_name} did not trigger"
            )

    logging.info(
        f"[DIAG_ALL_CASES_FAILED] {symbol} bar {cur} ({df.loc[cur,'timestamp']}): "
        f"LGC+LGCR gate passed, {len(sweep_candidates)} sweep candidates tried, ALL cases failed"
    )
    return False, None, None, None