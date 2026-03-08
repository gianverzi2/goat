"""
00 — Direction configs (bull/bear) and comparison helpers.
All direction-dependent parameters live here.
"""


def compare(value, threshold, op):
    """Generic comparison: 'gt', 'lt', 'gte', 'lte'."""
    if op == "gt":
        return value > threshold
    elif op == "lt":
        return value < threshold
    elif op == "gte":
        return value >= threshold
    elif op == "lte":
        return value <= threshold
    return False


def sweep_reaches(wick_value, line_value, tolerance_factor):
    """Check if a wick reaches a line level within tolerance."""
    if tolerance_factor < 1:
        return wick_value >= line_value * tolerance_factor
    else:
        return wick_value <= line_value * tolerance_factor


def line_invalidated(close_value, line_level, dcfg):
    """Check if a close invalidates a line based on direction."""
    return compare(close_value, line_level, dcfg["invalidate_op"])


def get_direction_config(side, lgcr_selector_bear, lgcr_selector_bull,
                         lgcr_selector_multi_bear=None, lgcr_selector_multi_bull=None):
    """
    Return all direction-dependent parameters for bull or bear GOAT checks.
    Callers must pass in the LGCR selector functions to avoid circular imports.
    """
    if side == "BEAR":
        return {
            "side": "BEAR",
            "lgc_col": "bearish_LGC",
            "lgcr_col": "bearish_LGCR",
            "lgc_line_col": "bearish_LGC_line",
            # LGCR (Case 1): reference lines on the prior LGCR bar
            "lgcr_line1_col": "HA_Close",
            "lgcr_line2_col": "HA_High",
            # Sweep detected via HA_High (price wicks UP to resistance)
            "sweep_col": "HA_High",
            # Projection = opposite side of sweep bar
            "sweep_level_col": "HA_Low",
            # Line is above price for bear
            "line_above_price": True,
            # Both lines are resistance → invalidated by close ABOVE
            "invalidate_op": "gt",
            "secondary_invalidate_op": "gt",
            # Tolerance: wick >= line * 0.999999
            "tolerance_factor": 0.999999,
            # LGCR selector functions
            "lgcr_selector": lgcr_selector_bear,
            "lgcr_selector_multi": lgcr_selector_multi_bear,
            # Labels
            "lgcr_swept_label_1": "LGCR_HA_high_swept",
            "lgcr_swept_label_2": "LGCR_HA_close_swept",
            "pivot_swept_label": "pivot_high_swept",
        }
    else:
        return {
            "side": "BULL",
            "lgc_col": "bullish_LGC",
            "lgcr_col": "bullish_LGCR",
            "lgc_line_col": "bullish_LGC_line",
            "lgcr_line1_col": "HA_Close",
            "lgcr_line2_col": "HA_Low",
            "sweep_col": "HA_Low",
            "sweep_level_col": "HA_High",
            "line_above_price": False,
            # Both lines are support → invalidated by close BELOW
            "invalidate_op": "lt",
            "secondary_invalidate_op": "lt",
            "tolerance_factor": 1.000001,
            # LGCR selector functions
            "lgcr_selector": lgcr_selector_bull,
            "lgcr_selector_multi": lgcr_selector_multi_bull,
            "lgcr_swept_label_1": "LGCR_HA_low_swept",
            "lgcr_swept_label_2": "LGCR_HA_close_swept",
            "pivot_swept_label": "pivot_low_swept",
        }