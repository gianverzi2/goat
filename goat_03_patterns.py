"""
03 — Pattern detection wrapper: applies LG, LGC, LGCR to a Heikin-Ashi DataFrame.
"""

import pandas as pd

from function_LG_major_functions import detect_lg, detect_lgc, detect_lgcr


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all LG/LGC/LGCR patterns and annotate the DataFrame."""
    for col in ["bullish_LG", "bearish_LG", "bullish_LGC", "bearish_LGC",
                "bullish_LGCR", "bearish_LGCR"]:
        if col not in df.columns:
            df[col] = False

    for col in ["bullish_LGC_line", "bearish_LGC_line"]:
        if col not in df.columns:
            df[col] = float('nan')

    lg_levels = detect_lg(df)
    lgc_levels = detect_lgc(df)
    lgcr_levels = detect_lgcr(df, lg_levels)

    for lg in lg_levels:
        df.loc[lg["index"], "bullish_LG" if lg["type"] == "bullish" else "bearish_LG"] = True

    for lgc in lgc_levels:
        col_flag = "bullish_LGC" if lgc["type"] == "bullish" else "bearish_LGC"
        col_line = "bullish_LGC_line" if lgc["type"] == "bullish" else "bearish_LGC_line"
        df.loc[lgc["index"], col_flag] = True
        df.loc[lgc["index"], col_line] = lgc["line_level"]

    for lgcr in lgcr_levels:
        df.loc[lgcr["index"], "bullish_LGCR" if lgcr["type"] == "bullish" else "bearish_LGCR"] = True

    return df