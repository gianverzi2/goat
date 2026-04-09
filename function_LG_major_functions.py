# store major LG functions
import numpy as np

def calculate_heikin_ashi(df):
    if df.empty:
        print("DataFrame is empty. Skipping Heikin-Ashi calculation.")
        return df

    ha_df = df.copy()
    ha_df['HA_Close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    ha_df['HA_Open'] = np.nan

    if len(ha_df) > 0:
        ha_df.loc[0, 'HA_Open'] = (ha_df.loc[0, 'open'] + ha_df.loc[0, 'close']) / 2
        for i in range(1, len(ha_df)):
            ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2

        for i in range(len(ha_df)):
            ha_df.loc[i, 'HA_High'] = max(ha_df.loc[i, 'high'], max(ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close']))
            ha_df.loc[i, 'HA_Low'] = min(ha_df.loc[i, 'low'], min(ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close']))

    return ha_df

def detect_lg(df):
    lg_levels = []
    for i in range(1, len(df)):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]
        if prev_candle['HA_High'] < current_candle['HA_Close']:
            lg_levels.append({'type': 'bullish', 'index': i, 'price': current_candle['HA_Low']})
        if prev_candle['HA_Low'] > current_candle['HA_Close']:
            lg_levels.append({'type': 'bearish', 'index': i, 'price': current_candle['HA_High']})
    return lg_levels

def detect_lgc(df):
    lgc_levels = []
    for i in range(2, len(df)):
        two_candles_back = df.iloc[i - 2]
        one_candle_back = df.iloc[i - 1]
        current_candle = df.iloc[i]
        if (two_candles_back['HA_High'] < current_candle['HA_Low'] and one_candle_back['HA_Close'] > one_candle_back['HA_Open']):
            lgc_levels.append({'type': 'bullish', 'index': i - 1, 'line_level': two_candles_back['HA_High'], 'is_untested': True})
        if (two_candles_back['HA_Low'] > current_candle['HA_High'] and one_candle_back['HA_Close'] < one_candle_back['HA_Open']):
            lgc_levels.append({'type': 'bearish', 'index': i - 1, 'line_level': two_candles_back['HA_Low'], 'is_untested': True})
    return lgc_levels

def detect_lgcr(df, lg_levels):
    lgcr_levels = []
    ref_for_bullish_lgcr = None
    ref_for_bearish_lgcr = None
    bullish_lgcr_flag = False
    bearish_lgcr_flag = False

    for i in range(len(df)):
        current_candle = df.iloc[i]
        is_bullish_lg = any(lg['type'] == 'bullish' and lg['index'] == i for lg in lg_levels)
        is_bearish_lg = any(lg['type'] == 'bearish' and lg['index'] == i for lg in lg_levels)

        if is_bullish_lg:
            ref_for_bearish_lgcr = current_candle['HA_Low']
            bearish_lgcr_flag = False

        if is_bearish_lg:
            ref_for_bullish_lgcr = current_candle['HA_High']
            bullish_lgcr_flag = False

        if (is_bullish_lg and ref_for_bullish_lgcr is not None and current_candle['HA_Close'] > ref_for_bullish_lgcr and not bullish_lgcr_flag):
            lgcr_levels.append({'type': 'bullish', 'index': i, 'price': current_candle['HA_Low']})
            bullish_lgcr_flag = True

        if (is_bearish_lg and ref_for_bearish_lgcr is not None and current_candle['HA_Close'] < ref_for_bearish_lgcr and not bearish_lgcr_flag):
            lgcr_levels.append({'type': 'bearish', 'index': i, 'price': current_candle['HA_High']})
            bearish_lgcr_flag = True

    return lgcr_levels