import re
from functools import partial

import pandas as pd
import numpy as np

def drop_ohlcv_cols(df: pd.DataFrame):
    """drop ohlcv columns"""
    print('Dropping OHLCV columns')
    return df.drop(
        columns=["open", "high", "low", "close", "volume", "open_interest"], axis=1
    )


def split_features_target(df: pd.DataFrame):
    """split features and target"""
    print("Splitting features/target")
    X = df.drop(['pos_change_signal', 'net_pos_signal', 'desired_pos_change', 'desired_pos_rolling'], axis=1)
    y = df[['pos_change_signal', 'net_pos_signal', 'desired_pos_change', 'desired_pos_rolling']]

    return X, y


def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """K线合成"""
    df_resampled = df.resample(interval).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "open_interest": "last"
    })

    df_resampled = df_resampled.drop_duplicates()
    df.fillna(method='pad', inplace=True)

    return df_resampled


def keep_essentials(df: pd.DataFrame):
    """Keep only OHLCVT"""
    df.drop(
        columns=["exchange", "turnover","symbol"],
        axis=1,
        inplace=True,
    )
    return renaming(df)


def trim_df(df: pd.DataFrame, keep_symbol=False):
    """Trim everything other than OHLCV with option to keep symbol"""
    df.drop(
        columns=["exchange", "interval", "turnover", "datetime", "gateway_name"],
        axis=1,
        inplace=True,
    )

    if not keep_symbol:
        df.drop(["symbol"], axis=1, inplace=True)


def divide_means(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

    # Calculate the means of the arrays in each list.
    means1 = [np.mean(array) for array in list1]
    means2 = [np.mean(array) for array in list2]

    # Divide the means of the arrays in list1 by the means of the arrays in list2.
    return [mean1 / mean2 for mean1, mean2 in zip(means1, means2)]


def renaming(df: pd.DataFrame):
    """rename columns"""
    df.rename(
        columns={
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
            "open_price": "open",
        },
        inplace=True,
    )
    return df


def remove_infs_and_zeros(df: pd.DataFrame):
    """remove inf values and zeros"""
    df.replace([np.inf, -np.inf, np.zeros], 0.0001, inplace=True)
    return df


def drop_ohlcvt_cols(df: pd.DataFrame):
    """drop ohlcv and time columns"""
    return df.drop(
        columns=["open", "high", "low", "close", "volume", "open_interest", "datetime"], axis=1
    )


def generate_simple_features(df):
    df['open_change'] = df.open.pct_change(10)
    df['high_change'] = df.high.pct_change(10)
    df['low_change'] = df.low.pct_change(10)
    df['close_change'] = df.close.pct_change(10)
    df['volume_change'] = df.volume.pct_change(10)
    df.dropna(inplace=True)

