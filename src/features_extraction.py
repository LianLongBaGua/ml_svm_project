import pandas as pd
import numpy as np
import talib
import warnings


def generate_og_features_df(df: pd.DataFrame, lags: list):
    print("Generating original features...")
    for lag in lags:
        df["ADOSC_" + str(lag)] = talib.ADOSC(
            df["high"], df["low"], df["close"], df["volume"], lag, lag * 3
        )
        df["MFI_" + str(lag)] = talib.MFI(
            df["high"], df["low"], df["close"], df["volume"], lag
        )


def generate_mom_features_df(df: pd.DataFrame, lags: list):
    print("Generating momentum features...")
    for lag in lags:
        df["ROC_" + str(lag)] = talib.ROC(df["close"], lag)
        df["MOM_" + str(lag)] = talib.MOM(df["close"], lag)
        df["PLUS_DM_" + str(lag)] = talib.PLUS_DM(df["high"], df["low"], lag)
        df["MINUS_DM_" + str(lag)] = talib.MINUS_DM(df["high"], df["low"], lag)
        df["ADX_" + str(lag)] = talib.ADX(df["high"], df["low"], df["close"], lag)
        df["ADXR_" + str(lag)] = talib.ADXR(df["high"], df["low"], df["close"], lag)
        df["APO_" + str(lag)] = talib.APO(df["close"], lag, lag * 2)
        df["AROONOSC_" + str(lag)] = talib.AROONOSC(df["high"], df["low"], lag)

        df["CCI_" + str(lag)] = talib.CCI(df["high"], df["low"], df["close"], lag)
        df["CMO_" + str(lag)] = talib.CMO(df["close"], lag)
        df["DX_" + str(lag)] = talib.DX(df["high"], df["low"], df["close"], lag)
        df["STOCH_" + str(lag) + "slowk"], _ = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=lag,
            slowk_period=int(lag / 2),
            slowk_matype=0,
            slowd_period=int(lag / 2),
            slowd_matype=0,
        )
        df["STOCHF_" + str(lag) + "fastk"], _ = talib.STOCHF(
            df["high"], df["low"], df["close"], lag, int(lag / 2), 0
        )
        (_, df["MACDSIGNAL_" + str(lag)], _) = talib.MACD(
            df["close"], lag, lag * 2, int(lag / 2)
        )
        _, df["MACDSIGNALFIX_" + str(lag)], _ = talib.MACDFIX(df["close"], lag)
        df["PPO_" + str(lag)] = talib.PPO(df["close"], lag, lag * 2)
        df["RSI_" + str(lag)] = talib.RSI(df["close"], lag)
        df["ULTOSC_" + str(lag)] = talib.ULTOSC(
            df["high"], df["low"], df["close"], lag, lag * 2, lag * 3
        )
        df["WILLR_" + str(lag)] = talib.WILLR(df["high"], df["low"], df["close"], lag)
        df["STOCHRSI_" + str(lag) + "k"], _ = talib.STOCHRSI(df["close"], lag, 3, 3)
        df["NATR_" + str(lag)] = talib.NATR(df["high"], df["low"], df["close"], lag)
        df["ATR_" + str(lag)] = talib.ATR(df["high"], df["low"], df["close"], lag)
        df["TRANGE_" + str(lag)] = talib.TRANGE(df["high"], df["low"], df["close"])

    df["HT_TRENDLINE"] = talib.HT_TRENDLINE(df["close"])
    df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["close"])
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
    df["HT_DCPHASE"] = talib.HT_DCPHASE(df["close"])
    df["HT_PHASORinphase"], _= talib.HT_PHASOR(df["close"])
    df["HT_SINEsine"], _ = talib.HT_SINE(df["close"])


def generate_math_features_df(df: pd.DataFrame, lags: list):
    print("Generating math features...")
    for lag in lags:
        df["BETA_" + str(lag)] = talib.BETA(df["high"], df["low"], lag)
        df["CORREL_" + str(lag)] = talib.CORREL(df["high"], df["low"], lag)
        df["LINEARREG_" + str(lag)] = talib.LINEARREG(df["close"], lag)
        df["LINEARREG_ANGLE_" + str(lag)] = talib.LINEARREG_ANGLE(df["close"], lag)
        df["LINEARREG_INTERCEPT_" + str(lag)] = talib.LINEARREG_INTERCEPT(
            df["close"], lag
        )
        df["LINEARREG_SLOPE_" + str(lag)] = talib.LINEARREG_SLOPE(df["close"], lag)
        df["STDDEV_" + str(lag)] = talib.STDDEV(df["close"], lag)
        df["TSF_" + str(lag)] = talib.TSF(df["close"], lag)
        df["VAR_" + str(lag)] = talib.VAR(df["close"], lag)


def generate_time_features(df: pd.DataFrame):
    print("Generating time features...")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_hour"] = df["datetime"].dt.hour
    df["time_minute"] = df["datetime"].dt.minute
    df["time_day_of_week"] = df["datetime"].dt.dayofweek
    df["time_day_of_month"] = df["datetime"].dt.day
    df.drop(columns=["datetime"], inplace=True)


def normalize_features(df, window):
    """
    Normalize each column of features by minusing the average and divide by std
    """
    for col in df.columns:
        if col == "datetime":
            continue
        df[col] = (df[col] - talib.EMA(df[col], window)) / talib.STDDEV(df[col], window)


def generate_cossin_time_features(df):
    """
    Transform time columns to cos and sin
    """
    print("Generating cossin time features...")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df['time_hour_sin'] = talib.SIN(df['datetime'].dt.hour / 24 * 2 * np.pi)
    df['time_hour_cos'] = talib.COS(df['datetime'].dt.hour / 24 * 2 * np.pi)
    df['time_minute_sin'] = talib.SIN(df['datetime'].dt.minute / 60 * 2 * np.pi)
    df['time_minute_cos'] = talib.COS(df['datetime'].dt.minute / 60 * 2 * np.pi)
    df['time_day_of_week_sin'] = talib.SIN(df['datetime'].dt.dayofweek / 7 * 2 * np.pi)
    df['time_day_of_week_cos'] = talib.COS(df['datetime'].dt.dayofweek / 7 * 2 * np.pi)
    df['time_day_of_month_sin'] = talib.SIN(df['datetime'].dt.day / 30 * 2 * np.pi)
    df['time_day_of_month_cos'] = talib.COS(df['datetime'].dt.day / 30 * 2 * np.pi)
    df.drop(columns=["datetime"], inplace=True)

def generate_all_features_df(df: pd.DataFrame, lags: list, normalization_window=1000):
    warnings.filterwarnings("ignore")
    generate_og_features_df(df, lags)
    generate_mom_features_df(df, lags)
    generate_math_features_df(df, lags)
    normalize_features(df, normalization_window)
    generate_cossin_time_features(df)
    df.dropna(inplace=True)

    # sort by name
    df = df.reindex(sorted(df.columns), axis=1)
    return df
