import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

from aurum.config import MIN_BARS_FOR_FEATURES


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_BARS_FOR_FEATURES:
        return df

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_ind.macd()
    df["MACD_Signal"] = macd_ind.macd_signal()
    df["MACD_Diff"] = macd_ind.macd_diff()

    df["RSI_Overbought"] = df["RSI"].rolling(200).quantile(0.90)
    df["RSI_Oversold"] = df["RSI"].rolling(200).quantile(0.10)

    df["ATR"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Squeeze"] = (df["BB_High"] - df["BB_Low"]) / close

    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    df["EMA_9_Slope"] = df["EMA_9"].diff(3) / df["ATR"]
    df["EMA_50_Slope"] = df["EMA_50"].diff(5) / df["ATR"]

    rolling_window = 50
    corr_dxy = close.rolling(rolling_window).corr(df["DXY_Close"])
    corr_us10y = close.rolling(rolling_window).corr(df["US10Y_Close"])

    dxy_std = corr_dxy.rolling(200).std().replace(0, np.nan)
    us10y_std = corr_us10y.rolling(200).std().replace(0, np.nan)
    df["Corr_DXY_Z"] = (corr_dxy - corr_dxy.rolling(200).mean()) / dxy_std
    df["Corr_US10Y_Z"] = (corr_us10y - corr_us10y.rolling(200).mean()) / us10y_std

    df["POC"], df["VAH"], df["VAL"] = _volume_profile(close, volume)

    df["Swing_High"] = high.rolling(20, center=True).max().ffill()
    df["Swing_Low"] = low.rolling(20, center=True).min().ffill()

    df["POC_Dist"] = (close - df["POC"]) / df["ATR"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    df["Vol_Ratio"] = volume / vol_ma

    return df.dropna()


def _volume_profile(close: pd.Series, volume: pd.Series, window: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes_np = close.to_numpy()
    vols_np = volume.to_numpy()
    n = len(close)
    pocs = np.zeros(n)
    vahs = np.zeros(n)
    vals = np.zeros(n)

    for i in range(window, n):
        hist_bins = closes_np[i - window : i]
        hist_vols = vols_np[i - window : i]
        counts, bins = np.histogram(hist_bins, bins=20, weights=hist_vols)
        max_idx = int(np.argmax(counts))
        pocs[i] = (bins[max_idx] + bins[max_idx + 1]) / 2
        vahs[i] = bins[min(max_idx + 2, len(bins) - 1)]
        vals[i] = bins[max(max_idx - 2, 0)]

    return pocs, vahs, vals
