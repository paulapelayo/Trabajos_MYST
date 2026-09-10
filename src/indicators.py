"""
Indicadores tecnicos calculados directamente con pandas/numpy (sin la
libreria `ta`, que no compila en este entorno con setuptools moderno).
Todas las funciones son causales: solo usan datos hasta el instante t,
nunca datos futuros.

Convencion de columnas esperada en `df`: Open, High, Low, Close, Volume.
"""
import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Close"].rolling(window=window, min_periods=window).mean()


def ema(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Close"].ewm(span=window, adjust=False, min_periods=window).mean()


def bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(window=window, min_periods=window).mean()
    std = df["Close"].rolling(window=window, min_periods=window).std(ddof=0)
    high = mid + n_std * std
    low = mid - n_std * std
    pct = (df["Close"] - low) / (high - low)
    return pd.DataFrame({
        "bb_high": high,
        "bb_mid": mid,
        "bb_low": low,
        "bb_pct": pct,
    }, index=df.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26,
          window_signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=window_fast, adjust=False, min_periods=window_fast).mean()
    ema_slow = df["Close"].ewm(span=window_slow, adjust=False, min_periods=window_slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=window_signal, adjust=False, min_periods=window_signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
    }, index=df.index)


def stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    lowest_low = df["Low"].rolling(window=window, min_periods=window).min()
    highest_high = df["High"].rolling(window=window, min_periods=window).max()
    k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    k_smooth = k.rolling(window=smooth, min_periods=smooth).mean()
    d = k_smooth.rolling(window=smooth, min_periods=smooth).mean()
    return pd.DataFrame({
        "stoch_k": k_smooth,
        "stoch_d": d,
    }, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_10"] = sma(df, 10)
    out["sma_100"] = sma(df, 100)
    out["ema_20"] = ema(df, 20)
    out = out.join(bollinger(df, 20, 2.0))
    out["atr_14"] = atr(df, 14)
    out["rsi_14"] = rsi(df, 14)
    out = out.join(macd(df))
    out = out.join(stochastic(df))
    return out
