"""
17 technical indicators computed from raw OHLCV data (Section IV-B of the
paper). Combined with the 5 Kalman-filtered OHLCV values this produces the
paper's 22-dimensional state vector.

Indicators are computed on the *raw* price series (indicators and the
Kalman-filtered price are two separate branches feeding the same state
vector, as in the paper) and z-score normalised downstream in
data_loader.build_feature_matrix.

Everything here is vectorised with numpy/pandas (including CCI's mean
absolute deviation, via sliding_window_view) so it stays fast on
multi-million-row M1 files.
"""
import numpy as np
import pandas as pd
import config


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _stochastic(df: pd.DataFrame, window: int, smooth: int):
    lowest = df["low"].rolling(window).min()
    highest = df["high"].rolling(window).max()
    pct_k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    pct_k = pct_k.fillna(50.0)
    pct_d = pct_k.rolling(smooth).mean().fillna(50.0)
    return pct_k, pct_d


def _bollinger(close: pd.Series, window: int, n_std: float):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, lower


def _atr(df: pd.DataFrame, window: int):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _obv(df: pd.DataFrame):
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["tick_volume"]).cumsum()


def _vwap(df: pd.DataFrame, window: int):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["tick_volume"]
    return pv.rolling(window).sum() / df["tick_volume"].rolling(window).sum().replace(0, np.nan)


def _rolling_mad(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorised rolling mean-absolute-deviation via sliding_window_view
    (avoids pandas .rolling().apply(), which is very slow on M1-sized data)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    means = windows.mean(axis=1)
    mad = np.abs(windows - means[:, None]).mean(axis=1)
    out[window - 1:] = mad
    return out


def _cci(df: pd.DataFrame, window: int):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(window).mean()
    mad = pd.Series(_rolling_mad(typical.to_numpy(dtype=np.float64), window), index=df.index)
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def _williams_r(df: pd.DataFrame, window: int):
    highest = df["high"].rolling(window).max()
    lowest = df["low"].rolling(window).min()
    return -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)


def compute_indicators(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    close = df["close"]
    feats = {}

    for w in config.SMA_WINDOWS:
        feats[f"sma_{w}"] = close.rolling(w).mean()

    ema12 = close.ewm(span=config.EMA_WINDOWS[0], adjust=False).mean()
    ema26 = close.ewm(span=config.EMA_WINDOWS[1], adjust=False).mean()
    feats["ema_12"] = ema12
    feats["ema_26"] = ema26

    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_line"] = macd_line
    feats["macd_signal"] = macd_signal

    feats["rsi_14"] = _rsi(close, config.RSI_WINDOW)

    pct_k, pct_d = _stochastic(df, config.STOCH_WINDOW, config.STOCH_SMOOTH)
    feats["stoch_k"] = pct_k
    feats["stoch_d"] = pct_d

    bb_upper, bb_lower = _bollinger(close, config.BB_WINDOW, config.BB_STD)
    feats["bb_upper"] = bb_upper
    feats["bb_lower"] = bb_lower

    feats["atr_14"] = _atr(df, config.ATR_WINDOW)
    feats["obv"] = _obv(df)
    feats["vwap"] = _vwap(df, config.VWAP_WINDOW)
    feats["cci_20"] = _cci(df, config.CCI_WINDOW)
    feats["williams_r"] = _williams_r(df, config.WILLIAMS_WINDOW)

    out = pd.DataFrame(feats)
    assert out.shape[1] == 17, f"Expected 17 indicators, got {out.shape[1]}"
    return out
