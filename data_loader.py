import pandas as pd
import numpy as np
from pathlib import Path
from pykalman import KalmanFilter
import ta
import config

def load_csv(symbol, timeframe, data_dir=config.DATA_DIR, proxy_symbol=config.PROXY_SYMBOL):
    main_path = Path(data_dir) / f"{symbol}_{timeframe}.csv"
    if not main_path.exists(): raise FileNotFoundError(f"Data file not found: {main_path}")

    print(f"[*] Memuat data utama dari: {main_path.name}")
    
    df = pd.read_csv(main_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    if 'spread' in df.columns:
        df['spread_price'] = df['spread'] * 0.001
    else:
        df['spread_price'] = 0.250

    if proxy_symbol:
        proxy_path = Path(data_dir) / f"{proxy_symbol}_{timeframe}.csv"
        if proxy_path.exists():
            df_proxy = pd.read_csv(proxy_path)
            df_proxy['datetime'] = pd.to_datetime(df_proxy['datetime'])
            df_proxy.set_index('datetime', inplace=True)
            df_proxy.sort_index(inplace=True)
            df_proxy = df_proxy.add_prefix('proxy_')
            df = df.join(df_proxy[['proxy_close', 'proxy_tick_volume']], how='left')
            df.ffill(inplace=True)
            df.fillna(0, inplace=True)
        else:
            df['proxy_close'] = 0.0
            df['proxy_tick_volume'] = 0.0
    else:
        df['proxy_close'] = 0.0
        df['proxy_tick_volume'] = 0.0

    first_valid = df['proxy_close'].ne(0).idxmax()
    if first_valid != df.index[0]:
        df = df.loc[first_valid:]

    df.reset_index(inplace=True)
    return df

def apply_kalman_filter(df: pd.DataFrame, columns=config.KALMAN_COLUMNS, n_iter=config.KALMAN_EM_ITER, subsample_size=config.KALMAN_FIT_SUBSAMPLE, model=None):
    data = df[columns].values
    if model is None:
        kf = KalmanFilter(
            transition_matrices=np.eye(len(columns)), observation_matrices=np.eye(len(columns)),
            initial_state_mean=data[0], initial_state_covariance=np.eye(len(columns)),
            observation_covariance=np.eye(len(columns)), transition_covariance=np.eye(len(columns)) * 0.01
        )
        fit_data = data if len(data) <= subsample_size else data[:subsample_size]
        kf = kf.em(fit_data, n_iter=n_iter)
    else:
        kf = model
    filtered_states, _ = kf.filter(data)
    df_filtered = pd.DataFrame(filtered_states, columns=[f"{c}_kf" for c in columns], index=df.index)
    return df_filtered, kf

def add_technical_indicators(df: pd.DataFrame):
    close_col = 'close_kf' if 'close_kf' in df.columns else 'close'
    high_col = 'high_kf' if 'high_kf' in df.columns else 'high'
    low_col = 'low_kf' if 'low_kf' in df.columns else 'low'
    vol_col = 'tick_volume_kf' if 'tick_volume_kf' in df.columns else 'tick_volume'

    for w in config.SMA_WINDOWS: df[f'sma_{w}'] = ta.trend.sma_indicator(df[close_col], window=w)
    for w in config.EMA_WINDOWS: df[f'ema_{w}'] = ta.trend.ema_indicator(df[close_col], window=w)
    df['rsi'] = ta.momentum.rsi(df[close_col], window=config.RSI_WINDOW)
    stoch = ta.momentum.StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=config.STOCH_WINDOW, smooth_window=config.STOCH_SMOOTH)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    bb = ta.volatility.BollingerBands(close=df[close_col], window=config.BB_WINDOW, window_dev=config.BB_STD)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_m'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.average_true_range(high=df[high_col], low=df[low_col], close=df[close_col], window=config.ATR_WINDOW)
    df['cci'] = ta.trend.cci(high=df[high_col], low=df[low_col], close=df[close_col], window=config.CCI_WINDOW)
    df['williams_r'] = ta.momentum.williams_r(high=df[high_col], low=df[low_col], close=df[close_col], lbp=config.WILLIAMS_WINDOW)
    df['vwap'] = ta.volume.volume_weighted_average_price(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[vol_col], window=config.VWAP_WINDOW)
    df['macro_ema'] = ta.trend.ema_indicator(df[close_col], window=config.MACRO_EMA_SPAN)
    
    if 'proxy_close' in df.columns and df['proxy_close'].sum() != 0:
        df['proxy_ret'] = df['proxy_close'].pct_change()
        df['proxy_vol_sma'] = df['proxy_tick_volume'].rolling(window=10).mean()
    else:
        df['proxy_ret'] = 0.0
        df['proxy_vol_sma'] = 0.0

    df.ffill(inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_feature_matrix(df: pd.DataFrame, timeframe: str, use_kalman=True, kalman_model=None):
    # PERBAIKAN: Konversi otomatis spread MT5 ke spread_price saat berjalan di mode Live
    if 'spread_price' not in df.columns:
        if 'spread' in df.columns:
            df['spread_price'] = df['spread'] * 0.001
        else:
            df['spread_price'] = 0.250

    if use_kalman:
        df_filtered, kf_model = apply_kalman_filter(df, model=kalman_model)
        for c in df_filtered.columns: df[c] = df_filtered[c]
    else:
        kf_model = None
    df = add_technical_indicators(df)
    
    feature_cols = [f"{c}_kf" for c in config.KALMAN_COLUMNS] if use_kalman else config.KALMAN_COLUMNS
    tech_cols = ['rsi', 'stoch_k', 'stoch_d', 'atr', 'cci', 'williams_r', 'vwap', 'macro_ema', 'proxy_ret', 'proxy_vol_sma']
    for w in config.SMA_WINDOWS: feature_cols.append(f'sma_{w}')
    for w in config.EMA_WINDOWS: feature_cols.append(f'ema_{w}')
    feature_cols.extend(['bb_h', 'bb_l', 'bb_m'] + tech_cols)
    
    features = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    close_prices = df['close_kf'].values if use_kalman else df['close'].values
    spreads = df['spread_price'].values
    dates = df['datetime'].values
    return features, close_prices, spreads, dates, kf_model, feature_cols

def walk_forward_splits(features, prices, spreads, dates, n_splits=4):
    chunk_size = len(features) // n_splits
    splits = []
    for i in range(1, n_splits):
        train_end = i * chunk_size
        test_end = (i + 1) * chunk_size if i < n_splits - 1 else len(features)
        splits.append((
            (features[:train_end], prices[:train_end], spreads[:train_end], dates[:train_end]),
            (features[train_end:test_end], prices[train_end:test_end], spreads[train_end:test_end], dates[train_end:test_end])
        ))
    return splits