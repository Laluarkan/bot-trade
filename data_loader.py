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
        df['spread_price'] = df['spread'] * config.POINT
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
    """
    STRATEGI BARU: semua fitur yang masuk observation space dibuat RELATIF/
    STASIONER (normalisasi terhadap ATR, persentase, atau sudah bounded),
    bukan level harga absolut. Ini penting karena harga gold trending kuat
    lintas tahun -> fitur level-harga mentah membuat VecNormalize (running
    mean/std) selalu "kejar-kejaran" dengan level harga baru, sehingga model
    kehilangan sensitivitas persis saat market sedang trending / rezim baru
    (gejala "tidak adaptif" yang dikeluhkan).

    Kolom mentah (open/high/low/close/volume _kf) TETAP disimpan di df untuk
    dipakai kalkulasi harga & PnL (kolom 'close_kf' dipakai env sbg harga
    eksekusi), tapi TIDAK dimasukkan langsung ke feature_cols.
    """
    close_col = 'close_kf' if 'close_kf' in df.columns else 'close'
    high_col = 'high_kf' if 'high_kf' in df.columns else 'high'
    low_col = 'low_kf' if 'low_kf' in df.columns else 'low'
    open_col = 'open_kf' if 'open_kf' in df.columns else 'open'
    vol_col = 'tick_volume_kf' if 'tick_volume_kf' in df.columns else 'tick_volume'

    eps = 1e-8

    # --- ATR dihitung duluan: jadi basis normalisasi untuk fitur2 lain ---
    df['atr'] = ta.volatility.average_true_range(high=df[high_col], low=df[low_col], close=df[close_col], window=config.ATR_WINDOW)
    atr_safe = df['atr'].replace(0, np.nan).bfill().ffill().fillna(eps) + eps

    # --- 1. Return & candle shape (stasioner) ---
    df['log_ret'] = np.log(df[close_col].clip(lower=eps) / df[close_col].shift(1).clip(lower=eps))
    df['candle_body_norm'] = (df[close_col] - df[open_col]) / atr_safe
    df['candle_range_norm'] = (df[high_col] - df[low_col]) / atr_safe
    df['upper_wick_norm'] = (df[high_col] - df[[close_col, open_col]].max(axis=1)) / atr_safe
    df['lower_wick_norm'] = (df[[close_col, open_col]].min(axis=1) - df[low_col]) / atr_safe

    # --- 2. Moving averages -> jarak relatif ke ATR, bukan level absolut ---
    for w in config.SMA_WINDOWS:
        sma = ta.trend.sma_indicator(df[close_col], window=w)
        df[f'sma_{w}_dist'] = (df[close_col] - sma) / atr_safe
    for w in config.EMA_WINDOWS:
        ema = ta.trend.ema_indicator(df[close_col], window=w)
        df[f'ema_{w}_dist'] = (df[close_col] - ema) / atr_safe

    # jarak antar EMA cepat/lambat (momentum jangka pendek)
    ema_fast = ta.trend.ema_indicator(df[close_col], window=config.EMA_WINDOWS[0])
    ema_slow = ta.trend.ema_indicator(df[close_col], window=config.EMA_WINDOWS[1])
    df['ema_cross_norm'] = (ema_fast - ema_slow) / atr_safe

    # --- 3. Oscillator yang sudah bounded -> tinggal di-skala ke [-1, 1] ---
    df['rsi_norm'] = (ta.momentum.rsi(df[close_col], window=config.RSI_WINDOW) - 50.0) / 50.0
    stoch = ta.momentum.StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=config.STOCH_WINDOW, smooth_window=config.STOCH_SMOOTH)
    df['stoch_k_norm'] = (stoch.stoch() - 50.0) / 50.0
    df['stoch_d_norm'] = (stoch.stoch_signal() - 50.0) / 50.0
    df['cci_norm'] = (ta.trend.cci(high=df[high_col], low=df[low_col], close=df[close_col], window=config.CCI_WINDOW) / 200.0).clip(-3, 3)
    df['williams_r_norm'] = (ta.momentum.williams_r(high=df[high_col], low=df[low_col], close=df[close_col], lbp=config.WILLIAMS_WINDOW) + 50.0) / 50.0

    # --- 4. Bollinger Bands -> posisi relatif dalam band (0..1, center 0) ---
    bb = ta.volatility.BollingerBands(close=df[close_col], window=config.BB_WINDOW, window_dev=config.BB_STD)
    bb_h, bb_l = bb.bollinger_hband(), bb.bollinger_lband()
    bb_width = (bb_h - bb_l).replace(0, np.nan)
    df['bb_pos'] = (((df[close_col] - bb_l) / bb_width) * 2.0 - 1.0).clip(-3, 3)
    df['bb_width_norm'] = (bb_width / atr_safe).fillna(0.0)

    # --- 5. VWAP -> jarak relatif ke ATR ---
    vwap = ta.volume.volume_weighted_average_price(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[vol_col], window=config.VWAP_WINDOW)
    df['vwap_dist'] = (df[close_col] - vwap) / atr_safe

    # --- 6. Tren makro (EMA-500) -> disimpan sbg %, dipakai juga di env sbg filter ---
    macro_ema = ta.trend.ema_indicator(df[close_col], window=config.MACRO_EMA_SPAN)
    df['macro_ema'] = macro_ema  # disimpan mentah untuk dipakai trading_env.py sbg filter tren
    df['macro_dev_pct'] = ((df[close_col] - macro_ema) / macro_ema.replace(0, np.nan)) * 100.0
    df['macro_slope'] = (macro_ema.diff(10) / macro_ema.replace(0, np.nan)).fillna(0.0) * 100.0

    # --- 7. Volume -> z-score rolling (relatif, bukan level absolut) ---
    vol_mean = df[vol_col].rolling(window=config.VOL_ZSCORE_WINDOW).mean()
    vol_std = df[vol_col].rolling(window=config.VOL_ZSCORE_WINDOW).std().replace(0, np.nan)
    df['volume_zscore'] = ((df[vol_col] - vol_mean) / vol_std).clip(-5, 5)

    # --- 8. Proxy symbol (korelasi antar-pasar) -> sudah relatif ---
    if 'proxy_close' in df.columns and df['proxy_close'].sum() != 0:
        df['proxy_ret'] = df['proxy_close'].pct_change()
        proxy_vol_mean = df['proxy_tick_volume'].rolling(window=config.VOL_ZSCORE_WINDOW).mean()
        proxy_vol_std = df['proxy_tick_volume'].rolling(window=config.VOL_ZSCORE_WINDOW).std().replace(0, np.nan)
        df['proxy_vol_zscore'] = ((df['proxy_tick_volume'] - proxy_vol_mean) / proxy_vol_std).clip(-5, 5)
    else:
        df['proxy_ret'] = 0.0
        df['proxy_vol_zscore'] = 0.0

    # --- 9. Waktu (siklikal, membantu model mengenali sesi/toxic hours sendiri) ---
    if 'datetime' in df.columns:
        hour = pd.to_datetime(df['datetime']).dt.hour
        dow = pd.to_datetime(df['datetime']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    else:
        df['hour_sin'] = 0.0; df['hour_cos'] = 0.0
        df['dow_sin'] = 0.0; df['dow_cos'] = 0.0

    df.ffill(inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_feature_matrix(df: pd.DataFrame, timeframe: str, use_kalman=True, kalman_model=None):
    if 'spread_price' not in df.columns:
        if 'spread' in df.columns:
            df['spread_price'] = df['spread'] * config.POINT
        else:
            df['spread_price'] = 0.250

    if use_kalman:
        df_filtered, kf_model = apply_kalman_filter(df, model=kalman_model)
        for c in df_filtered.columns: df[c] = df_filtered[c]
    else:
        kf_model = None
        # kalau tidak pakai kalman, tetap sediakan kolom *_kf sbg alias supaya
        # add_technical_indicators konsisten
        for c in config.KALMAN_COLUMNS:
            df[f"{c}_kf"] = df[c]

    df = add_technical_indicators(df)

    # feature_cols SEKARANG SEMUA RELATIF/STASIONER (lihat add_technical_indicators)
    feature_cols = [
        'log_ret', 'candle_body_norm', 'candle_range_norm', 'upper_wick_norm', 'lower_wick_norm',
    ]
    for w in config.SMA_WINDOWS: feature_cols.append(f'sma_{w}_dist')
    for w in config.EMA_WINDOWS: feature_cols.append(f'ema_{w}_dist')
    feature_cols += [
        'ema_cross_norm', 'rsi_norm', 'stoch_k_norm', 'stoch_d_norm', 'cci_norm', 'williams_r_norm',
        'bb_pos', 'bb_width_norm', 'vwap_dist', 'macro_dev_pct', 'macro_slope',
        'volume_zscore', 'proxy_ret', 'proxy_vol_zscore',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    raw_features = df[feature_cols].values.astype(np.float64)
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
    # clip akhir supaya outlier ekstrem (mis. gap besar) tidak merusak statistik VecNormalize
    features = np.clip(raw_features, -config.FEATURE_CLIP, config.FEATURE_CLIP)

    close_prices = df['close_kf'].values if use_kalman else df['close'].values
    spreads = df['spread_price'].values
    dates = df['datetime'].values

    # dikembalikan tambahan: macro_ema/trend_ema/macro_slope, dipakai trading_env.py
    # sbg filter tren (bukan fitur observasi, tapi sinyal kontrol). Dihitung SEKALI
    # di sini dengan konsisten baik untuk training (histori penuh) maupun live
    # (window LIVE_LOOKBACK_BARS) -> tidak ada lagi duplikasi logika di trading_env.py
    # yang bisa membuat train & live melihat sinyal tren yang berbeda.
    close_col_final = 'close_kf' if use_kalman else 'close'
    trend_ema = ta.trend.ema_indicator(df[close_col_final], window=config.TREND_EMA_SPAN)
    extra = {
        'macro_ema': df['macro_ema'].values,
        'macro_slope': df['macro_slope'].values,
        'trend_ema': np.nan_to_num(trend_ema.values, nan=0.0),
        'atr': df['atr'].values,
    }
    return features, close_prices, spreads, dates, kf_model, feature_cols, extra

def walk_forward_splits(features, prices, spreads, dates, extra=None, n_splits=4):
    chunk_size = len(features) // n_splits
    splits = []
    for i in range(1, n_splits):
        train_end = i * chunk_size
        test_end = (i + 1) * chunk_size if i < n_splits - 1 else len(features)

        def _slice_extra(a, b):
            if extra is None:
                return None
            return {k: v[a:b] for k, v in extra.items()}

        splits.append((
            (features[:train_end], prices[:train_end], spreads[:train_end], dates[:train_end], _slice_extra(0, train_end)),
            (features[train_end:test_end], prices[train_end:test_end], spreads[train_end:test_end], dates[train_end:test_end], _slice_extra(train_end, test_end))
        ))
    return splits