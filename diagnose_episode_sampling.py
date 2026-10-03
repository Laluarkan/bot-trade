"""
diagnose_episode_sampling.py
=============================
Menguji LANGSUNG GoldTradingEnv.reset() berkali-kali (seperti yang dilakukan
evaluate_model lewat 10 episode), TANPA model/VecNormalize sama sekali, supaya
kita tahu apakah start_idx yang disampel memang bias ke bagian awal data,
atau masalahnya ada di tempat lain (mis. evaluate_model/VecNormalize).

Cara pakai:
    python diagnose_episode_sampling.py
"""
import pandas as pd
import numpy as np
import config
from data_loader import load_csv, build_feature_matrix, walk_forward_splits
from trading_env import GoldTradingEnv

SYMBOL = config.DEFAULT_SYMBOL
TF = "1M"
EPISODE_LENGTH = config.EPISODE_LENGTH

df = load_csv(SYMBOL, TF, data_dir=config.DATA_DIR)
features, prices, spreads, dates, kf_model, feature_cols, extra = build_feature_matrix(df, TF, use_kalman=True)
splits = walk_forward_splits(features, prices, spreads, dates, extra=extra, n_splits=4)
_, (test_f, test_p, test_s, test_d, test_e) = splits[-1]

dates_pd = pd.to_datetime(test_d)
n_steps = len(test_f)
max_start_expected = max(1, n_steps - EPISODE_LENGTH - 1)
print(f"Test chunk: n_steps={n_steps}, episode_length={EPISODE_LENGTH}")
print(f"max_start yang DIHARAPKAN = {max_start_expected}")
print(f"Tanggal test chunk: {dates_pd.min()} -> {dates_pd.max()}")
print()

for seed in [42, 7, 123]:
    print(f"--- eval_seed={seed} ---")
    env = GoldTradingEnv(
        test_f, test_p, test_s, dates=test_d, extra=test_e,
        episode_length=EPISODE_LENGTH, random_start=True, use_margin_call=True,
        timeframe=TF, eval_seed=seed,
    )
    print(f"  max_start AKTUAL di env: {max(1, env.n_steps - env.episode_length - 1)}  (env.n_steps={env.n_steps})")
    for ep in range(10):
        obs, _ = env.reset()
        start_idx = env.start_idx
        end_idx = env.end_idx
        start_date = dates_pd[start_idx] if start_idx < len(dates_pd) else None
        end_date = dates_pd[end_idx] if end_idx < len(dates_pd) else None
        pct_through = start_idx / max(1, env.n_steps - env.episode_length - 1) * 100
        print(f"  Eps {ep+1}: start_idx={start_idx:>6} ({pct_through:5.1f}% dari range) | {start_date}  ->  {end_date}")
    print()