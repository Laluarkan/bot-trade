"""
random_baseline.py
===================
Menjalankan kebijakan ACAK (BUY/SELL/FLAT seragam) di simulator yang sama persis
dengan backtest -- aturan MIN_HOLD, cooldown, SL/TP, spread, slippage semuanya
sama -- pada dua jenis harga eksekusi: "kalman" dan "raw".

Gunanya: memberi tahu seberapa besar kerugian yang SEMATA-MATA disebabkan biaya
dan mekanik simulator. Bandingkan dengan hasil backtest modelmu:

  * model jauh LEBIH BAIK dari acak  -> model punya kemampuan, tapi mungkin tipis
  * model SAMA dengan acak           -> model tidak punya kemampuan sama sekali
  * model LEBIH BURUK dari acak      -> model aktif merugikan (ia belajar pola palsu)

Cara pakai (satu timeframe, model/artifacts tidak wajib ada):
    python random_baseline.py --timeframe 1M --episodes 30 --seed 42
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import config
from data_loader import load_csv, build_feature_matrix, walk_forward_splits
from trading_env import GoldTradingEnv


def run(timeframe, symbol, episodes, seed, price_source, kalman_model):
    config.EXECUTION_PRICE_SOURCE = price_source
    df = load_csv(symbol, timeframe, data_dir=config.DATA_DIR)
    f, p, s, d, _, _, extra = build_feature_matrix(df, timeframe, use_kalman=True, kalman_model=kalman_model)
    splits = walk_forward_splits(f, p, s, d, extra=extra, n_splits=4)
    _, (tf_, tp_, ts_, td_, te_) = splits[-1]          # kuartal terakhir = sama seperti `main.py backtest`
    env = GoldTradingEnv(tf_, tp_, ts_, dates=td_, extra=te_, episode_length=config.EPISODE_LENGTH,
                         random_start=True, use_margin_call=True, timeframe=timeframe, eval_seed=seed)
    rng = np.random.default_rng(seed)
    pnls, trades = [], []
    for _ in range(episodes):
        env.reset()
        done, n_tr, prev = False, 0, 0.0
        while not done:
            _, _, term, trunc, info = env.step(int(rng.integers(0, 3)))
            done = term or trunc
            if info["position"] != prev and info["position"] != 0.0:
                n_tr += 1
            prev = info["position"]
        pnls.append(info["portfolio_value"] - config.INITIAL_BALANCE)
        trades.append(n_tr)
    return np.array(pnls), np.array(trades), pd.to_datetime(td_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1M")
    ap.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tf = a.timeframe.upper()
    config.apply_timeframe_profile(tf)

    # Pakai parameter Kalman milik model kalau ada, supaya harga "kalman" sama dengan backtest.
    kalman_model = None
    art = config.MODEL_DIR / f"{config.MODEL_PREFIX}_{a.symbol}_{tf}_artifacts.pkl"
    if art.exists():
        with open(art, "rb") as fh:
            kalman_model = pickle.load(fh)["kalman_model"]
        print(f"[*] Memakai parameter Kalman dari {art.name}")
    else:
        print(f"[!] {art.name} tidak ada -> Kalman di-fit ulang (hasil harga 'kalman' sedikit berbeda dari backtest)")

    print(f"\nKebijakan ACAK | {tf} | {a.episodes} episode | seed {a.seed} | slippage {config.SLIPPAGE_PTS} pts\n")
    print(f"{'harga eksekusi':<16}{'total P&L':>12}{'rata2/episode':>15}{'median':>10}{'% episode +':>13}{'trade/episode':>15}")
    out = {}
    for src in ("kalman", "raw"):
        pn, tr, dts = run(tf, a.symbol, a.episodes, a.seed, src, kalman_model)
        out[src] = pn
        print(f"{src:<16}{pn.sum():>12.2f}{pn.mean():>15.2f}{np.median(pn):>10.2f}{(pn > 0).mean() * 100:>12.1f}%{tr.mean():>15.0f}")
    print(f"\nRentang data test: {dts.min()} s/d {dts.max()}")
    print("\nBandingkan 'total P&L' baris yang sesuai dengan hasil `main.py backtest` modelmu")
    print("(jumlah episode dan seed yang sama). Lihat penjelasan skor di docstring file ini.")


if __name__ == "__main__":
    main()