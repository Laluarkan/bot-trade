"""
diagnose_data_range.py
=======================
Jalankan ini di folder yang sama dengan main.py untuk mencari TEPAT di tahap
mana rentang tanggal data terpotong di sekitar 9-11 September, padahal CSV
mentah sudah sampai 1 Oktober.

Cara pakai:
    python diagnose_data_range.py
"""
import pandas as pd
from pathlib import Path
import config
from data_loader import load_csv, build_feature_matrix, walk_forward_splits

SYMBOL = config.DEFAULT_SYMBOL
TF = "1M"

print("=" * 70)
print("STEP 0: Baca CSV mentah langsung (tanpa lewat load_csv)")
print("=" * 70)
raw_main = pd.read_csv(Path(config.DATA_DIR) / f"{SYMBOL}_{TF}.csv")
raw_main['datetime'] = pd.to_datetime(raw_main['datetime'])
print(f"XAU mentah : {raw_main['datetime'].min()}  ->  {raw_main['datetime'].max()}  (n={len(raw_main)})")
dup_main = raw_main['datetime'].duplicated().sum()
print(f"  Baris duplikat timestamp di XAU mentah : {dup_main}")
gaps_main = raw_main['datetime'].diff().dt.total_seconds().div(60)
big_gaps_main = gaps_main[gaps_main > 5]
print(f"  Jumlah gap > 5 menit di XAU mentah : {len(big_gaps_main)}")
if len(big_gaps_main) > 0:
    top_gaps_idx = big_gaps_main.sort_values(ascending=False).head(10).index
    print("  10 gap terbesar di XAU (menit, dan kapan terjadi):")
    for idx in top_gaps_idx:
        gap_start = raw_main['datetime'].iloc[idx-1]
        gap_end = raw_main['datetime'].iloc[idx]
        print(f"    {gaps_main[idx]:>10.0f} menit  |  {gap_start}  ->  {gap_end}")

proxy_path = Path(config.DATA_DIR) / f"{config.PROXY_SYMBOL}_{TF}.csv"
if proxy_path.exists():
    raw_proxy = pd.read_csv(proxy_path)
    raw_proxy['datetime'] = pd.to_datetime(raw_proxy['datetime'])
    print(f"Proxy mentah ({config.PROXY_SYMBOL}): {raw_proxy['datetime'].min()}  ->  {raw_proxy['datetime'].max()}  (n={len(raw_proxy)})")
    dup_proxy = raw_proxy['datetime'].duplicated().sum()
    print(f"  Baris duplikat timestamp di proxy mentah : {dup_proxy}")
    gaps = raw_proxy['datetime'].diff().dt.total_seconds().div(60)
    big_gaps = gaps[gaps > 5]
    print(f"  Jumlah gap > 5 menit di proxy mentah : {len(big_gaps)}")
    if len(big_gaps) > 0:
        print(f"  5 gap terbesar (menit): {sorted(big_gaps.tolist(), reverse=True)[:5]}")
else:
    print(f"[!] File proxy {proxy_path} TIDAK DITEMUKAN.")

print()
print("=" * 70)
print("STEP 1: load_csv() -- setelah join proxy + trim awal")
print("=" * 70)
df = load_csv(SYMBOL, TF, data_dir=config.DATA_DIR)
print(f"Setelah load_csv : {df['datetime'].min()}  ->  {df['datetime'].max()}  (n={len(df)})")
dup_after_join = df['datetime'].duplicated().sum()
print(f"  Baris duplikat timestamp setelah join : {dup_after_join}")
zero_proxy = (df['proxy_close'] == 0.0).sum()
print(f"  Baris dengan proxy_close == 0.0 : {zero_proxy} / {len(df)} ({zero_proxy/len(df)*100:.1f}%)")

print()
print("=" * 70)
print("STEP 2: build_feature_matrix() -- setelah indikator & kalman")
print("=" * 70)
features, prices, spreads, dates, kf_model, feature_cols, extra = build_feature_matrix(df, TF, use_kalman=True)
dates_pd = pd.to_datetime(dates)
print(f"Setelah build_feature_matrix : {dates_pd.min()}  ->  {dates_pd.max()}  (n={len(dates_pd)})")
if len(dates_pd) != len(df):
    print(f"  [!] JUMLAH BARIS BERUBAH: load_csv={len(df)} vs build_feature_matrix={len(dates_pd)}")

print()
print("=" * 70)
print("STEP 3: walk_forward_splits() -- batas tanggal tiap kuartal")
print("=" * 70)
splits = walk_forward_splits(features, prices, spreads, dates, extra=extra, n_splits=4)
for i, ((train_f, train_p, train_s, train_d, train_e), (test_f, test_p, test_s, test_d, test_e)) in enumerate(splits):
    test_d_pd = pd.to_datetime(test_d)
    print(f"Split {i+1}: TRAIN n={len(train_f):>6} | TEST n={len(test_f):>6}  "
          f"TEST range: {test_d_pd.min()}  ->  {test_d_pd.max()}")

print()
print("=" * 70)
print("KESIMPULAN")
print("=" * 70)
last_test_end = pd.to_datetime(splits[-1][1][3]).max()
print(f"Tanggal terakhir di TEST split paling akhir (yang dipakai `main.py backtest`): {last_test_end}")
print(f"Tanggal terakhir di CSV mentah XAU: {raw_main['datetime'].max()}")
if (raw_main['datetime'].max() - last_test_end).days > 3:
    print("[!] ADA SELISIH SIGNIFIKAN -> data hilang/terpotong di salah satu tahap di atas.")
    print("    Lihat STEP mana yang nilainya sudah telat sebelum STEP ini.")
else:
    print("[OK] Rentang tanggal konsisten sampai akhir -- backtest/evaluate_model")
    print("     kemungkinan cuma kebetulan tidak sampling episode ke bagian paling akhir.")