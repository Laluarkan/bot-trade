"""
Script diagnostik: cek ATR & spread per-jam dari data historis, supaya nilai
MIN_SL_PTS / MAX_SL_PTS / MIN_TP_PTS dan TOXIC_HOURS di config.py didasarkan
pada data asli (bukan tebakan), untuk tiap timeframe (M1, M5, dst) yang mau
dipakai.

Cara pakai:
    python check_market_params.py --symbol XAUUSDm --timeframe 5M
    python check_market_params.py --symbol XAUUSDm --timeframe 1M
    python check_market_params.py --symbol XAUUSDm --timeframe 5M --timeframe2 1M   (cek dua-duanya sekaligus)
"""
import argparse
import numpy as np
import pandas as pd
import config
from data_loader import load_csv, build_feature_matrix

PIP_SIZE = 100 * config.POINT  # 1 pip = 100 point di POINT=0.001 (konvensi gold Exness)


def fmt_pip(dollar_value):
    return dollar_value / PIP_SIZE


def analyze_timeframe(symbol: str, timeframe: str):
    print("\n" + "=" * 70)
    print(f"  ANALISIS {symbol} — TIMEFRAME {timeframe}")
    print("=" * 70)

    try:
        df = load_csv(symbol, timeframe)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    features, prices, spreads, dates, kf_model, feature_names, extra = build_feature_matrix(
        df, timeframe, use_kalman=True
    )
    dates = pd.to_datetime(dates)
    atr = pd.Series(extra["atr"])
    # buang bagian awal yang masih warm-up indikator (belum stabil)
    warmup = max(config.MACRO_EMA_SPAN, 500)
    atr = atr.iloc[warmup:].reset_index(drop=True)

    raw_sl = atr * config.SL_ATR_MULTIPLIER
    raw_tp = atr * config.TP_ATR_MULTIPLIER

    print(f"\n[Jumlah candle dipakai (setelah buang warm-up {warmup} bar)]: {len(atr):,}")
    print(f"[Rentang tanggal]: {dates[warmup]} s/d {dates[-1]}")

    def line(label, series_dollar):
        p = series_dollar
        print(
            f"  {label:<22} | $ mean={p.mean():7.3f} median={p.median():7.3f} "
            f"p10={p.quantile(.10):7.3f} p25={p.quantile(.25):7.3f} "
            f"p75={p.quantile(.75):7.3f} p90={p.quantile(.90):7.3f} p95={p.quantile(.95):7.3f} "
            f"| pip mean={fmt_pip(p.mean()):6.1f} p25={fmt_pip(p.quantile(.25)):6.1f} "
            f"p90={fmt_pip(p.quantile(.90)):6.1f}"
        )

    print("\n--- ATR & Rentang SL/TP Dinamis (ATR x multiplier) ---")
    line("ATR mentah", atr)
    line(f"raw_SL (ATRx{config.SL_ATR_MULTIPLIER})", raw_sl)
    line(f"raw_TP (ATRx{config.TP_ATR_MULTIPLIER})", raw_tp)

    print("\n--- Config SEKARANG (di config.py) ---")
    print(f"  MIN_SL_PTS = {config.MIN_SL_PTS:>6} pts = ${config.MIN_SL_PTS*config.POINT:6.2f} = {config.MIN_SL_PTS*config.POINT/PIP_SIZE:6.1f} pip")
    print(f"  MAX_SL_PTS = {config.MAX_SL_PTS:>6} pts = ${config.MAX_SL_PTS*config.POINT:6.2f} = {config.MAX_SL_PTS*config.POINT/PIP_SIZE:6.1f} pip")
    print(f"  MIN_TP_PTS = {config.MIN_TP_PTS:>6} pts = ${config.MIN_TP_PTS*config.POINT:6.2f} = {config.MIN_TP_PTS*config.POINT/PIP_SIZE:6.1f} pip")

    pct_below_floor = (raw_sl < (config.MIN_SL_PTS * config.POINT)).mean() * 100
    pct_above_ceiling = (raw_sl > (config.MAX_SL_PTS * config.POINT)).mean() * 100
    print(f"\n  -> raw_SL di BAWAH floor MIN_SL_PTS pada {pct_below_floor:5.1f}% candle (floor akan dominan di sini)")
    print(f"  -> raw_SL di ATAS ceiling MAX_SL_PTS pada {pct_above_ceiling:5.1f}% candle (ceiling akan dominan di sini)")
    if pct_below_floor > 80:
        print("  [WARNING] Floor SL kemungkinan besar SELALU dominan -> SL efektif jadi statis, bukan adaptif ke ATR.")
    if pct_above_ceiling > 30:
        print("  [WARNING] Ceiling SL sering kepotong -> ATR asli lebih besar dari batas atas yang di-set.")

    print("\n--- SARAN (berdasarkan persentil raw_SL/raw_TP data ini) ---")
    sug_min_sl = raw_sl.quantile(0.25)
    sug_max_sl = raw_sl.quantile(0.90)
    sug_min_tp = raw_tp.quantile(0.25)
    print("  (pakai p25 sbg floor & p90 sbg ceiling -> floor/ceiling dominan di <=25%/10% candle, sisanya benar2 dinamis)")
    print(f"  MIN_SL_PTS ~= {int(sug_min_sl/config.POINT):>6} pts  (${sug_min_sl:5.2f} = {fmt_pip(sug_min_sl):5.1f} pip)")
    print(f"  MAX_SL_PTS ~= {int(sug_max_sl/config.POINT):>6} pts  (${sug_max_sl:5.2f} = {fmt_pip(sug_max_sl):5.1f} pip)")
    print(f"  MIN_TP_PTS ~= {int(sug_min_tp/config.POINT):>6} pts  (${sug_min_tp:5.2f} = {fmt_pip(sug_min_tp):5.1f} pip)")

    # --- Spread per jam ---
    print("\n--- Spread rata-rata per jam (server time di data) ---")
    spread_series = pd.Series(spreads, name="spread_dollar")
    hours = dates.hour
    sdf = pd.DataFrame({"hour": hours, "spread_dollar": spread_series})
    hourly = sdf.groupby("hour")["spread_dollar"].agg(["mean", "median", "count"])
    hourly["mean_pip"] = hourly["mean"] / PIP_SIZE
    overall_mean_pip = hourly["mean_pip"].mean()

    print(f"  {'Jam':>4} | {'Mean(pip)':>10} | {'Median(pip)':>12} | {'N candle':>9} | {'vs rata2':>9} | Status")
    print("  " + "-" * 66)
    for h in range(24):
        if h not in hourly.index:
            continue
        row = hourly.loc[h]
        ratio = row["mean_pip"] / overall_mean_pip if overall_mean_pip > 0 else 1.0
        flagged_now = "[TOXIC_HOURS saat ini]" if h in config.TOXIC_HOURS else ""
        status = "-> SPREAD LEBAR" if ratio >= 1.3 else ""
        print(f"  {h:>4} | {row['mean_pip']:>10.2f} | {row['median']/PIP_SIZE:>12.2f} | {int(row['count']):>9} | {ratio:>8.2f}x | {status} {flagged_now}")

    suggested_toxic = hourly[hourly["mean_pip"] >= overall_mean_pip * 1.3].index.tolist()
    print(f"\n  Rata-rata spread keseluruhan: {overall_mean_pip:.2f} pip")
    print(f"  Jam dengan spread >=1.3x rata-rata (kandidat toxic hour dari DATA): {suggested_toxic}")
    print(f"  TOXIC_HOURS di config.py saat ini                                : {config.TOXIC_HOURS}")
    if set(suggested_toxic) != set(config.TOXIC_HOURS):
        print("  [INFO] Daftar jam di config TIDAK cocok dengan pola spread di data -> pertimbangkan update TOXIC_HOURS di atas, atau ENABLE_TOXIC_HOUR_FILTER=False kalau memang datanya rata.")
    else:
        print("  [OK] Daftar TOXIC_HOURS di config konsisten dengan pola spread di data.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    p.add_argument("--timeframe", default="5M")
    p.add_argument("--timeframe2", default=None, help="Timeframe kedua opsional, mis. 1M, biar dicek sekaligus")
    args = p.parse_args()

    analyze_timeframe(args.symbol, args.timeframe.upper())
    if args.timeframe2:
        analyze_timeframe(args.symbol, args.timeframe2.upper())


if __name__ == "__main__":
    main()