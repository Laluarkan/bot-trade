"""
edge_scan.py  --  Fase 1: CARI EDGE SEBELUM MENGOPTIMASI APA PUN
=================================================================
Pertanyaan yang dijawab: adakah fitur HARGA MENTAH (tanpa Kalman, tanpa RL) yang
memprediksi return ke depan LEBIH BESAR dari biaya trade, secara out-of-sample?

Cara kerja (ringkas)
--------------------
* Fitur kausal dari OHLC/volume mentah (+ EURUSD kalau ada). Tidak ada Kalman.
* Return ke depan dihitung dari close mentah (bid). Biaya per trade = 1x spread di bar
  masuk (long beli di ask / short tutup di ask -> 1x spread per round trip) + 2x slippage.
* Walk-forward EXPANDING dengan embargo: tiap kombinasi (fitur x regime x horizon) dilatih di
  fold-latih (hanya untuk menentukan ARAH hubungan dan ambang kuantil), lalu dievaluasi di
  fold-uji yang belum pernah dilihat. Hasil semua fold uji digabung.
* t-stat memakai batch-means (blok = panjang horizon) supaya return yang saling tumpang tindih
  tidak menggelembungkan signifikansi.
* Banyak kombinasi diuji sekaligus -> koreksi Benjamini-Hochberg (FDR 5%). Jumlah uji dihitung
  dari SEMUA kombinasi yang dicoba, termasuk yang dilewati.
* Data uji TERKUNCI (holdout) tidak disentuh saat pemindaian. Konfirmasi hanya lewat
  --confirm --unlock-holdout, dan hanya BISA SEKALI per timeframe (ada berkas penanda).

Cara pakai
----------
    python edge_scan.py --timeframe 5M                      # pemindaian
    python edge_scan.py --timeframe 5M --horizons 30 60 120 240
    python edge_scan.py --timeframe 5M --confirm ret5:V2_tinggi:60 --unlock-holdout

Yang TIDAK dilakukan skrip ini (batasan jujur)
---------------------------------------------
* Bukan simulasi portofolio: sinyal pada bar berdekatan saling tumpang tindih, dan tidak ada
  manajemen posisi/SL/TP. Ini detektor edge, bukan backtest strategi.
* Entry diasumsikan di close bar sinyal (selaras dengan hasil live: slippage masuk terukur ~0).
* "Lolos" di sini artinya LAYAK diteliti lebih lanjut (Fase 2), bukan siap trading.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data_loader import load_csv

WARMUP = 300           # bar awal dibuang (indikator belum stabil)
MIN_TRAIN = 500
MIN_TEST = 100


# --------------------------------------------------------------------------- fitur
def build_features(df: pd.DataFrame, use_proxy: bool):
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["tick_volume"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr100 = tr.rolling(100).mean()
    sgn = np.sign(c.diff())

    F = {}
    for k in (1, 3, 5, 10, 20, 60):
        F[f"ret{k}"] = (c - c.shift(k)) / atr
    for k in (20, 60, 200):
        F[f"zscore{k}"] = (c - c.rolling(k).mean()) / c.rolling(k).std()
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    F["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50
    F["stoch14"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()) - 0.5
    for k in (20, 50, 200):
        F[f"emadist{k}"] = (c - c.ewm(span=k, adjust=False).mean()) / atr
    e50 = c.ewm(span=50, adjust=False).mean()
    F["emaslope50"] = (e50 - e50.shift(10)) / atr
    F["rangepos60"] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()) - 0.5
    F["volsigned"] = sgn * (v - v.rolling(50).mean()) / v.rolling(50).std()
    F["rangeexp_signed"] = sgn * (tr / atr100)
    day = df["datetime"].dt.date
    F["dayopen_dist"] = (c - c.groupby(day).transform("first")) / atr
    if use_proxy and df["proxy_close"].abs().sum() > 0:
        pcl = df["proxy_close"].astype(float)
        for k in (1, 5, 20):
            r = pcl - pcl.shift(k)
            F[f"eur_ret{k}"] = r / r.rolling(500).std()
    F["drift"] = pd.Series(1.0, index=c.index)      # tebak arah searah rata-rata train (musiman/drift)

    out = {}
    for name, s in F.items():
        a = s.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float).copy()
        a[:WARMUP] = np.nan
        out[name] = a
    return out, atr


def build_regimes(df: pd.DataFrame, atr: pd.Series):
    hr = df["datetime"].dt.hour.to_numpy()
    q33 = atr.rolling(2000, min_periods=500).quantile(1 / 3).to_numpy()
    q66 = atr.rolling(2000, min_periods=500).quantile(2 / 3).to_numpy()
    a = atr.to_numpy()
    with np.errstate(invalid="ignore"):
        R = {
            "all": np.ones(len(df), dtype=bool),
            "S0_00-06": hr < 6,
            "S1_06-12": (hr >= 6) & (hr < 12),
            "S2_12-18": (hr >= 12) & (hr < 18),
            "S3_18-24": hr >= 18,
            "V0_rendah": a <= q33,
            "V1_sedang": (a > q33) & (a <= q66),
            "V2_tinggi": a > q66,
        }
    return R


# --------------------------------------------------------------------------- statistik
def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _signed_ic(f, fwd):
    if len(f) < 50:
        return 0.0
    rf, ry = _rank(f), _rank(fwd)
    rf -= rf.mean(); ry -= ry.mean()
    den = math.sqrt((rf ** 2).sum() * (ry ** 2).sum())
    return float((rf * ry).sum() / den) if den > 0 else 0.0


def ratio_t(idx, x, h):
    """Rata-rata per trade + t-stat via batch-means (blok = h bar) untuk data yang tumpang tindih."""
    if len(idx) < 30:
        return float("nan"), float("nan"), float("nan")
    blk = idx // max(int(h), 1)
    _, inv = np.unique(blk, return_inverse=True)
    B = inv.max() + 1
    X = np.bincount(inv, weights=x, minlength=B)
    S = np.bincount(inv, minlength=B).astype(float)
    R = X.sum() / S.sum()
    z = X - R * S
    if B < 3:
        return float(R), float("nan"), float("nan")
    se = math.sqrt((B / (B - 1)) * (z ** 2).sum()) / S.sum()
    return float(R), (float(R / se) if se > 0 else float("nan")), float(se)


def sf(t):
    return 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else 1.0


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        i = order[rank_i]
        prev = min(prev, p[i] * m / (rank_i + 1))
        q[i] = prev
    return q


# --------------------------------------------------------------------------- inti evaluasi
def eval_combo(name, f, fwd, cost, valid, regime, train_m, test_m, q):
    """Latih (arah+ambang) di train_m, nilai di test_m. Mengembalikan (idx, gross, net) atau None."""
    base = valid & regime & np.isfinite(f)
    tr = train_m & base
    te = test_m & base
    if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
        return None
    if name == "drift":
        s = 1.0 if fwd[tr].mean() >= 0 else -1.0
        idx = np.flatnonzero(te)
        sig = np.full(len(idx), s)
    else:
        ic = _signed_ic(f[tr], fwd[tr])
        s = 1.0 if ic >= 0 else -1.0
        lo, hi = np.quantile(f[tr], [q, 1 - q])
        if not (hi > lo):
            return None
        fl = f[te]
        long_ = (fl >= hi) if s > 0 else (fl <= lo)
        short_ = (fl <= lo) if s > 0 else (fl >= hi)
        take = long_ | short_
        idx = np.flatnonzero(te)[take]
        sig = np.where(long_[take], 1.0, -1.0)
    if len(idx) == 0:
        return None
    gross = sig * fwd[idx]
    return idx, gross, gross - cost[idx]


def summarize(parts, h):
    idx = np.concatenate([p[0] for p in parts])
    gross = np.concatenate([p[1] for p in parts])
    net = np.concatenate([p[2] for p in parts])
    order = np.argsort(idx)
    idx, gross, net = idx[order], gross[order], net[order]
    gm, gt, _ = ratio_t(idx, gross, h)
    nm, nt, nse = ratio_t(idx, net, h)
    return dict(n=len(idx), gross=gm, gross_t=gt, net=nm, net_t=nt, net_se=nse,
                win=float((net > 0).mean() * 100), cost=float((gross - net).mean()))


# --------------------------------------------------------------------------- program utama
def main():
    ap = argparse.ArgumentParser(description="Pemindai edge (Fase 1)")
    ap.add_argument("--timeframe", default="5M")
    ap.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="Horizon dalam MENIT (default: 15 30 60 120)")
    ap.add_argument("--folds", type=int, default=4, help="Jumlah fold uji walk-forward")
    ap.add_argument("--tail", type=float, default=0.2, help="Fraksi ekor kuantil yang diperdagangkan (0.2 = 20%%)")
    ap.add_argument("--holdout-days", type=int, default=90, help="Hari terakhir yang DIKUNCI")
    ap.add_argument("--min-trades", type=int, default=200)
    ap.add_argument("--slippage-pts", type=float, default=None, help="Default: config.SLIPPAGE_PTS")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--confirm", default=None,
                    help="Daftar 'fitur:regime:menit' dipisah koma, dievaluasi di holdout")
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--force", action="store_true", help="Abaikan penanda holdout (merusak integritas uji!)")
    a = ap.parse_args()

    tf = a.timeframe.upper()
    tf_min = config.TIMEFRAME_MINUTES[tf]
    horizons_min = a.horizons or [15, 30, 60, 120]
    horizons = sorted({max(1, int(round(m / tf_min))) for m in horizons_min})
    slip = (config.SLIPPAGE_PTS if a.slippage_pts is None else a.slippage_pts) * config.POINT

    df = load_csv(a.symbol, tf, data_dir=a.data_dir).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    N = len(df)
    times = df["datetime"].to_numpy()
    close = df["close"].to_numpy(dtype=float)
    spread = (df["spread_price"] if "spread_price" in df.columns else df["spread"] * config.POINT).to_numpy(dtype=float)
    cost = spread + 2.0 * slip                               # $/oz per trade (1x spread + slippage dua sisi)

    holdout_start = int(np.searchsorted(times, times[-1] - np.timedelta64(a.holdout_days, "D")))
    scan_end = holdout_start
    feats, atr = build_features(df, use_proxy=not a.no_proxy)
    regimes = build_regimes(df, atr)

    print("=" * 74)
    print(f" PEMINDAI EDGE  |  {a.symbol} {tf}  |  {N:,} bar")
    print("=" * 74)
    print(f" Data        : {pd.Timestamp(times[0])}  ->  {pd.Timestamp(times[-1])}")
    print(f" Pemindaian  : s/d {pd.Timestamp(times[scan_end - 1])}  ({scan_end:,} bar)")
    print(f" TERKUNCI    : {pd.Timestamp(times[holdout_start])}  ->  akhir  ({N - holdout_start:,} bar, {a.holdout_days} hari)")
    print(f" Biaya       : spread rata-rata ${spread.mean():.3f}/oz + slippage 2x${slip:.3f}  = ${cost.mean():.3f}/oz per trade")
    print(f" Horizon     : {[h * tf_min for h in horizons]} menit  ({horizons} bar)")

    # ---- gerbang fwd per horizon
    fwd_h, valid_h = {}, {}
    ar = np.arange(N)
    for h in horizons:
        fw = np.full(N, np.nan)
        fw[:-h] = close[h:] - close[:-h]
        dtm = np.full(N, np.nan)
        dtm[:-h] = (times[h:] - times[:-h]) / np.timedelta64(1, "m")
        ok = np.isfinite(fw) & (dtm <= h * tf_min * 1.6)    # buang lompatan weekend/libur
        fwd_h[h], valid_h[h] = fw, ok

    # ------------------------------------------------------------------ MODE KONFIRMASI
    if a.confirm:
        marker = config.RESULTS_DIR / f"edge_scan_holdout_used_{tf}.json"
        if not a.unlock_holdout:
            sys.exit("\n[!] Data terkunci. Tambahkan --unlock-holdout kalau kamu benar-benar siap (HANYA SEKALI).")
        if marker.exists() and not a.force:
            sys.exit(f"\n[!] Holdout {tf} SUDAH PERNAH dibuka ({marker.name}). Memakainya lagi berarti "
                     f"memilih-milih di data uji. Hentikan, atau --force kalau kamu sadar risikonya.")
        items = [x.strip() for x in a.confirm.split(",") if x.strip()]
        print(f"\n KONFIRMASI di data terkunci untuk {len(items)} kandidat "
              f"(koreksi Bonferroni x{len(items)}):\n")
        print(f" {'kandidat':<34}{'n':>7}{'bruto':>9}{'biaya':>8}{'neto':>9}{'t':>7}{'p-adj':>8}  verdict")
        results = []
        for it in items:
            fname, rname, mm = it.split(":")
            h = max(1, int(round(int(mm) / tf_min)))
            if h not in fwd_h:
                sys.exit(f"[!] Horizon {mm} menit tidak ada di daftar --horizons.")
            train_m = ar < (scan_end - h)
            test_m = ar >= holdout_start
            r = eval_combo(fname, feats[fname], fwd_h[h], cost, valid_h[h], regimes[rname], train_m, test_m, a.tail)
            if r is None:
                print(f" {it:<34}  (data tidak cukup)")
                continue
            s = summarize([r], h)
            padj = min(1.0, sf(s["net_t"]) * len(items))
            ok = (s["net"] > 0) and (padj < 0.05)
            results.append(dict(kandidat=it, **s, p_adj=padj, lolos=ok))
            print(f" {it:<34}{s['n']:>7}{s['gross']:>+9.3f}{s['cost']:>8.3f}{s['net']:>+9.3f}{s['net_t']:>7.2f}{padj:>8.3f}  "
                  f"{'LOLOS' if ok else 'GAGAL'}")
        marker.write_text(json.dumps(dict(dibuka=datetime.now().isoformat(timespec="seconds"),
                                          timeframe=tf, kandidat=items, hasil=results), indent=2), encoding="utf-8")
        print(f"\n Penanda holdout ditulis: {marker}. Data terkunci SUDAH TERPAKAI untuk timeframe ini.")
        return

    # ------------------------------------------------------------------ MODE PEMINDAIAN
    K = a.folds
    edges = np.linspace(WARMUP, scan_end, K + 2).astype(int)
    rows = []
    n_attempt = len(feats) * len(regimes) * len(horizons)
    print(f" Fitur       : {len(feats)}  | Regime: {len(regimes)}  | Jumlah uji: {n_attempt:,}  "
          f"(dikoreksi Benjamini-Hochberg)")
    print(f" Walk-forward: {K} fold uji, expanding, embargo = horizon\n")
    t0 = time.time()
    done = 0
    for h in horizons:
        fw, ok_h = fwd_h[h], valid_h[h] & (ar + h < scan_end)
        for fname, f in feats.items():
            for rname, rmask in regimes.items():
                parts, fold_net = [], []
                for i in range(1, K + 1):
                    train_m = (ar >= WARMUP) & (ar < edges[i] - h)
                    test_m = (ar >= edges[i]) & (ar < edges[i + 1])
                    r = eval_combo(fname, f, fw, cost, ok_h, rmask, train_m, test_m, a.tail)
                    if r is not None:
                        parts.append(r)
                        fold_net.append(r[2].mean())
                    else:
                        fold_net.append(np.nan)
                done += 1
                if not parts:
                    continue
                s = summarize(parts, h)
                rows.append(dict(fitur=fname, regime=rname, menit=h * tf_min, **s,
                                 fold_pos=int(np.nansum(np.array(fold_net) > 0)), folds=K))
        print(f"   horizon {h * tf_min:>4} menit selesai  ({done:,}/{n_attempt:,} uji, {time.time() - t0:.0f} dtk)")

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("\n[!] Tidak ada kombinasi dengan data cukup. Periksa data / ukuran jendela.")
    res["p"] = [sf(t) for t in res["net_t"]]
    # kombinasi yang dilewati dihitung p=1 dalam BH (jumlah uji = n_attempt)
    p_all = np.concatenate([res["p"].to_numpy(), np.ones(n_attempt - len(res))])
    res["q"] = bh_qvalues(p_all)[:len(res)]
    res["lolos"] = ((res["q"] < 0.05) & (res["net"] > 0) & (res["n"] >= a.min_trades) &
                    (res["fold_pos"] >= math.ceil(0.75 * K)))
    out_csv = config.RESULTS_DIR / f"edge_scan_{tf}.csv"
    res.sort_values("net_t", ascending=False).to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    show = res.sort_values("net_t", ascending=False).head(a.top).copy()
    show["kandidat"] = show["fitur"] + ":" + show["regime"] + ":" + show["menit"].astype(str)
    print(f"\n {a.top} kombinasi terbaik menurut t-stat NETO (setelah biaya), data uji walk-forward saja:\n")
    print(f" {'kandidat':<34}{'n':>7}{'bruto':>9}{'biaya':>8}{'neto':>9}{'t':>7}{'win%':>6}{'fold+':>7}{'q(BH)':>8}")
    for _, r in show.iterrows():
        print(f" {r['kandidat']:<34}{int(r['n']):>7}{r['gross']:>+9.3f}{r['cost']:>8.3f}{r['net']:>+9.3f}"
              f"{r['net_t']:>7.2f}{r['win']:>6.1f}{int(r['fold_pos']):>4}/{K}{r['q']:>8.3f}")
    print("   (bruto/biaya/neto dalam $ per oz per trade)")

    res["p_gross"] = [sf(t) for t in res["gross_t"]]
    pg_all = np.concatenate([res["p_gross"].to_numpy(), np.ones(n_attempt - len(res))])
    res["q_gross"] = bh_qvalues(pg_all)[:len(res)]
    res.sort_values("net_t", ascending=False).to_csv(out_csv, index=False)
    n_ok = int(res["lolos"].sum())
    # "edge ada tapi dimakan biaya" HARUS signifikan setelah koreksi banyak-uji, bukan sekadar t>3
    gross_sig = int(((res["q_gross"] < 0.05) & (res["gross"] > 0) & (res["net"] <= 0)).sum())
    print("\n" + "=" * 74)
    print(f" Lolos penuh (q<0.05, neto>0, n>={a.min_trades}, >={math.ceil(0.75 * K)}/{K} fold positif): {n_ok}")
    print(f" Bruto signifikan setelah koreksi (q<0.05) tetapi neto <= 0 (dimakan biaya): {gross_sig}")
    print(f" Hasil lengkap: {out_csv}")
    print("=" * 74)
    print(" SENSITIVITAS: edge NETO minimum yang bisa terdeteksi (kasar, ~3.8 x galat baku):")
    sens = res.groupby("menit")["net_se"].median() * 3.8
    print("   " + "   ".join(f"{int(m)} mnt: ${v:.2f}/oz" for m, v in sens.items()))
    print("   -> edge yang lebih kecil dari angka ini TIDAK bisa dibedakan dari noise dengan data ini.")
    if (show["fitur"] == "drift").any():
        print(" CATATAN 'drift': itu BUKAN prediksi, melainkan posisi searah rata-rata periode latih")
        print("   (eksposur ke tren harga). Kalau muncul di puncak, artinya harga cenderung searah tren")
        print("   selama periode ini -- bukan keunggulan yang bisa diandalkan ke depan.")
    if n_ok == 0:
        print(" VERDICT: TIDAK ADA edge yang lolos pada fitur-fitur ini di timeframe ini.")
        if gross_sig:
            print("          Ada sinyal bruto yang nyata tapi lebih kecil dari biaya -> coba horizon/timeframe")
            print("          LEBIH BESAR, atau hanya trade saat gerak yang diharapkan >> biaya.")
        else:
            print("          Bahkan sebelum biaya tidak ada sinyal yang meyakinkan. Jangan lanjut ke RL/Fase 2")
            print("          dengan fitur yang sama; cari sumber informasi baru (lintas-aset, event, timeframe lain).")
    else:
        surv = res[res["lolos"]].sort_values("net_t", ascending=False)
        names = ",".join(surv["fitur"] + ":" + surv["regime"] + ":" + surv["menit"].astype(str))
        print(f" VERDICT: {n_ok} kandidat LAYAK DITELITI (belum berarti layak trading).")
        print(" Langkah berikut: pilih paling banyak beberapa kandidat yang masuk akal secara logika pasar,")
        print(" lalu buka data terkunci SEKALI:")
        print(f"   python edge_scan.py --timeframe {tf} --confirm {names.split(',')[0]} --unlock-holdout")
    print(" Catatan: pada data acak murni, BH 5% tetap memberi ~5% peluang satu 'penemuan' palsu;")
    print(" karena itu syarat fold-positif dan konfirmasi holdout WAJIB.")


if __name__ == "__main__":
    main()