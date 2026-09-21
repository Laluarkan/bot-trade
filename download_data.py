"""
Download histori MT5 dalam JUMLAH BESAR (>100.000 bar) dengan cara chunk/
paginasi. copy_rates_from_pos() dibatasi terminal MT5 (umumnya mentok di
~100.000 bar per SATU panggilan, tidak peduli count yang diminta) -> jadi
di sini kita tarik berkali-kali dengan start_pos digeser, lalu digabung.

PENTING SEBELUM JALAN (gotcha yang sering kelewat):
1. mt5.copy_rates_from_pos() hanya baca dari cache HISTORI LOKAL terminal,
   bukan langsung dari server. Kalau kalian belum pernah scroll chart M1
   symbol itu jauh ke belakang, cache lokalnya mungkin belum punya histori
   sedalam itu -> hasil tetap kepotong meski sudah di-chunk di sini.
   Fix: buka chart M1 symbol tsb di MT5, tekan Home lalu scroll/drag terus
   ke kiri (atau klik kanan chart > "Show all" beberapa kali) sampai MT5
   selesai memuat histori lama dari server, BARU jalankan script ini.
2. Tools > Options > Charts > "Max bars in chart" & "Max bars in history"
   -> set ke maksimum (biasanya "Unlimited"/angka besar), lalu restart MT5.
"""
import time
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# --- KONFIGURASI ---
SYMBOLS = ["XAUUSDm", "EURUSDm"]
TIMEFRAME = mt5.TIMEFRAME_M1
TIMEFRAME_STR = "1M"
TARGET_BARS = 200_000
CHUNK_SIZE = 50_000       # aman di bawah batas ~100rb per panggilan
MAX_RETRIES_PER_CHUNK = 3
OUTPUT_DIR = "data"


def fetch_in_chunks(symbol: str, timeframe, target_bars: int, chunk_size: int) -> pd.DataFrame:
    all_chunks = []
    start_pos = 0
    fetched = 0

    while fetched < target_bars:
        remaining = target_bars - fetched
        this_chunk_size = min(chunk_size, remaining)

        rates = None
        for attempt in range(1, MAX_RETRIES_PER_CHUNK + 1):
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, this_chunk_size)
            if rates is not None and len(rates) > 0:
                break
            print(f"    [retry {attempt}/{MAX_RETRIES_PER_CHUNK}] chunk start_pos={start_pos} kosong, coba lagi...")
            time.sleep(1)

        if rates is None or len(rates) == 0:
            print(f"    [*] Berhenti di start_pos={start_pos} -> tidak ada data lagi (histori lokal MT5 habis).")
            break

        df_chunk = pd.DataFrame(rates)
        all_chunks.append(df_chunk)
        n_got = len(df_chunk)
        fetched += n_got
        oldest_in_chunk = pd.to_datetime(df_chunk['time'].iloc[0], unit='s')
        print(f"    [+] chunk start_pos={start_pos:>7} -> {n_got:>6} bar diterima (total {fetched:>7}/{target_bars}), tertua: {oldest_in_chunk}")

        if n_got < this_chunk_size:
            # server/terminal sudah tidak punya bar lebih tua lagi dari ini
            print(f"    [*] Chunk terakhir cuma {n_got} (< {this_chunk_size} diminta) -> histori sudah mentok, berhenti.")
            break

        start_pos += n_got

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df.drop_duplicates(subset='time', inplace=True)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def download_historical_data():
    print("=" * 60)
    print(f"📥 MEMULAI UNDUH DATA HISTORI MT5 (chunked, target {TARGET_BARS:,} bar) 📥")
    print("=" * 60)

    if not mt5.initialize():
        print(f"[!] Gagal menghubungkan ke MetaTrader 5.")
        print(f"    Pesan Error: {mt5.last_error()}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for symbol in SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            print(f"[!] Simbol {symbol} tidak ditemukan di Market Watch MT5. Melewati...")
            continue

        print(f"\n[*] Menarik {symbol} ({TIMEFRAME_STR}) secara chunk (@{CHUNK_SIZE:,} bar/panggilan)...")
        df = fetch_in_chunks(symbol, TIMEFRAME, TARGET_BARS, CHUNK_SIZE)

        if df.empty:
            print(f"[!] Gagal menarik {symbol} sama sekali. Cek koneksi & symbol select.")
            continue

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'time': 'datetime'}, inplace=True)

        output_filename = f"{symbol}_{TIMEFRAME_STR}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df.to_csv(output_path, index=False)

        pct_of_target = len(df) / TARGET_BARS * 100
        print(f"[*] SELESAI {symbol}!")
        print(f"    - Jumlah Baris : {len(df):,} ({pct_of_target:.1f}% dari target {TARGET_BARS:,})")
        print(f"    - Tanggal Awal : {df['datetime'].iloc[0]}")
        print(f"    - Tanggal Akhir: {df['datetime'].iloc[-1]}")
        print(f"    - Tersimpan di : {output_path}")
        if pct_of_target < 95:
            print(f"    [WARNING] Kurang dari target -> kemungkinan histori lokal MT5 belum sedalam itu.")
            print(f"              Lihat catatan di atas file ini (scroll chart M1 {symbol} ke belakang dulu di MT5).")

    print("\n" + "=" * 60)
    mt5.shutdown()


if __name__ == "__main__":
    download_historical_data()