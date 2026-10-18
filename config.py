from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (MODEL_DIR, LOG_DIR, RESULTS_DIR, MODEL_DIR / "checkpoints"):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOL = "XAUUSDm"
PROXY_SYMBOL = "EURUSDm"   # PERBAIKAN: sebelumnya "EURUSDc" -- tidak cocok dengan simbol broker
                            # (yang pakai suffix "m", sama seperti XAUUSDm) maupun nama file yang
                            # dihasilkan download_data.py (EURUSDm_1M.csv). Akibatnya data_loader.py
                            # tidak pernah menemukan file proxy-nya dan diam-diam mengisi
                            # proxy_ret/proxy_vol_sma dengan NOL di semua training sebelumnya --
                            # fitur korelasi EUR/USD efektif tidak pernah benar-benar dipakai model.
DEFAULT_TIMEFRAME = "1M"
TRAIN_SPLIT = 0.7

MODEL_PREFIX = "ppo_v4_relative"   # versi baru -> jangan ditimpa ke model lama

TIMEFRAME_MINUTES = {
    "1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240,
}

# =====================================================================
# !!! WAJIB DIVERIFIKASI SEBELUM TRAIN !!!
# Buka MT5 -> klik kanan symbol XAUUSDm -> Specification -> lihat "Point".
# Kalau quote broker 2 desimal (mis. 2650.32) -> point biasanya 0.01
# Kalau quote broker 3 desimal (mis. 2650.325) -> point biasanya 0.001
# Nilai ini WAJIB SAMA baik saat training (trading_env.py) maupun saat
# live (live_env.py). Sebelumnya ini hardcode beda -> jadi biang error
# SL/TP & lot size beda 10x lipat antara backtest dan live/demo.
# =====================================================================
POINT = 0.001

# Berapa banyak candle historis yang ditarik live_env.py setiap step untuk
# menghitung indikator (termasuk EMA-500/macro_ema & Kalman filter).
# Sebelumnya cuma 1000 -> EMA-500 belum converge -> distribusi fitur beda
# jauh dari training (yang dihitung dari histori bertahun-tahun).
# Aturan kasar: minimal ~5x window terpanjang (macro_ema span).
LIVE_LOOKBACK_BARS = 3000

SMA_WINDOWS = (5, 10, 20)
EMA_WINDOWS = (9, 21)
RSI_WINDOW = 7
STOCH_WINDOW = 14
STOCH_SMOOTH = 3
BB_WINDOW = 14
BB_STD = 2.0
ATR_WINDOW = 14          # dinaikkan dari 7 -> ATR lebih stabil sbg basis normalisasi fitur
CCI_WINDOW = 14
WILLIAMS_WINDOW = 14
VWAP_WINDOW = 14
VOL_ZSCORE_WINDOW = 50   # window rolling utk z-score volume

KALMAN_EM_ITER = 15
KALMAN_FIT_SUBSAMPLE = 50_000
KALMAN_COLUMNS = ["open", "high", "low", "close", "tick_volume"]

# Batas clipping fitur akhir (sebelum masuk VecNormalize). Karena sekarang
# fitur sudah relatif/stasioner, range ini realistis (bukan level harga).
FEATURE_CLIP = 5.0

INITIAL_BALANCE = 1000.0

RISK_PER_TRADE = 10.0
MAX_SIZE_OZ = 100.0

MACRO_EMA_SPAN = 500
TREND_EMA_SPAN = 200     # EMA dipakai filter tren cepat (dulu dihitung ulang di env dari raw prices)

# --- ATR DINAMIS DENGAN BATAS BAWAH (FLOOR) & ATAS (CEILING), DALAM POINT ---
# !!! NILAI DASAR DI SECTION INI = NILAI M1 (yang sedang jalan live) !!!
# Nilai untuk timeframe lain (M5, dst) TIDAK diedit di sini, tapi di
# TIMEFRAME_PROFILES di bagian paling bawah file ini. Profil dipilih otomatis
# dari argumen --timeframe (lihat apply_timeframe_profile di bawah).
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 4.0
MIN_SL_PTS = 3600        # M1: $3.6
MIN_TP_PTS = 7200        # M1: $7.2 (rasio TP:SL = 2:1)
MAX_SL_PTS = 9000       # M1: $9.0

BE_TRIGGER_PCT = 0.50
BE_BUFFER_PTS = 200

# Sebelumnya SATU flag (ENABLE_TIME_FIREWALL) mengontrol proteksi weekend-gap
# DAN proteksi toxic-hours sekaligus -> kalau salah satu dimatikan, yang lain
# ikut mati tanpa disadari. Sekarang dipisah jadi 2 flag independen.
ENABLE_WEEKEND_FIREWALL = True    # proteksi gap weekend (hanya berlaku di training/backtest, live sudah otomatis ditutup broker)
ENABLE_TOXIC_HOUR_FILTER = False  # blokir jam-jam tertentu (spread/likuiditas buruk) -> saat ini dimatikan sesuai keputusan kalian
TOXIC_HOURS = [4, 5, 7, 11, 17]

# --- FILTER TREN (EMA-200 & macro_slope) ---
# Sekarang bisa dimatikan/dilonggarkan lewat config, tidak hardcode block
# paksa. TREND_FILTER_PENALTY dipakai sebagai penalti reward (soft guidance)
# bukan pemblokiran aksi total, supaya agen tetap punya ruang belajar sendiri.
ENABLE_TREND_FILTER = True
TREND_FILTER_HARD_BLOCK = False   # False = hanya kasih penalti, tidak paksa flat
TREND_FILTER_PENALTY = 0.3

# --- PENCEGAH OVERTRADING & CHURNING ---
# !!! NILAI DASAR = M1 (satuan: jumlah STEP/candle, bukan menit) !!!
# Untuk M5 nilainya dibagi 5 supaya durasi riil (menit) sama -> lihat
# TIMEFRAME_PROFILES["5M"] di bawah.
MIN_HOLD_STEPS = 15           # M1: 15 step = 15 menit
LOSS_COOLDOWN_STEPS = 10       # M1: 10 step = 10 menit
MAX_HOLD_LOSS_STEPS = 600     # M1: 600 step = 600 menit
CHURN_PENALTY = 2.0      # base penalty, dikalikan gamma dari REWARD_COEFFS
IDLE_PENALTY_AFTER_STEPS = 20  # M1: setelah 20 step (=20 menit) flat, mulai kena penalti idle `delta` (dulu hardcode 20 di trading_env.py)

WEEKEND_START_DAY = 5
WEEKEND_START_HOUR = 4
WEEKEND_END_DAY = 0
WEEKEND_END_HOUR = 5

# --- CIRCUIT BREAKER HARIAN (backstop utama, peduli $ apa pun mekanismenya) ---
# Rugi harian tembus batas -> bot_control TIDAK diubah (tetap milik tombol
# manual), tapi risk_state.daily_breaker_date diisi tanggal hari ini ->
# live_env.py otomatis paksa flat sampai tanggal kalender berganti.
ENABLE_DAILY_LOSS_BREAKER = True
DAILY_LOSS_LIMIT_PCT = 0.05      # auto-stop kalau rugi harian >= 5% saldo awal hari ini
DAILY_LOSS_LIMIT_USD = None      # isi angka $ tetap kalau mau override % (mis. 50.0), None = pakai % saja
# False = rugi harian dihitung dari selisih BALANCE AKUN (semua bot di akun yang
#         sama ikut terhitung -> kalau 2 bot jalan di akun yg sama, rugi bot A bisa
#         memicu breaker bot B).
# True  = rugi harian dihitung HANYA dari trade closed milik bot ini (dari
#         trade_details di DB-nya sendiri). WAJIB True kalau M1 & M5 jalan
#         bersamaan di akun yang sama.
DAILY_BREAKER_PER_BOT = False

# --- CIRCUIT BREAKER SL BERUNTUN (deteksi cepat regime choppy) ---
# Hitungan KUMULATIF, bukan harus berturut-turut: SL-di-breakeven (net_profit
# mendekati 0) dan SIGNAL_CHANGE dilewati/tidak mereset counter; hanya SL
# ASLI (net_profit <= -STREAK_LOSS_MIN_USD) yang menambah counter, dan hanya
# TP asli yang mereset ke 0. Validasi dari data live 13 hari: dengan definisi
# ini, limit 8 akan trigger jam 10:05 di hari terburuk (21 Sep), jauh sebelum
# kerugian terbesar jam 10:40-12:44.
ENABLE_STREAK_BREAKER = True
STREAK_SL_LIMIT = 8
STREAK_LOSS_MIN_USD = 1.0        # net_profit lebih negatif dari -$1 dianggap SL asli, bukan SL-BE
STREAK_COOLDOWN_MINUTES = 120    # setelah limit tercapai, flat paksa selama ini, lalu resume otomatis

# --- PROTEKSI WEEKEND EKSPLISIT DI LIVE ---
# Sebelumnya live_env.py TIDAK menutup posisi menjelang weekend sendiri --
# cuma asumsi "broker sudah handle otomatis". Ini bikin proteksi gap weekend
# tergantung asumsi yang belum tentu benar. Sekarang ditutup eksplisit oleh
# bot X menit sebelum WEEKEND_START_HOUR, dan entry baru diblokir sampai
# WEEKEND_END_HOUR hari Minggu (dua-duanya dari WEEKEND_START_DAY/HOUR &
# WEEKEND_END_DAY/HOUR di atas).
WEEKEND_CLOSE_BUFFER_MINUTES = 30

COMMISSION_RATE = 0.0000
SPREAD_COST_RATE = 0.0001
# PERBAIKAN: slippage sekarang dalam satuan POINT (sama seperti MIN_SL_PTS dkk),
# diterapkan PER-OZ seperti spread_cost -- bukan persentase dari notional value.
# SLIPPAGE_COEFF lama (trade_value * 0.0005) salah basis: untuk posisi notional
# ribuan dolar, itu jadi beberapa dolar per transaksi -- lebih besar dari
# RISK_PER_TRADE ($10) itu sendiri -> otomatis margin call di semua episode.
# Mulai dari nilai konservatif realistis utk gold M1 (1-3 point), lalu naikkan
# bertahap (mis. 5, 10) untuk uji sensitivitas -- JANGAN loncat ke angka besar
# langsung supaya kita bisa lihat di titik berapa edge-nya mulai goyah.
SLIPPAGE_PTS = 20.0

EPISODE_LENGTH = 6000    # M1: 6000 step = 6000 menit (~4,2 hari)
INCLUDE_POSITION_IN_STATE = True

MARGIN_CALL_DRAWDOWN = 0.99
MARGIN_CALL_PENALTY = 5.0

# --- REWARD COEFFICIENTS (SEKARANG BENAR-BENAR DIPAKAI DI trading_env.py) ---
# alpha : bobot reward dari return portofolio (PnL)
# beta  : bobot penalti drawdown per-step (risk-adjusted, mendorong equity curve halus)
# gamma : pengali CHURN_PENALTY (ganti posisi terlalu cepat)
# delta : penalti idle/menganggur kecil, mendorong agen tidak "diam" terus-menerus
#         kalau ada peluang (nilai kecil supaya tidak memaksa overtrading)
REWARD_COEFFS_BY_TIMEFRAME = {
    "1M":  dict(alpha=1.0, beta=0.20, gamma=1.0, delta=0.01),
    "5M":  dict(alpha=1.0, beta=0.15, gamma=1.0, delta=0.01),
    "15M": dict(alpha=1.0, beta=0.15, gamma=0.8, delta=0.01),
    "1H":  dict(alpha=1.0, beta=0.10, gamma=0.6, delta=0.005),
}
DEFAULT_REWARD_COEFFS = dict(alpha=1.0, beta=0.15, gamma=1.0, delta=0.01)
REWARD_SCALE = 1000.0    # skala dasar reward (dulu magic number 5000 tersebar di banyak tempat)

GAMMA = 0.999
TOTAL_TIMESTEPS = 6_000_000   # dinaikkan; user siap tunggu training lebih lama

PPO_KW = dict(
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=10,
    gamma=GAMMA,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
)

def reward_coeffs_for(tf):
    return REWARD_COEFFS_BY_TIMEFRAME.get(tf, DEFAULT_REWARD_COEFFS)


# =====================================================================
# PROFIL PER-TIMEFRAME
# ---------------------------------------------------------------------
# Semua nilai di atas adalah nilai DASAR = M1 (yang sedang live).
# Profil di bawah hanya menimpa nilai yang memang bergantung timeframe.
# Dipilih otomatis oleh main.py lewat apply_timeframe_profile(args.timeframe)
# SEBELUM training / backtest / live dimulai -> training, backtest, dan live
# untuk timeframe yang sama SELALU memakai angka yang sama.
#
# Profil "1M" sengaja kosong = tidak mengubah apa pun dari nilai dasar,
# jadi bot M1 yang sedang berjalan perilakunya tetap identik.
# =====================================================================
TIMEFRAME_PROFILES = {
    "1M": {},
    "5M": {
        # --- satuan STEP: nilai M1 / 5 supaya durasi riil (menit) SAMA ---
        "EPISODE_LENGTH": 1200,            # 1200 step x 5 mnt = 6000 mnt (sama dgn M1)
        "MIN_HOLD_STEPS": 3,               # 3 x 5 mnt  = 15 mnt
        "LOSS_COOLDOWN_STEPS": 2,          # 2 x 5 mnt  = 10 mnt
        "MAX_HOLD_LOSS_STEPS": 120,        # 120 x 5 mnt = 600 mnt
        "IDLE_PENALTY_AFTER_STEPS": 4,     # 4 x 5 mnt  = 20 mnt

        # --- SL/TP (point; POINT=0.001 -> 1000 pts = $1) ---
        # Angka di bawah = angka yang tercatat sbg hasil validasi ATR M5 di
        # komentar config lama (p25 raw_SL 33 pip, p90 140 pip, TP:SL 2:1).
        # WAJIB dicek ulang pada data 5M TERBARU sebelum training:
        #     python check_market_params.py --symbol XAUUSDm --timeframe 5M
        # lalu salin angka "SARAN" dari output-nya ke 3 baris ini.
        "MIN_SL_PTS": 7600,                # $3.3
        "MAX_SL_PTS": 14300,               # $14
        "MIN_TP_PTS": 15200,                # $6.6

        # --- training: bar M5 5x lebih sedikit per satuan waktu ---
        "TOTAL_TIMESTEPS": 3_000_000,

        # --- breaker harian dihitung per-bot (aman jalan bareng bot M1) ---
        "DAILY_BREAKER_PER_BOT": True,
    },
}


# Snapshot nilai DASAR (M1) untuk semua key yang pernah ditimpa profil manapun.
# Diambil saat import, sebelum profil apa pun diterapkan. Dipakai untuk
# memulihkan nilai dasar sebelum menerapkan profil baru, supaya pindah
# profil (mis. 5M lalu 1M dalam satu proses) tidak meninggalkan sisa nilai.
_PROFILE_KEYS = {k for _p in TIMEFRAME_PROFILES.values() for k in _p}
_BASE_VALUES = {k: globals()[k] for k in _PROFILE_KEYS}


def apply_timeframe_profile(timeframe: str) -> dict:
    """Timpa nilai config global sesuai profil timeframe.
    Selalu mulai dari nilai dasar (M1) lalu timpa dengan profil -> aman
    dipanggil berkali-kali / berpindah profil. Mengembalikan dict nilai
    AKTIF yang relevan (untuk dicetak di log)."""
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_PROFILES:
        raise ValueError(
            f"Timeframe '{tf}' belum punya profil di config.TIMEFRAME_PROFILES "
            f"(tersedia: {list(TIMEFRAME_PROFILES)}). Tambahkan profilnya dulu."
        )
    g = globals()
    for k, v in _BASE_VALUES.items():
        g[k] = v
    for k, v in TIMEFRAME_PROFILES[tf].items():
        g[k] = v
    g["DEFAULT_TIMEFRAME"] = tf
    keys = ["EPISODE_LENGTH", "MIN_HOLD_STEPS", "LOSS_COOLDOWN_STEPS", "MAX_HOLD_LOSS_STEPS",
            "IDLE_PENALTY_AFTER_STEPS", "MIN_SL_PTS", "MAX_SL_PTS", "MIN_TP_PTS",
            "TOTAL_TIMESTEPS", "DAILY_BREAKER_PER_BOT", "SLIPPAGE_PTS"]
    return {k: g[k] for k in keys}