from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (MODEL_DIR, LOG_DIR, RESULTS_DIR, MODEL_DIR / "checkpoints"):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOL = "XAUUSDm"
PROXY_SYMBOL = "EURUSDc"
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
# NILAI AKTIF DI BAWAH INI UNTUK M5 (hasil validasi ATR asli via
# check_market_params.py, bukan tebakan lagi). Nilai M1 dikomentari di
# sebelahnya untuk swap cepat kalau mau retrain M1.
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 4.0
MIN_SL_PTS = 3600        # M5 (data tervalidasi, p25 raw_SL): 33 pip = $3.3.  UNTUK M1 pakai: 3600 (36 pip)
MIN_TP_PTS = 7200        # M5: 66 pip = $6.6, rasio TP:SL=2:1 tetap dijaga.   UNTUK M1 pakai: 7200 (72 pip)
MAX_SL_PTS = 9000       # M5 (p90 raw_SL): 140 pip = $14.                    UNTUK M1 pakai: 9000 (90 pip)

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
# NILAI AKTIF DI BAWAH INI UNTUK M5. Kalau mau retrain M1, pakai nilai yang
# dikomentari di sebelahnya (candle M1 5x lebih rapat dari M5).
MIN_HOLD_STEPS = 15           # UNTUK M1 pakai: 15  (M1: 15 step = 15 menit, setara 3 step M5 = 15 menit)
LOSS_COOLDOWN_STEPS = 10       # UNTUK M1 pakai: 10  (10 step M1 = 10 menit, setara 2 step M5 = 10 menit)
MAX_HOLD_LOSS_STEPS = 600     # UNTUK M1 pakai: 600 (600 step M1 = 600 menit, setara 120 step M5 = 600 menit)
CHURN_PENALTY = 2.0      # base penalty, dikalikan gamma dari REWARD_COEFFS

WEEKEND_START_DAY = 5
WEEKEND_START_HOUR = 4
WEEKEND_END_DAY = 0
WEEKEND_END_HOUR = 5

COMMISSION_RATE = 0.0000
SPREAD_COST_RATE = 0.0001
SLIPPAGE_COEFF = 0.0000

EPISODE_LENGTH = 6000    # UNTUK M1 pakai: 6000 (6000 menit, setara 1500 step M5 x5 menit = 7500 menit; kurang lebih sepadan)
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