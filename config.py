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
DEFAULT_TIMEFRAME = "5M"          
TRAIN_SPLIT = 0.7                  

MODEL_PREFIX = "ppo_v3_dynamic"

TIMEFRAME_MINUTES = {
    "1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240,
}

SMA_WINDOWS = (5, 10, 20)
EMA_WINDOWS = (9, 21)
RSI_WINDOW = 7
STOCH_WINDOW = 14
STOCH_SMOOTH = 3
BB_WINDOW = 14
BB_STD = 2.0
ATR_WINDOW = 7
CCI_WINDOW = 14
WILLIAMS_WINDOW = 14
VWAP_WINDOW = 14                   

KALMAN_EM_ITER = 15
KALMAN_FIT_SUBSAMPLE = 50_000      
KALMAN_COLUMNS = ["open", "high", "low", "close", "tick_volume"]

INITIAL_BALANCE = 1000.0

RISK_PER_TRADE = 10.0              
MAX_SIZE_OZ = 100.0               

MACRO_EMA_SPAN = 500              

# --- KEMBALI KE ATR DINAMIS DENGAN BATAS BAWAH (FLOOR) & ATAS (CEILING) ---
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 4.0
MIN_SL_PTS = 2500        # Batas minimal 25 pips agar tidak tersentuh noise
MIN_TP_PTS = 5000        # Batas minimal 50 pips
MAX_SL_PTS = 6000        # Batas maksimal 60 pips agar tidak rugi terlalu dalam saat volatilitas gila

BE_TRIGGER_PCT = 0.50    # Breakeven saat 50% TP tercapai
BE_BUFFER_PTS = 200      # 2 pips buffer

ENABLE_TIME_FIREWALL = True
TOXIC_HOURS = [4, 5, 7, 11, 17]

# --- PENCEGAH OVERTRADING & CHURNING (DI-SINKRONKAN UNTUK TRAINING & LIVE) ---
MIN_HOLD_STEPS = 3       # Bot WAJIB tahan posisi minimal 3 candle sebelum boleh cut-loss manual/ganti arah
LOSS_COOLDOWN_STEPS = 2  # Jika kena SL, tunggu 2 candle sebelum buka posisi baru
MAX_HOLD_LOSS_STEPS = 120          
CHURN_PENALTY = 2.0      # Hukuman AI jika terlalu cepat ganti sinyal

WEEKEND_START_DAY = 5              
WEEKEND_START_HOUR = 4             
WEEKEND_END_DAY = 0                
WEEKEND_END_HOUR = 5               

COMMISSION_RATE = 0.0000           
SPREAD_COST_RATE = 0.0001
SLIPPAGE_COEFF = 0.0000           

EPISODE_LENGTH = 1500              
INCLUDE_POSITION_IN_STATE = True   

MARGIN_CALL_DRAWDOWN = 0.99        
MARGIN_CALL_PENALTY = 5.0          

REWARD_COEFFS_BY_TIMEFRAME = {
    "1M":  dict(alpha=1.0, beta=0.10, gamma=0.50, delta=0.0),
    "5M":  dict(alpha=3.0, beta=0.05, gamma=0.10, delta=0.0), 
}
DEFAULT_REWARD_COEFFS = dict(alpha=1.0, beta=0.10, gamma=0.50, delta=0.0)

GAMMA = 0.999
TOTAL_TIMESTEPS = 3_000_000        

# --- ENTROPY COEF DITURUNKAN AGAR AI LEBIH TEGAS ---
PPO_KW = dict(
    learning_rate=1e-4,            
    n_steps=4096,                  
    batch_size=512,                
    n_epochs=10,                   
    gamma=GAMMA,
    gae_lambda=0.95, 
    clip_range=0.2, 
    ent_coef=0.01,        # DITURUNKAN dari 0.08 menjadi 0.01 agar tidak flip-flop (labil)
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])), 
)

def reward_coeffs_for(tf):
    return REWARD_COEFFS_BY_TIMEFRAME.get(tf, DEFAULT_REWARD_COEFFS)