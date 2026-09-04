import argparse
import pickle
import sys
import subprocess
import numpy as np
from stable_baselines3 import PPO
import MetaTrader5 as mt5

import config
from live_env import LiveMT5Env

def main():
    parser = argparse.ArgumentParser(description="Live On-Policy Fine-Tuning with Gemini")
    parser.add_argument("--mode", type=str, choices=["demo", "real"], default="demo", help="Pilih mode: demo atau real")
    args = parser.parse_args()

    # Menjalankan Gemini API Process di background
    print("[*] Memulai Proses Background: Gemini Analyst...")
    gemini_process = subprocess.Popen([sys.executable, "gemini_analyst.py"])

    if args.mode == "real":
        ACTIVE_SYMBOL = "XAUUSDc"
        MT5_PATH = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
    else:
        ACTIVE_SYMBOL = "XAUUSDm"
        MT5_PATH = r"C:\Program Files\MetaTrader 5 EXNESS 1\terminal64.exe"
        
    if not mt5.initialize(path=MT5_PATH):
        print(f"[!] Gagal inisialisasi MT5 dari path: {MT5_PATH}")
        gemini_process.terminate()
        return
        
    mt5.symbol_select(ACTIVE_SYMBOL, True)
    
    tag = f"ppo_kalman_{ACTIVE_SYMBOL}_{config.DEFAULT_TIMEFRAME}"
    model_path = config.MODEL_DIR / f"{tag}.zip"
    artifact_path = config.MODEL_DIR / f"{tag}_artifacts.pkl"
    
    if not model_path.exists():
        print(f"[!] Model {model_path} tidak ditemukan. Lakukan pre-training (train.py) terlebih dahulu.")
        mt5.shutdown()
        gemini_process.terminate()
        return

    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    kalman_model = art["kalman_model"]
    
    env = LiveMT5Env(ACTIVE_SYMBOL, kalman_model)
    
    print(f"[*] Memuat otak AI dari {model_path} ...")
    model = PPO.load(model_path, env=env, custom_objects={"learning_rate": 5e-5, "ent_coef": 0.005})
    
    print("\n=======================================================")
    print(f"🚀 HYBRID AI TRADER DIMULAI PADA AKUN {args.mode.upper()} ({ACTIVE_SYMBOL}) 🚀")
    print("Agent: PPO Kalman (Eksekutor Taktis M1)")
    print("Supervisor: Gemini 2.5 Flash (Analis Makro H1)")
    print("Tekan [Ctrl + C] kapan saja jika Anda ingin mematikan bot.")
    print("=======================================================\n")
    
    try:
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n\n[!] Sinyal interupsi (Ctrl+C) terdeteksi. Memulai protokol penyimpanan aman...")
    finally:
        print(f"[*] Menyimpan pembaruan bobot otak ke {model_path} ...")
        model.save(model_path)
        print("[*] Model berhasil disimpan! Ingatan bot hari ini sudah terekam.")
        
        positions = mt5.positions_get(symbol=ACTIVE_SYMBOL)
        if positions:
            print("[*] Menutup posisi yang masih terbuka di MT5...")
            from live_env import close_all_positions
            close_all_positions(ACTIVE_SYMBOL)
            
        mt5.shutdown()
        gemini_process.terminate()
        print("[*] MT5 & Gemini Shutdown. Selamat beristirahat.")
        sys.exit(0)

if __name__ == "__main__":
    main()