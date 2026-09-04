import argparse
import pickle
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import MetaTrader5 as mt5
import config
from data_loader import load_csv, build_feature_matrix, walk_forward_splits
from trading_env import GoldTradingEnv
from live_env import LiveMT5Env

DB_FILE = "bot_database.db"

def init_eval_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS model_evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, action_type TEXT, symbol TEXT,
                        timesteps INTEGER, net_profit REAL, total_trades INTEGER,
                        win_rate REAL, eval_start_time TEXT, eval_end_time TEXT,
                        duration_seconds REAL, episodes_run INTEGER, actual_trades INTEGER,
                        avg_profit_per_trade REAL
                    )''')
        conn.commit()

def save_evaluation(action_type, symbol, timesteps, net_profit, total_trades, win_rate, start_t, end_t, duration, episodes, actual_trades, avg_profit):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''INSERT INTO model_evaluations 
                     (timestamp, action_type, symbol, timesteps, net_profit, total_trades, win_rate, eval_start_time, eval_end_time, duration_seconds, episodes_run, actual_trades, avg_profit_per_trade)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (now, action_type, symbol, timesteps, net_profit, total_trades, win_rate, start_t, end_t, round(duration, 2), episodes, actual_trades, round(avg_profit, 2)))
        conn.commit()

def evaluate_model(model, env_fn, symbol, episodes=1, dataset_dates=None, mode_name="BACKTEST", step_idx=None):
    env = DummyVecEnv([env_fn])
    vec_path = config.MODEL_DIR / f"{config.MODEL_PREFIX}_{symbol}_{config.DEFAULT_TIMEFRAME}_vecnormalize.pkl"
    
    if vec_path.exists():
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
    total_profit = 0.0
    wins = 0
    losses = 0
    actual_trades_count = 0
    
    hist_start_dates = []
    hist_end_dates = []
    start_time_exec = time.time()
    
    all_pnl_histories = []
    for _ in range(episodes):
        obs = env.reset()
        current_pos = 0.0
        ep_pnl = []
        
        try:
            step = env.envs[0].unwrapped.current_step
            hist_start_dates.append(env.envs[0].unwrapped.dates[step])
        except: pass
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act_val = int(action[0])
            target_pos = -1.0 if act_val == 0 else (0.0 if act_val == 1 else 1.0)
            
            if target_pos != current_pos and target_pos != 0.0:
                actual_trades_count += 1
            current_pos = target_pos
            obs, reward, done_arr, info = env.step(action)
            done = done_arr[0]
            
            p_val = info[0].get('portfolio_value', config.INITIAL_BALANCE)
            ep_pnl.append(p_val - config.INITIAL_BALANCE)
            
            if done:
                try:
                    step = env.envs[0].unwrapped.current_step
                    hist_end_dates.append(env.envs[0].unwrapped.dates[step])
                except: pass
                profit = p_val - config.INITIAL_BALANCE
                total_profit += profit
                if profit > 0:
                    wins += 1
                elif profit < 0:
                    losses += 1
                    
        all_pnl_histories.append(ep_pnl)
    end_time_exec = time.time()
    duration = end_time_exec - start_time_exec
    total_episodes_run = wins + losses
    win_rate = (wins / total_episodes_run * 100) if total_episodes_run > 0 else 0.0
    avg_profit_per_trade = total_profit / actual_trades_count if actual_trades_count > 0 else 0.0
    
    if hist_start_dates and hist_end_dates:
        start_str = pd.to_datetime(min(hist_start_dates)).strftime("%Y-%m-%d %H:%M")
        end_str = pd.to_datetime(max(hist_end_dates)).strftime("%Y-%m-%d %H:%M")
    elif dataset_dates is not None and len(dataset_dates) > 0:
        start_str = pd.to_datetime(dataset_dates[0]).strftime("%Y-%m-%d %H:%M")
        end_str = pd.to_datetime(dataset_dates[-1]).strftime("%Y-%m-%d %H:%M")
    else:
        start_str = datetime.fromtimestamp(start_time_exec).strftime("%Y-%m-%d %H:%M:%S")
        end_str = datetime.fromtimestamp(end_time_exec).strftime("%Y-%m-%d %H:%M:%S")
        
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :]) 
    ax2 = fig.add_subplot(gs[1, 0]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    final_pnls = []
    for i, pnl in enumerate(all_pnl_histories):
        ax1.plot(pnl, label=f'Eps {i+1}', alpha=0.7)
        final_pnls.append(pnl[-1] if len(pnl) > 0 else 0)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    title_suffix = f" (Step {step_idx})" if step_idx is not None else ""
    ax1.set_title(f"Pergerakan Net Profit per Episode ({symbol}) - {mode_name}{title_suffix}", fontweight='bold')
    ax1.set_xlabel("Langkah (Steps)")
    ax1.set_ylabel("Net Profit (USD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=5, fontsize=8)
    colors = ['mediumseagreen' if p >= 0 else 'lightcoral' for p in final_pnls]
    ax2.bar(range(1, episodes+1), final_pnls, color=colors)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("Total Profit per Episode (Distribusi Sesi)", fontweight='bold')
    ax2.set_xlabel("Sesi (Episode)")
    ax2.set_ylabel("Profit (USD)")
    ax2.set_xticks(range(1, episodes+1))
    ax2.grid(True, alpha=0.3)
    ax3.axis('off')
    stats_text = (
        f"📊 RINGKASAN {mode_name} MODEL\n\n"
        f"🔹 Simbol         : {symbol} ({config.DEFAULT_TIMEFRAME})\n"
        f"🔹 Sesi Disimulasi: {episodes} Episode\n"
        f"🔹 Total Eksekusi : {actual_trades_count} Trades\n"
        f"🔹 Win Rate Sesi  : {win_rate:.1f}%\n"
        f"🔹 Avg Profit/Trd : ${avg_profit_per_trade:.2f}\n"
        f"🔹 TOTAL NET PNL  : ${total_profit:.2f}\n\n"
        f"Rentang Waktu Data :\n{start_str} s/d {end_str}\n"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=14, va='center', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=1.0, edgecolor='silver'))
    plt.tight_layout()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_str = f"_Step{step_idx}" if step_idx is not None else ""
    plot_path = config.RESULTS_DIR / f"{mode_name}_Report_{symbol}{step_str}_{timestamp_str}.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
        
    return total_profit, total_episodes_run, win_rate, start_str, end_str, duration, actual_trades_count, avg_profit_per_trade

class TradingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_profits = []
        self.episode_rewards = []
        self.episodes = 0
        self.hist_exp_var = []
        self.hist_ent_loss = []
        self.hist_pg_loss = []
        self.hist_val_loss = []
    def _on_step(self) -> bool:
        if "dones" in self.locals and "infos" in self.locals:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "portfolio_value" in info:
                        profit = info["portfolio_value"] - config.INITIAL_BALANCE
                        self.episode_profits.append(profit)
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                    self.episodes += 1
                    if self.logger:
                        self.hist_exp_var.append(self.logger.name_to_value.get("train/explained_variance", np.nan))
                        self.hist_ent_loss.append(self.logger.name_to_value.get("train/entropy_loss", np.nan))
                        self.hist_pg_loss.append(self.logger.name_to_value.get("train/policy_gradient_loss", np.nan))
                        self.hist_val_loss.append(self.logger.name_to_value.get("train/value_loss", np.nan))
                    else:
                        for lst in [self.hist_exp_var, self.hist_ent_loss, self.hist_pg_loss, self.hist_val_loss]:
                            lst.append(np.nan)
                    if self.episodes % 10 == 0:
                        avg_prof = np.mean(self.episode_profits[-10:])
                        avg_rew = np.mean(self.episode_rewards[-10:])
                        print(f"[TRAINING] Episode {self.episodes} | Avg Profit (last 10): ${avg_prof:,.2f} | Avg Reward: {avg_rew:,.2f}")
                        self._save_live_plot()
        return True
        
    def _save_live_plot(self) -> None:
        if len(self.episode_profits) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            def plot_panel(ax, data, title, ylabel, color, rolling_color, hline=None):
                data_clean = pd.Series(data).dropna()
                if len(data_clean) > 0:
                    ax.plot(data_clean.index, data_clean.values, alpha=0.4, color=color)
                    rolling_window = max(1, len(data_clean)//20)
                    ax.plot(data_clean.index, data_clean.rolling(rolling_window).mean(), color=rolling_color, linewidth=2)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel("Episode", fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(True, alpha=0.3)
                if hline is not None:
                    ax.axhline(hline, color='crimson', linestyle='--', linewidth=1.5)
            plot_panel(axes[0, 0], self.episode_rewards, "Evolusi Reward", "Reward Score", 'royalblue', 'navy')
            plot_panel(axes[0, 1], self.episode_profits, "Evolusi Profit", "Net Profit (USD)", 'mediumseagreen', 'darkgreen', hline=0)
            plot_panel(axes[0, 2], self.hist_exp_var, "Explained Variance", "Variance (Target: 1.0)", 'mediumpurple', 'indigo', hline=1.0)
            plot_panel(axes[1, 0], self.hist_ent_loss, "Entropy Loss", "Loss (Mendekati 0)", 'orange', 'darkorange')
            plot_panel(axes[1, 1], self.hist_pg_loss, "Policy Gradient Loss", "PG Loss", 'lightcoral', 'darkred')
            plot_panel(axes[1, 2], self.hist_val_loss, "Value Loss", "Loss (Target: 0)", 'skyblue', 'darkblue')
            plt.tight_layout()
            chart_path = config.RESULTS_DIR / f"{config.MODEL_PREFIX}_learning_curve.png"
            fig.savefig(str(chart_path), dpi=150)
            plt.close(fig)

def make_env_fn(features, prices, spreads, dates, episode_length):
    def _init():
        env = GoldTradingEnv(
            features, prices, spreads, dates=dates,
            episode_length=episode_length, 
            random_start=True, 
            use_margin_call=True
        )
        return Monitor(env)
    return _init

def run_pretrain(args):
    print("="*50)
    if getattr(args, 'resume', False):
        print(f"🚀 MEMULAI CONTINUOUS LEARNING / RESUME ({args.symbol}_{args.timeframe.upper()})")
    else:
        print(f"🚀 MEMULAI WALK-FORWARD PRE-TRAINING BARU ({args.symbol}_{args.timeframe.upper()})")
    print("="*50)
    
    config.DEFAULT_TIMEFRAME = args.timeframe.upper()
    init_eval_db()
    df = load_csv(args.symbol, args.timeframe.upper(), data_dir=args.data_dir)
    features, prices, spreads, dates, kalman_model, feature_names = build_feature_matrix(df, args.timeframe.upper(), use_kalman=True)
    
    splits = walk_forward_splits(features, prices, spreads, dates, n_splits=4)
    tag = f"{config.MODEL_PREFIX}_{args.symbol}_{args.timeframe.upper()}"
    model_path = config.MODEL_DIR / f"{tag}.zip"
    vec_path = config.MODEL_DIR / f"{tag}_vecnormalize.pkl"
    model = None
    
    for i, ((train_f, train_p, train_s, train_d), (test_f, test_p, test_s, test_d)) in enumerate(splits):
        print(f"\n[{i+1}/4] Memulai Siklus Pelatihan Walk-Forward...")
        
        env = DummyVecEnv([make_env_fn(train_f, train_p, train_s, train_d, args.episode_length)])
        
        if i == 0:
            if getattr(args, 'resume', False) and model_path.exists() and vec_path.exists():
                print(f"[*] RESUME AKTIF: Memuat memori lama dari {model_path.name}...")
                env = VecNormalize.load(str(vec_path), env)
                env.training = True
                model = PPO.load(str(model_path), env=env)
            else:
                print(f"[*] RESUME TIDAK AKTIF / MODEL TIDAK DITEMUKAN: Melatih dari nol...")
                env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=config.GAMMA)
                model = PPO(policy="MlpPolicy", env=env, verbose=0, tensorboard_log=str(config.LOG_DIR), **config.PPO_KW)
                model.set_random_seed(42)
        else:
            env = VecNormalize.load(str(vec_path), env)
            env.training = True
            model.set_env(env)
            
        ckpt_cb = CheckpointCallback(save_freq=max(10_000, args.timesteps // 20), save_path=str(config.MODEL_DIR / "checkpoints"), name_prefix=f"{tag}_step{i+1}")
        cb_list = CallbackList([ckpt_cb, TradingCallback()])
        
        step_timesteps = args.timesteps // len(splits)
        model.learn(total_timesteps=step_timesteps, callback=cb_list, reset_num_timesteps=False)
        
        model.save(str(config.MODEL_DIR / f"{tag}.zip"))
        env.save(str(config.MODEL_DIR / f"{tag}_vecnormalize.pkl"))
        
        with open(config.MODEL_DIR / f"{tag}_artifacts.pkl", "wb") as f:
            pickle.dump({
                "kalman_model": kalman_model,
                "feature_names": feature_names,
                "test_features": test_f,
                "test_prices": test_p,
                "test_dates": test_d,
                "episode_length": args.episode_length,
            }, f)
            
        print(f"[*] Menguji Model pada Unseen Data di Walk-Forward Step {i+1}...")
        test_env_fn = make_env_fn(test_f, test_p, test_s, test_d, args.episode_length)
        
        eval_episodes = 10 
        profit, trades, win_rate, s_time, e_time, dur, act_trades, avg_prof = evaluate_model(
            model, test_env_fn, args.symbol, episodes=eval_episodes, dataset_dates=test_d, mode_name="PRETRAIN", step_idx=i+1
        )
        
        save_evaluation("PRETRAIN", args.symbol, step_timesteps, profit, trades, win_rate, s_time, e_time, dur, eval_episodes, act_trades, avg_prof)
        print(f"[*] Step {i+1} Selesai! Avg Eval Profit: ${profit:,.2f} | Win Rate: {win_rate:.1f}%")
        
    print("\n[*] Seluruh Siklus Walk-Forward Selesai!")

def run_backtest(args):
    print("="*50)
    print(f"🔬 MEMULAI OFFLINE BACKTEST ({args.symbol}_{args.timeframe.upper()})")
    print("="*50)
    config.DEFAULT_TIMEFRAME = args.timeframe.upper()
    init_eval_db()
    
    df = load_csv(args.symbol, args.timeframe.upper(), data_dir=args.data_dir)
    tag = f"{config.MODEL_PREFIX}_{args.symbol}_{args.timeframe.upper()}"
    model_path = config.MODEL_DIR / f"{tag}.zip"
    vecnorm_path = config.MODEL_DIR / f"{tag}_vecnormalize.pkl"
    artifacts_path = config.MODEL_DIR / f"{tag}_artifacts.pkl"
    
    if not model_path.exists():
        print(f"[!] File model {tag} tidak ditemukan! Lakukan pretrain terlebih dahulu.")
        return
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        kalman_model = artifacts["kalman_model"]
        
    features, prices, spreads, dates, _, _ = build_feature_matrix(df, args.timeframe.upper(), use_kalman=True, kalman_model=kalman_model)
    
    splits = walk_forward_splits(features, prices, spreads, dates, n_splits=4)
    _, (test_f, test_p, test_s, test_d) = splits[-1] 
    
    if args.data_split == "test":
        eval_f, eval_p, eval_s, eval_d = test_f, test_p, test_s, test_d
    else:
        eval_f, eval_p, eval_s, eval_d = features, prices, spreads, dates
    env_fn = make_env_fn(eval_f, eval_p, eval_s, eval_d, args.episode_length)
    model = PPO.load(model_path)
    
    profit, trades, win_rate, s_time, e_time, dur, act_trades, avg_prof = evaluate_model(model, env_fn, args.symbol, episodes=args.episodes, dataset_dates=eval_d, mode_name="BACKTEST")
    save_evaluation("BACKTEST", args.symbol, 0, profit, trades, win_rate, s_time, e_time, dur, args.episodes, act_trades, avg_prof)
    
    print("\n" + "="*50)
    print(f"[*] TOTAL NET PROFIT : ${profit:,.2f} | Win Rate: {win_rate:.1f}%")
    print("="*50)

def run_live(args):
    ACTIVE_SYMBOL = "XAUUSDc" if args.mode == "real" else "XAUUSDm"
    
    # PERBAIKAN: Kita hapus hardcode path MT5. 
    # mt5.initialize() tanpa parameter otomatis mendeteksi MetaTrader 5 yang sedang menyala di AWS.
    if not mt5.initialize():
        print(f"[!] GAGAL MENGHUBUNGKAN KE MT5.")
        print(f"    Pesan Error: {mt5.last_error()}")
        print(f"    Pastikan aplikasi MetaTrader 5 di AWS sudah terbuka dan Algo Trading aktif!")
        return
        
    mt5.symbol_select(ACTIVE_SYMBOL, True)
    if config.PROXY_SYMBOL: mt5.symbol_select(config.PROXY_SYMBOL, True)
    
    tag = f"{config.MODEL_PREFIX}_{args.symbol}_{args.timeframe.upper()}"
    model_path = config.MODEL_DIR / f"{tag}.zip"
    vecnorm_path = config.MODEL_DIR / f"{tag}_vecnormalize.pkl"
    
    if not model_path.exists():
        print(f"[!] Model {tag} tidak ditemukan.")
        mt5.shutdown(); return
        
    with open(config.MODEL_DIR / f"{tag}_artifacts.pkl", "rb") as f:
        kalman_model = pickle.load(f)["kalman_model"]
        
    base_env = LiveMT5Env(ACTIVE_SYMBOL, kalman_model, timeframe=args.timeframe.upper(), db_file=args.db, magic=args.magic)
    env = DummyVecEnv([lambda: base_env])
    
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False 
        env.norm_reward = False
        
    model = PPO.load(model_path, custom_objects={"learning_rate": 0.0}) 
    
    print("\n=======================================================")
    print(f"🚀 AI TRADER [INFERENCE MODE] - {args.mode.upper()} ({ACTIVE_SYMBOL}) {args.timeframe.upper()} 🚀")
    print(f"Database : {args.db}")
    print(f"Magic No : {args.magic}")
    print(f"Model Aktif : {model_path.name}")
    print("Agent: PPO V2 Hybrid + Cloud Supervisor (FinBERT) Latar Belakang")
    print("=======================================================\n")
    
    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            base_env.last_normalized_obs = obs[0]
            obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        mt5.shutdown()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_pre = subparsers.add_parser("pretrain")
    p_pre.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    p_pre.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p_pre.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    p_pre.add_argument("--episode-length", type=int, default=config.EPISODE_LENGTH)
    p_pre.add_argument("--data-dir", default=str(config.DATA_DIR))
    p_pre.add_argument("--resume", action="store_true", help="Lanjutkan training dari model sebelumnya tanpa menghapus ingatan")
    
    p_back = subparsers.add_parser("backtest")
    p_back.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    p_back.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p_back.add_argument("--episode-length", type=int, default=config.EPISODE_LENGTH)
    p_back.add_argument("--data-dir", default=str(config.DATA_DIR))
    p_back.add_argument("--episodes", type=int, default=10)
    p_back.add_argument("--data-split", type=str, choices=["test", "all"], default="test")
    
    p_live = subparsers.add_parser("live")
    p_live.add_argument("--mode", type=str, choices=["demo", "real"], default="demo")
    p_live.add_argument("--symbol", type=str, default=config.DEFAULT_SYMBOL)
    p_live.add_argument("--timeframe", type=str, default="5M") 
    p_live.add_argument("--db", type=str, default="bot_database.db", help="Nama file database SQLite untuk bot ini")
    p_live.add_argument("--magic", type=int, default=10001, help="Magic Number unik MT5 untuk bot ini")
    
    args = parser.parse_args()
    if args.command == "pretrain": run_pretrain(args)
    elif args.command == "backtest": run_backtest(args)
    elif args.command == "live": run_live(args)

if __name__ == "__main__":
    main()