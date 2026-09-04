import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

from stable_baselines3 import DQN, PPO
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

import config
from trading_env import GoldTradingEnv

def load_model(algo, path):
    cls = {"dqn": DQN, "ddqn": DQN, "ppo": PPO, "rppo": RecurrentPPO}[algo]
    return cls.load(path)

def analyze_trades(positions, sizes, portfolio_values, prices, dates):
    trades = []
    current_trade = None
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        if prev_pos == 0.0 and curr_pos != 0.0:
            current_trade = {
                "type": "BUY" if curr_pos > 0 else "SELL",
                "entry_idx": i,
                "entry_price": prices[i],
                "entry_time": dates[i],
                "size_oz": sizes[i],
                "prices_during_trade": [prices[i]]
            }
            
        elif prev_pos == curr_pos and curr_pos != 0.0 and current_trade is not None:
            current_trade["prices_during_trade"].append(prices[i])
            
        elif prev_pos != curr_pos and prev_pos != 0.0 and current_trade is not None:
            current_trade["prices_during_trade"].append(prices[i])
            current_trade["exit_idx"] = i
            current_trade["exit_price"] = prices[i]
            current_trade["exit_time"] = dates[i]
            
            price_diff = current_trade["exit_price"] - current_trade["entry_price"]
            if current_trade["type"] == "SELL":
                price_diff = -price_diff
                
            current_trade["profit_pts"] = price_diff
            current_trade["profit_dollar"] = price_diff * current_trade["size_oz"]
            current_trade["is_win"] = current_trade["profit_dollar"] > 0
            
            trade_prices = np.array(current_trade["prices_during_trade"])
            if current_trade["type"] == "BUY":
                mfe_price = trade_prices.max()
                mae_price = trade_prices.min()
                current_trade["mfe_pts"] = mfe_price - current_trade["entry_price"]
                current_trade["mae_pts"] = current_trade["entry_price"] - mae_price
            else:
                mfe_price = trade_prices.min()
                mae_price = trade_prices.max()
                current_trade["mfe_pts"] = current_trade["entry_price"] - mfe_price
                current_trade["mae_pts"] = mae_price - current_trade["entry_price"]
                
            trades.append(current_trade)
            
            if curr_pos != 0.0:
                current_trade = {
                    "type": "BUY" if curr_pos > 0 else "SELL",
                    "entry_idx": i,
                    "entry_price": prices[i],
                    "entry_time": dates[i],
                    "size_oz": sizes[i],
                    "prices_during_trade": [prices[i]]
                }
            else:
                current_trade = None

    return trades

def print_advanced_metrics(trades, equity_curve, tag, test_dates):
    if not trades:
        print(f"\n[{tag}] Tidak ada transaksi yang dieksekusi.")
        return

    net_profit = equity_curve[-1] - equity_curve[0]
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown_dollars = running_peak - equity_curve
    max_dd_dollar = drawdown_dollars.max()
    
    dd_durations = []
    current_dd_start = None
    for i in range(len(equity_curve)):
        if drawdown_dollars[i] > 0:
            if current_dd_start is None:
                current_dd_start = pd.to_datetime(test_dates[i])
        else:
            if current_dd_start is not None:
                dd_durations.append((pd.to_datetime(test_dates[i]) - current_dd_start).total_seconds())
                current_dd_start = None
    if current_dd_start is not None:
        dd_durations.append((pd.to_datetime(test_dates[-1]) - current_dd_start).total_seconds())
    
    max_dd_duration_str = str(timedelta(seconds=int(max(dd_durations)))) if dd_durations else "0"
    max_dd_duration_str = max_dd_duration_str.replace("days", "days ")

    start_str = (pd.to_datetime(test_dates[0]) + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    end_str = (pd.to_datetime(test_dates[-1]) + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    
    winning_trades = [t for t in trades if t["is_win"]]
    losing_trades = [t for t in trades if not t["is_win"]]
    buy_trades = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    
    buy_win = sum(1 for t in buy_trades if t["is_win"])
    sell_win = sum(1 for t in sell_trades if t["is_win"])
    
    gross_win = sum(t["profit_dollar"] for t in winning_trades)
    gross_loss = sum(t["profit_dollar"] for t in losing_trades) 
    profit_factor = gross_win / abs(gross_loss) if gross_loss != 0 else float('inf')
    
    avg_win = gross_win / len(winning_trades) if winning_trades else 0
    avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
    
    total_lot_oz = sum(t["size_oz"] for t in trades)
    avg_profit_per_001_lot = (net_profit / total_lot_oz) if total_lot_oz > 0 else 0
    
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    streak_type = None
    
    for t in trades:
        if t["is_win"]:
            if streak_type == "WIN": current_streak += 1
            else: streak_type = "WIN"; current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "LOSS": current_streak += 1
            else: streak_type = "LOSS"; current_streak = 1
            max_loss_streak = max(max_loss_streak, current_streak)
            
    win_durations = [(pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).total_seconds() for t in winning_trades]
    loss_durations = [(pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).total_seconds() for t in losing_trades]
    
    avg_win_duration = str(timedelta(seconds=int(np.mean(win_durations)))) if win_durations else "0"
    avg_loss_duration = str(timedelta(seconds=int(np.mean(loss_durations)))) if loss_durations else "0"
    avg_win_duration = avg_win_duration.replace("days", "days ")
    avg_loss_duration = avg_loss_duration.replace("days", "days ")
    
    trade_efficiencies = [(t["mfe_pts"] / (t["mfe_pts"] + t["mae_pts"])) for t in winning_trades if (t["mfe_pts"] + t["mae_pts"]) > 0]
    avg_efficiency = np.mean(trade_efficiencies) * 100 if trade_efficiencies else 0
    
    r_dist = config.SL_PRICE_DIST
    mfe_025r = sum(1 for t in trades if t["mfe_pts"] >= 0.25 * r_dist) / len(trades) * 100 if trades else 0
    mfe_05r = sum(1 for t in trades if t["mfe_pts"] >= 0.50 * r_dist) / len(trades) * 100 if trades else 0
    mfe_075r = sum(1 for t in trades if t["mfe_pts"] >= 0.75 * r_dist) / len(trades) * 100 if trades else 0
    mfe_1r = sum(1 for t in trades if t["mfe_pts"] >= 1.0 * r_dist) / len(trades) * 100 if trades else 0
    
    loss_but_green = sum(1 for t in losing_trades if t["mfe_pts"] > 0)
    loss_025r = sum(1 for t in losing_trades if t["mfe_pts"] >= 0.25 * r_dist)
    loss_05r = sum(1 for t in losing_trades if t["mfe_pts"] >= 0.50 * r_dist)
    
    total_loss = len(losing_trades)
    pct_loss_green = (loss_but_green / total_loss * 100) if total_loss else 0
    pct_loss_025r = (loss_025r / total_loss * 100) if total_loss else 0
    pct_loss_05r = (loss_05r / total_loss * 100) if total_loss else 0
    
    df_trades = pd.DataFrame(trades)
    df_trades['entry_time_wita'] = pd.to_datetime(df_trades['entry_time']) + pd.Timedelta(hours=8)
    df_trades['hour'] = df_trades['entry_time_wita'].dt.hour
    df_trades['day_name'] = df_trades['entry_time_wita'].dt.day_name()
    
    profit_by_hour = df_trades.groupby('hour')['profit_dollar'].sum()
    profit_by_day = df_trades.groupby('day_name')['profit_dollar'].sum()
    
    print("\n=====================================================================")
    print(f" LAPORAN EVALUASI MENDALAM BOT RL: {tag.upper()}")
    print("=====================================================================")
    print(f" Periode (WITA)   : {start_str}  ->  {end_str}")
    print(f" Profit Bersih    : ${net_profit:,.2f}")
    print(f" Drawdown Maks    : -${max_dd_dollar:,.2f}  |  Rekor Lama DD: {max_dd_duration_str}")
    print(f" Profit Factor    : {profit_factor:.2f}  (Gross Win: ${gross_win:,.2f} / Gross Loss: -${abs(gross_loss):,.2f})")
    print(f" Total Trades     : {len(trades)}  ( BUY: {len(buy_trades)} | SELL: {len(sell_trades)} )")
    
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    buy_wr = (buy_win / len(buy_trades) * 100) if buy_trades else 0
    sell_wr = (sell_win / len(sell_trades) * 100) if sell_trades else 0
    print(f" Win Rate         : GLOBAL {win_rate:.2f}%  ( BUY WR: {buy_wr:.2f}% | SELL WR: {sell_wr:.2f}% )")
    print(f" Avg Win / Loss   : Win ${avg_win:,.2f} / Loss -${abs(avg_loss):,.2f}")
    print(f" Avg profit/0.01lot: ${avg_profit_per_001_lot:.3f}  <- metrik FAIR lintas mode")
    print(f" Streaks          : Menang Beruntun Maks {max_win_streak}x | Kalah Beruntun Maks {max_loss_streak}x")
    print(f" > Trade Duration : Rata-rata Win ({avg_win_duration}) | Rata-rata Loss ({avg_loss_duration})")
    print(f" > Trade Efficency: {avg_efficiency:.1f}% (Seberapa bersih posisi Win meluncur ke target tanpa menyentuh MAE)")
    print(f" MFE (Favorable)  : {mfe_025r:.1f}% capai 0.25R | {mfe_05r:.1f}% capai 0.5R | {mfe_075r:.1f}% capai 0.75R | {mfe_1r:.1f}% capai 1R")
    print(f" Rugi tp Sempat + : {pct_loss_green:.1f}% rugi padahal sempat hijau | {pct_loss_025r:.1f}% rugi padahal capai 0.25R | {pct_loss_05r:.1f}% rugi padahal capai 0.5R")
    
    print("\n --- ANALISA PERFORMA BERDASARKAN HARI (WAKTU WITA) ---")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for day in days_order:
        val = profit_by_day.get(day, 0)
        print(f" {day:<10}: ${val:>9,.2f}")
        
    print("\n --- ANALISA PERFORMA BERDASARKAN JAM WITA (TOP 3 & BOTTOM 3) ---")
    sorted_hours = profit_by_hour.sort_values(ascending=False)
    print(" Jam Terbaik  : " + " | ".join([f"{h:02d}:00 (${v:,.0f})" for h, v in sorted_hours.head(3).items()]))
    print(" Jam Terburuk : " + " | ".join([f"{h:02d}:00 (${v:,.0f})" for h, v in sorted_hours.tail(3).items()]))
    
    print("\n --- TOP 5 WORST TRADES (WAKTU WITA) ---")
    worst_trades = sorted(trades, key=lambda x: x["profit_dollar"])[:5]
    for i, wt in enumerate(worst_trades):
        wita_time = pd.to_datetime(wt['entry_time']) + pd.Timedelta(hours=8)
        print(f" {i+1}. {wita_time.strftime('%Y-%m-%d %H:%M:%S')} | {wt['type']} | Size: {wt['size_oz']:.2f} oz | Loss: ${wt['profit_dollar']:,.2f}")
    print("=====================================================================\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p.add_argument("--algos", nargs="+", default=["ppo", "rppo"], help="List algoritma yang ingin dievaluasi")
    args = p.parse_args()

    model_files = list(config.MODEL_DIR.glob("*_artifacts.pkl"))
    if not model_files:
        print("Tidak ada model yang ditemukan di folder models/. Jalankan train.py terlebih dahulu.")
        return

    for artifact_path in sorted(model_files):
        tag = artifact_path.stem.replace("_artifacts", "")
        model_path = config.MODEL_DIR / f"{tag}.zip"
        if not model_path.exists(): continue
        
        algo = tag.split("_")[0]  
        
        if algo not in args.algos:
            continue

        with open(artifact_path, "rb") as f: art = pickle.load(f)

        model = load_model(algo, str(model_path))
        env = GoldTradingEnv(
            art["test_features"], art["test_prices"], dates=art["test_dates"],
            episode_length=len(art["test_features"]) - 1, random_start=False,
            use_margin_call=False 
        )
        
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        equity_curve = [env.portfolio_value]
        positions = [env.position]
        sizes = [env.current_size_oz]
        
        done = False
        while not done:
            if algo == "rppo":
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            equity_curve.append(info["portfolio_value"])
            positions.append(info["position"])
            sizes.append(info.get("size_oz", 0.0))

        trades = analyze_trades(positions, sizes, equity_curve, art["test_prices"], art["test_dates"])
        print_advanced_metrics(trades, equity_curve, tag, art["test_dates"])

        plot_dates_wita = pd.to_datetime(art["test_dates"][:len(equity_curve)]) + pd.Timedelta(hours=8)
        equity_arr = np.array(equity_curve)
        running_peak = np.maximum.accumulate(equity_arr)
        drawdown_pct = np.where(running_peak > 0, (running_peak - equity_arr) / running_peak * 100, 0)
        
        fig, axes = plt.subplots(6, 1, figsize=(14, 26), gridspec_kw={'height_ratios': [3, 1.5, 1, 2, 2, 2]})
        
        axes[0].plot(plot_dates_wita, equity_arr, label="Strategy Equity", color='royalblue', linewidth=1.2)
        axes[0].set_title(f"DASHBOARD ANALITIK BOT (WITA): {tag.upper()}", fontsize=16, fontweight='bold')
        axes[0].set_ylabel("Balance (USD)", fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        axes[1].fill_between(plot_dates_wita, 0, drawdown_pct, color='crimson', alpha=0.6, label="Drawdown (%)")
        axes[1].set_ylabel("Drawdown %", fontsize=10)
        axes[1].set_ylim(max(drawdown_pct) * 1.1 if max(drawdown_pct) > 0 else 1, 0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left")

        axes[2].plot(plot_dates_wita, positions, label="Position (1=Buy, -1=Sell)", drawstyle="steps-post", color='darkorange')
        axes[2].set_ylabel("Position", fontsize=10)
        axes[2].set_yticks([-1, 0, 1])
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper left")
        
        if trades:
            exit_dates_wita = [pd.to_datetime(t["exit_time"]) + pd.Timedelta(hours=8) for t in trades]
            profits = [t["profit_dollar"] for t in trades]
            colors = ['limegreen' if p > 0 else 'crimson' for p in profits]
            
            axes[3].scatter(exit_dates_wita, profits, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=30)
            axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[3].set_ylabel("Trade Profit (USD)", fontsize=10)
            axes[3].set_title("Distribusi Hasil Transaksi (Scatter Plot - WITA)", fontsize=12)
            axes[3].grid(True, alpha=0.3)
            
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            df_trades = pd.DataFrame(trades)
            df_trades['entry_time_wita'] = pd.to_datetime(df_trades['entry_time']) + pd.Timedelta(hours=8)
            df_trades['hour'] = df_trades['entry_time_wita'].dt.hour
            df_trades['day_name'] = df_trades['entry_time_wita'].dt.day_name()
            
            hourly_profit = df_trades.groupby('hour')['profit_dollar'].sum().reindex(range(24), fill_value=0)
            
            bar_colors = ['limegreen' if val > 0 else 'crimson' for val in hourly_profit.values]
            axes[4].bar(hourly_profit.index, hourly_profit.values, color=bar_colors, alpha=0.8, edgecolor='black')
            axes[4].set_title("Distribusi Profit Berdasarkan Jam Entry (WITA: 00:00 - 23:00)", fontsize=12)
            axes[4].set_xlabel("Jam (WITA)", fontsize=10)
            axes[4].set_ylabel("Net Profit (USD)", fontsize=10)
            axes[4].set_xticks(range(24))
            axes[4].grid(True, alpha=0.3, axis='y')
            
            df_trades['day_name'] = pd.Categorical(df_trades['day_name'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], ordered=True)
            daily_profit = df_trades.groupby('day_name')['profit_dollar'].sum()
            day_colors = ['limegreen' if val > 0 else 'crimson' for val in daily_profit.values]
            
            axes[5].bar(daily_profit.index, daily_profit.values, color=day_colors, alpha=0.8, edgecolor='black')
            axes[5].set_title("Distribusi Profit Berdasarkan Hari (WITA)", fontsize=12)
            axes[5].set_xlabel("Hari (WITA)", fontsize=10)
            axes[5].set_ylabel("Net Profit (USD)", fontsize=10)
            axes[5].grid(True, alpha=0.3, axis='y')
            
            df_trades['date_wita'] = df_trades['entry_time_wita'].dt.date
            daily_stats = df_trades.groupby('date_wita').agg(
                trades=('profit_dollar', 'count'),
                profit=('profit_dollar', 'sum'),
                wins=('is_win', 'sum')
            )
            daily_stats['winrate'] = (daily_stats['wins'] / daily_stats['trades'] * 100).fillna(0)
            
            fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            profit_colors = ['limegreen' if val > 0 else 'crimson' for val in daily_stats['profit']]
            axes2[0].bar(daily_stats.index, daily_stats['profit'], color=profit_colors, alpha=0.8, edgecolor='black')
            axes2[0].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes2[0].set_title(f"Harian: Net Profit (USD) - {tag.upper()}", fontsize=12, fontweight='bold')
            axes2[0].set_ylabel("Profit (USD)", fontsize=10)
            axes2[0].grid(True, alpha=0.3, axis='y')

            axes2[1].bar(daily_stats.index, daily_stats['trades'], color='cornflowerblue', alpha=0.8, edgecolor='black')
            axes2[1].set_title("Harian: Jumlah Transaksi (Trades)", fontsize=12, fontweight='bold')
            axes2[1].set_ylabel("Jumlah Trades", fontsize=10)
            axes2[1].grid(True, alpha=0.3, axis='y')

            axes2[2].plot(daily_stats.index, daily_stats['winrate'], marker='o', color='darkorchid', linewidth=2)
            axes2[2].axhline(50, color='crimson', linestyle='--', alpha=0.8, label="Batas Kritis 50%")
            axes2[2].set_title("Harian: Win Rate (%)", fontsize=12, fontweight='bold')
            axes2[2].set_ylabel("Win Rate (%)", fontsize=10)
            axes2[2].set_xlabel("Tanggal (WITA)", fontsize=10)
            axes2[2].set_ylim(0, 105)
            axes2[2].grid(True, alpha=0.3)
            axes2[2].legend(loc="lower right")

            axes2[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes2[2].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes2[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            chart_path_daily = config.RESULTS_DIR / f"{tag}_daily_performance.png"
            fig2.savefig(chart_path_daily, dpi=200, bbox_inches='tight')
            print(f"[*] Grafik Performa Harian (Date, Trades, WinRate, Profit) disimpan di: {chart_path_daily}\n")
            plt.close(fig2)

        plt.tight_layout()
        chart_path = config.RESULTS_DIR / f"{tag}_advanced_dashboard.png"
        fig.savefig(chart_path, dpi=200, bbox_inches='tight')
        print(f"[*] Dashboard analitik 6 Panel disimpan di: {chart_path}\n")
        plt.close(fig)

if __name__ == "__main__": 
    main()