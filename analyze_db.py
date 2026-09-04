import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# PERBAIKAN: Menyesuaikan path karena script berada di folder yang sama dengan database
DB_PATH = "db2_m5.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    
    df = pd.read_sql_query("""
        SELECT * FROM trade_details 
        WHERE status = 'CLOSED' 
        ORDER BY timestamp_close ASC
    """, conn)
    
    conn.close()

    if df.empty:
        print("Belum ada data trade CLOSED di database.")
        return

    df['timestamp_close'] = pd.to_datetime(df['timestamp_close'])
    df['date'] = df['timestamp_close'].dt.date
    df['hour'] = df['timestamp_close'].dt.hour
    df['is_win'] = df['net_profit'] > 0

    df['balance_curve'] = df['balance_after']
    df['peak'] = df['balance_curve'].cummax()
    df['drawdown_usd'] = df['peak'] - df['balance_curve']
    df['drawdown_pct'] = (df['drawdown_usd'] / df['peak']) * 100

    start_balance = df['balance_before'].iloc[0]
    end_balance = df['balance_after'].iloc[-1]
    total_net_profit = df['net_profit'].sum()
    max_dd_pct = df['drawdown_pct'].max()
    max_dd_usd = df['drawdown_usd'].max()
    total_trades = len(df)
    win_rate = (df['is_win'].sum() / total_trades) * 100

    daily_stats = df.groupby('date').agg(
        total_trades=('id', 'count'),
        wins=('is_win', 'sum'),
        net_profit=('net_profit', 'sum')
    ).reset_index()
    daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['total_trades']) * 100

    hourly_stats = df.groupby('hour').agg(
        total_trades=('id', 'count'),
        wins=('is_win', 'sum'),
        net_profit=('net_profit', 'sum')
    ).reset_index()
    hourly_stats['win_rate'] = (hourly_stats['wins'] / hourly_stats['total_trades']) * 100

    print("="*50)
    print("📊 LAPORAN PERFORMA BOT (M5 STANDALONE) 📊")
    print("="*50)
    print(f"Saldo Awal      : ${start_balance:,.2f}")
    print(f"Saldo Akhir     : ${end_balance:,.2f}")
    print(f"Total Profit    : ${total_net_profit:,.2f}")
    print(f"Max Drawdown    : {max_dd_pct:.2f}% (${max_dd_usd:,.2f})")
    print(f"Total Trades    : {total_trades}")
    print(f"Win Rate Keseluruhan: {win_rate:.2f}%\n")

    print("📅 STATISTIK PER HARI 📅")
    print("-" * 50)
    print(f"{'Tanggal':<12} | {'Trades':<6} | {'WinRate':<8} | {'Net Profit'}")
    print("-" * 50)
    for _, row in daily_stats.iterrows():
        print(f"{str(row['date']):<12} | {row['total_trades']:<6.0f} | {row['win_rate']:<7.2f}% | ${row['net_profit']:,.2f}")
    print("\n")

    print("⏰ STATISTIK PER JAM (WAKTU SERVER) ⏰")
    print("-" * 50)
    print(f"{'Jam':<5} | {'Trades':<6} | {'WinRate':<8} | {'Net Profit'}")
    print("-" * 50)
    hourly_sorted = hourly_stats.sort_values(by='net_profit', ascending=False)
    for _, row in hourly_sorted.iterrows():
        print(f"{row['hour']:02.0f}:00 | {row['total_trades']:<6.0f} | {row['win_rate']:<7.2f}% | ${row['net_profit']:,.2f}")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analisis Kinerja Quant Bot M5', fontsize=16)

    axes[0, 0].plot(df['timestamp_close'], df['balance_curve'], color='#10b981', linewidth=2)
    axes[0, 0].set_title('Kurva Pertumbuhan Saldo (Equity Curve)')
    axes[0, 0].set_ylabel('Balance (USD)')
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].fill_between(df['timestamp_close'], df['drawdown_pct'], color='#ef4444', alpha=0.5)
    axes[0, 1].set_title('Drawdown (%)')
    axes[0, 1].set_ylabel('Drop dari Peak (%)')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.2)

    colors = ['#10b981' if p > 0 else '#ef4444' for p in daily_stats['net_profit']]
    axes[1, 0].bar(daily_stats['date'].astype(str), daily_stats['net_profit'], color=colors)
    axes[1, 0].set_title('Net Profit Harian')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.2, axis='y')

    axes[1, 1].bar(hourly_stats['hour'], hourly_stats['net_profit'], color='#38bdf8')
    axes[1, 1].set_title('Distribusi Profit per Jam (Server Time)')
    axes[1, 1].set_xlabel('Jam (0-23)')
    axes[1, 1].set_xticks(range(0, 24))
    axes[1, 1].grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()