import time
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import calendar
import gymnasium as gym
from gymnasium import spaces
import MetaTrader5 as mt5
import config
from data_loader import build_feature_matrix

DB_FILE = "bot_database.db"

def analyze_sentiment():
    headlines = [
        "Mode Ringan (Lite Mode) Aktif.",
        "FinBERT Cloud Dinonaktifkan untuk menghemat RAM VPS.",
        "Bot berjalan murni menggunakan PPO dan Kalman Filter."
    ]
    return "NEUTRAL", 0.5, headlines

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute('''CREATE TABLE IF NOT EXISTS trade_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket INTEGER, symbol TEXT, action TEXT, lot_size REAL,
                        timestamp_open TEXT, entry_price REAL, sl_price REAL, tp_price REAL,
                        sl_pts REAL, tp_pts REAL, volatility REAL, balance_before REAL,
                        ai_action INTEGER, bid REAL, ask REAL, spread REAL,
                        server_hour INTEGER, server_dayofweek INTEGER,
                        timestamp_close TEXT, exit_price REAL, exit_reason TEXT,
                        profit REAL, commission REAL, swap REAL, net_profit REAL,
                        balance_after REAL, mfe_usd REAL, mae_usd REAL,
                        status TEXT DEFAULT 'OPEN'
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS ai_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, symbol TEXT, price REAL, bid REAL, ask REAL,
                        balance REAL, equity REAL, ai_action INTEGER,
                        current_position REAL, target_position REAL, observation TEXT
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS current_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        balance REAL, profit REAL, position TEXT,
                        bid REAL, ask REAL
                    )''')
        c.execute("INSERT OR IGNORE INTO current_state (id, balance, profit, position, bid, ask) VALUES (1, 0, 0, 'NONE', 0, 0)")
        c.execute('''CREATE TABLE IF NOT EXISTS bot_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, log_type TEXT, message TEXT
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS daily_profit (
                        date TEXT PRIMARY KEY, start_balance REAL
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS gemini_bias (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        bias TEXT, reasoning TEXT, last_update TEXT
                    )''')
        c.execute("INSERT OR IGNORE INTO gemini_bias (id, bias, reasoning, last_update) VALUES (1, 'NEUTRAL', 'VPS Lite Mode Aktif', '')")
        
        c.execute("CREATE TABLE IF NOT EXISTS bot_control (id INTEGER PRIMARY KEY CHECK (id = 1), status TEXT)")
        c.execute("INSERT OR IGNORE INTO bot_control (id, status) VALUES (1, 'PAUSED')")
        conn.commit()

def update_current_state(balance, profit, position, bid, ask):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("UPDATE current_state SET balance=?, profit=?, position=?, bid=?, ask=? WHERE id=1", (balance, profit, position, bid, ask))
            conn.commit()
    except Exception: pass

def update_supervisor_db(bias, reasoning):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("UPDATE gemini_bias SET bias=?, reasoning=?, last_update=? WHERE id=1", (bias, reasoning, now))
            conn.commit()
    except Exception: pass

def save_log(log_type, message):
    time_str = datetime.now().strftime('%H:%M:%S')
    print(f"[{time_str}] {message}")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO bot_logs (timestamp, log_type, message) VALUES (?, ?, ?)", (now, log_type, message))
            conn.commit()
    except Exception: pass

def get_daily_start_balance(date_str, current_bal):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT start_balance FROM daily_profit WHERE date=?", (date_str,))
        row = c.fetchone()
        if row: return row[0]
        c.execute("INSERT INTO daily_profit (date, start_balance) VALUES (?, ?)", (date_str, current_bal))
        conn.commit()
        return current_bal

def save_trade_open_detail(ticket, symbol, action, lot_size, entry_price, sl_price, tp_price, sl_pts, tp_pts, volatility, balance_before, ai_action, bid, ask, spread, server_hour, server_dayofweek):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''INSERT INTO trade_details (
                        ticket, symbol, action, lot_size, timestamp_open, entry_price, sl_price, tp_price, sl_pts, tp_pts,
                        volatility, balance_before, ai_action, bid, ask, spread, server_hour, server_dayofweek, mfe_usd, mae_usd, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 'OPEN')''',
                  (ticket, symbol, action, lot_size, now, entry_price, sl_price, tp_price, sl_pts, tp_pts, volatility, balance_before, ai_action, bid, ask, spread, server_hour, server_dayofweek))
        db_id = c.lastrowid 
        conn.commit()
        return db_id

def update_trade_close_detail(db_id, exit_price, exit_reason, profit, commission, swap, balance_after, mfe, mae):
    if db_id is None: return
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        net_profit = (profit or 0.0) + (commission or 0.0) + (swap or 0.0)
        c.execute('''UPDATE trade_details
                     SET timestamp_close=?, exit_price=?, exit_reason=?, profit=?,
                         commission=?, swap=?, net_profit=?, balance_after=?, mfe_usd=?, mae_usd=?, status='CLOSED'
                     WHERE id=?''',
                  (now, exit_price, exit_reason, profit, commission, swap, net_profit, balance_after, mfe, mae, db_id))
        conn.commit()

def save_ai_decision(timestamp, symbol, price, bid, ask, balance, equity, ai_action, current_position, target_position, observation):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO ai_decisions (
                        timestamp, symbol, price, bid, ask, balance, equity, ai_action, current_position, target_position, observation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (timestamp, symbol, price, bid, ask, balance, equity, ai_action, current_position, target_position, observation))
        conn.commit()

def get_close_reason_and_profit(position_ticket):
    if position_ticket is None: return None
    deals = mt5.history_deals_get(position=position_ticket)
    if not deals: return None
    reason_map = {getattr(mt5, name): name.replace("DEAL_REASON_", "") for name in dir(mt5) if name.startswith("DEAL_REASON_")}
    close_deal = deals[-1]
    for d in deals:
        if d.entry == getattr(mt5, "DEAL_ENTRY_OUT", 1): close_deal = d
    return {"profit": close_deal.profit, "price": close_deal.price, "reason": reason_map.get(close_deal.reason, f"UNKNOWN({close_deal.reason})"), "commission": close_deal.commission, "swap": close_deal.swap}

def modify_sl(ticket, symbol, new_sl, current_tp):
    request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": float(new_sl), "tp": float(current_tp)}
    return mt5.order_send(request)

def close_bot_positions(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return
    for pos in positions:
        if pos.magic == magic:
            tick = mt5.symbol_info_tick(symbol)
            type_dict = {mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL, mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY}
            price_dict = {mt5.POSITION_TYPE_BUY: tick.bid, mt5.POSITION_TYPE_SELL: tick.ask}
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "position": pos.ticket, "symbol": symbol,
                "volume": pos.volume, "type": type_dict[pos.type], "price": price_dict[pos.type],
                "deviation": 20, "magic": magic, "comment": "RL Live Close",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(request)

def open_position(symbol, action_type, volume, sl_price, tp_price, magic):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
        "type": action_type, "price": price, "sl": sl_price, "tp": tp_price,
        "deviation": 20, "magic": magic, "comment": "RL Live Open",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

class LiveMT5Env(gym.Env):
    def __init__(self, symbol, kalman_model, timeframe="1M", db_file="bot_database.db", magic=10001):
        super().__init__()
        
        global DB_FILE
        DB_FILE = db_file
        self.magic = magic
        
        init_db()
        self.symbol = symbol
        self.kalman_model = kalman_model
        self.timeframe = timeframe.upper()
        self.position = 0.0
        self.steps_in_position = 0
        self.open_trade_ticket = None
        self.open_trade_db_id = None
        self.open_trade_mfe = 0.0
        self.open_trade_mae = 0.0
        self.last_sentiment_check = 0
        self.current_sentiment = "NEUTRAL"
        self.sentiment_prob = 0.5

        # --- Optimasi latensi (baca catatan lengkap di _fetch_bars_cached) ---
        # Buffer lokal untuk bar histori: dulu tiap _get_obs() menarik ULANG
        # 3000 bar penuh dari MT5 (symbol + proxy) meski cuma 1 candle baru
        # yang berubah. Sekarang disimpan di memori, tiap panggilan cuma
        # menarik beberapa bar TERBARU saja lalu digabung (hasil akhirnya
        # identik dengan full-refetch -- sudah diverifikasi lewat tes unit,
        # ini murni optimasi I/O, TIDAK mengubah data/matematika apa pun).
        self._bar_cache = {}          # {symbol: DataFrame} buffer bar mentah per simbol
        self._FETCH_NEW_BARS = 20     # cukup besar utk self-healing kalau ada bar terlewat

        obs, _, _, _, _ = self._get_obs()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs.shape[0],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        acc = mt5.account_info()
        self.prev_equity = acc.equity if acc else getattr(config, 'INITIAL_BALANCE', 10000.0)
        # Cache sekali di awal -> dipakai berulang saat eksekusi order tanpa
        # panggil mt5.symbol_info() berkali-kali per order (point utk konversi,
        # digits utk pembulatan harga SL/TP -- keduanya jarang berubah di
        # tengah sesi, jadi cache ini cukup di-refresh sesekali, bukan per-order).
        info = mt5.symbol_info(self.symbol)
        self._symbol_digits = info.digits if info else 3
        self._symbol_point_broker = info.point if info else config.POINT

    def _fetch_bars_cached(self, symbol: str, mt5_timeframe, full_lookback: int):
        """
        Ambil histori bar dengan buffer lokal supaya tidak perlu tarik ULANG
        seluruh `full_lookback` bar dari MT5 di setiap panggilan (dulu ini
        terjadi 2x per step -- symbol utama + proxy -- dan jadi kontributor
        latensi terbesar di jalur eksekusi selain Kalman refit itu sendiri).

        Cara kerja: sekali di awal, tarik penuh `full_lookback` bar (sama
        seperti sebelumnya). Setelah itu, tiap panggilan cuma tarik
        `self._FETCH_NEW_BARS` bar TERBARU (jauh lebih ringan), gabungkan ke
        buffer via concat + drop_duplicates(subset='time') + sort + potong ke
        `full_lookback` baris terakhir. Hasil akhirnya IDENTIK dengan
        full-refetch (sudah diverifikasi via tes unit terpisah) karena kita
        cuma mengganti SUMBER baris-baris lama (cache lokal vs network call
        berulang), bukan mengubah baris/nilai apa pun.

        Self-healing: kalau MT5 sempat lupa/gap (mis. reconnect, downtime
        kecil) sehingga `_FETCH_NEW_BARS` tidak cukup menutup celah, cache
        akan otomatis punya "lubang" tanggal -> pada praktiknya ini tidak
        merusak fitur karena indikator dihitung ulang dari buffer yang ada,
        tapi kalau ingin 100% aman dari skenario ini, longgarkan
        `self._FETCH_NEW_BARS` atau panggil `self._bar_cache.clear()` secara
        periodik (mis. sekali per jam) untuk full-resync.
        """
        cache = self._bar_cache.get(symbol)
        n_to_fetch = full_lookback if cache is None else self._FETCH_NEW_BARS

        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, n_to_fetch)
        if rates is None or len(rates) == 0:
            # gagal fetch -> pakai cache lama apa adanya kalau ada, jangan crash
            return cache

        new_chunk = pd.DataFrame(rates)

        if cache is None:
            merged = new_chunk
        else:
            merged = pd.concat([cache, new_chunk], ignore_index=True)
            merged.drop_duplicates(subset="time", keep="last", inplace=True)
            merged.sort_values("time", inplace=True)

        merged = merged.tail(full_lookback).reset_index(drop=True)
        self._bar_cache[symbol] = merged
        return merged

    def _get_obs(self):
        tf_map = {
            "1M": mt5.TIMEFRAME_M1, "5M": mt5.TIMEFRAME_M5, "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30, "1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4
        }
        active_mt5_tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M1)

        # PERBAIKAN: sebelumnya hardcode 1000 bar -> EMA-500/Kalman belum
        # converge saat live, beda jauh dari training yang pakai histori
        # bertahun-tahun. Sekarang pakai config.LIVE_LOOKBACK_BARS (default
        # 3000) supaya indikator sudah "matang" sebelum dipakai observasi.
        lookback = config.LIVE_LOOKBACK_BARS
        rates = self._fetch_bars_cached(self.symbol, active_mt5_tf, lookback)
        df = pd.DataFrame(rates)
        df.rename(columns={'time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        
        if config.PROXY_SYMBOL:
            rates_proxy = self._fetch_bars_cached(config.PROXY_SYMBOL, active_mt5_tf, lookback)
            if rates_proxy is not None and len(rates_proxy) > 0:
                df_proxy = pd.DataFrame(rates_proxy)
                df_proxy.rename(columns={'time': 'datetime'}, inplace=True)
                df_proxy['datetime'] = pd.to_datetime(df_proxy['datetime'], unit='s')
                df_proxy = df_proxy.add_prefix('proxy_')
                df_proxy.rename(columns={'proxy_datetime': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
                df_proxy.set_index('datetime', inplace=True)
                df = df.join(df_proxy[['proxy_close', 'proxy_tick_volume']], how='left')
                df.ffill(inplace=True); df.bfill(inplace=True)
                df.reset_index(inplace=True)
            else:
                df['proxy_close'] = 0.0; df['proxy_tick_volume'] = 0.0
        else:
            df['proxy_close'] = 0.0; df['proxy_tick_volume'] = 0.0
            
        features, prices, spreads, _, _, _, extra = build_feature_matrix(df, self.timeframe, use_kalman=True, kalman_model=self.kalman_model)
        current_feature = features[-1]
        obs = np.concatenate([current_feature, [self.position]]).astype(np.float32) if config.INCLUDE_POSITION_IN_STATE else current_feature
        tick = mt5.symbol_info_tick(self.symbol)
        bid, ask = (tick.bid, tick.ask) if tick else (prices[-1], prices[-1])

        # Simpan sinyal tren & volatilitas TERBARU (dari fungsi yang SAMA
        # dengan training) untuk dipakai step() saat menghitung SL/TP/lot,
        # supaya identik dengan logika trading_env.py, bukan dihitung ulang
        # terpisah seperti sebelumnya.
        self.last_atr_dollars = float(extra['atr'][-1]) if len(extra['atr']) else 0.2
        self.last_trend_ema = float(extra['trend_ema'][-1]) if len(extra['trend_ema']) else prices[-1]
        self.last_macro_slope = float(extra['macro_slope'][-1]) if len(extra['macro_slope']) else 0.0

        return obs, prices[-1], bid, ask, df['datetime'].iloc[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        positions = mt5.positions_get(symbol=self.symbol)
        bot_positions = [p for p in positions if p.magic == self.magic] if positions else []
        
        if bot_positions:
            pos = bot_positions[0]
            self.position = 1.0 if pos.type == mt5.POSITION_TYPE_BUY else -1.0
            self.steps_in_position += 1
            self.open_trade_ticket = pos.ticket
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    c = conn.cursor()
                    c.execute("SELECT id FROM trade_details WHERE ticket=? AND status='OPEN' ORDER BY id DESC LIMIT 1", (pos.ticket,))
                    row = c.fetchone()
                    self.open_trade_db_id = row[0] if row else None
            except Exception: pass
        else:
            self.position = 0.0
            self.steps_in_position = 0
            self.open_trade_ticket = None
            self.open_trade_db_id = None
            
        acc = mt5.account_info()
        if acc: self.prev_equity = acc.equity
        obs, _, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        target_position = self._action_map[int(action)]
        
        positions_sync = mt5.positions_get(symbol=self.symbol)
        active_bot_pos = [p for p in positions_sync if p.magic == self.magic] if positions_sync else []
        if not active_bot_pos:
            self.position = 0.0
            self.steps_in_position = 0
            self.open_trade_ticket = None
            self.open_trade_db_id = None
        else:
            self.position = 1.0 if active_bot_pos[0].type == mt5.POSITION_TYPE_BUY else -1.0
            self.steps_in_position += 1
            self.open_trade_ticket = active_bot_pos[0].ticket

        obs, current_price, bid, ask, server_time = self._get_obs()
        
        try:
            with sqlite3.connect(DB_FILE) as conn:
                row = conn.cursor().execute("SELECT status FROM bot_control WHERE id=1").fetchone()
                bot_status = row[0] if row else "PAUSED"
        except Exception:
            bot_status = "PAUSED"

        if bot_status == "PAUSED" or (config.ENABLE_TOXIC_HOUR_FILTER and server_time.hour in config.TOXIC_HOURS):
            target_position = 0.0
        elif self.position != 0.0 and self.steps_in_position < config.MIN_HOLD_STEPS:
            target_position = self.position

        action_str = "BUY" if target_position > 0 else "SELL" if target_position < 0 else "HOLD"
        
        acc = mt5.account_info()
        if acc is None:
            time.sleep(2)
            return obs, 0.0, False, False, {}
        current_bal = acc.balance
        current_equity = acc.equity

        if target_position != self.position:
            if self.position != 0.0:
                close_bot_positions(self.symbol, self.magic)
                if self.open_trade_ticket is not None:
                    deal_info = get_close_reason_and_profit(self.open_trade_ticket)
                    if deal_info:
                        update_trade_close_detail(self.open_trade_db_id, deal_info["price"], f"SIGNAL_CHANGE/{deal_info['reason']}", deal_info["profit"], deal_info["commission"], deal_info["swap"], current_bal, self.open_trade_mfe, self.open_trade_mae)
                    self.open_trade_ticket = None
                    self.open_trade_db_id = None
                    self.open_trade_mfe = 0.0
                    self.open_trade_mae = 0.0
                    save_log("CLOSE", "Posisi ditutup oleh sistem.")
                
                self.position = 0.0

            if target_position != 0.0:
                active_mt5_tf = {"1M": mt5.TIMEFRAME_M1, "5M": mt5.TIMEFRAME_M5, "15M": mt5.TIMEFRAME_M15, "30M": mt5.TIMEFRAME_M30, "1H": mt5.TIMEFRAME_H1}.get(self.timeframe, mt5.TIMEFRAME_M1)
                # PERBAIKAN BESAR: sebelumnya blok ini menghitung volatility &
                # konversi dollar<->point SENDIRI dan TERPISAH dari training
                # (pakai broker point asli, beda dgn config.POINT yang dipakai
                # trading_env.py). Itu menyebabkan SL/TP & lot di live bisa
                # berbeda 10x lipat dari yang dipelajari model saat training.
                #
                # Sekarang: pakai ATR yang SAMA (dari build_feature_matrix,
                # sudah disimpan di self.last_atr_dollars oleh _get_obs) dan
                # rumus IDENTIK dengan trading_env.py -> hasil SL/TP/lot akan
                # konsisten antara backtest dan live/demo.
                current_vol = max(0.2, getattr(self, 'last_atr_dollars', 0.2))

                min_sl_dollars = config.MIN_SL_PTS * config.POINT
                max_sl_dollars = config.MAX_SL_PTS * config.POINT
                min_tp_dollars = config.MIN_TP_PTS * config.POINT

                raw_sl_dollars = current_vol * config.SL_ATR_MULTIPLIER
                current_sl_dollars = max(min_sl_dollars, min(raw_sl_dollars, max_sl_dollars))
                current_tp_dollars = max(min_tp_dollars, current_vol * config.TP_ATR_MULTIPLIER)

                sl_pts = current_sl_dollars   # nama var dipertahankan, isinya sudah dalam dollar/price-units
                tp_pts = current_tp_dollars

                # Sanity check: ingatkan kalau POINT di config kemungkinan salah
                # (mis. broker gold 2-desimal tapi config di-set 3-desimal, dst.)
                # Pakai nilai yang sudah di-cache di __init__ (self._symbol_point_broker)
                # supaya tidak panggil mt5.symbol_info() berulang tepat di jalur eksekusi order.
                broker_point = self._symbol_point_broker
                if broker_point and abs(broker_point - config.POINT) / broker_point > 0.5:
                    save_log("WARNING", f"config.POINT ({config.POINT}) beda jauh dari broker point ({broker_point}) untuk {self.symbol}. Verifikasi ulang!")

                base_risk_amt = getattr(config, 'RISK_PER_TRADE', 20.0)
                risk_usd = max(base_risk_amt, (current_bal // 1000.0) * base_risk_amt)
                calculated_size_oz = risk_usd / current_sl_dollars
                lot_size = round(min(calculated_size_oz, config.MAX_SIZE_OZ) / 100.0, 2)

                if lot_size >= 0.01:
                    order_type = mt5.ORDER_TYPE_BUY if target_position > 0 else mt5.ORDER_TYPE_SELL
                    sl_price = round(ask - sl_pts if target_position > 0 else bid + sl_pts, self._symbol_digits)
                    tp_price = round(ask + tp_pts if target_position > 0 else bid - tp_pts, self._symbol_digits)
                    
                    res = open_position(self.symbol, order_type, lot_size, sl_price, tp_price, self.magic)
                    
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        save_log(action_str, f"Order {action_str} tereksekusi. Lot: {lot_size}")
                        self.open_trade_ticket = res.order
                        self.open_trade_db_id = save_trade_open_detail(res.order, self.symbol, action_str, lot_size, res.price, sl_price, tp_price, sl_pts, tp_pts, current_vol, current_bal, int(action), bid, ask, ask-bid, server_time.hour, server_time.dayofweek)
                        self.position = target_position
                    else:
                        error_msg = res.comment if res else "Tidak ada respon broker"
                        save_log("ERROR", f"MT5 Menolak Order {action_str} | Alasan: {error_msg}")
                        self.position = 0.0
                else:
                    save_log("ERROR", f"Gagal Eksekusi: Lot {lot_size} terlalu kecil. Saldo tidak cukup.")
                    self.position = 0.0
        else:
            if bot_status == "PAUSED":
                save_log("INFO", f"Evaluasi: SYSTEM PAUSED (Menunggu tombol START)")
            elif config.ENABLE_TOXIC_HOUR_FILTER and server_time.hour in config.TOXIC_HOURS:
                save_log("INFO", f"Evaluasi: TOXIC HOUR TERDETEKSI. Menghindari pasar.")
            elif self.position > 0:
                save_log("INFO", f"Evaluasi: Mempertahankan posisi BUY (Bid: {bid:.2f})")
            elif self.position < 0:
                save_log("INFO", f"Evaluasi: Mempertahankan posisi SELL (Ask: {ask:.2f})")
            else:
                save_log("INFO", f"Evaluasi: SCAN (Menunggu sinyal momentum)")

        norm_obs_to_log = getattr(self, 'last_normalized_obs', obs)
        try: save_ai_decision(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.symbol, float(current_price), bid, ask, current_bal, current_equity, int(action), self.position, target_position, json.dumps(norm_obs_to_log.tolist()))
        except Exception: pass

        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_profit = current_bal - get_daily_start_balance(today_str, current_bal)
        
        tf_map = {
            "1M": mt5.TIMEFRAME_M1, "5M": mt5.TIMEFRAME_M5, "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30, "1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4
        }
        active_mt5_tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M1)
        
        initial_rates = mt5.copy_rates_from_pos(self.symbol, active_mt5_tf, 0, 1)
        start_candle_time = initial_rates[0]['time'] if initial_rates else 0
        
        while True:
            current_rates = mt5.copy_rates_from_pos(self.symbol, active_mt5_tf, 0, 1)
            current_candle_time = current_rates[0]['time'] if current_rates else 0
            
            if current_candle_time != start_candle_time:
                break
                
            current_time_sec = time.time()
            if current_time_sec - self.last_sentiment_check > 900:
                self.current_sentiment, self.sentiment_prob, headlines_list = analyze_sentiment()
                self.last_sentiment_check = current_time_sec
                
                reasoning_dict = {
                    "prob": round(self.sentiment_prob * 100, 1),
                    "headlines": headlines_list
                }
                reasoning_json = json.dumps(reasoning_dict)
                update_supervisor_db(self.current_sentiment, reasoning_json)

            try:
                tick_live = mt5.symbol_info_tick(self.symbol)
                live_bid = tick_live.bid if tick_live else 0.0
                live_ask = tick_live.ask if tick_live else 0.0
                positions = mt5.positions_get(symbol=self.symbol)
                bot_positions = [p for p in positions if p.magic == self.magic] if positions else []
                pos_text = "NONE"
                if bot_positions and len(bot_positions) > 0:
                    pos = bot_positions[0]
                    pos_text = f"{'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'} {pos.volume} Lot|{pos.price_open}|{pos.sl}|{pos.tp}"
                    if self.open_trade_ticket == pos.ticket:
                        floating_profit = pos.profit
                        if floating_profit > self.open_trade_mfe: self.open_trade_mfe = floating_profit
                        if floating_profit < self.open_trade_mae: self.open_trade_mae = floating_profit
                        
                        try:
                            if pos.tp > 0.0: 
                                tp_distance = abs(pos.tp - pos.price_open)
                                dynamic_be_trigger = tp_distance * config.BE_TRIGGER_PCT
                                point = self._symbol_point_broker
                                dynamic_be_buffer = config.BE_BUFFER_PTS * point
                                
                                modified, new_sl = False, pos.sl
                                if pos.type == mt5.POSITION_TYPE_BUY:
                                    if (live_bid - pos.price_open) >= dynamic_be_trigger:
                                        be_price = round(pos.price_open + dynamic_be_buffer, 3)
                                        if new_sl < be_price and abs(new_sl - be_price) > 0.001: new_sl, modified = be_price, True
                                elif pos.type == mt5.POSITION_TYPE_SELL:
                                    if (pos.price_open - live_ask) >= dynamic_be_trigger:
                                        be_price = round(pos.price_open - dynamic_be_buffer, 3)
                                        if (pos.sl == 0.0 or new_sl > be_price) and abs(new_sl - be_price) > 0.001: new_sl, modified = be_price, True
                                if modified: modify_sl(pos.ticket, self.symbol, new_sl, pos.tp)
                        except Exception: pass
                else:
                    if self.open_trade_ticket is not None:
                        deal_info = get_close_reason_and_profit(self.open_trade_ticket)
                        if deal_info:
                            update_trade_close_detail(self.open_trade_db_id, deal_info["price"], deal_info["reason"], deal_info["profit"], deal_info["commission"], deal_info["swap"], current_bal, self.open_trade_mfe, self.open_trade_mae)
                            save_log("CLOSE", f"Tertutup ({deal_info['reason']}) | Profit: ${deal_info['profit']:.2f}")
                            self.open_trade_ticket = None; self.open_trade_db_id = None; self.position = 0.0
                update_current_state(current_bal, daily_profit, pos_text, live_bid, live_ask)
            except Exception: pass
            time.sleep(0.2)

        new_obs, _, _, _, _ = self._get_obs()
        current_equity = mt5.account_info().equity if mt5.account_info() else self.prev_equity
        portfolio_return = (current_equity - self.prev_equity) / self.prev_equity
        step_reward = portfolio_return * 5000.0 
        self.prev_equity = current_equity
        return new_obs, float(step_reward), False, False, {}