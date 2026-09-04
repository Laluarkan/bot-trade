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
        
        obs, _, _, _, _ = self._get_obs()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs.shape[0],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        acc = mt5.account_info()
        self.prev_equity = acc.equity if acc else getattr(config, 'INITIAL_BALANCE', 10000.0)

    def _get_obs(self):
        tf_map = {
            "1M": mt5.TIMEFRAME_M1, "5M": mt5.TIMEFRAME_M5, "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30, "1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4
        }
        active_mt5_tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M1)

        rates = mt5.copy_rates_from_pos(self.symbol, active_mt5_tf, 0, 1000)
        df = pd.DataFrame(rates)
        df.rename(columns={'time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        
        if config.PROXY_SYMBOL:
            rates_proxy = mt5.copy_rates_from_pos(config.PROXY_SYMBOL, active_mt5_tf, 0, 1000)
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
            
        features, prices, spreads, _, _, _ = build_feature_matrix(df, self.timeframe, use_kalman=True, kalman_model=self.kalman_model)
        current_feature = features[-1]
        obs = np.concatenate([current_feature, [self.position]]).astype(np.float32) if config.INCLUDE_POSITION_IN_STATE else current_feature
        tick = mt5.symbol_info_tick(self.symbol)
        bid, ask = (tick.bid, tick.ask) if tick else (prices[-1], prices[-1])
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

        if bot_status == "PAUSED" or (config.ENABLE_TIME_FIREWALL and server_time.hour in config.TOXIC_HOURS):
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
                rates = mt5.copy_rates_from_pos(self.symbol, active_mt5_tf, 0, 100)
                prices_arr = pd.DataFrame(rates)['close'].values
                volatility = pd.Series(prices_arr).diff().abs().rolling(14).mean().bfill().values[-1]
                volatility = max(0.2, volatility)
                
                point = mt5.symbol_info(self.symbol).point
                
                # --- PERBAIKAN LOGIKA: Konversi nilai Dollar ke Point sebelum divalidasi dengan config ---
                vol_in_pts = volatility / point
                
                raw_sl_pts = vol_in_pts * config.SL_ATR_MULTIPLIER
                sl_pts_val = max(config.MIN_SL_PTS, min(raw_sl_pts, config.MAX_SL_PTS))
                tp_pts_val = max(config.MIN_TP_PTS, vol_in_pts * config.TP_ATR_MULTIPLIER)
                
                sl_pts = sl_pts_val * point
                tp_pts = tp_pts_val * point
                
                base_risk_amt = getattr(config, 'RISK_PER_TRADE', 20.0)
                risk_usd = max(base_risk_amt, (current_bal // 1000.0) * base_risk_amt)
                lot_size = round(min(risk_usd / sl_pts, config.MAX_SIZE_OZ) / 100.0, 2)
                
                if lot_size >= 0.01:
                    order_type = mt5.ORDER_TYPE_BUY if target_position > 0 else mt5.ORDER_TYPE_SELL
                    sl_price = ask - sl_pts if target_position > 0 else bid + sl_pts
                    tp_price = ask + tp_pts if target_position > 0 else bid - tp_pts
                    
                    res = open_position(self.symbol, order_type, lot_size, sl_price, tp_price, self.magic)
                    
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        save_log(action_str, f"Order {action_str} tereksekusi. Lot: {lot_size}")
                        self.open_trade_ticket = res.order
                        self.open_trade_db_id = save_trade_open_detail(res.order, self.symbol, action_str, lot_size, res.price, sl_price, tp_price, sl_pts, tp_pts, volatility, current_bal, int(action), bid, ask, ask-bid, server_time.hour, server_time.dayofweek)
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
            elif config.ENABLE_TIME_FIREWALL and server_time.hour in config.TOXIC_HOURS:
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
                                point = mt5.symbol_info(self.symbol).point
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
            time.sleep(0.5)

        new_obs, _, _, _, _ = self._get_obs()
        current_equity = mt5.account_info().equity if mt5.account_info() else self.prev_equity
        portfolio_return = (current_equity - self.prev_equity) / self.prev_equity
        step_reward = portfolio_return * 5000.0 
        self.prev_equity = current_equity
        return new_obs, float(step_reward), False, False, {}