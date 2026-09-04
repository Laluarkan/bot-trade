import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config

class GoldTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    
    def __init__(self, features: np.ndarray, prices: np.ndarray, spreads: np.ndarray, dates=None,
                 episode_length: int = None, random_start: bool = True,
                 include_position: bool = None, timeframe: str = None,
                 use_margin_call: bool = True):
        super().__init__()
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.spreads = spreads.astype(np.float64)
        self.dates = pd.to_datetime(dates) if dates is not None else None
        
        self.n_steps = len(features)
        self.episode_length = episode_length or config.EPISODE_LENGTH
        self.random_start = random_start
        self.include_position = (config.INCLUDE_POSITION_IN_STATE if include_position is None else include_position)
        self.use_margin_call = use_margin_call
        self.reward_coeffs = config.reward_coeffs_for(timeframe)
        
        self.ema_200 = pd.Series(self.prices).ewm(span=200, adjust=False).mean().values
        self.macro_ema = pd.Series(self.prices).ewm(span=config.MACRO_EMA_SPAN, adjust=False).mean().values
        self.macro_slope = (pd.Series(self.macro_ema).diff(10) / pd.Series(self.macro_ema)).fillna(0).values * 100.0

        self.volatility = pd.Series(self.prices).diff().abs().rolling(window=14).mean().bfill().values
        self.volatility = np.clip(self.volatility, a_min=0.2, a_max=None)
        
        n_features = self.features.shape[1] + (1 if self.include_position else 0)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(n_features,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(3)  
        self._action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(1, self.n_steps - self.episode_length - 1)
        self.start_idx = int(self.np_random.integers(0, max_start)) if self.random_start else 0
        self.t = self.start_idx
        self.end_idx = min(self.start_idx + self.episode_length, self.n_steps - 1)
        
        self.balance = config.INITIAL_BALANCE
        self.position = 0.0                
        self.entry_price = None
        self.current_size_oz = 0.0          
        self.portfolio_value = self.balance
        self.peak_value = self.balance
        self.prev_position = 0.0
        
        self.dynamic_sl_price = 0.0
        self.mfe_price = 0.0
        self.be_activated = False
        self.cooldown_timer = 0
        self.steps_in_position = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.features[self.t]
        if self.include_position:
            obs = np.concatenate([obs, [self.position]]).astype(np.float32)
        return obs

    def _unrealized_pnl_dollar(self, price):
        if self.position == 0.0 or self.entry_price is None:
            return 0.0
        return (price - self.entry_price) * self.position * self.current_size_oz

    def step(self, action):
        target_position = self._action_map[int(action)]
        price = self.prices[self.t]
        prev_portfolio_value = max(self.portfolio_value, 1e-8)
        forced_close = False
        step_reward = 0.0 
        
        if self.position != 0.0:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0

        if config.ENABLE_TIME_FIREWALL and self.dates is not None:
            current_time = self.dates[self.t]
            is_weekend = False
            
            if current_time.dayofweek == 5 and current_time.hour >= 4:
                is_weekend = True
            elif current_time.dayofweek == 6:
                is_weekend = True
            elif current_time.dayofweek == 0 and current_time.hour < 5:
                is_weekend = True

            if is_weekend or current_time.hour in config.TOXIC_HOURS:
                target_position = 0.0
                if self.position != 0.0:
                    forced_close = True

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            if self.position == 0.0:
                target_position = 0.0  

        if self.position != 0.0 and target_position != self.position and not forced_close:
            if self.steps_in_position < config.MIN_HOLD_STEPS:
                target_position = self.position

        current_ema = self.ema_200[self.t]
        current_slope = self.macro_slope[self.t]

        if target_position > 0:
            if price < current_ema or current_slope <= 0.001:
                target_position = 0.0  
                step_reward -= 1.0  
        elif target_position < 0:
            if price > current_ema or current_slope >= -0.001:
                target_position = 0.0  
                step_reward -= 1.0

        current_vol = self.volatility[self.t]
        raw_sl_dollars = current_vol * config.SL_ATR_MULTIPLIER
        
        # --- PERBAIKAN LOGIKA: Konversi batas Point config menjadi nilai Dollar ---
        POINT = 0.001
        min_sl_dollars = config.MIN_SL_PTS * POINT
        max_sl_dollars = config.MAX_SL_PTS * POINT
        min_tp_dollars = config.MIN_TP_PTS * POINT
        
        current_sl_dollars = max(min_sl_dollars, min(raw_sl_dollars, max_sl_dollars))
        current_tp_dollars = max(min_tp_dollars, current_vol * config.TP_ATR_MULTIPLIER)
        
        dynamic_be_trigger = current_tp_dollars * config.BE_TRIGGER_PCT
        dynamic_be_buffer = config.BE_BUFFER_PTS * POINT

        if self.position != 0.0 and self.entry_price is not None:
            current_pnl = self._unrealized_pnl_dollar(price)
            
            if self.steps_in_position >= config.MAX_HOLD_LOSS_STEPS and current_pnl < 0:
                forced_close = True

            if self.position > 0:
                self.mfe_price = max(self.mfe_price, price)
                mfe_pts = self.mfe_price - self.entry_price
            else:
                self.mfe_price = min(self.mfe_price, price)
                mfe_pts = self.entry_price - self.mfe_price

            if mfe_pts >= current_tp_dollars:
                forced_close = True
            elif mfe_pts >= dynamic_be_trigger and not self.be_activated:
                self.be_activated = True
                if self.position > 0:
                    self.dynamic_sl_price = self.entry_price + dynamic_be_buffer
                else:
                    self.dynamic_sl_price = self.entry_price - dynamic_be_buffer

            if not forced_close:
                if self.position > 0 and price <= self.dynamic_sl_price:
                    forced_close = True
                elif self.position < 0 and price >= self.dynamic_sl_price:
                    forced_close = True
        
        if forced_close:
            target_position = 0.0

        transaction_cost = 0.0
        
        if target_position != self.position:
            if self.position != 0.0 and target_position != 0.0:
                step_reward -= config.CHURN_PENALTY

            if self.position != 0.0 and self.entry_price is not None:
                pnl = self._unrealized_pnl_dollar(price)
                self.balance += pnl
                
                if pnl > 0:
                    step_reward += (pnl / self.balance) * 5000.0  
                else:
                    step_reward -= (abs(pnl) / self.balance) * 5000.0 

                if pnl < 0 and not forced_close:
                    self.cooldown_timer = config.LOSS_COOLDOWN_STEPS
                    target_position = 0.0  
                
            if target_position != 0.0:
                calculated_size_oz = config.RISK_PER_TRADE / current_sl_dollars 
                new_size_oz = min(calculated_size_oz, config.MAX_SIZE_OZ)
                trade_value = new_size_oz * price
                
                self.mfe_price = price
                self.be_activated = False
                self.steps_in_position = 0
                if target_position > 0:
                    self.dynamic_sl_price = price - current_sl_dollars
                else:
                    self.dynamic_sl_price = price + current_sl_dollars
            else:
                new_size_oz = 0.0
                trade_value = self.current_size_oz * price 
                self.dynamic_sl_price = 0.0
                self.mfe_price = 0.0
                self.be_activated = False
                self.steps_in_position = 0
            
            transacted_size = new_size_oz if target_position != 0.0 else self.current_size_oz
            commission = trade_value * config.COMMISSION_RATE
            spread_cost = transacted_size * self.spreads[self.t]
            impact = trade_value * config.SLIPPAGE_COEFF  
            transaction_cost = commission + spread_cost + impact
            
            self.balance -= transaction_cost
            self.current_size_oz = new_size_oz
            self.entry_price = price if target_position != 0.0 else None

        self.position = target_position
        self.t += 1
        terminated = False
        truncated = self.t >= self.end_idx
        
        mark_price = self.prices[self.t] if self.t < self.n_steps else price
        self.portfolio_value = self.balance + self._unrealized_pnl_dollar(mark_price)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        step_reward += portfolio_return * 5000.0 
        
        if target_position != self.prev_position:
            step_reward -= (transaction_cost / self.balance) * 5000.0
        else:
            if self.position != 0.0:
                current_pnl = self._unrealized_pnl_dollar(price)
                if current_pnl > 0:
                    step_reward += 0.2  
                else:
                    step_reward -= 0.1

        drawdown = max(0.0, (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8))
        
        if self.use_margin_call and drawdown >= config.MARGIN_CALL_DRAWDOWN:
            terminated = True
            step_reward -= config.MARGIN_CALL_PENALTY * 1000.0
            self.balance = self.portfolio_value  
        
        if target_position == 0.0:
            self.balance = self.portfolio_value

        self.prev_position = target_position

        done = terminated or truncated
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if done else self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "transaction_cost": transaction_cost,
            "drawdown": drawdown,
            "size_oz": self.current_size_oz
        }
        return obs, float(step_reward), terminated, truncated, info