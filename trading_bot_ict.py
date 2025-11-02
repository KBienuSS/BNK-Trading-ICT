import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import threading
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_ict.log', encoding='utf-8')
    ]
)

class ICTTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # ML Model Components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_columns = []
        
        # STRATEGIA ICT/SMC
        self.max_simultaneous_positions = 3  # Mniej pozycji, lepsze zarządzanie
        
        # ALOKACJA KAPITAŁU
        self.asset_allocation = {
            'BTCUSDT': 0.30,  # 30%
            'ETHUSDT': 0.25,  # 25%
            'SOLUSDT': 0.20,  # 20%
            'BNBUSDT': 0.15,  # 15%
            'XRPUSDT': 0.10,  # 10%
        }
        
        self.priority_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
        
        # PARAMETRY ICT
        self.min_confidence = 0.75  # Wyższy próg dla ICT
        self.risk_reward_ratio = 2.0  # MIN 1:3 Risk/Reward
        
        # GODZINY HANDLU (Sessiony ICT)
        self.trading_sessions = {
            'asian_range': {'start': 0, 'end': 6},    # 00:00-06:00 UTC
            'london_open': {'start': 7, 'end': 10},   # 07:00-10:00 UTC
            'new_york_open': {'start': 13, 'end': 16}, # 13:00-16:00 UTC
            'enabled': True
        }
        
        # BIAS STRATEGII
        self.long_bias = 0.60  # 60% bias na LONG
        self.risk_tolerance = "MEDIUM"
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0,
            'ict_trades': 0,
            'portfolio_utilization': 0,
            'average_rr': 0,
            'win_rate': 0
        }
        
        # Dashboard
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'last_update': datetime.now(),
            'trading_session': 'CLOSED'
        }
        
        self.initialize_ml_model()
        
        self.logger.info("🎯 ICT/SMC TRADING BOT - SMART MONEY CONCEPTS")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")
        self.logger.info("📊 Risk/Reward: 1:3 | Min Confidence: 75%")
        self.logger.info("🕒 Trading Sessions: Asian(00-06), London(07-10), NY(13-16) UTC")

    def initialize_ml_model(self):
        """Initialize ML model with Random Forest"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.logger.info("✅ ML Model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing ML model: {e}")

    def get_current_trading_session(self):
        """Zwraca aktualną sesję handlową"""
        if not self.trading_sessions['enabled']:
            return "ALL_DAY"
            
        current_hour = datetime.utcnow().hour
        
        if self.trading_sessions['asian_range']['start'] <= current_hour <= self.trading_sessions['asian_range']['end']:
            return "ASIAN"
        elif self.trading_sessions['london_open']['start'] <= current_hour <= self.trading_sessions['london_open']['end']:
            return "LONDON"
        elif self.trading_sessions['new_york_open']['start'] <= current_hour <= self.trading_sessions['new_york_open']['end']:
            return "NEW_YORK"
        else:
            return "CLOSED"

    def should_enter_trade(self):
        """Decyzja o wejściu w trade z uwzględnieniem bias"""
        return random.random() < self.long_bias

    def get_binance_klines(self, symbol: str, interval: str = '5m', limit: int = 200):
        """Pobiera dane z Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Konwersja typów danych
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"✅ Binance Data for {symbol}: {len(df)} rows, Last: ${df['close'].iloc[-1]:.2f}")
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching Binance data for {symbol}: {e}")
        
        return None

    def get_current_price(self, symbol: str):
        """Pobiera aktualną cenę z Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current_price = float(data['price'])
                return current_price
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching current price for {symbol}: {e}")
        
        return None

    # ========== ICT/SMC CORE FUNCTIONS ==========

    def detect_ict_smc_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał oparty na zasadach ICT i SMC"""
        try:
            df = self.get_binance_klines(symbol, '5m', 200)
            if df is None or len(df) < 100:
                return "HOLD", 0.5
            
            # Oblicz poziomy ICT
            df = self.calculate_ict_levels(df)
            
            current_price = df['close'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_high = df['high'].iloc[-1]
            
            # 1. IDENTYFIKACJA FAZY RYNKU
            market_structure = self.identify_market_structure(df)
            
            # 2. OBLASTI LIKWIDACJI
            liquidity_zones = self.find_liquidity_zones(df)
            
            # 3. ORDER BLOCKS
            order_blocks = self.find_order_blocks(df)
            
            # 4. FAIR VALUE GAP
            fvg_signals = self.find_fair_value_gaps(df)
            
            confidence = 0.0
            signal = "HOLD"
            
            # SYGNAŁ KUPNA (BULLISH)
            bullish_conditions = 0
            
            # Warunek 1: Cena testuje obszar likwidacji poniżej
            if self.is_testing_liquidity_below(df, liquidity_zones):
                bullish_conditions += 1
                confidence += 0.30
            
            # Warunek 2: Cena wraca do bullish order block
            if self.is_at_bullish_order_block(current_price, order_blocks):
                bullish_conditions += 1
                confidence += 0.25
            
            # Warunek 3: FVG działa jako support
            if self.is_fvg_support(current_price, fvg_signals):
                bullish_conditions += 1
                confidence += 0.20
            
            # Warunek 4: Zmiana struktury rynku na bullish
            if market_structure in ["BULLISH", "BULLISH_BOS"]:
                bullish_conditions += 1
                confidence += 0.25
            
            # SYGNAŁ SPRZEDAŻY (BEARISH)
            bearish_conditions = 0
            
            # Warunek 1: Cena testuje obszar likwidacji powyżej
            if self.is_testing_liquidity_above(df, liquidity_zones):
                bearish_conditions += 1
                confidence += 0.30
            
            # Warunek 2: Cena wraca do bearish order block
            if self.is_at_bearish_order_block(current_price, order_blocks):
                bearish_conditions += 1
                confidence += 0.25
            
            # Warunek 3: FVG działa jako resistance
            if self.is_fvg_resistance(current_price, fvg_signals):
                bearish_conditions += 1
                confidence += 0.20
            
            # Warunek 4: Zmiana struktury rynku na bearish
            if market_structure in ["BEARISH", "BEARISH_BOS"]:
                bearish_conditions += 1
                confidence += 0.25
            
            # DECYZJA O SYGNALE
            if bullish_conditions >= 3 and bearish_conditions <= 1:
                signal = "LONG"
                confidence = min(confidence, 0.95)
            elif bearish_conditions >= 3 and bullish_conditions <= 1:
                signal = "SHORT" 
                confidence = min(confidence, 0.95)
            else:
                signal = "HOLD"
                confidence = max(confidence * 0.7, 0.3)
            
            self.logger.info(f"🎯 ICT/SMC {symbol}: {signal} (Conf: {confidence:.1%}, Bull: {bullish_conditions}, Bear: {bearish_conditions})")
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error in ICT/SMC signal detection for {symbol}: {e}")
            return "HOLD", 0.5

    def calculate_ict_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza poziomy ICT"""
        # Weekly High/Low
        df['weekly_high'] = df['high'].rolling(2016).max()
        df['weekly_low'] = df['low'].rolling(2016).min()
        
        # Daily High/Low
        df['daily_high'] = df['high'].rolling(288).max()
        df['daily_low'] = df['low'].rolling(288).min()
        
        # Swing Highs/Lows
        df['swing_high'] = df['high'].rolling(5, center=True).max()
        df['swing_low'] = df['low'].rolling(5, center=True).min()
        
        # Previous Day High/Low
        df['prev_daily_high'] = df['daily_high'].shift(288)
        df['prev_daily_low'] = df['daily_low'].shift(288)
        
        return df

    def identify_market_structure(self, df: pd.DataFrame) -> str:
        """Identyfikuje strukturę rynku według ICT"""
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        # Higher Highs & Higher Lows (Uptrend)
        if (recent_highs.iloc[-1] > recent_highs.iloc[-3] and 
            recent_lows.iloc[-1] > recent_lows.iloc[-3]):
            return "BULLISH"
        
        # Lower Highs & Lower Lows (Downtrend)
        elif (recent_highs.iloc[-1] < recent_highs.iloc[-3] and 
              recent_lows.iloc[-1] < recent_lows.iloc[-3]):
            return "BEARISH"
        
        # Break of Structure
        elif recent_highs.iloc[-1] > df['swing_high'].iloc[-15]:
            return "BULLISH_BOS"
        
        elif recent_lows.iloc[-1] < df['swing_low'].iloc[-15]:
            return "BEARISH_BOS"
        
        return "RANGE"

    def find_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """Znajduje obszary likwidacji"""
        zones = {
            'above': [],
            'below': []
        }
        
        # Likwidacja powyżej (stopy longów)
        recent_high = df['high'].tail(20).max()
        if recent_high > df['high'].iloc[-50:-20].max():
            zones['above'].append(recent_high)
        
        # Likwidacja poniżej (stopy shortów)
        recent_low = df['low'].tail(20).min()
        if recent_low < df['low'].iloc[-50:-20].min():
            zones['below'].append(recent_low)
        
        # Poziomy weekly/daily
        zones['above'].extend([
            df['weekly_high'].iloc[-1],
            df['daily_high'].iloc[-1],
            df['prev_daily_high'].iloc[-1]
        ])
        
        zones['below'].extend([
            df['weekly_low'].iloc[-1],
            df['daily_low'].iloc[-1],
            df['prev_daily_low'].iloc[-1]
        ])
        
        return zones

    def find_order_blocks(self, df: pd.DataFrame) -> Dict:
        """Znajduje bloki zleceń (Order Blocks)"""
        blocks = {
            'bullish': [],
            'bearish': []
        }
        
        for i in range(2, min(50, len(df))):
            # Bullish Order Block
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['low'].iloc[i] > df['low'].iloc[i-1]):
                blocks['bullish'].append({
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'price': (df['high'].iloc[i-1] + df['low'].iloc[i-1]) / 2
                })
            
            # Bearish Order Block
            if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['high'].iloc[i] < df['high'].iloc[i-1]):
                blocks['bearish'].append({
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'price': (df['high'].iloc[i-1] + df['low'].iloc[i-1]) / 2
                })
        
        return blocks

    def find_fair_value_gaps(self, df: pd.DataFrame) -> Dict:
        """Znajduje Fair Value Gaps (FVG)"""
        fvgs = {
            'bullish': [],
            'bearish': []
        }
        
        for i in range(1, min(20, len(df))):
            current_low = df['low'].iloc[i]
            prev_high = df['high'].iloc[i-1]
            current_high = df['high'].iloc[i]
            prev_low = df['low'].iloc[i-1]
            
            # Bullish FVG
            if df['close'].iloc[i] > prev_high and current_low > prev_high:
                fvgs['bullish'].append({
                    'top': current_low,
                    'bottom': prev_high
                })
            
            # Bearish FVG
            if df['close'].iloc[i] < prev_low and current_high < prev_low:
                fvgs['bearish'].append({
                    'top': prev_low,
                    'bottom': current_high
                })
        
        return fvgs

    # Helper functions dla warunków wejścia
    def is_testing_liquidity_below(self, df: pd.DataFrame, liquidity_zones: Dict) -> bool:
        current_low = df['low'].iloc[-1]
        for zone in liquidity_zones['below']:
            if abs(current_low - zone) / zone < 0.002:
                return True
        return False

    def is_testing_liquidity_above(self, df: pd.DataFrame, liquidity_zones: Dict) -> bool:
        current_high = df['high'].iloc[-1]
        for zone in liquidity_zones['above']:
            if abs(current_high - zone) / zone < 0.002:
                return True
        return False

    def is_at_bullish_order_block(self, current_price: float, order_blocks: Dict) -> bool:
        for block in order_blocks['bullish'][-3:]:
            if block['low'] <= current_price <= block['high']:
                return True
        return False

    def is_at_bearish_order_block(self, current_price: float, order_blocks: Dict) -> bool:
        for block in order_blocks['bearish'][-3:]:
            if block['low'] <= current_price <= block['high']:
                return True
        return False

    def is_fvg_support(self, current_price: float, fvg_signals: Dict) -> bool:
        for fvg in fvg_signals['bullish'][-2:]:
            if fvg['bottom'] <= current_price <= fvg['top']:
                return True
        return False

    def is_fvg_resistance(self, current_price: float, fvg_signals: Dict) -> bool:
        for fvg in fvg_signals['bearish'][-2:]:
            if fvg['bottom'] <= current_price <= fvg['top']:
                return True
        return False

    def calculate_ict_exit_levels(self, entry_price: float, side: str, df: pd.DataFrame) -> Dict:
        """Oblicza poziomy wyjścia z risk/reward 1:3"""
        liquidity_zones = self.find_liquidity_zones(df)
        
        if side == 'LONG':
            # Stop Loss poniżej ostatniego swing low
            recent_lows = df['low'].tail(10)
            stop_loss = recent_lows.min() * 0.998  # 0.2% poniżej
            
            # Risk calculation
            risk = entry_price - stop_loss
            
            # Take Profit z RR 1:3
            take_profit = entry_price + (risk * self.risk_reward_ratio)
            
            # Sprawdź czy TP nie jest zbyt blisko likwidacji
            if liquidity_zones['above']:
                nearest_liquidity = min(liquidity_zones['above'], key=lambda x: abs(x - entry_price))
                if nearest_liquidity > take_profit:
                    take_profit = nearest_liquidity
            
        else:  # SHORT
            # Stop Loss powyżej ostatniego swing high
            recent_highs = df['high'].tail(10)
            stop_loss = recent_highs.max() * 1.002  # 0.2% powyżej
            
            # Risk calculation
            risk = stop_loss - entry_price
            
            # Take Profit z RR 1:3
            take_profit = entry_price - (risk * self.risk_reward_ratio)
            
            # Sprawdź czy TP nie jest zbyt blisko likwidacji
            if liquidity_zones['below']:
                nearest_liquidity = min(liquidity_zones['below'], key=lambda x: abs(x - entry_price))
                if nearest_liquidity < take_profit:
                    take_profit = nearest_liquidity
        
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'invalidation': stop_loss * (0.99 if side == 'LONG' else 1.01),
            'risk_reward': self.risk_reward_ratio
        }

    # ========== TRADING ENGINE ==========

    def calculate_position_size(self, symbol: str, price: float, confidence: float):
        """Oblicza wielkość pozycji z uwzględnieniem risk management"""
        try:
            # Bazowa alokacja
            base_allocation = self.asset_allocation.get(symbol, 0.15)
            
            # Modyfikator confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            # Wartość pozycji
            position_value = (self.virtual_capital * base_allocation * confidence_multiplier)
            
            # Limit maksymalnej pozycji (25% kapitału)
            max_position_value = self.virtual_capital * 0.25
            position_value = min(position_value, max_position_value)
            
            quantity = position_value / price
            margin_required = position_value / self.leverage
            
            return quantity, position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0, 0, 0

    def open_ict_position(self, symbol: str):
        """Otwiera pozycję według zasad ICT/SMC"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.info(f"❌ No current price for {symbol}")
            return None
        
        # Sprawdź sesję handlową
        current_session = self.get_current_trading_session()
        if current_session == "CLOSED" and self.trading_sessions['enabled']:
            self.logger.info(f"⏹️ Market closed for {symbol}")
            return None
        
        signal, confidence = self.detect_ict_smc_signal(symbol)
        
        # Sprawdź warunki wejścia
        if confidence < self.min_confidence:
            self.logger.info(f"⏹️ {symbol} - Confidence too low: {confidence:.1%}")
            return None
        
        if signal not in ["LONG", "SHORT"]:
            self.logger.info(f"⏹️ {symbol} - No valid signal: {signal}")
            return None
        
        # Sprawdź limit pozycji
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        # Oblicz wielkość pozycji
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        # Oblicz poziomy wyjścia
        df = self.get_binance_klines(symbol, '5m', 200)
        exit_levels = self.calculate_ict_exit_levels(current_price, signal, df)
        
        # Sprawdź czy RR jest przynajmniej 1:3
        if signal == 'LONG':
            risk = current_price - exit_levels['stop_loss']
            reward = exit_levels['take_profit'] - current_price
        else:
            risk = exit_levels['stop_loss'] - current_price
            reward = current_price - exit_levels['take_profit']
        
        actual_rr = reward / risk if risk > 0 else 0
        if actual_rr < self.risk_reward_ratio:
            self.logger.info(f"⏹️ {symbol} - RR too low: {actual_rr:.1f}:1")
            return None
        
        liquidation_price = current_price * (1 - 0.9 / self.leverage) if signal == 'LONG' else current_price * (1 + 0.9 / self.leverage)
        
        position_id = f"ict_{self.position_id}"
        self.position_id += 1
        
        position = {
            'symbol': symbol,
            'side': signal,
            'entry_price': current_price,
            'quantity': quantity,
            'leverage': self.leverage,
            'margin': margin_required,
            'liquidation_price': liquidation_price,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'unrealized_pnl': 0,
            'confidence': confidence,
            'risk_reward': actual_rr,
            'exit_plan': exit_levels
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        self.stats['ict_trades'] += 1
        self.logger.info(f"🎯 ICT OPEN: {signal} {quantity:.4f} {symbol} @ ${current_price:.2f}")
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.2f} | SL: ${exit_levels['stop_loss']:.2f}")
        self.logger.info(f"   💰 Position: ${position_value:.2f} | RR: {actual_rr:.1f}:1")
        self.logger.info(f"   🤖 Confidence: {confidence:.1%} | Session: {current_session}")
        
        return position_id

    def should_close_ict_position(self, symbol: str, position: dict) -> Tuple[bool, str]:
        """Sprawdza warunki zamknięcia pozycji według ICT"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return False, ""
            
            entry_price = position['entry_price']
            side = position['side']
            
            # 1. Take Profit
            if ((side == 'LONG' and current_price >= position['exit_plan']['take_profit']) or
                (side == 'SHORT' and current_price <= position['exit_plan']['take_profit'])):
                return True, "TAKE_PROFIT"
            
            # 2. Stop Loss
            if ((side == 'LONG' and current_price <= position['exit_plan']['stop_loss']) or
                (side == 'SHORT' and current_price >= position['exit_plan']['stop_loss'])):
                return True, "STOP_LOSS"
            
            # 3. Invalidation
            if ((side == 'LONG' and current_price <= position['exit_plan']['invalidation']) or
                (side == 'SHORT' and current_price >= position['exit_plan']['invalidation'])):
                return True, "INVALIDATION"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"❌ Error in ICT exit check for {symbol}: {e}")
            return False, ""

    def update_positions_pnl(self):
        """Update P&L for all positions"""
        total_unrealized = 0
        total_margin = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            position['current_price'] = current_price
            
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        if self.virtual_capital > 0:
            portfolio_utilization = (total_margin * self.leverage) / self.virtual_capital
            self.stats['portfolio_utilization'] = portfolio_utilization
        
        self.dashboard_data['last_update'] = datetime.now()
        self.dashboard_data['trading_session'] = self.get_current_trading_session()

    def check_exit_conditions(self):
        """Check exit conditions with ICT rules"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
            
            should_close, exit_reason = self.should_close_ict_position(position['symbol'], position)
            
            if should_close:
                current_price = self.get_current_price(position['symbol'])
                if current_price:
                    positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Close a position"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        self.virtual_balance += position['margin'] + realized_pnl_after_fee
        self.virtual_capital += realized_pnl_after_fee
        
        # Update statistics
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
            if realized_pnl_after_fee > self.stats['biggest_win']:
                self.stats['biggest_win'] = realized_pnl_after_fee
        else:
            self.stats['losing_trades'] += 1
            if realized_pnl_after_fee < self.stats['biggest_loss']:
                self.stats['biggest_loss'] = realized_pnl_after_fee
        
        # Update RR statistics
        total_trades = self.stats['total_trades']
        if total_trades > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / total_trades) * 100
            self.stats['average_rr'] = (self.stats['average_rr'] * (total_trades - 1) + position['risk_reward']) / total_trades
        
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'realized_pnl': realized_pnl_after_fee,
            'exit_reason': exit_reason,
            'confidence': position.get('confidence', 0),
            'risk_reward': position.get('risk_reward', 0),
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        position['status'] = 'CLOSED'
        
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} - Reason: {exit_reason}")

    def get_dashboard_data(self):
        """Prepare dashboard data"""
        active_positions = []
        
        # Pobierz aktualne ceny dla aktywnych pozycji
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_current_price(position['symbol'])
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                active_positions.append({
                    'position_id': position_id,
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'margin': position['margin'],
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position.get('confidence', 0),
                    'risk_reward': position.get('risk_reward', 0)
                })
        
        # CONFIDENCE LEVELS
        confidence_levels = {}
        for symbol in self.priority_symbols:
            try:
                signal, confidence = self.detect_ict_smc_signal(symbol)
                confidence_percent = round(confidence * 100, 1)
                confidence_levels[symbol] = confidence_percent
            except Exception as e:
                self.logger.error(f"❌ Error calculating confidence for {symbol}: {e}")
                confidence_levels[symbol] = 0
        
        recent_trades = []
        for trade in self.trade_history[-10:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'risk_reward': trade.get('risk_reward', 0),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'confidence': trade.get('confidence', 0)
            })
        
        current_session = self.get_current_trading_session()
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'total_fees': round(self.stats['total_fees'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2)
            },
            'performance_metrics': {
                'avg_leverage': self.leverage,
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1),
                'ict_trades': self.stats['ict_trades'],
                'win_rate': round(self.stats['win_rate'], 1),
                'total_trades': self.stats['total_trades'],
                'biggest_win': round(self.stats['biggest_win'], 2),
                'biggest_loss': round(self.stats['biggest_loss'], 2),
                'average_rr': round(self.stats['average_rr'], 1)
            },
            'trading_session': {
                'current': current_session,
                'sessions': self.trading_sessions
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'strategy_profile': {
                'risk_reward': f"1:{self.risk_reward_ratio}",
                'min_confidence': f"{self.min_confidence*100:.0f}%",
                'max_positions': self.max_simultaneous_positions,
                'trading_sessions': self.trading_sessions['enabled']
            },
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_ict_strategy(self):
        """Główna pętla strategii ICT/SMC"""
        self.logger.info("🚀 ICT/SMC STRATEGY STARTED")
        self.logger.info("📊 Risk/Reward: 1:3 | Min Confidence: 75%")
        self.logger.info("🕒 Trading Sessions: Asian(00-06), London(07-10), NY(13-16) UTC")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now()
                current_session = self.get_current_trading_session()
                
                self.logger.info(f"\n🔄 ICT Iteration #{iteration} | Session: {current_session}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. SPRAWDŹ SYGNAŁY WEJŚCIA
                active_symbols = [p['symbol'] for p in self.positions.values() 
                                if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                self.logger.info(f"📊 Active positions: {active_count}/{self.max_simultaneous_positions}")
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            # Sprawdź bias
                            if not self.should_enter_trade():
                                self.logger.info(f"⏹️ Skipping {symbol} due to bias")
                                continue
                            
                            # Sprawdź sesję handlową
                            if current_session == "CLOSED" and self.trading_sessions['enabled']:
                                continue
                            
                            position_id = self.open_ict_position(symbol)
                            if position_id:
                                self.logger.info(f"✅ SUCCESS: Opened ICT position for {symbol}")
                                time.sleep(2)  # Opóźnienie między pozycjami
                
                # 4. Loguj status
                portfolio_value = self.dashboard_data['account_value']
                active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
                
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active: {active_count}/{self.max_simultaneous_positions}")
                self.logger.info(f"🕒 Session: {current_session} | Win Rate: {self.stats['win_rate']:.1f}%")
                
                # 5. Czekaj przed następną iteracją
                wait_time = 60  # 1 minuta
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in ICT trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Start ICT/SMC trading"""
        self.is_running = True
        self.run_ict_strategy()

    def stop_trading(self):
        """Stop ICT/SMC trading"""
        self.is_running = False
        self.logger.info("🛑 ICT/SMC Trading stopped")

# Global ICT bot instance
ict_trading_bot = ICTTradingBot(initial_capital=10000, leverage=10)
