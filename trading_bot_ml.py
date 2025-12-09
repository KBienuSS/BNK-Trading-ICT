# trading_bot_ml_bybit_strategy_v3.py
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
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import hmac
import hashlib
import base64

# Spróbuj zaimportować pybit
try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    logging.warning("⚠️ Biblioteka pybit nie jest zainstalowana. Użyj: pip install pybit")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_trading_bot.log', encoding='utf-8')
    ]
)

class LLMTradingBot:
    def __init__(self, api_key=None, api_secret=None, initial_capital=10000, leverage=10):
        # Inicjalizacja loggera NAJPIERW
        self.logger = logging.getLogger(__name__)
        
        # Konfiguracja Bybit API - dla otwierania pozycji
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.base_url = "https://api.bybit.com"
        self.testnet = False
        
        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        
        # Konfiguracja Binance API - tylko do odczytu danych
        self.binance_base_url = "https://api.binance.com/api/v3"
        
        # Sprawdź czy klucze API są dostępne
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ Brak kluczy API Bybit - bot będzie działał w trybie wirtualnym")
            self.real_trading = False
        else:
            self.real_trading = True
            self.logger.info("🔑 Klucze API Bybit załadowane - REAL TRADING ENABLED")
            
        # Kapitał - TERAZ UŻYWAMY RZECZYWISTEGO SALDA Z BYBIT
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        # Cache cen
        self.price_cache = {}
        self.price_history = {}
        
        # --- KONFIGURACJA STRATEGII TECHNICZNEJ (Z BOTA V3) ---
        self.timeframe = '15m'  # Interwał świecowy
        self.ema_short_period = 9
        self.ema_long_period = 21
        self.rsi_period = 14
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM
        self.llm_profiles = {
            'Claude': {
                'risk_appetite': 'MEDIUM',
                'confidence_bias': 0.6,
                'short_frequency': 0.1,
                'holding_bias': 'LONG',
                'trade_frequency': 'LOW',
                'position_sizing': 'CONSERVATIVE',
                'max_holding_hours': (2, 8)
            },
            'Gemini': {
                'risk_appetite': 'HIGH', 
                'confidence_bias': 0.7,
                'short_frequency': 0.35,
                'holding_bias': 'SHORT',
                'trade_frequency': 'HIGH',
                'position_sizing': 'AGGRESSIVE',
                'max_holding_hours': (1, 4)
            },
            'GPT': {
                'risk_appetite': 'LOW',
                'confidence_bias': 0.3,
                'short_frequency': 0.4,
                'holding_bias': 'NEUTRAL',
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'CONSERVATIVE',
                'max_holding_hours': (3, 12)
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE',
                'max_holding_hours': (1, 6)
            }
        }
        
        # AKTYWNY PROFIL
        self.active_profile = 'Qwen'
        
        # PARAMETRY OPERACYJNE
        self.max_simultaneous_positions = 4
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT']
        
        # STATYSTYKI
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'long_trades': 0,
            'short_trades': 0,
            'win_long_trades': 0,
            'win_short_trades': 0,
            'avg_holding_time': 0,
            'portfolio_utilization': 0
        }
        
        # DASHBOARD
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'average_confidence': 0,
            'portfolio_diversity': 0,
            'last_update': datetime.now(),
            'active_profile': self.active_profile
        }
        
        # Dane wykresu
        self.chart_data = {
            'labels': [],
            'values': []
        }
        
        # Inicjalizacja sesji HTTP dla pybit
        self.session = None
        if self.real_trading and PYBIT_AVAILABLE:
            try:
                self.session = HTTP(
                    testnet=self.testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
                self.logger.info("✅ Sesja HTTP pybit zainicjalizowana")
            except Exception as e:
                self.logger.error(f"❌ Błąd inicjalizacji sesji pybit: {e}")
                self.session = None
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - PRO VERSION (Bybit Execution)")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📊 Technicals: EMA({self.ema_short_period}/{self.ema_long_period}), RSI({self.rsi_period}) on {self.timeframe}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")

    def get_historical_data(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """POBIERA ŚWIECE (OHLCV) Z BINANCE - NOWA FUNKCJA Z BOTA V3"""
        try:
            url = f"{self.binance_base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': self.timeframe,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            # Tworzymy DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
            
            # Konwersja typów na liczby
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            self.logger.error(f"❌ Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBLICZA WSKAŹNIKI TECHNICZNE (EMA, RSI, ATR) - NOWA FUNKCJA Z BOTA V3"""
        if df.empty:
            return df
            
        # 1. EMA (Exponential Moving Average) - Trend
        df['EMA_short'] = df['close'].ewm(span=self.ema_short_period, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=self.ema_long_period, adjust=False).mean()
        
        # 2. RSI (Relative Strength Index) - Momentum
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR (Average True Range) - Zmienność dla Stop Loss
        df['TR'] = np.maximum(
            df['high'] - df['low'], 
            np.maximum(
                abs(df['high'] - df['close'].shift(1)), 
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę (Wrapper używający świec)"""
        try:
            df = self.get_historical_data(symbol, limit=1)
            if not df.empty:
                price = df['close'].iloc[-1]
                self.price_cache[symbol] = {'price': price, 'timestamp': datetime.now()}
                return price
            
            # Fallback do cache
            if symbol in self.price_cache:
                return self.price_cache[symbol]['price']
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał na podstawie ANALIZY TECHNICZNEJ (EMA/RSI) - Z BOTA V3"""
        
        # 1. Pobierz dane i oblicz wskaźniki
        df = self.get_historical_data(symbol)
        if df.empty or len(df) < 30:
            return "HOLD", 0.0

        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        
        # Wartości wskaźników
        rsi = last_row['RSI']
        ema_short = last_row['EMA_short']
        ema_long = last_row['EMA_long']
        price = last_row['close']
        
        # Pobierz profil ryzyka
        profile = self.get_current_profile()
        
        signal = "HOLD"
        confidence = 0.5 # Bazowe confidence
        
        # --- LOGIKA STRATEGII "PROFIT" ---
        
        # Wykrywanie trendu
        is_uptrend = ema_short > ema_long
        trend_strength = abs(ema_short - ema_long) / price * 1000 
        
        # WARUNKI WEJŚCIA
        if is_uptrend:
            # LONG: Cena nad EMA, RSI nie jest "przegrzane"
            if 40 < rsi < 70: 
                signal = "LONG"
                confidence = 0.6 + (0.1 if rsi < 60 else 0) + min(trend_strength, 0.2)
        else:
            # SHORT: Cena pod EMA, RSI nie jest "wyprzedane"
            if 30 < rsi < 60:
                signal = "SHORT"
                confidence = 0.6 + (0.1 if rsi > 40 else 0) + min(trend_strength, 0.2)

        # MODYFIKACJA PRZEZ "OSOBOWOŚĆ" LLM
        if profile['holding_bias'] == 'LONG' and signal == 'SHORT':
            confidence -= 0.15 
        elif profile['holding_bias'] == 'SHORT' and signal == 'LONG':
            confidence -= 0.15 
            
        confidence = (confidence + profile['confidence_bias']) / 2
            
        if confidence < 0.60:
            signal = "HOLD"
        
        trend_str = "UP 🟢" if is_uptrend else "DOWN 🔴"
        self.logger.info(f"📊 {symbol} | RSI: {rsi:.1f} | Trend: {trend_str} | Signal: {signal} ({confidence:.1%})")
        
        # Dodajemy ATR do cache (trick, aby użyć w exit_plan)
        self.price_cache[symbol + '_ATR'] = last_row['ATR'] if not pd.isna(last_row['ATR']) else price * 0.01
        
        return signal, final_confidence

    def get_account_balance(self) -> Optional[float]:
        """Pobiera rzeczywiste saldo konta z Bybit"""
        if not self.real_trading:
            return self.initial_capital
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return None

        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] == 0:
                total_equity = float(response['result']['list'][0]['totalEquity'])
                # self.logger.info(f"💰 Rzeczywiste saldo konta z Bybit: ${total_equity:.2f}")
                return total_equity
            else:
                self.logger.warning("⚠️ Nie udało się pobrać salda konta z Bybit")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance from Bybit: {e}")
            return None

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji UŻYWAJĄC RZECZYWISTEGO SALDA"""
        profile = self.get_current_profile()
        
        base_allocation = {
            'Claude': 0.15, 'Gemini': 0.25, 'GPT': 0.10, 'Qwen': 0.30
        }.get(self.active_profile, 0.15)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        sizing_multiplier = {
            'CONSERVATIVE': 0.8, 'AGGRESSIVE': 1.2, 'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        # TERAZ: Użyj rzeczywistego salda z Bybit
        real_balance = self.get_account_balance()
        if real_balance is None:
            self.logger.warning("⚠️ Could not get real balance, using initial capital")
            real_balance = self.initial_capital
        
        position_value = (real_balance * base_allocation * confidence_multiplier * sizing_multiplier)
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """DYNAMICZNY PLAN WYJŚCIA (Risk Management) w oparciu o ATR - Z BOTA V3"""
        profile = self.get_current_profile()
        
        # Pobieramy ATR (zmienność) z cache
        atr = self.price_cache.get(f"{self.active_profile}_temp_symbol_ATR", entry_price * 0.015)
        
        risk_factor = 1.0
        if profile['risk_appetite'] == 'HIGH': risk_factor = 1.5
        if profile['risk_appetite'] == 'LOW': risk_factor = 0.8

        # Risk:Reward Ratio
        rr_ratio = 2.0 if risk_factor < 1.0 else 1.5
        
        # Stop Loss = 2x ATR * risk_factor
        stop_distance = atr * 2.0 * risk_factor
        
        if stop_distance == 0 or pd.isna(stop_distance):
             stop_distance = entry_price * 0.02
        
        profit_distance = stop_distance * rr_ratio
        
        if side == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
            invalidation = stop_loss * 0.995
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
            invalidation = stop_loss * 1.005
            
        return {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': round(invalidation, 4),
            'max_holding_hours': random.randint(4, 24) # Dłuższy czas dla strategii trendowej
        }

    def should_enter_trade(self) -> bool:
        """Filtr częstotliwości"""
        profile = self.get_current_profile()
        frequency_chance = {'LOW': 0.4, 'MEDIUM': 0.7, 'HIGH': 0.9}.get(profile['trade_frequency'], 0.7)
        return random.random() < frequency_chance

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Ustawia dźwignię na Bybit"""
        if not self.real_trading: return True
        if not self.session: return False
        try:
            response = self.session.set_leverage(
                category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage),
            )
            if response['retCode'] == 0: return True
            if response['retCode'] == 110043: return True # Już ustawiona
            self.logger.error(f"❌ Error setting leverage: {response.get('retMsg')}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Exception setting leverage: {e}")
            return False

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatuje ilość pod specyfikację Bybit"""
        lot_size_rules = {'BTCUSDT': 0.001, 'ETHUSDT': 0.01, 'SOLUSDT': 0.01, 'XRPUSDT': 1, 'BNBUSDT': 0.001, 'DOGEUSDT': 1}
        lot_size = lot_size_rules.get(symbol, 0.001)
        formatted_quantity = round(quantity / lot_size) * lot_size
        if lot_size >= 1: formatted_quantity = int(formatted_quantity)
        elif lot_size == 0.001: formatted_quantity = round(formatted_quantity, 3)
        elif lot_size == 0.01: formatted_quantity = round(formatted_quantity, 2)
        else: formatted_quantity = round(formatted_quantity, 6)
        if formatted_quantity <= 0: formatted_quantity = lot_size
        return str(formatted_quantity)

    def place_bybit_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[str]:
        """Składa zlecenie na Bybit"""
        self.logger.info(f"🚀 PLACE_BYBIT_ORDER: {symbol} {side} Qty: {quantity:.6f}")
        if not self.real_trading: return f"virtual_{int(time.time())}"
        if not self.session: return None
            
        try:
            self.set_leverage(symbol, self.leverage)
            quantity_str = self.format_quantity(symbol, quantity)
            
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side="Buy" if side == "LONG" else "Sell",
                orderType="Market",
                qty=quantity_str,
                timeInForce="GTC",
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"✅ ORDER SUCCESS: {symbol} {side} - ID: {order_id}")
                return order_id
            else:
                self.logger.error(f"❌ ORDER FAILED: {response.get('retMsg')}")
                return None
        except Exception as e:
            self.logger.error(f"💥 CRITICAL ERROR in place_bybit_order: {e}")
            return None

    def close_bybit_position(self, symbol: str, side: str, quantity: float) -> bool:
        """Zamyka pozycję na Bybit"""
        if not self.real_trading: return True
        if not self.session: return False

        try:
            close_side = 'Sell' if side == 'LONG' else 'Buy'
            quantity_str = self.format_quantity(symbol, quantity)
            
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=quantity_str,
                timeInForce="GTC",
                reduceOnly=True,
            )
            
            if response['retCode'] == 0:
                self.logger.info(f"✅ Pozycja zamknięta: {symbol}")
                return True
            else:
                self.logger.error(f"❌ Błąd zamykania: {response.get('retMsg')}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Exception closing position: {e}")
            return False

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję - INTEGRACJA LOGIKI V3 I EGZEKUCJI BYBIT"""
        if not self.should_enter_trade():
            return None
            
        # 1. Analiza Techniczna (Generuje też cache ATR i cenę)
        signal, confidence = self.generate_llm_signal(symbol)
        
        if signal == "HOLD" or confidence < 0.6: # Próg wejścia z V3
            return None
        
        # 2. Pobranie ceny
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        # Trick z cache ATR dla calculate_position_size/exit_plan
        df = self.get_historical_data(symbol, limit=20)
        df = self.calculate_indicators(df)
        if not df.empty:
            self.price_cache[f"{self.active_profile}_temp_symbol_ATR"] = df['ATR'].iloc[-1]

        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        # 3. Obliczenie wielkości (na podstawie REALNEGO salda Bybit)
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        # Sprawdzenie salda (Realnego)
        available_balance = self.get_account_balance()
        if available_balance and margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance. Avail: ${available_balance:.2f}, Req: ${margin_required:.2f}")
            return None
            
        # 4. Plan Wyjścia (Na podstawie ATR z V3)
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        liquidation_price = current_price * (1 - 0.9 / self.leverage) if signal == "LONG" else current_price * (1 + 0.9 / self.leverage)
        
        # 5. Egzekucja na Bybit
        order_id = None
        if self.real_trading:
            order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
            if not order_id:
                return None
        
        position_id = f"llm_{self.position_id}"
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
            'llm_profile': self.active_profile,
            'exit_plan': exit_plan,
            'order_id': order_id,
            'real_trading': self.real_trading
        }
        
        self.positions[position_id] = position
        
        if signal == "LONG": self.stats['long_trades'] += 1
        else: self.stats['short_trades'] += 1
        
        tp_dist = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_dist = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        self.logger.info(f"🎯 OPEN {symbol} {signal} | Conf: {confidence:.1%} | TP: {tp_dist:+.2f}% | SL: {sl_dist:+.2f}% | ATR-Based")
        
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L i Saldo (używając rzeczywistego salda)"""
        total_unrealized = 0
        total_margin = 0
        confidence_count = 0
        total_confidence = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE': continue
                
            current_price = self.get_current_price(position['symbol'])
            if not current_price: continue
            
            entry = position['entry_price']
            qty = position['quantity']
            
            if position['side'] == 'LONG':
                pnl = (current_price - entry) / entry * qty * entry
            else:
                pnl = (entry - current_price) / entry * qty * entry
            
            position['unrealized_pnl'] = pnl
            position['current_price'] = current_price
            total_unrealized += pnl
            total_margin += position['margin']
            total_confidence += position['confidence']
            confidence_count += 1
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        
        # Pobierz rzeczywiste saldo do dashboardu
        real_balance = self.get_account_balance()
        if real_balance:
            self.dashboard_data['account_value'] = real_balance + total_unrealized
            self.dashboard_data['available_cash'] = real_balance
            self.stats['portfolio_utilization'] = total_margin / real_balance
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
            
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia (SL/TP/Czas)"""
        positions_to_close = []
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE': continue
            current_price = position.get('current_price', self.get_current_price(position['symbol']))
            if not current_price: continue
                
            exit_reason = None
            plan = position['exit_plan']
            side = position['side']
            
            if side == 'LONG':
                if current_price >= plan['take_profit']: exit_reason = "TAKE_PROFIT"
                elif current_price <= plan['stop_loss']: exit_reason = "STOP_LOSS"
                elif current_price <= plan['invalidation']: exit_reason = "INVALIDATION"
            else:
                if current_price <= plan['take_profit']: exit_reason = "TAKE_PROFIT"
                elif current_price >= plan['stop_loss']: exit_reason = "STOP_LOSS"
                elif current_price >= plan['invalidation']: exit_reason = "INVALIDATION"
            
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_time > plan['max_holding_hours']: exit_reason = "TIME_EXPIRED"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję (Logika + Bybit)"""
        position = self.positions[position_id]
        entry = position['entry_price']
        qty = position['quantity']
        
        if position['side'] == 'LONG':
            pnl = (exit_price - entry) / entry * qty * entry
        else:
            pnl = (entry - exit_price) / entry * qty * entry
            
        realized_pnl = pnl - (abs(pnl) * 0.001) # Fee approx
        
        # Zamknij na Bybit
        if position.get('real_trading', False):
            self.close_bybit_position(position['symbol'], position['side'], position['quantity'])
            
        self.trade_history.append({
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': entry,
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
            'exit_reason': exit_reason,
            'exit_time': datetime.now(),
            'confidence': position['confidence'],
            'holding_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        })
        
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl
        if realized_pnl > 0: self.stats['winning_trades'] += 1
        else: self.stats['losing_trades'] += 1
        
        position['status'] = 'CLOSED'
        pnl_color = "🟢" if realized_pnl > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSE {position['symbol']} | P&L: ${realized_pnl:.2f} | {exit_reason}")

    def get_portfolio_diversity(self) -> float:
        try:
            active = [p for p in self.positions.values() if p['status'] == 'ACTIVE']
            if not active: return 0
            total_margin = sum(p['margin'] for p in active)
            if total_margin == 0: return 0
            conc = sum((p['margin'] / total_margin) ** 2 for p in active)
            return 1 - conc
        except: return 0

    def get_current_profile(self): return self.llm_profiles[self.active_profile]

    def set_active_profile(self, profile_name: str):
        if profile_name in self.llm_profiles:
            self.active_profile = profile_name
            self.dashboard_data['active_profile'] = profile_name
            return True
        return False

    def get_dashboard_data(self):
        # Skrócona wersja dla czytelności - zachowuje strukturę API
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
                'real_trading': self.real_trading
            },
            'active_positions': [
                {
                    'symbol': p['symbol'], 'side': p['side'], 'entry_price': p['entry_price'],
                    'current_price': p.get('current_price', 0), 'unrealized_pnl': p.get('unrealized_pnl', 0),
                    'confidence': p['confidence'], 'tp_distance_pct': 0, 'sl_distance_pct': 0 # Uproszczone
                } for p in self.positions.values() if p['status'] == 'ACTIVE'
            ],
            'recent_trades': self.trade_history[-10:],
            'llm_config': {
                'active_profile': self.active_profile,
                'real_trading': self.real_trading
            },
             'performance_metrics': { # Dodanie brakujących kluczy, aby frontend nie wyrzucił błędu
                'total_return_pct': 0,
                'win_rate': 0,
                'total_trades': self.stats['total_trades'],
                 'portfolio_diversity': 0,
                 'avg_confidence': 0
            },
             'confidence_levels': {},
             'last_update': datetime.now().isoformat()
        }

    def save_chart_data(self, chart_data: Dict):
        self.chart_data = chart_data
        return True

    def load_chart_data(self) -> Dict:
        return self.chart_data

    def run_llm_trading_strategy(self):
        self.logger.info("🚀 STARTING STRATEGY (V3 Logic + Bybit Execution)")
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 Iteration #{iteration}")
                self.update_positions_pnl()
                
                # Sprawdź wyjścia
                for pid, reason, price in self.check_exit_conditions():
                    self.close_position(pid, reason, price)
                
                # Sprawdź wejścia
                active_syms = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                if len(active_syms) < self.max_simultaneous_positions:
                    for sym in self.assets:
                        if sym not in active_syms:
                            if self.open_llm_position(sym):
                                time.sleep(1)
                
                time.sleep(20) # Czekaj na następną świecę/cykl
            except Exception as e:
                self.logger.error(f"Loop Error: {e}")
                time.sleep(30)

    def start_trading(self):
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()

    def stop_trading(self):
        self.is_running = False

# FLASK APP
app = Flask(__name__)
CORS(app)
trading_bot = LLMTradingBot(initial_capital=10000, leverage=10)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/trading-data')
def get_trading_data(): return jsonify(trading_bot.get_dashboard_data())

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    trading_bot.start_trading()
    return jsonify({'status': 'started'})

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    trading_bot.stop_trading()
    return jsonify({'status': 'stopped'})

@app.route('/api/change-profile', methods=['POST'])
def change_profile():
    data = request.get_json()
    trading_bot.set_active_profile(data.get('profile'))
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🚀 BYBIT PRO BOT STARTED (EMA/RSI/ATR Strategy)")
    app.run(debug=True, host='0.0.0.0', port=5000)
