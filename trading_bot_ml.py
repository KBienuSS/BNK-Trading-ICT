# trading_bot_ml.py
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_trading_bot.log', encoding='utf-8')
    ]
)

class LLMTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # API Binance
        self.binance_base_url = "https://api.binance.com/api/v3"
        
        # Cache cen
        self.price_cache = {}
        self.price_history = {}
        
        # KONFIGURACJA STRATEGII TECHNICZNEJ (NOWOŚĆ)
        self.timeframe = '15m'  # Interwał świecowy
        self.ema_short_period = 9
        self.ema_long_period = 21
        self.rsi_period = 14
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM (wg Alpha Arena)
        self.llm_profiles = {
            'Claude': {
                'risk_appetite': 'MEDIUM',
                'confidence_bias': 0.6,
                'short_frequency': 0.1,
                'holding_bias': 'LONG',
                'trade_frequency': 'LOW',
                'position_sizing': 'CONSERVATIVE'
            },
            'Gemini': {
                'risk_appetite': 'HIGH', 
                'confidence_bias': 0.7,
                'short_frequency': 0.35,
                'holding_bias': 'SHORT',
                'trade_frequency': 'HIGH',
                'position_sizing': 'AGGRESSIVE'
            },
            'GPT': {
                'risk_appetite': 'LOW',
                'confidence_bias': 0.3,
                'short_frequency': 0.4,
                'holding_bias': 'NEUTRAL',
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'CONSERVATIVE'
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE'
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
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - PRO VERSION")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📊 Technicals: EMA({self.ema_short_period}/{self.ema_long_period}), RSI({self.rsi_period}) on {self.timeframe}")

    def get_historical_data(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """POBIERA ŚWIECE (OHLCV) ZAMIAST SAMEJ CENY - NOWA FUNKCJA"""
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
        """OBLICZA WSKAŹNIKI TECHNICZNE (EMA, RSI, ATR) - NOWA FUNKCJA"""
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
        """Pobiera aktualną cenę (Wrapper dla kompatybilności)"""
        try:
            df = self.get_historical_data(symbol, limit=1)
            if not df.empty:
                price = df['close'].iloc[-1]
                self.price_cache[symbol] = {'price': price, 'timestamp': datetime.now()}
                return price
            
            # Fallback do cache jeśli API zawiedzie
            if symbol in self.price_cache:
                return self.price_cache[symbol]['price']
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał na podstawie ANALIZY TECHNICZNEJ, a nie losowości"""
        
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
        
        # Wykrywanie trendu (Golden Cross / Death Cross)
        is_uptrend = ema_short > ema_long
        trend_strength = abs(ema_short - ema_long) / price * 1000 # Znormalizowana siła trendu
        
        # WARUNKI WEJŚCIA
        if is_uptrend:
            # LONG: Cena nad EMA, RSI nie jest "przegrzane" (>70)
            # Unikamy kupowania na szczycie
            if 40 < rsi < 70: 
                signal = "LONG"
                confidence = 0.6 + (0.1 if rsi < 60 else 0) + min(trend_strength, 0.2)
        else:
            # SHORT: Cena pod EMA, RSI nie jest "wyprzedane" (<30)
            # Unikamy sprzedawania w dołku
            if 30 < rsi < 60:
                signal = "SHORT"
                confidence = 0.6 + (0.1 if rsi > 40 else 0) + min(trend_strength, 0.2)

        # MODYFIKACJA PRZEZ "OSOBOWOŚĆ" LLM
        if profile['holding_bias'] == 'LONG' and signal == 'SHORT':
            confidence -= 0.15 # Profil "Byka" niechętnie shortuje
        elif profile['holding_bias'] == 'SHORT' and signal == 'LONG':
            confidence -= 0.15 # Profil "Niedźwiedzia" niechętnie longuje
            
        # Wpływ "Confidence Bias" z profilu
        confidence = (confidence + profile['confidence_bias']) / 2
            
        # Filtrowanie słabych sygnałów (musi być min 60% pewności)
        if confidence < 0.60:
            signal = "HOLD"
        
        # Logowanie analizy
        trend_str = "UP 🟢" if is_uptrend else "DOWN 🔴"
        self.logger.info(f"📊 {symbol} | RSI: {rsi:.1f} | Trend: {trend_str} | Signal: {signal} ({confidence:.1%})")
        
        # Dodajemy ATR do cache (trick, aby użyć w exit_plan)
        self.price_cache[symbol + '_ATR'] = last_row['ATR'] if not pd.isna(last_row['ATR']) else price * 0.01
        
        return signal, confidence

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji (Bez zmian logicznych, tylko czystszy kod)"""
        profile = self.get_current_profile()
        
        base_allocation = {
            'Claude': 0.15, 'Gemini': 0.25, 'GPT': 0.10, 'Qwen': 0.30
        }.get(self.active_profile, 0.15)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        sizing_multiplier = {
            'CONSERVATIVE': 0.8, 'AGGRESSIVE': 1.2, 'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        position_value = (self.virtual_capital * base_allocation * confidence_multiplier * sizing_multiplier)
        
        max_position_value = self.virtual_capital * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """DYNAMICZNY PLAN WYJŚCIA (Risk Management) w oparciu o Zmienność (ATR)"""
        profile = self.get_current_profile()
        
        # Pobieramy ATR (zmienność) z cache (zapisane przy generowaniu sygnału)
        # Jeśli brak, zakładamy standardowe 1.5% zmienności
        atr = self.price_cache.get(f"{self.active_profile}_temp_symbol_ATR", entry_price * 0.015)
        
        # Ustalanie mnożnika ryzyka w zależności od profilu
        risk_factor = 1.0
        if profile['risk_appetite'] == 'HIGH': risk_factor = 1.5
        if profile['risk_appetite'] == 'LOW': risk_factor = 0.8

        # Stosunek Zysk:Ryzyko (Risk:Reward Ratio)
        # Dla bezpieczniejszych profili chcemy większy zysk za ryzyko (1:2)
        # Dla agresywnych akceptujemy mniejszy (1:1.5) dla szybszych ruchów
        rr_ratio = 2.0 if risk_factor < 1.0 else 1.5
        
        # Obliczanie odległości SL na podstawie ATR (np. 2x ATR)
        stop_distance = atr * 2.0 * risk_factor
        
        # Jeśli ATR nie jest dostępny (np. start bota), użyj procentowego fail-safe
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
            'max_holding_hours': random.randint(4, 24) # Dłuższy czas dla trendów
        }

    def should_enter_trade(self) -> bool:
        """Sprawdza czy wejść (filtr częstotliwości)"""
        profile = self.get_current_profile()
        frequency_chance = {'LOW': 0.4, 'MEDIUM': 0.7, 'HIGH': 0.9}.get(profile['trade_frequency'], 0.7)
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję używając nowej logiki"""
        if not self.should_enter_trade():
            return None
            
        # Tutaj trick: generate_llm_signal pobiera też cenę do cache
        signal, confidence = self.generate_llm_signal(symbol)
        
        if signal == "HOLD" or confidence < 0.6: # Podniesiony próg wejścia
            return None
        
        # Pobierz cenę (już powinna być w cache po generate_signal)
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        # Zapisz ATR dla tego symbolu do obliczeń exit planu
        # (W normalnym kodzie przekazywalibyśmy to jako argument, tu używamy cache obiektu dla prostoty)
        df = self.get_historical_data(symbol, limit=20)
        df = self.calculate_indicators(df)
        if not df.empty:
            self.price_cache[f"{self.active_profile}_temp_symbol_ATR"] = df['ATR'].iloc[-1]
            
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
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
            'exit_plan': exit_plan
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        self.logger.info(f"🎯 {self.active_profile} OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L (bez zmian, używa get_current_price)"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
                
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
            total_confidence += position['confidence']
            confidence_count += 1
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        if self.virtual_capital > 0:
            self.stats['portfolio_utilization'] = total_margin / self.virtual_capital
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia (bez zmian)"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = position.get('current_price', self.get_current_price(position['symbol']))
            if not current_price:
                continue
                
            exit_reason = None
            exit_plan = position['exit_plan']
            
            if position['side'] == 'LONG':
                if current_price >= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price <= exit_plan['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price <= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            else:
                if current_price <= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price >= exit_plan['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price >= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            
            # Zwiększyliśmy max_holding_hours w exit planie, więc tu bez zmian
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_time > exit_plan['max_holding_hours']:
                exit_reason = "TIME_EXPIRED"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję (bez zmian)"""
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
        
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'realized_pnl': realized_pnl_after_fee,
            'exit_reason': exit_reason,
            'llm_profile': position['llm_profile'],
            'confidence': position['confidence'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'holding_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        }
        
        self.trade_history.append(trade_record)
        
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        total_holding = sum((t['exit_time'] - t['entry_time']).total_seconds() 
                          for t in self.trade_history) / 3600
        self.stats['avg_holding_time'] = total_holding / len(self.trade_history) if self.trade_history else 0
        
        position['status'] = 'CLOSED'
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        margin_return = pnl_pct * self.leverage * 100
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}% margin) - Reason: {exit_reason}")

    def get_portfolio_diversity(self) -> float:
        """Oblicza dywersyfikację (bez zmian)"""
        try:
            active_positions = [p for p in self.positions.values() if p['status'] == 'ACTIVE']
            if not active_positions:
                return 0
            
            total_margin = sum(p['margin'] for p in active_positions)
            if total_margin == 0:
                return 0
            
            concentration_index = sum((p['margin'] / total_margin) ** 2 for p in active_positions)
            diversity = 1 - concentration_index
            
            return diversity
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio diversity: {e}")
            return 0

    def get_current_profile(self):
        """Zwraca aktywny profil LLM"""
        return self.llm_profiles[self.active_profile]

    def set_active_profile(self, profile_name: str):
        """Zmienia aktywny profil"""
        if profile_name in self.llm_profiles:
            self.active_profile = profile_name
            self.dashboard_data['active_profile'] = profile_name
            self.logger.info(f"🔄 Changed LLM profile to: {profile_name}")
            return True
        return False

    def get_dashboard_data(self):
        """Dane dashboardu (bez zmian)"""
        active_positions = []
        total_unrealized_pnl = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                if not current_price:
                    continue
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                if position['side'] == 'LONG':
                    tp_distance_pct = ((position['exit_plan']['take_profit'] - current_price) / current_price) * 100
                    sl_distance_pct = ((current_price - position['exit_plan']['stop_loss']) / current_price) * 100
                else:
                    tp_distance_pct = ((current_price - position['exit_plan']['take_profit']) / current_price) * 100
                    sl_distance_pct = ((position['exit_plan']['stop_loss'] - current_price) / current_price) * 100
                
                active_positions.append({
                    'position_id': position_id,
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'margin': position['margin'],
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position['confidence'],
                    'llm_profile': position['llm_profile'],
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'exit_plan': position['exit_plan'],
                    'tp_distance_pct': tp_distance_pct,
                    'sl_distance_pct': sl_distance_pct
                })
                
                total_unrealized_pnl += unrealized_pnl
        
        # Confidence levels - teraz używają nowej logiki
        confidence_levels = {}
        for symbol in self.assets:
            try:
                # Nie generuj pełnego logu przy odświeżaniu dashboardu, tylko prosty check
                # W realnej aplikacji cache'owalibyśmy to, tutaj wywołujemy
                # Ale uwaga: generate_llm_signal robi request do API, więc ostrożnie
                # Dla dashboardu możemy zwrócić 0 lub ostatnią znaną wartość
                confidence_levels[symbol] = 0 
            except:
                confidence_levels[symbol] = 0
        
        recent_trades = []
        for trade in self.trade_history[-10:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'llm_profile': trade['llm_profile'],
                'confidence': trade['confidence'],
                'holding_hours': round(trade['holding_hours'], 2),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S')
            })
        
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        total_return_pct = ((self.dashboard_data['account_value'] - 10000) / 10000) * 100
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2),
                'unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2)
            },
            'performance_metrics': {
                'total_return_pct': round(total_return_pct, 2),
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'long_trades': self.stats['long_trades'],
                'short_trades': self.stats['short_trades'],
                'avg_holding_hours': round(self.stats['avg_holding_time'], 2),
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1),
                'portfolio_diversity': round(self.get_portfolio_diversity() * 100, 1),
                'avg_confidence': round(self.dashboard_data['average_confidence'] * 100, 1)
            },
            'llm_config': {
                'active_profile': self.active_profile,
                'available_profiles': list(self.llm_profiles.keys()),
                'max_positions': self.max_simultaneous_positions,
                'leverage': self.leverage
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': total_unrealized_pnl,
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def save_chart_data(self, chart_data: Dict):
        """Zapisuje dane wykresu"""
        try:
            self.chart_data = chart_data
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving chart data: {e}")
            return False

    def load_chart_data(self) -> Dict:
        """Ładuje dane wykresu"""
        return self.chart_data

    def run_llm_trading_strategy(self):
        """Główna pętla strategii"""
        self.logger.info("🚀 STARTING LLM-STYLE TRADING STRATEGY")
        self.logger.info(f"🎯 Active Profile: {self.active_profile}")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 LLM Trading Iteration #{iteration}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Sprawdź możliwości wejścia
                active_symbols = [p['symbol'] for p in self.positions.values() 
                                if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.assets:
                        if symbol not in active_symbols:
                            position_id = self.open_llm_position(symbol)
                            if position_id:
                                time.sleep(1) # Delikatne opóźnienie między orderami
                
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                # Czekaj na następny cykl
                # Ponieważ używamy świec 15m, nie musimy spamować co chwila, ale dla demo 20s jest ok
                wait_time = 20 
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in LLM trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Rozpoczyna trading"""
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("🚀 LLM Trading Bot started")

    def stop_trading(self):
        """Zatrzymuje trading"""
        self.is_running = False
        self.logger.info("🛑 LLM Trading Bot stopped")


# FLASK APP
app = Flask(__name__)
CORS(app)

# Inicjalizacja bota
trading_bot = LLMTradingBot(initial_capital=10000, leverage=10)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

# API endpoints
@app.route('/api/trading-data')
def get_trading_data():
    try:
        data = trading_bot.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot-status')
def get_bot_status():
    status = 'running' if trading_bot.is_running else 'stopped'
    return jsonify({'status': status})

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    try:
        trading_bot.start_trading()
        return jsonify({'status': 'Bot started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    try:
        trading_bot.stop_trading()
        return jsonify({'status': 'Bot stopped successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/change-profile', methods=['POST'])
def change_profile():
    try:
        data = request.get_json()
        profile_name = data.get('profile')
        if trading_bot.set_active_profile(profile_name):
            return jsonify({'status': f'Profile changed to {profile_name}'})
        else:
            return jsonify({'error': 'Invalid profile name'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-update', methods=['POST'])
def force_update():
    try:
        trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    try:
        data = request.get_json()
        if trading_bot.save_chart_data(data):
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Failed to save chart data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-chart-data')
def load_chart_data():
    try:
        chart_data = trading_bot.load_chart_data()
        return jsonify({
            'status': 'success',
            'chartData': chart_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting LLM Trading Bot Server (PRO VERSION)...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 Profiles: Claude, Gemini, GPT, Qwen")
    print("📈 Analysis: RSI + EMA Cross + ATR")
    app.run(debug=True, host='0.0.0.0', port=5000)
