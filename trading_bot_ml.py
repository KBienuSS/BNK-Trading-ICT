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
        
        # Konfiguracja Bybit API
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.base_url = "https://api.bybit.com"
        self.testnet = False  # Ustaw na True dla testnet
        
        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        
        # Sprawdź czy klucze API są dostępne
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ Brak kluczy API Bybit - bot będzie działał w trybie wirtualnym")
            self.real_trading = False
        else:
            self.real_trading = True
            self.logger.info("🔑 Klucze API Bybit załadowane - REAL TRADING ENABLED")
        
        # Kapitał wirtualny (fallback)
        self.initial_capital = initial_capital
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        # Cache cen
        self.price_cache = {}
        self.price_history = {}
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM (wg Alpha Arena)
        self.llm_profiles = {
            'Claude': {
                'risk_appetite': 'MEDIUM',
                'confidence_bias': 0.6,
                'short_frequency': 0.1,
                'holding_bias': 'LONG',
                'trade_frequency': 'LOW',
                'position_sizing': 'CONSERVATIVE',
                'max_position_size_pct': 0.15,
                'min_confidence': 0.3,
                'momentum_threshold': 0.003
            },
            'Gemini': {
                'risk_appetite': 'HIGH', 
                'confidence_bias': 0.7,
                'short_frequency': 0.35,
                'holding_bias': 'SHORT',
                'trade_frequency': 'HIGH',
                'position_sizing': 'AGGRESSIVE',
                'max_position_size_pct': 0.25,
                'min_confidence': 0.3,
                'momentum_threshold': 0.002
            },
            'GPT': {
                'risk_appetite': 'LOW',
                'confidence_bias': 0.3,
                'short_frequency': 0.4,
                'holding_bias': 'NEUTRAL',
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'CONSERVATIVE',
                'max_position_size_pct': 0.10,
                'min_confidence': 0.3,
                'momentum_threshold': 0.004
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE',
                'max_position_size_pct': 0.30,
                'min_confidence': 0.3,
                'momentum_threshold': 0.002
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
            'portfolio_utilization': 0,
            'won_long_trades': 0,
            'won_short_trades': 0
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
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - Alpha Arena Inspired")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📈 Trading assets: {', '.join(self.assets)}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Ustawia dźwignię dla symbolu używając pybit"""
        if not self.real_trading:
            return True
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return False

        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            
            if response['retCode'] == 0:
                self.logger.info(f"✅ Ustawiono dźwignię {leverage}x dla {symbol}")
                return True
            else:
                # Błąd 110043 oznacza, że dźwignia jest już ustawiona - traktuj jako sukces
                if response['retCode'] == 110043:
                    self.logger.info(f"ℹ️ Dźwignia już ustawiona na {leverage}x dla {symbol}")
                    return True
                else:
                    error_msg = response.get('retMsg', 'Unknown error')
                    self.logger.error(f"❌ Błąd ustawiania dźwigni dla {symbol}: {error_msg}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error setting leverage for {symbol}: {e}")
            return False

    def check_available_categories(self):
        """Sprawdza dostępne kategorie dla konta używając pybit"""
        self.logger.info("🔍 Checking available categories...")
        
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return []
            
        categories_to_test = ['spot', 'linear', 'inverse', 'option']
        available_categories = []
        
        for category in categories_to_test:
            try:
                response = self.session.get_tickers(
                    category=category,
                    symbol='BTCUSDT'  # Sprawdź dla konkretnego symbolu
                )
                
                if response['retCode'] == 0:
                    available_categories.append(category)
                    self.logger.info(f"✅ Category '{category}' is available")
                else:
                    self.logger.info(f"❌ Category '{category}' is NOT available")
            except Exception as e:
                self.logger.info(f"❌ Category '{category}' is NOT available: {e}")
        
        self.logger.info(f"📊 Available categories: {available_categories}")
        return available_categories

    def sync_all_positions_with_bybit(self):
        """Kompletna synchronizacja pozycji z Bybit"""
        if not self.real_trading:
            self.logger.info("🔄 SYNC: Virtual mode - no Bybit sync needed")
            return
            
        self.logger.info("🔄 FULL SYNC: Synchronizing all positions with Bybit...")
        
        try:
            # 1. Pobierz wszystkie pozycje z Bybit
            bybit_positions = self.get_bybit_positions()
            self.logger.info(f"📊 BYBIT POSITIONS: Found {len(bybit_positions)} positions on Bybit")
            
            # Jeśli nie ma pozycji, sprawdź czy to problem z API
            if len(bybit_positions) == 0:
                self.logger.warning("⚠️ No positions found on Bybit - checking API connection...")
                api_status = self.test_bybit_api_connection()
                self.logger.info(f"🔧 API Status: {api_status}")
            
            # 2. Wyczyść stare pozycje i dodaj nowe z Bybit
            old_count = len(self.positions)
            self.positions = {}  # Reset lokalnych pozycji
            
            for i, bybit_pos in enumerate(bybit_positions):
                position_id = f"bybit_sync_{i}_{int(time.time())}"
                
                current_price = self.get_current_price(bybit_pos['symbol'])
                if not current_price:
                    current_price = bybit_pos['entry_price']
                    self.logger.warning(f"⚠️ Could not get current price for {bybit_pos['symbol']}, using entry price")
                
                # Oblicz unrealized P&L
                unrealized_pnl = bybit_pos.get('unrealised_pnl', 0)
                
                self.positions[position_id] = {
                    'symbol': bybit_pos['symbol'],
                    'side': bybit_pos['side'],
                    'entry_price': bybit_pos['entry_price'],
                    'quantity': bybit_pos['size'],
                    'leverage': bybit_pos['leverage'],
                    'entry_time': bybit_pos.get('created_time', datetime.now()),
                    'status': 'ACTIVE',
                    'order_id': f"bybit_{bybit_pos['symbol']}",
                    'real_trading': True,
                    'llm_profile': self.active_profile,
                    'confidence': 0.5,
                    'margin': bybit_pos.get('position_margin', bybit_pos['size'] * bybit_pos['entry_price'] / bybit_pos['leverage']),
                    'exit_plan': self.calculate_llm_exit_plan(bybit_pos['entry_price'], 0.5, bybit_pos['side']),
                    'liquidation_price': bybit_pos.get('liq_price'),
                    'unrealized_pnl': unrealized_pnl,
                    'current_price': current_price
                }
                
                self.logger.info(f"✅ SYNCED: {bybit_pos['symbol']} {bybit_pos['side']} - Size: {bybit_pos['size']}, Entry: ${bybit_pos['entry_price']}, PnL: ${unrealized_pnl:.2f}")
            
            self.logger.info(f"🎯 SYNC COMPLETE: {old_count} -> {len(self.positions)} positions synchronized with Bybit")
            
        except Exception as e:
            self.logger.error(f"❌ SYNC ERROR: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
        
    def get_account_balance(self) -> Optional[float]:
        """Pobiera rzeczywiste saldo konta z Bybit używając pybit"""
        if not self.real_trading:
            return self.virtual_balance
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return None

        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response['retCode'] == 0:
                total_equity = float(response['result']['list'][0]['totalEquity'])
                self.logger.info(f"💰 Rzeczywiste saldo konta z Bybit: ${total_equity:.2f}")
                return total_equity
            else:
                self.logger.warning("⚠️ Nie udało się pobrać salda konta z Bybit")
                return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance from Bybit: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera cenę futures TYLKO przez PUBLIC API - bez autoryzacji"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {
                'category': 'linear',
                'symbol': symbol
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('retCode') == 0:
                    result = data.get('result', {})
                    if 'list' in result and len(result['list']) > 0:
                        price_str = result['list'][0].get('lastPrice')
                        if price_str:
                            price = float(price_str)
                            return price
            return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting PUBLIC futures price for {symbol}: {e}")
            return None

    def place_bybit_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[str]:
        """Składa zlecenie futures na Bybit używając pybit"""
        
        self.logger.info(f"🚀 PLACE_BYBIT_ORDER: {symbol} {side} Qty: {quantity:.6f}")
        
        if not self.real_trading:
            order_id = f"virtual_{int(time.time())}"
            self.logger.info(f"🔄 Virtual order: {order_id}")
            return order_id
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return None
            
        try:
            # 1. Ustaw dźwignię
            self.set_leverage(symbol, self.leverage)
    
            # 2. Formatowanie quantity
            quantity_str = self.format_quantity(symbol, quantity)
            
            # 3. Złóż zlecenie
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
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ ORDER FAILED: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"💥 CRITICAL ERROR in place_bybit_order: {e}")
            import traceback
            self.logger.error(f"💥 Stack trace: {traceback.format_exc()}")
            return None

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na podstawie rzeczywistych danych z Bybit API"""
        try:
            # Użyj historii cen do obliczenia momentum
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return random.uniform(-0.02, 0.02)
            
            history = self.price_history[symbol]
            current_price = history[-1]['price']
            
            # Oblicz momentum na podstawie ostatnich punktów
            lookback = min(5, len(history) - 1)
            past_price = history[-lookback]['price']
            
            momentum = (current_price - past_price) / past_price
            
            # Normalizuj momentum
            momentum = max(min(momentum, 0.03), -0.03)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return random.uniform(-0.02, 0.02)

    def check_volume_activity(self, symbol: str) -> bool:
        """Sprawdza aktywność wolumenu na podstawie zmienności cen z Bybit API"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return random.random() < 0.6
            
            # Oblicz zmienność na podstawie rzeczywistej historii cen
            prices = [entry['price'] for entry in self.price_history[symbol][-10:]]
            volatility = np.std(prices) / np.mean(prices)
            
            # Wyższa zmienność = wyższa aktywność
            return volatility > 0.002
            
        except Exception as e:
            self.logger.error(f"❌ Error checking volume activity for {symbol}: {e}")
            return random.random() < 0.6

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu LLM na podstawie rzeczywistych danych z Bybit API"""
        profile = self.get_current_profile()
        
        # Podstawowe obserwacje na podstawie rzeczywistych cen
        momentum = self.analyze_simple_momentum(symbol)
        volume_active = self.check_volume_activity(symbol)
        
        # Confidence bazowe z profilu
        base_confidence = profile['confidence_bias']
        
        # Modyfikatory confidence na podstawie rzeczywistych danych
        confidence_modifiers = 0
        
        if momentum > 0.008:  # Silny pozytywny momentum
            confidence_modifiers += 0.2
        elif momentum > 0.003:  # Umiarkowany pozytywny momentum
            confidence_modifiers += 0.1
        elif momentum < -0.008:  # Silny negatywny momentum
            confidence_modifiers += 0.15
        elif momentum < -0.003:  # Umiarkowany negatywny momentum
            confidence_modifiers += 0.08
            
        if volume_active:
            confidence_modifiers += 0.1
            
        # Final confidence z losowością
        final_confidence = min(base_confidence + confidence_modifiers + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.1)
        
        # Decyzja o kierunku na podstawie rzeczywistego momentum
        if momentum > 0.01 and volume_active:
            signal = "LONG"
        elif momentum < -0.01 and volume_active:
            if random.random() < profile['short_frequency']:
                signal = "SHORT"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
            
        current_price = self.get_current_price(symbol)
        price_display = f"${current_price:.4f}" if current_price else "N/A"
        self.logger.info(f"🎯 {self.active_profile} SIGNAL: {symbol} -> {signal} (Price: {price_display}, Conf: {final_confidence:.1%}, Mom: {momentum:.2%})")
        
        return signal, final_confidence

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu LLM"""
        profile = self.get_current_profile()
        
        base_allocation = {
            'Claude': 0.15,
            'Gemini': 0.25, 
            'GPT': 0.10,
            'Qwen': 0.30
        }.get(self.active_profile, 0.15)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        sizing_multiplier = {
            'CONSERVATIVE': 0.8,
            'AGGRESSIVE': 1.2,
            'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        # Pobierz rzeczywiste saldo konta
        real_balance = self.get_account_balance()
        if real_balance is None:
            real_balance = self.virtual_balance
        
        position_value = (real_balance * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu LLM"""
        profile = self.get_current_profile()
        
        if confidence > 0.7:
            if side == "LONG":
                take_profit = entry_price * 1.018
                stop_loss = entry_price * 0.992
            else:
                take_profit = entry_price * 0.982
                stop_loss = entry_price * 1.008
        elif confidence > 0.5:
            if side == "LONG":
                take_profit = entry_price * 1.012
                stop_loss = entry_price * 0.994
            else:
                take_profit = entry_price * 0.988
                stop_loss = entry_price * 1.006
        else:
            if side == "LONG":
                take_profit = entry_price * 1.008
                stop_loss = entry_price * 0.996
            else:
                take_profit = entry_price * 0.992
                stop_loss = entry_price * 1.004
        
        risk_multiplier = {
            'LOW': 0.8,
            'MEDIUM': 1.0,
            'HIGH': 1.2
        }.get(profile['risk_appetite'], 1.0)
        
        if side == "LONG":
            take_profit = entry_price + (take_profit - entry_price) * risk_multiplier
            stop_loss = entry_price - (entry_price - stop_loss) * risk_multiplier
        else:
            take_profit = entry_price - (entry_price - take_profit) * risk_multiplier
            stop_loss = entry_price + (stop_loss - entry_price) * risk_multiplier
        
        return {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': entry_price * 0.98 if side == "LONG" else entry_price * 1.02,
            'max_holding_hours': random.randint(1, 6)
        }

    def should_enter_trade(self) -> bool:
        """Decyduje czy wejść w transakcję wg profilu częstotliwości"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7
        }.get(profile['trade_frequency'], 0.5)
        
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM używając rzeczywistych cen z API Bybit"""
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Could not get price for {symbol} - skipping trade")
            return None
            
        signal, confidence = self.generate_llm_signal(symbol)
        if signal == "HOLD" or confidence < 0.3:
            return None
            
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        # Sprawdź dostępny kapitał
        if self.real_trading:
            real_balance = self.get_account_balance()
            available_balance = real_balance if real_balance else self.virtual_balance
        else:
            available_balance = self.virtual_balance
            
        if margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        # SPRÓBUJ ZŁOŻYĆ ZLECENIE NA BYBIT
        order_id = None
        if self.real_trading:
            order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
            if not order_id:
                self.logger.error(f"❌ Failed to place order on Bybit for {symbol}")
                return None
        else:
            order_id = f"virtual_{int(time.time())}"
        
        position_id = order_id
        
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
        
        # Aktualizuj saldo tylko w trybie wirtualnym
        if not self.real_trading:
            self.virtual_balance -= margin_required
        
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        trading_mode = "REAL" if self.real_trading else "VIRTUAL"
        self.logger.info(f"🎯 {trading_mode} {self.active_profile} OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L wszystkich pozycji używając rzeczywistych danych z Bybit"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        # Synchronizuj z Bybit jeśli używamy real trading
        if self.real_trading:
            self.sync_with_bybit()
        else:
            # Dla trybu wirtualnego, oblicz normalnie
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
        
        # Użyj rzeczywistego salda konta
        real_balance = self.get_account_balance()
        if real_balance is not None:
            account_value = real_balance + total_unrealized
            available_cash = real_balance
        else:
            account_value = self.virtual_capital + total_unrealized
            available_cash = self.virtual_balance
        
        # Dla real trading, użyj bezpośrednio z Bybit
        if self.real_trading:
            total_unrealized = self.get_bybit_unrealized_pnl()
            self.dashboard_data['unrealized_pnl'] = total_unrealized
            if real_balance:
                account_value = real_balance + total_unrealized
            else:
                account_value = self.virtual_capital + total_unrealized
        
        self.dashboard_data['account_value'] = account_value
        self.dashboard_data['available_cash'] = available_cash
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        if self.virtual_capital > 0:
            self.stats['portfolio_utilization'] = total_margin / self.virtual_capital
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji używając rzeczywistych cen z Bybit API"""
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
            
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_time > exit_plan['max_holding_hours']:
                exit_reason = "TIME_EXPIRED"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        # Zamknij pozycję na Bybit jeśli to real trading
        if position.get('real_trading', False):
            success = self.close_bybit_position(position['symbol'], position['side'], position['quantity'])
            if not success:
                self.logger.error(f"❌ Failed to close position on Bybit: {position_id}")
                return
        
        # Aktualizuj saldo tylko w trybie wirtualnym
        if not self.real_trading:
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
            'holding_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600,
            'real_trading': position.get('real_trading', False)
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
        trading_mode = "REAL" if position.get('real_trading', False) else "VIRTUAL"
        self.logger.info(f"{pnl_color} {trading_mode} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}% margin) - Reason: {exit_reason}")

    # ... (reszta metod pozostaje bez zmian - get_portfolio_diversity, get_current_profile, set_active_profile, etc.)

    def run_llm_trading_strategy(self):
        """Główna pętla strategii LLM"""
        self.logger.info("🚀 STARTING LLM TRADING STRATEGY")
        
        # NAJPIERW PRZETESTOJ API
        self.logger.info("🔧 RUNNING API TESTS...")
        api_ok = self.debug_api_connection()
        
        if not api_ok:
            self.logger.error("❌ API TESTS FAILED - stopping bot")
            self.is_running = False
            return
        
        self.logger.info("✅ API TESTS PASSED - starting trading")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 LLM Trading Iteration #{iteration}")
                
                # 1. Aktualizuj P&L używając rzeczywistych cen
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
                                time.sleep(1)
                
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                wait_time = random.randint(30, 90)
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in LLM trading loop: {e}")
                time.sleep(30)

    # ... (reszta metod pozostaje bez zmian)

# FLASK APP - bez zmian
app = Flask(__name__)
CORS(app)

# Inicjalizacja bota
trading_bot = LLMTradingBot(initial_capital=10000, leverage=10)

# Routes - bez zmian
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

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
    print("🚀 Starting LLM Trading Bot Server...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 LLM Profiles: Claude, Gemini, GPT, Qwen")
    print("📈 Trading assets: BTC, ETH, SOL, XRP, BNB, DOGE")
    print("💹 Using REAL-TIME prices from Bybit API only")
    app.run(debug=True, host='0.0.0.0', port=5000)
