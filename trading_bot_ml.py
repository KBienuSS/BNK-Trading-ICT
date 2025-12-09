# trading_bot_ml_binance_bybit_real_balance.py
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
            
        # Kapitał wirtualny - IDENTYCZNIE JAK W PIERWSZYM BOCIE
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
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM - IDENTYCZNIE JAK W PIERWSZYM BOCIE
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
        
        # DASHBOARD - IDENTYCZNIE JAK W PIERWSZYM BOCIE
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
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - Binance Data + Bybit Execution")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📈 Trading assets: {', '.join(self.assets)}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")
        self.logger.info("📊 Using Binance API for price data & analysis")
        self.logger.info("⚡ Using Bybit API for order execution")

    def open_real_position(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """Otwiera pozycję z ręcznie określoną ilością"""
        try:
            self.logger.info(f"🎯 MANUAL OPEN: {symbol} {side} Qty: {quantity}")
            
            # Pobierz aktualną cenę
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.logger.warning(f"❌ Could not get price for {symbol}")
                return None
            
            # Sprawdź czy symbol jest wspierany
            if symbol not in self.assets:
                self.logger.warning(f"⚠️ Symbol {symbol} not in assets, adding it")
                self.assets.append(symbol)
            
            # Sprawdź dostępny balans
            available_balance = self.get_account_balance()
            if available_balance is None:
                self.logger.warning("⚠️ Could not get real balance, using virtual")
                available_balance = self.virtual_balance
            
            # Oblicz wartość i margin
            position_value = quantity * current_price
            margin_required = position_value / self.leverage
            
            # Sprawdź czy wystarczy środków
            if margin_required > available_balance:
                self.logger.warning(f"💰 Insufficient balance: Available ${available_balance:.2f}, Required ${margin_required:.2f}")
                return None
            
            # Liquidation price
            if side == "LONG":
                liquidation_price = current_price * (1 - 0.9 / self.leverage)
            else:
                liquidation_price = current_price * (1 + 0.9 / self.leverage)
            
            # Plan wyjścia
            exit_plan = self.calculate_llm_exit_plan(current_price, 0.8, side)
            
            # Składanie zlecenia na Bybit
            order_id = None
            if self.real_trading and hasattr(self, 'place_bybit_order'):
                try:
                    order_id = self.place_bybit_order(symbol, side, quantity, current_price)
                    self.logger.info(f"📝 Bybit order placed: {order_id}")
                except Exception as e:
                    self.logger.error(f"❌ Bybit order failed: {e}")
                    if not self.real_trading:  # Jeśli to tryb wirtualny, kontynuuj
                        order_id = f"virtual_{int(time.time())}"
                    else:
                        return None
            else:
                order_id = f"virtual_{int(time.time())}"
            
            # Utwórz pozycję
            position_id = f"manual_{self.position_id}"
            self.position_id += 1
            
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'quantity': quantity,
                'leverage': self.leverage,
                'margin': margin_required,
                'liquidation_price': liquidation_price,
                'entry_time': datetime.now(),
                'status': 'ACTIVE',
                'unrealized_pnl': 0,
                'confidence': 0.8,
                'llm_profile': f"MANUAL ({self.active_profile})",
                'exit_plan': exit_plan,
                'order_id': order_id,
                'real_trading': self.real_trading,
                'manual': True
            }
            
            self.positions[position_id] = position
            
            # Aktualizuj wirtualny balans jeśli nie jest real trading
            if not self.real_trading:
                self.virtual_balance -= margin_required
            
            # Logowanie
            tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
            sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
            
            trading_mode = "REAL" if self.real_trading else "VIRTUAL"
            
            self.logger.info(f"✅ MANUAL OPEN SUCCESS: {position_id}")
            self.logger.info(f"   {symbol} {side} @ ${current_price:.4f}")
            self.logger.info(f"   Qty: {quantity}, Value: ${position_value:.2f}")
            self.logger.info(f"   Margin: ${margin_required:.2f}")
            self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
            self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"💥 Error in open_real_position: {e}")
            import traceback
            self.logger.error(f"💥 Stack trace: {traceback.format_exc()}")
            return None
            
    def get_binance_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę z API Binance - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        try:
            url = f"{self.binance_base_url}/ticker/price"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data['price'])
            
            # Zapisz w cache
            self.price_cache[symbol] = {
                'price': price,
                'timestamp': datetime.now()
            }
            
            # Zapisz w historii dla analizy
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': price,
                'timestamp': datetime.now()
            })
            
            # Ogranicz historię do ostatnich 50 punktów
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]
            
            return price
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ API Error getting price for {symbol}: {e}")
            if symbol in self.price_cache:
                cache_age = (datetime.now() - self.price_cache[symbol]['timestamp']).total_seconds()
                if cache_age < 300:  # 5 minut
                    self.logger.info(f"🔄 Using cached price for {symbol} (age: {cache_age:.1f}s)")
                    return self.price_cache[symbol]['price']
            
            self.logger.warning(f"⚠️ Could not get price for {symbol} and no recent cache")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error getting price for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę - WYŁĄCZNIE Z API BINANCE"""
        return self.get_binance_price(symbol)

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na podstawie rzeczywistych danych z API Binance - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return random.uniform(-0.02, 0.02)
            
            history = self.price_history[symbol]
            current_price = history[-1]['price']
            
            lookback = min(5, len(history) - 1)
            past_price = history[-lookback]['price']
            
            momentum = (current_price - past_price) / past_price
            momentum = max(min(momentum, 0.03), -0.03)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return random.uniform(-0.02, 0.02)

    def check_volume_activity(self, symbol: str) -> bool:
        """Sprawdza aktywność wolumenu na podstawie zmienności cen z API Binance - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return random.random() < 0.6
            
            prices = [entry['price'] for entry in self.price_history[symbol][-10:]]
            volatility = np.std(prices) / np.mean(prices)
            
            return volatility > 0.002
            
        except Exception as e:
            self.logger.error(f"❌ Error checking volume activity for {symbol}: {e}")
            return random.random() < 0.6

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu LLM na podstawie rzeczywistych danych z API Binance - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        profile = self.get_current_profile()
        
        momentum = self.analyze_simple_momentum(symbol)
        volume_active = self.check_volume_activity(symbol)
        
        base_confidence = profile['confidence_bias']
        
        confidence_modifiers = 0
        
        if momentum > 0.008:
            confidence_modifiers += 0.2
        elif momentum > 0.003:
            confidence_modifiers += 0.1
        elif momentum < -0.008:
            confidence_modifiers += 0.15
        elif momentum < -0.003:
            confidence_modifiers += 0.08
            
        if volume_active:
            confidence_modifiers += 0.1
            
        final_confidence = min(base_confidence + confidence_modifiers + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.1)
        
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

    def get_account_balance(self) -> Optional[float]:
        """Pobiera rzeczywiste saldo konta z Bybit używając pybit"""
        if not self.real_trading:
            return None
            
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

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu LLM - TERAZ Z RZECZYWISTYM SALDEM Z BYBIT i 10x WIĘKSZE"""
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
        
        # TERAZ: Użyj rzeczywistego salda z Bybit do obliczeń
        real_balance = self.get_account_balance()
        if real_balance is None:
            self.logger.warning("⚠️ Could not get real balance from Bybit, using initial capital")
            real_balance = self.virtual_capital  # fallback do wirtualnego kapitału
        
        # ZWIĘKSZENIE 10x - mnożymy przez 10
        position_value = (real_balance * base_allocation * 
                         confidence_multiplier * sizing_multiplier * 10)  # 10x większe
        
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu LLM - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
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
            'max_holding_hours': random.randint(1, 6)  # IDENTYCZNIE JAK W PIERWSZYM BOCIE
        }

    def should_enter_trade(self) -> bool:
        """Decyduje czy wejść w transakcję wg profilu częstotliwości - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7
        }.get(profile['trade_frequency'], 0.5)
        
        return random.random() < frequency_chance

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Ustawia dźwignię dla symbolu używając Bybit API"""
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

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatuje ilość zgodnie z wymaganiami Bybit dla każdego symbolu"""
        lot_size_rules = {
            'BTCUSDT': 0.001,
            'ETHUSDT': 0.01,  
            'SOLUSDT': 0.01,
            'XRPUSDT': 1,
            'BNBUSDT': 0.001,
            'DOGEUSDT': 1,
        }
        
        lot_size = lot_size_rules.get(symbol, 0.001)
        formatted_quantity = round(quantity / lot_size) * lot_size
        
        if lot_size >= 1:
            formatted_quantity = int(formatted_quantity)
        elif lot_size == 0.001:
            formatted_quantity = round(formatted_quantity, 3)
        elif lot_size == 0.01:
            formatted_quantity = round(formatted_quantity, 2)
        else:
            formatted_quantity = round(formatted_quantity, 6)
        
        if formatted_quantity <= 0:
            formatted_quantity = lot_size
        
        return str(formatted_quantity)

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
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ ORDER FAILED: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"💥 CRITICAL ERROR in place_bybit_order: {e}")
            import traceback
            self.logger.error(f"💥 Stack trace: {traceback.format_exc()}")
            return None

    def close_bybit_position(self, symbol: str, side: str, quantity: float) -> bool:
        """Zamyka pozycję na Bybit używając pybit"""
        if not self.real_trading:
            self.logger.info(f"🔄 Tryb wirtualny - symulacja zamknięcia pozycji {symbol}")
            return True
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return False

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
                self.logger.info(f"✅ Pozycja zamknięta na Bybit: {symbol} - ID: {response['result']['orderId']}")
                return True
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ Błąd zamykania pozycji na Bybit dla {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error closing Bybit position: {e}")
            return False

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM - TERAZ Z RZECZYWISTYM SALDEM Z BYBIT"""
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Could not get price for {symbol} from Binance - skipping trade")
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
        
        # Sprawdź dostępny balans - TERAZ Z RZECZYWISTYM SALDEM Z BYBIT
        available_balance = self.get_account_balance()
        if available_balance is None:
            self.logger.warning("⚠️ Could not get available balance - skipping trade")
            return None
            
        if margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}. Available: ${available_balance:.2f}, Required: ${margin_required:.2f}")
            return None
            
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        # Składanie zlecenia na Bybit
        order_id = None
        if self.real_trading:
            order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
            if not order_id:
                return None
        
        # IDENTYCZNIE JAK W PIERWSZYM BOCIE: Zapisywanie pozycji
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
        
        # NIE ODEJMUJEMY z wirtualnego balansu - używamy rzeczywistego salda z Bybit
        # self.virtual_balance -= margin_required
        
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        trading_mode = "REAL" if self.real_trading else "VIRTUAL"
        available_balance_after = available_balance - margin_required
        self.logger.info(f"🎯 {trading_mode} OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   💰 Balance: ${available_balance:.2f} -> ${available_balance_after:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L wszystkich pozycji używając rzeczywistych cen z API - POPRAWIONE LICZENIE"""
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
                # POPRAWIONE: NIE mnożymy przez dźwignię - to jest realny P&L
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                # POPRAWIONE: NIE mnożymy przez dźwignię - to jest realny P&L
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
            total_confidence += position['confidence']
            confidence_count += 1
        
        # Zaktualizuj dane dashboardu używając rzeczywistego salda
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        
        # TERAZ: Użyj rzeczywistego salda z Bybit do aktualizacji wartości konta
        real_balance = self.get_account_balance()
        if real_balance is not None:
            self.dashboard_data['account_value'] = real_balance + total_unrealized
            self.dashboard_data['available_cash'] = real_balance
            # Aktualizuj też wirtualny kapitał dla spójności
            self.virtual_capital = real_balance + total_unrealized
            self.virtual_balance = real_balance
        else:
            # Fallback do wirtualnego kapitału
            self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
            self.dashboard_data['available_cash'] = self.virtual_balance
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        # Portfolio utilization na podstawie rzeczywistego salda
        if real_balance and real_balance > 0:
            self.stats['portfolio_utilization'] = total_margin / real_balance
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji używając rzeczywistych cen z API - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
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
        """Zamyka pozycję - POPRAWIONE LICZENIE P&L"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # POPRAWIONE: NIE mnożymy przez dźwignię - to jest realny P&L
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        # Zamknięcie na Bybit
        if position.get('real_trading', False):
            success = self.close_bybit_position(position['symbol'], position['side'], position['quantity'])
            if not success:
                self.logger.error(f"❌ Failed to close position on Bybit: {position_id}")
        
        # NIE AKTUALIZUJEMY WIRTUALNEGO BALANSU - zostanie zaktualizowany przy następnym get_account_balance()
        
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
        
        # POPRAWIONE: Pokazujemy rzeczywisty % zysku/straty bez dźwigni
        margin_return = pnl_pct * 100  # Realny procent
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        trading_mode = "REAL" if position.get('real_trading', False) else "VIRTUAL"
        
        self.logger.info(f"{pnl_color} {trading_mode} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}%) - Reason: {exit_reason}")

    def get_portfolio_diversity(self) -> float:
        """Oblicza dywersyfikację portfela - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
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
        """Zmienia aktywny profil zachowania"""
        if profile_name in self.llm_profiles:
            self.active_profile = profile_name
            self.dashboard_data['active_profile'] = profile_name
            self.logger.info(f"🔄 Changed LLM profile to: {profile_name}")
            return True
        return False

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu używając rzeczywistych cen z API"""
        active_positions = []
        total_unrealized_pnl = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                if not current_price:
                    continue
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    # POPRAWIONE: NIE mnożymy przez dźwignię
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    # POPRAWIONE: NIE mnożymy przez dźwignię
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price']
                
                # Oblicz odległości do TP/SL
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
                    'sl_distance_pct': sl_distance_pct,
                    'real_trading': position.get('real_trading', False)
                })
                
                total_unrealized_pnl += unrealized_pnl
        
        confidence_levels = {}
        for symbol in self.assets:
            try:
                signal, confidence = self.generate_llm_signal(symbol)
                confidence_levels[symbol] = round(confidence * 100, 1)
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
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'real_trading': trade.get('real_trading', False)
            })
        
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        total_return_pct = ((self.dashboard_data['account_value'] - 10000) / 10000) * 100
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2),
                'unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
                'real_trading': self.real_trading
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
                'leverage': self.leverage,
                'real_trading': self.real_trading
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': total_unrealized_pnl,
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def save_chart_data(self, chart_data: Dict):
        """Zapisuje dane wykresu - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        try:
            self.chart_data = chart_data
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving chart data: {e}")
            return False

    def load_chart_data(self) -> Dict:
        """Ładuje dane wykresu - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        return self.chart_data

    def run_llm_trading_strategy(self):
        """Główna pętla strategii LLM używająca rzeczywistych cen z API - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        self.logger.info("🚀 STARTING LLM-STYLE TRADING STRATEGY")
        self.logger.info(f"🎯 Active Profile: {self.active_profile}")
        self.logger.info("📊 Data Source: Binance API")
        self.logger.info("⚡ Execution: Bybit API (if real trading enabled)")
        
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
                
                # 3. Sprawdź możliwości wejścia - IDENTYCZNIE JAK W PIERWSZYM BOCIE
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

    def start_trading(self):
        """Rozpoczyna trading - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("🚀 LLM Trading Bot started")

    def stop_trading(self):
        """Zatrzymuje trading - IDENTYCZNIE JAK W PIERWSZYM BOCIE"""
        self.is_running = False
        self.logger.info("🛑 LLM Trading Bot stopped")


# FLASK APP
app = Flask(__name__)
CORS(app)

# Inicjalizacja bota
trading_bot = LLMTradingBot(initial_capital=10000, leverage=10)

# Routes do renderowania stron
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

@app.route('/api/open-position', methods=['POST'])
def open_position():
    """Ręczne otwieranie pozycji"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        side = data.get('side')
        quantity = float(data.get('quantity', 0))
        
        if not symbol or not side or quantity <= 0:
            return jsonify({'error': 'Symbol, side and valid quantity are required'}), 400
        
        position_id = trading_bot.open_real_position(symbol, side, quantity)
        
        if position_id:
            return jsonify({'status': 'Position opened successfully', 'position_id': position_id})
        else:
            return jsonify({'error': 'Failed to open position'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting LLM Trading Bot Server...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 LLM Profiles: Claude, Gemini, GPT, Qwen")
    print("📈 Trading assets: BTC, ETH, SOL, XRP, BNB, DOGE")
    print("📊 Using BINANCE API for price data & analysis")
    print("⚡ Using BYBIT API for order execution")
    print("💰 Position size: 10x larger than before")
    print("📈 P&L calculation: Fixed (no leverage multiplier)")
    print("🔗 Real Trading: Enabled (with Bybit API)" if trading_bot.real_trading else "🔗 Real Trading: Disabled (Virtual Mode)")
    app.run(debug=True, host='0.0.0.0', port=5000)
