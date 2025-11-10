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
            return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting PUBLIC futures price for {symbol}: {e}")
            # Tylko cache jako fallback
            if symbol in self.price_cache:
                cache_age = (datetime.now() - self.price_cache[symbol]['timestamp']).total_seconds()
                if cache_age < 300:  # 5 minut
                    self.logger.info(f"🔄 Using cached price for {symbol} (age: {cache_age:.1f}s)")
                    return self.price_cache[symbol]['price']
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
        """Generuje sygnał w stylu LLM na podstawie rzeczywistych danych z Bybit API - DOKŁADNIE JAK W DRUGIM BOCIE"""
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
        
        # DOKŁADNIE TAKA SAMA LOGIKA OTWIERANIA POZYCJI JAK W DRUGIM BOCIE
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
        """Oblicza wielkość pozycji w stylu LLM - DOKŁADNIE JAK W DRUGIM BOCIE"""
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
        
        # Użyj rzeczywistego salda konta
        real_balance = self.get_account_balance()
        if real_balance is None:
            real_balance = self.virtual_balance
        
        position_value = (real_balance * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        # Minimalna wartość pozycji to $5
        min_position_value = 5
        if position_value < min_position_value:
            self.logger.warning(f"⚠️ Calculated position value too small: ${position_value:.2f}, using minimum: ${min_position_value}")
            position_value = min_position_value
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        self.logger.info(f"📏 POSITION SIZE: Value: ${position_value:.2f}, Qty: {quantity:.6f}, Margin: ${margin_required:.2f}, Confidence: {confidence:.1%}")
        
        return quantity, position_value, margin_required

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatuje ilość zgodnie z wymaganiami Bybit dla każdego symbolu"""
        # Wymagania lot size dla różnych symboli
        lot_size_rules = {
            'BTCUSDT': 0.001,   # 0.001 BTC
            'ETHUSDT': 0.01,    # 0.01 ETH  
            'SOLUSDT': 0.01,    # 0.01 SOL
            'XRPUSDT': 1,       # 1 XRP
            'BNBUSDT': 0.001,   # 0.001 BNB
            'DOGEUSDT': 1,      # 1 DOGE
        }
        
        lot_size = lot_size_rules.get(symbol, 0.001)
        
        # Zaokrąglij do najbliższej wielokrotności lot size
        formatted_quantity = round(quantity / lot_size) * lot_size
        
        # Formatuj do odpowiedniej liczby miejsc po przecinku
        if lot_size >= 1:
            formatted_quantity = int(formatted_quantity)
        elif lot_size == 0.001:
            formatted_quantity = round(formatted_quantity, 3)
        elif lot_size == 0.01:
            formatted_quantity = round(formatted_quantity, 2)
        else:
            formatted_quantity = round(formatted_quantity, 6)
        
        # Upewnij się, że nie jest zerowe
        if formatted_quantity <= 0:
            formatted_quantity = lot_size
        
        return str(formatted_quantity)

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
                reduceOnly=True,  # Ważne: tylko redukcja istniejącej pozycji
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

    def get_bybit_positions(self) -> List[Dict]:
        """Pobiera aktywne pozycje z Bybit używając pybit"""
        if not self.real_trading:
            self.logger.info("🔄 Virtual mode - no Bybit positions")
            return []
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return []
    
        try:
            self.logger.info("🔍 Fetching positions from Bybit...")
            
            response = self.session.get_positions(
                category="linear",
                symbol=""  # Pobierz wszystkie pozycje
            )
            
            self.logger.info(f"📨 Bybit API Response Code: {response['retCode']}")
            self.logger.info(f"📨 Bybit API Message: {response.get('retMsg', 'No message')}")
            
            if response['retCode'] == 0:
                active_positions = []
                result_list = response['result'].get('list', [])
                self.logger.info(f"📊 Found {len(result_list)} position entries in response")
                
                for i, pos in enumerate(result_list):
                    size = float(pos['size'])
                    symbol = pos['symbol']
                    side = 'LONG' if pos['side'] == 'Buy' else 'SHORT'
                    
                    self.logger.info(f"  📍 Entry {i}: {symbol} {side} - Size: {size}")
                    
                    if size > 0:  # Tylko pozycje z wielkością > 0
                        # Konwertuj timestamp na datetime
                        created_time = datetime.fromtimestamp(int(pos['createdTime']) / 1000) if pos.get('createdTime') else datetime.now()
                        
                        position_data = {
                            'symbol': symbol,
                            'side': side,
                            'size': size,
                            'entry_price': float(pos['avgPrice']),
                            'leverage': float(pos['leverage']),
                            'unrealised_pnl': float(pos['unrealisedPnl']),
                            'liq_price': float(pos['liqPrice']) if pos['liqPrice'] and pos['liqPrice'] != '' else None,
                            'position_value': float(pos['positionValue']),
                            'position_margin': float(pos['positionIM']),
                            'created_time': created_time,
                            'mark_price': float(pos['markPrice']) if pos.get('markPrice') else float(pos['avgPrice'])
                        }
                        
                        active_positions.append(position_data)
                        self.logger.info(f"  ✅ ADDED: {symbol} {side} Size: {size}, Entry: ${position_data['entry_price']}")
                    else:
                        self.logger.info(f"  ❌ SKIPPED: {symbol} {side} - Zero size: {size}")
                
                self.logger.info(f"✅ Final: {len(active_positions)} active positions on Bybit")
                return active_positions
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ Błąd pobierania pozycji z Bybit: {error_msg}")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ Error getting Bybit positions: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            return []

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu LLM - DOKŁADNIE JAK W DRUGIM BOCIE"""
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
        """Decyduje czy wejść w transakcję wg profilu częstotliwości - DOKŁADNIE JAK W DRUGIM BOCIE"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7
        }.get(profile['trade_frequency'], 0.5)
        
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM używając rzeczywistych cen z API - DOKŁADNIE JAK W DRUGIM BOCIE"""
        
        # 1. Sprawdź częstotliwość tradingu
        if not self.should_enter_trade():
            return None
            
        # 2. Pobierz cenę
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Could not get price for {symbol} - skipping trade")
            return None
            
        # 3. Wygeneruj sygnał
        signal, confidence = self.generate_llm_signal(symbol)
        
        # 4. Sprawdź warunki wejścia - DOKŁADNIE JAK W DRUGIM BOCIE
        if signal == "HOLD" or confidence < 0.3:
            return None
            
        # 5. Sprawdź liczbę aktywnych pozycji
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
        
        # 6. Oblicz wielkość pozycji
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        # 7. Sprawdź dostępność kapitału
        real_balance = self.get_account_balance()
        if real_balance is None:
            real_balance = self.virtual_balance
            
        if margin_required > real_balance * 0.8:  # Zostaw 20% bufor
            self.logger.warning(f"💰 Insufficient margin for {symbol}. Required: ${margin_required:.2f}, Available: ${real_balance:.2f}")
            return None
        
        # 8. Sprawdź minimalną wartość zlecenia
        min_order_value = 5
        order_value = quantity * current_price
        if order_value < min_order_value:
            self.logger.warning(f"❌ Order value too small for {symbol}. Required: ${min_order_value}, Actual: ${order_value:.2f}")
            return None
            
        # 9. SPRÓBUJ ZŁOŻYĆ ZLECENIE
        self.logger.info(f"🚀 ATTEMPTING ORDER: {symbol} {signal} Qty: {quantity:.6f}")
        order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
        
        if order_id:
            self.logger.info(f"🎉 SUCCESS! Order placed: {order_id}")
            
            # Oblicz plan wyjścia
            exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
            
            if signal == "LONG":
                liquidation_price = current_price * (1 - 0.9 / self.leverage)
            else:
                liquidation_price = current_price * (1 + 0.9 / self.leverage)
            
            # Zapisz pozycję
            position_id = order_id
            self.positions[position_id] = {
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
                'real_trading': self.real_trading
            }
            
            # Aktualizuj saldo wirtualne
            if not self.real_trading:
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
        else:
            self.logger.error(f"❌ FAILED to place order")
            return None

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

    def get_bybit_unrealized_pnl(self) -> float:
        """Pobiera unrealized P&L bezpośrednio z Bybit"""
        if not self.real_trading:
            # Dla trybu wirtualnego, oblicz normalnie
            total_unrealized = 0
            for position in self.positions.values():
                if position['status'] == 'ACTIVE':
                    current_price = self.get_current_price(position['symbol'])
                    if current_price:
                        if position['side'] == 'LONG':
                            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                            unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                        else:
                            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                            unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                        total_unrealized += unrealized_pnl
            return total_unrealized
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return 0.0
    
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=""
            )
            
            if response['retCode'] == 0:
                total_unrealized = 0.0
                for pos in response['result']['list']:
                    unrealised_pnl = float(pos.get('unrealisedPnl', 0))
                    total_unrealized += unrealised_pnl
                
                self.logger.info(f"📊 Real Unrealized P&L from Bybit: ${total_unrealized:.2f}")
                return total_unrealized
            else:
                self.logger.error(f"❌ Błąd pobierania unrealized P&L z Bybit: {response.get('retMsg', 'Unknown')}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Error getting unrealized P&L from Bybit: {e}")
            return 0.0

    def sync_with_bybit(self):
        """Synchronizuje stan z rzeczywistymi pozycjami na Bybit"""
        if not self.real_trading:
            return
            
        try:
            # Pobierz aktywne pozycje z Bybit
            bybit_positions = self.get_bybit_positions()
            
            # Aktualizuj saldo konta
            real_balance = self.get_account_balance()
            if real_balance:
                self.virtual_balance = real_balance
                self.virtual_capital = real_balance
            
            # Pobierz unrealized P&L bezpośrednio z Bybit
            real_unrealized_pnl = self.get_bybit_unrealized_pnl()
            self.dashboard_data['unrealized_pnl'] = real_unrealized_pnl
            
            # Synchronizuj lokalne pozycje z Bybit
            self.sync_local_positions_with_bybit(bybit_positions)
            
            # Log synchronizacji
            self.logger.info(f"🔄 Zsynchronizowano z Bybit - Pozycje: {len(bybit_positions)}, Saldo: ${real_balance:.2f}, Unrealized P&L: ${real_unrealized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing with Bybit: {e}")

    def sync_local_positions_with_bybit(self, bybit_positions: List[Dict]):
        """Synchronizuje lokalne pozycje z pozycjami z Bybit"""
        # Znajdź pozycje, które są na Bybit ale nie ma ich lokalnie
        bybit_symbols = {pos['symbol'] for pos in bybit_positions}
        local_active_symbols = {pos['symbol'] for pos in self.positions.values() if pos['status'] == 'ACTIVE'}
        
        # Dodaj brakujące pozycje do lokalnego śledzenia
        for bybit_pos in bybit_positions:
            if bybit_pos['symbol'] not in local_active_symbols:
                position_id = f"bybit_sync_{bybit_pos['symbol']}_{int(time.time())}"
                self.positions[position_id] = {
                    'symbol': bybit_pos['symbol'],
                    'side': bybit_pos['side'],
                    'entry_price': bybit_pos['entry_price'],
                    'quantity': bybit_pos['size'],
                    'leverage': bybit_pos['leverage'],
                    'entry_time': bybit_pos['created_time'],
                    'status': 'ACTIVE',
                    'order_id': f"bybit_sync_{bybit_pos['symbol']}",
                    'real_trading': True,
                    'llm_profile': self.active_profile,
                    'confidence': 0.5,  # Domyślne confidence dla zsynchronizowanych pozycji
                    'margin': bybit_pos['position_margin'],
                    'exit_plan': self.calculate_llm_exit_plan(bybit_pos['entry_price'], 0.5, bybit_pos['side']),
                    'liquidation_price': bybit_pos['liq_price'],
                    'unrealized_pnl': bybit_pos['unrealised_pnl'],
                    'current_price': self.get_current_price(bybit_pos['symbol'])
                }
                self.logger.info(f"🔄 Dodano zsynchronizowaną pozycję z Bybit: {bybit_pos['symbol']} {bybit_pos['side']}")
        
        # Oznacz pozycje jako zamknięte jeśli nie ma ich na Bybit
        for position_id, position in list(self.positions.items()):
            if (position['status'] == 'ACTIVE' and position.get('real_trading', False) and 
                position['symbol'] not in bybit_symbols):
                position['status'] = 'CLOSED'
                self.logger.info(f"🔄 Oznaczono pozycję jako zamkniętą: {position['symbol']}")

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

    def get_portfolio_diversity(self) -> float:
        """Oblicza dywersyfikację portfela"""
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

    def debug_api_connection(self):
        """Testuje połączenie z API i próbuje złożyć testowe zlecenie używając pybit"""
        
        self.logger.info("🔧 DEBUG API CONNECTION")
        
        # 1. Test pobierania cen
        symbol = "BTCUSDT"
        price = self.get_current_price(symbol)
        if price:
            self.logger.info(f"✅ Price OK: {symbol} = ${price}")
        else:
            self.logger.error(f"❌ Price FAILED")
            return False
    
        # 2. Test pobierania salda
        balance = self.get_account_balance()
        if balance:
            self.logger.info(f"✅ Balance OK: ${balance:.2f}")
        else:
            self.logger.error(f"❌ Balance FAILED")
            return False
    
        # 3. Test ustawienia dźwigni
        leverage_ok = self.set_leverage(symbol, self.leverage)
        if leverage_ok:
            self.logger.info(f"✅ Leverage OK: {self.leverage}x")
        else:
            self.logger.warning(f"⚠️ Leverage may have failed")
    
        # 4. Test złożenia MAŁEGO zlecenia
        self.logger.info("🚀 TESTING ORDER PLACEMENT...")
        test_quantity = 0.001  # Bardzo małe
        order_id = self.place_bybit_order(symbol, "LONG", test_quantity, price)
        
        if order_id:
            self.logger.info(f"🎉 ORDER TEST SUCCESS! ID: {order_id}")
            return True
        else:
            self.logger.error("❌ ORDER TEST FAILED")
            return False

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

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu używając rzeczywistych danych z Bybit"""
        self.logger.info("🔄 Generating dashboard data...")
        
        # Sprawdź status API i pobierz saldo
        api_status = self.check_api_status()
        
        # SYNCHRONIZUJ POZYCJE Z BYBIT PRZED POBRANIEM DANYCH
        if self.real_trading:
            self.sync_all_positions_with_bybit()
        
        # Pobierz unrealized P&L bezpośrednio z Bybit dla real trading
        total_unrealized_pnl = self.get_bybit_unrealized_pnl()
        self.dashboard_data['unrealized_pnl'] = total_unrealized_pnl
        
        # Przygotuj aktywne pozycje z lokalnego stanu (który został zsynchronizowany)
        active_positions = []
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price') or self.get_current_price(position['symbol'])
                if not current_price:
                    continue
                
                # Użyj rzeczywistego unrealized P&L z Bybit jeśli dostępny
                unrealized_pnl = position.get('unrealized_pnl', 0)
                
                # Oblicz odległości do TP/SL
                exit_plan = position.get('exit_plan', self.calculate_llm_exit_plan(position['entry_price'], position.get('confidence', 0.5), position['side']))
                
                if position['side'] == 'LONG':
                    tp_distance_pct = ((exit_plan['take_profit'] - current_price) / current_price) * 100
                    sl_distance_pct = ((current_price - exit_plan['stop_loss']) / current_price) * 100
                else:
                    tp_distance_pct = ((current_price - exit_plan['take_profit']) / current_price) * 100
                    sl_distance_pct = ((exit_plan['stop_loss'] - current_price) / current_price) * 100
                
                active_positions.append({
                    'position_id': position_id,
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'margin': position.get('margin', 0),
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position.get('confidence', 0.5) * 100,  # Konwertuj na procenty
                    'llm_profile': position.get('llm_profile', self.active_profile),
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'exit_plan': exit_plan,
                    'tp_distance_pct': round(tp_distance_pct, 2),
                    'sl_distance_pct': round(sl_distance_pct, 2),
                    'real_trading': position.get('real_trading', False)
                })
        
        self.logger.info(f"📊 ACTIVE POSITIONS: {len(active_positions)} positions to display")
        
        # Oblicz confidence levels dla każdego assetu
        confidence_levels = {}
        for symbol in self.assets:
            try:
                signal, confidence = self.generate_llm_signal(symbol)
                confidence_levels[symbol] = round(confidence * 100, 1)
            except:
                confidence_levels[symbol] = 0
        
        # Ostatnie transakcje
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
                'llm_profile': trade['llm_profile'],
                'confidence': trade['confidence'],
                'holding_hours': round(trade['holding_hours'], 2),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'real_trading': trade.get('real_trading', False)
            })
        
        # Metryki wydajności
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        # Użyj rzeczywistego salda konta dla obliczeń
        current_balance = api_status['balance'] if api_status['balance_available'] else self.virtual_balance
        
        if current_balance:
            total_return_pct = ((current_balance + total_unrealized_pnl - self.initial_capital) / self.initial_capital) * 100
        else:
            total_return_pct = 0
            
        self.logger.info(f"📊 Dashboard data - Positions: {len(active_positions)}, Trades: {len(recent_trades)}, Unrealized P&L: ${total_unrealized_pnl:.2f}")    
        
        return {
            'account_summary': {
                'total_value': round(current_balance + total_unrealized_pnl, 2) if current_balance else 0,
                'available_cash': round(current_balance, 2) if current_balance else 0,
                'total_fees': round(self.stats.get('total_fees', 0), 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2),
                'unrealized_pnl': round(total_unrealized_pnl, 2),
                'real_trading': self.real_trading,
                'real_balance_available': api_status['balance_available']
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
                'real_trading': self.real_trading,
                'real_balance_available': api_status['balance_available'],
                'api_connected': api_status['api_connected']
            },
            'api_status': api_status,
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': total_unrealized_pnl,
            'last_update': datetime.now().isoformat()
        }

    def check_api_status(self) -> Dict:
        """Sprawdza status połączenia z Bybit API"""
        status = {
            'real_trading': self.real_trading,
            'api_connected': False,
            'balance_available': False,
            'testnet': self.testnet,
            'message': '',
            'balance': 0
        }
        
        if not self.real_trading:
            status['message'] = '🔄 Tryb wirtualny - brak kluczy API'
            status['balance'] = self.virtual_balance
            return status
        
        try:
            # Spróbuj pobrać saldo
            balance = self.get_account_balance()
            
            if balance is not None:
                status['api_connected'] = True
                status['balance_available'] = True
                status['balance'] = balance
            else:
                status['api_connected'] = False
                status['message'] = '❌ Błąd połączenia z Bybit - sprawdź klucze API'
                    
        except Exception as e:
            status['api_connected'] = False
            status['message'] = f'❌ Błąd API: {str(e)}'
        
        return status

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
                
                # 3. Sprawdź możliwości wejścia - DOKŁADNIE JAK W DRUGIM BOCIE
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

# Routes do renderowania stron
@app.route('/')
def index():
    """Strona główna - renderuje template index.html"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard - również renderuje index.html (lub inny template jeśli masz)"""
    return render_template('index.html')

# API endpoints
@app.route('/api/trading-data')
def get_trading_data():
    """Zwraca dane tradingowe dla dashboardu"""
    try:
        data = trading_bot.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot-status')
def get_bot_status():
    """Zwraca status bota"""
    status = 'running' if trading_bot.is_running else 'stopped'
    return jsonify({'status': status})

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    """Uruchamia bota"""
    try:
        trading_bot.start_trading()
        return jsonify({'status': 'Bot started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    """Zatrzymuje bota"""
    try:
        trading_bot.stop_trading()
        return jsonify({'status': 'Bot stopped successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/change-profile', methods=['POST'])
def change_profile():
    """Zmienia profil LLM"""
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
    """Wymusza aktualizację danych"""
    try:
        trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    """Zapisuje dane wykresu"""
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
    """Ładuje dane wykresu"""
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
    print("💹 Using REAL Bybit Futures API with REAL MONEY")
    app.run(debug=True, host='0.0.0.0', port=5000)
