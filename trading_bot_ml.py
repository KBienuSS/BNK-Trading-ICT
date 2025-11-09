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
        # Konfiguracja Bybit API
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.base_url = "https://api.bybit.com"
        self.testnet = False  # Ustaw na True dla testnet
        
        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        
        # Sprawdź czy klucze API są dostępne
        if not self.api_key or not self.api_secret:
            logging.warning("⚠️ Brak kluczy API Bybit - bot będzie działał w trybie wirtualnym")
            self.real_trading = False
        else:
            self.real_trading = True
            logging.info("🔑 Klucze API Bybit załadowane - REAL TRADING ENABLED")
        
        # Kapitał wirtualny (fallback)
        self.initial_capital = initial_capital
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
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
        self.active_profile = 'Claude'
        
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
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - Alpha Arena Inspired")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📈 Trading assets: {', '.join(self.assets)}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")

    def generate_bybit_signature(self, params: Dict, timestamp: str) -> str:
        """Generuje signature dla Bybit API"""
        param_str = timestamp + self.api_key + "5000"
        if params:
            sorted_params = sorted(params.items())
            param_str += "&".join([f"{k}={v}" for k, v in sorted_params])
        
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def bybit_request(self, method: str, endpoint: str, params: Dict = None, private: bool = False) -> Optional[Dict]:
        """Wykonuje request do Bybit API"""
        if not self.real_trading and private:
            self.logger.warning("⚠️ Tryb wirtualny - pomijam request do Bybit")
            return None
            
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if private:
            timestamp = str(int(time.time() * 1000))
            signature = self.generate_bybit_signature(params, timestamp)
            headers = {
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': '5000',
                'Content-Type': 'application/json'
            }
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                self.logger.error(f"❌ Nieobsługiwana metoda HTTP: {method}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('retCode') != 0:
                self.logger.error(f"❌ Bybit API Error: {data.get('retMsg', 'Unknown error')} (Code: {data.get('retCode')})")
                return None
                
            return data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Bybit API Request Error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in Bybit request: {e}")
            return None

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
            # Spróbuj pobrać saldo - to sprawdzi czy API działa
            balance = self.get_account_balance()
            
            if balance is not None:
                status['api_connected'] = True
                status['balance_available'] = True
                status['balance'] = balance
                status['message'] = f'✅ Połączono z Bybit - Saldo: ${balance:.2f}'
            else:
                status['api_connected'] = False
                status['message'] = '❌ Błąd połączenia z Bybit - sprawdź klucze API'
                
        except Exception as e:
            status['api_connected'] = False
            status['message'] = f'❌ Błąd API: {str(e)}'
        
        return status

    def get_bybit_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę z Bybit API"""
        try:
            endpoint = "/v5/market/tickers"
            params = {'category': 'linear', 'symbol': symbol}
            
            data = self.bybit_request('GET', endpoint, params)
            if data and 'list' in data and len(data['list']) > 0:
                price = float(data['list'][0]['lastPrice'])
                
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
            else:
                self.logger.warning(f"⚠️ Brak danych cenowych dla {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting Bybit price for {symbol}: {e}")
            return None

    def get_account_balance(self) -> Optional[float]:
        """Pobiera rzeczywiste saldo konta z Bybit"""
        if not self.real_trading:
            # Tryb wirtualny - zwróć saldo wirtualne
            return self.virtual_balance
            
        try:
            endpoint = "/v5/account/wallet-balance"
            params = {'accountType': 'UNIFIED'}
            
            data = self.bybit_request('GET', endpoint, params, private=True)
            if data and 'list' in data and len(data['list']) > 0:
                # Pobierz całkowite saldo
                total_equity = float(data['list'][0]['totalEquity'])
                
                self.logger.info(f"💰 Rzeczywiste saldo konta z Bybit: ${total_equity:.2f}")
                return total_equity
            else:
                self.logger.warning("⚠️ Nie udało się pobrać salda konta z Bybit")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance from Bybit: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę - WYŁĄCZNIE Z BYBIT API"""
        return self.get_bybit_price(symbol)

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
        
        # Pobierz rzeczywiste saldo konta
        api_status = self.check_api_status()
        if api_status['balance_available']:
            real_balance = api_status['balance']
        else:
            real_balance = self.virtual_balance
        
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
        
        position_value = (real_balance * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def place_bybit_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[str]:
        """Składa rzeczywiste zlecenie na Bybit"""
        if not self.real_trading:
            self.logger.info(f"🔄 Tryb wirtualny - symulacja zlecenia {side} dla {symbol}")
            return f"virtual_order_{int(time.time())}"
            
        try:
            endpoint = "/v5/order/create"
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': 'Buy' if side == 'LONG' else 'Sell',
                'orderType': 'Market',  # Zlecenie rynkowe dla prostoty
                'qty': str(round(quantity, 4)),  # Zaokrąglenie do 4 miejsc
                'price': str(price),
                'timeInForce': 'GTC',
                'leverage': str(self.leverage),
                'orderFilter': 'Order'
            }
            
            data = self.bybit_request('POST', endpoint, params, private=True)
            if data and 'orderId' in data:
                self.logger.info(f"✅ Zlecenie złożone na Bybit: {symbol} {side} - ID: {data['orderId']}")
                return data['orderId']
            else:
                self.logger.error(f"❌ Błąd składania zlecenia na Bybit dla {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error placing Bybit order: {e}")
            return None

    def close_bybit_position(self, symbol: str, side: str, quantity: float) -> bool:
        """Zamyka pozycję na Bybit"""
        if not self.real_trading:
            self.logger.info(f"🔄 Tryb wirtualny - symulacja zamknięcia pozycji {symbol}")
            return True
            
        try:
            endpoint = "/v5/order/create"
            
            # Dla zamknięcia pozycji używamy przeciwnego side
            close_side = 'Sell' if side == 'LONG' else 'Buy'
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': close_side,
                'orderType': 'Market',  # Market order dla szybkiego zamknięcia
                'qty': str(round(quantity, 4)),
                'reduceOnly': True,  # Tylko redukcja pozycji
                'orderFilter': 'Order'
            }
            
            data = self.bybit_request('POST', endpoint, params, private=True)
            if data and 'orderId' in data:
                self.logger.info(f"✅ Pozycja zamknięta na Bybit: {symbol} - ID: {data['orderId']}")
                return True
            else:
                self.logger.error(f"❌ Błąd zamykania pozycji na Bybit dla {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error closing Bybit position: {e}")
            return False

    def get_bybit_positions(self) -> List[Dict]:
        """Pobiera aktywne pozycje z Bybit"""
        if not self.real_trading:
            return []
            
        try:
            endpoint = "/v5/position/list"
            params = {'category': 'linear'}
            
            data = self.bybit_request('GET', endpoint, params, private=True)
            if data and 'list' in data:
                active_positions = []
                for pos in data['list']:
                    if float(pos['size']) > 0:  # Tylko pozycje z wielkością > 0
                        active_positions.append({
                            'symbol': pos['symbol'],
                            'side': 'LONG' if pos['side'] == 'Buy' else 'SHORT',
                            'size': float(pos['size']),
                            'entry_price': float(pos['avgPrice']),
                            'leverage': float(pos['leverage']),
                            'unrealised_pnl': float(pos['unrealisedPnl']),
                            'liq_price': float(pos['liqPrice']) if pos['liqPrice'] else None
                        })
                return active_positions
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Error getting Bybit positions: {e}")
            return []

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
            
            # Log synchronizacji
            self.logger.info(f"🔄 Zsynchronizowano z Bybit - Pozycje: {len(bybit_positions)}, Saldo: ${real_balance:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing with Bybit: {e}")

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

    def check_minimum_balance(self, symbol: str, price: float) -> bool:
        """Sprawdza czy saldo jest wystarczające dla danego assetu"""
        # Minimalna wartość pozycji (przykładowo $10)
        min_position_value = 5
        
        # Sprawdź czy można otworzyć minimalną pozycję
        min_margin = min_position_value / self.leverage
        
        api_status = self.check_api_status()
        available_balance = api_status['balance'] if api_status['balance_available'] else self.virtual_balance
        
        if available_balance < min_margin:
            self.logger.warning(f"💰 Insufficient balance for {symbol}. Available: ${available_balance:.2f}, Required: ${min_margin:.2f}")
            return False
        
        return True
    
    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM używając rzeczywistych cen z Bybit API"""
        
        self.logger.info(f"🔍 DEBUG ENTRY CHECK for {symbol}")
        
        if not self.should_enter_trade():
            self.logger.info(f"   ❌ Profile frequency check failed")
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
            
        signal, confidence = self.generate_llm_signal(symbol)
        self.logger.info(f"   🎯 Signal: {signal}, Confidence: {confidence:.1%}")
        
        if signal == "HOLD" or confidence < 0.3:
            self.logger.info(f"   ❌ Signal is HOLD or confidence too low")
            return None
            
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        self.logger.info(f"   📊 Active Positions: {active_positions}/{self.max_simultaneous_positions}")
        
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"   ❌ Max positions reached")
            return None
        
        self.logger.info(f"   ✅ ALL CONDITIONS MET - OPENING POSITION")
        # Reszta kodu...
            
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        # Sprawdź dostępne saldo
        api_status = self.check_api_status()
        available_balance = api_status['balance'] if api_status['balance_available'] else self.virtual_balance
        
        if margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}. Required: ${margin_required:.2f}, Available: ${available_balance:.2f}")
            return None
        
        # Składanie rzeczywistego zlecenia na Bybit
        order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
        if not order_id and self.real_trading:
            self.logger.error(f"❌ Failed to place order on Bybit for {symbol}")
            return None
            
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        position_id = order_id if order_id else f"virtual_{self.position_id}"
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
        """Aktualizuje P&L wszystkich pozycji używając rzeczywistych cen z Bybit API"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        # Synchronizuj z Bybit jeśli używamy real trading
        if self.real_trading:
            self.sync_with_bybit()
        
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
        api_status = self.check_api_status()
        if api_status['balance_available']:
            account_value = api_status['balance'] + total_unrealized
            available_cash = api_status['balance']
        else:
            account_value = self.virtual_capital + total_unrealized
            available_cash = self.virtual_balance
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
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

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu używając rzeczywistych cen z Bybit API"""
        # Sprawdź status API i pobierz saldo
        api_status = self.check_api_status()
        
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
            total_return_pct = ((current_balance - self.initial_capital) / self.initial_capital) * 100
        else:
            total_return_pct = 0
        
        return {
            'account_summary': {
                'total_value': round(current_balance, 2) if current_balance else 0,
                'available_cash': round(current_balance, 2) if current_balance else 0,
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
        """Główna pętla strategii LLM używająca rzeczywistych cen z Bybit API"""
        self.logger.info("🚀 STARTING LLM-STYLE TRADING STRATEGY")
        self.logger.info(f"🎯 Active Profile: {self.active_profile}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")
        
        # Sprawdź status API na starcie
        api_status = self.check_api_status()
        self.logger.info(f"📊 API Status: {api_status['message']}")
        
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

    def start_trading(self):
        """Rozpoczyna trading"""
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("🚀 LLM Trading Bot started")

    def stop_trading(self):
        """Zatrzymuje trading"""
        self.is_running = False
        self.logger.info("🛑 LLM Trading Bot stopped")
