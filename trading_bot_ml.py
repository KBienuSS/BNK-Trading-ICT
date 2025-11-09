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
    def __init__(self, api_key=None, api_secret=None, leverage=10):
        # Konfiguracja Bybit API
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.base_url = "https://api.bybit.com"
        
        # Sprawdź czy klucze API są dostępne
        if not self.api_key or not self.api_secret:
            raise Exception("❌ BRAK KLUCZY API BYBIT - wymagane dla real trading")
        
        self.real_trading = True
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Cache cen
        self.price_cache = {}
        self.price_history = {}
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM
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
            'avg_holding_time': 0
        }
        
        # DASHBOARD
        self.dashboard_data = {
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'average_confidence': 0,
            'portfolio_diversity': 0,
            'last_update': datetime.now(),
            'active_profile': self.active_profile
        }
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - REAL FUTURES TRADING")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📈 Trading assets: {', '.join(self.assets)}")
        self.logger.info(f"🔗 Real Trading: {self.real_trading}")

    def generate_bybit_signature(self, params: Dict, timestamp: str, recv_window: str = "5000") -> str:
        """Generuje signature dla Bybit API v5"""
        try:
            if params:
                # Konwertuj wszystkie wartości do string i posortuj
                string_params = {str(k): str(v) for k, v in params.items()}
                sorted_params = sorted(string_params.items())
                param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
                signature_payload = timestamp + self.api_key + recv_window + param_str
            else:
                signature_payload = timestamp + self.api_key + recv_window
            
            signature = hmac.new(
                bytes(self.api_secret, "utf-8"),
                signature_payload.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signature: {e}")
            return ""

    def bybit_request(self, method: str, endpoint: str, params: Dict = None, private: bool = False) -> Optional[Dict]:
        """Wykonuje request do Bybit API"""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        try:
            if private:
                timestamp = str(int(time.time() * 1000))
                recv_window = "5000"
                
                signature = self.generate_bybit_signature(params, timestamp, recv_window)
                if not signature:
                    self.logger.error("❌ Failed to generate signature")
                    return None
                    
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': recv_window,
                    'Content-Type': 'application/json'
                }
            
            self.logger.info(f"🌐 Making {method} request to: {url}")
            
            start_time = time.time()
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                self.logger.error(f"❌ Nieobsługiwana metoda HTTP: {method}")
                return None
            
            response_time = time.time() - start_time
            self.logger.info(f"📨 Response received in {response_time:.2f}s, status: {response.status_code}")
            
            if not response.text:
                self.logger.error("❌ Empty response from Bybit API")
                return None
                
            response_data = response.json()
            
            if response_data.get('retCode') != 0:
                error_msg = response_data.get('retMsg', 'Unknown error')
                error_code = response_data.get('retCode')
                self.logger.error(f"❌ Bybit API Error: {error_msg} (Code: {error_code})")
                return None
                
            return response_data.get('result', {})
            
        except Exception as e:
            self.logger.error(f"❌ Error in Bybit request: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Ustawia dźwignię dla symbolu"""
        try:
            endpoint = "/v5/position/set-leverage"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            }
            
            data = self.bybit_request('POST', endpoint, params, private=True)
            if data:
                self.logger.info(f"✅ Ustawiono dźwignię {leverage}x dla {symbol}")
                return True
            else:
                self.logger.error(f"❌ Błąd ustawiania dźwigni dla {symbol}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error setting leverage for {symbol}: {e}")
            return False

    def get_account_balance(self) -> Optional[float]:
        """Pobiera rzeczywiste saldo konta z Bybit"""
        try:
            endpoint = "/v5/account/wallet-balance"
            params = {'accountType': 'UNIFIED'}
            
            data = self.bybit_request('GET', endpoint, params, private=True)
            if data and 'list' in data and len(data['list']) > 0:
                total_equity = float(data['list'][0]['totalEquity'])
                self.logger.info(f"💰 Rzeczywiste saldo konta: ${total_equity:.2f}")
                return total_equity
            else:
                self.logger.error("❌ Nie udało się pobrać salda konta")
                return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę futures"""
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
            self.logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatuje ilość zgodnie z wymaganiami Bybit"""
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
        
        if formatted_quantity <= 0:
            formatted_quantity = lot_size
        
        return str(formatted_quantity)

    def place_bybit_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """Składa rzeczywiste zlecenie futures na Bybit"""
        try:
            # Ustaw dźwignię przed złożeniem zlecenia
            self.set_leverage(symbol, self.leverage)
            
            endpoint = "/v5/order/create"
            quantity_str = self.format_quantity(symbol, quantity)
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': 'Buy' if side == 'LONG' else 'Sell',
                'orderType': 'Market',
                'qty': quantity_str,
                'timeInForce': 'GTC',
                'leverage': str(self.leverage)
            }
            
            data = self.bybit_request('POST', endpoint, params, private=True)
            
            if data and 'orderId' in data:
                self.logger.info(f"✅ Zlecenie złożone: {symbol} {side} - ID: {data['orderId']}")
                return data['orderId']
            else:
                self.logger.error(f"❌ Błąd składania zlecenia dla {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return None

    def close_bybit_position(self, symbol: str, side: str, quantity: float) -> bool:
        """Zamyka pozycję na Bybit"""
        try:
            endpoint = "/v5/order/create"
            close_side = 'Sell' if side == 'LONG' else 'Buy'
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': close_side,
                'orderType': 'Market',
                'qty': self.format_quantity(symbol, quantity),
                'reduceOnly': True,
                'timeInForce': 'GTC'
            }
            
            data = self.bybit_request('POST', endpoint, params, private=True)
            if data and 'orderId' in data:
                self.logger.info(f"✅ Pozycja zamknięta: {symbol} - ID: {data['orderId']}")
                return True
            else:
                self.logger.error(f"❌ Błąd zamykania pozycji dla {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
            return False

    def get_bybit_positions(self) -> List[Dict]:
        """Pobiera aktywne pozycje z Bybit"""
        try:
            endpoint = "/v5/position/list"
            params = {'category': 'linear'}
            
            data = self.bybit_request('GET', endpoint, params, private=True)
            if data and 'list' in data:
                active_positions = []
                for pos in data['list']:
                    if float(pos['size']) > 0:
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
            self.logger.error(f"❌ Error getting positions: {e}")
            return []

    def sync_with_bybit(self):
        """Synchronizuje stan z rzeczywistymi pozycjami na Bybit"""
        try:
            # Pobierz aktywne pozycje z Bybit
            bybit_positions = self.get_bybit_positions()
            
            # Aktualizuj nasze pozycje na podstawie danych z Bybit
            self.positions = {}
            for i, pos in enumerate(bybit_positions):
                position_id = f"bybit_{i}"
                self.positions[position_id] = {
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'quantity': pos['size'],
                    'leverage': pos['leverage'],
                    'unrealized_pnl': pos['unrealised_pnl'],
                    'status': 'ACTIVE',
                    'entry_time': datetime.now(),
                    'real_trading': True
                }
            
            self.logger.info(f"🔄 Zsynchronizowano z Bybit - Pozycje: {len(bybit_positions)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing with Bybit: {e}")

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na podstawie danych z Bybit"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return random.uniform(-0.02, 0.02)
            
            # Symulacja prostego momentum (w prawdziwym bocie użyj prawdziwych danych historycznych)
            return random.uniform(-0.03, 0.03)
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return random.uniform(-0.02, 0.02)

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu LLM"""
        profile = self.get_current_profile()
        
        momentum = self.analyze_simple_momentum(symbol)
        base_confidence = profile['confidence_bias']
        
        # Final confidence z losowością
        final_confidence = min(base_confidence + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.1)
        
        # Decyzja o kierunku
        if momentum > 0.01:
            signal = "LONG"
        elif momentum < -0.01 and random.random() < profile['short_frequency']:
            signal = "SHORT"
        else:
            signal = "HOLD"
            
        current_price = self.get_current_price(symbol)
        price_display = f"${current_price:.4f}" if current_price else "N/A"
        self.logger.info(f"🎯 {self.active_profile} SIGNAL: {symbol} -> {signal} (Conf: {final_confidence:.1%})")
        
        return signal, final_confidence

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji"""
        profile = self.get_current_profile()
        
        # Pobierz rzeczywiste saldo konta
        real_balance = self.get_account_balance()
        if not real_balance:
            raise Exception("❌ Nie można pobrać salda konta")
        
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
            'max_holding_hours': random.randint(1, 6)
        }

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM"""
        try:
            # 1. Pobierz aktualną cenę
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.logger.warning(f"❌ Could not get price for {symbol}")
                return None
            
            # 2. Wygeneruj sygnał
            signal, confidence = self.generate_llm_signal(symbol)
            if signal == "HOLD":
                return None
            
            # 3. Sprawdź aktywne pozycje
            active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
            if active_positions >= self.max_simultaneous_positions:
                self.logger.warning(f"❌ Max positions reached: {active_positions}")
                return None
            
            # 4. Kalkulacja wielkości pozycji
            quantity, position_value, margin_required = self.calculate_position_size(
                symbol, current_price, confidence
            )
            
            # 5. Sprawdź minimalną wielkość zlecenia
            min_order_value = quantity * current_price
            if min_order_value < 5:  # Minimalne $5 dla Bybit
                self.logger.warning(f"❌ Order value too small: ${min_order_value:.2f} < $5")
                return None
            
            # 6. Składanie zlecenia na Bybit
            order_id = self.place_bybit_order(symbol, signal, quantity)
            if not order_id:
                self.logger.error(f"❌ Failed to place order for {symbol}")
                return None
            
            # 7. Tworzenie rekordu pozycji
            exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
            
            position = {
                'symbol': symbol,
                'side': signal,
                'entry_price': current_price,
                'quantity': quantity,
                'leverage': self.leverage,
                'margin': margin_required,
                'entry_time': datetime.now(),
                'status': 'ACTIVE',
                'unrealized_pnl': 0,
                'confidence': confidence,
                'llm_profile': self.active_profile,
                'exit_plan': exit_plan,
                'order_id': order_id,
                'real_trading': True
            }
            
            position_id = order_id
            self.positions[position_id] = position
            
            if signal == "LONG":
                self.stats['long_trades'] += 1
            else:
                self.stats['short_trades'] += 1
            
            # Logowanie sukcesu
            self.logger.info(f"🎉 REAL POSITION OPENED: {symbol} {signal} @ ${current_price:.4f}")
            self.logger.info(f"   📊 Size: ${position_value:.2f} | Margin: ${margin_required:.2f}")
            self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} | SL: {exit_plan['stop_loss']:.4f}")
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"❌ Error opening position: {e}")
            return None

    def update_positions_pnl(self):
        """Aktualizuje P&L wszystkich pozycji"""
        # Synchronizuj z Bybit
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

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji"""
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
            else:
                if current_price <= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
            
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_time > exit_plan['max_holding_hours']:
                exit_reason = "TIME_EXPIRED"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję"""
        position = self.positions[position_id]
        
        # Zamknij pozycję na Bybit
        success = self.close_bybit_position(position['symbol'], position['side'], position['quantity'])
        if not success:
            self.logger.error(f"❌ Failed to close position on Bybit: {position_id}")
            return
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
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
            'real_trading': True
        }
        
        self.trade_history.append(trade_record)
        
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        position['status'] = 'CLOSED'
        
        margin_return = pnl_pct * self.leverage * 100
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} REAL POSITION CLOSED: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} - Reason: {exit_reason}")

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

    def run_llm_trading_strategy(self):
        """Główna pętla strategii LLM"""
        self.logger.info("🚀 STARTING REAL FUTURES TRADING STRATEGY")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 Trading Iteration #{iteration}")
                
                # 1. Synchronizuj z Bybit i aktualizuj P&L
                self.sync_with_bybit()
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
                                time.sleep(2)  # Przerwa między zleceniami
                
                self.logger.info(f"📊 Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                wait_time = random.randint(60, 180)
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Rozpoczyna trading"""
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("🚀 Trading Bot started")

    def stop_trading(self):
        """Zatrzymuje trading"""
        self.is_running = False
        self.logger.info("🛑 Trading Bot stopped")

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu"""
        # Pobierz saldo konta
        account_balance = self.get_account_balance()
        
        active_positions = []
        total_unrealized_pnl = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                if not current_price:
                    continue
                
                active_positions.append({
                    'position_id': position_id,
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'confidence': position['confidence'],
                    'llm_profile': position['llm_profile'],
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'exit_plan': position['exit_plan'],
                    'real_trading': True
                })
                
                total_unrealized_pnl += position['unrealized_pnl']
        
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
                'holding_hours': round(trade['holding_hours'], 2),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'real_trading': True
            })
        
        # Metryki wydajności
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'account_summary': {
                'total_balance': round(account_balance, 2) if account_balance else 0,
                'unrealized_pnl': round(total_unrealized_pnl, 2),
                'net_realized': round(self.stats['total_pnl'], 2),
                'real_trading': True
            },
            'performance_metrics': {
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'long_trades': self.stats['long_trades'],
                'short_trades': self.stats['short_trades'],
                'total_pnl': round(self.stats['total_pnl'], 2)
            },
            'llm_config': {
                'active_profile': self.active_profile,
                'available_profiles': list(self.llm_profiles.keys()),
                'max_positions': self.max_simultaneous_positions,
                'leverage': self.leverage,
                'real_trading': True
            },
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'last_update': datetime.now().isoformat()
        }
