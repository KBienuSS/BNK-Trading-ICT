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
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM (wg Alpha Arena) - ZMODYFIKOWANE DLA QWEN
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
                'momentum_threshold': 0.003,
                'min_holding_hours': 2,
                'max_holding_hours': 8,
                'tp_multiplier': 1.0,
                'sl_multiplier': 1.0
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
                'momentum_threshold': 0.002,
                'min_holding_hours': 1,
                'max_holding_hours': 6,
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.9
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
                'momentum_threshold': 0.004,
                'min_holding_hours': 2,
                'max_holding_hours': 10,
                'tp_multiplier': 0.8,
                'sl_multiplier': 1.1
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE',
                'max_position_size_pct': 0.40,  # Zwiększone z 0.30
                'min_confidence': 0.4,         # Niższy próg wejścia
                'momentum_threshold': 0.002,
                'min_holding_hours': 4,        # Wydłużone minimum
                'max_holding_hours': 24,       # Wydłużone maksimum
                'tp_multiplier': 1.3,          # Szersze TP
                'sl_multiplier': 1.2,          # Szersze SL
                'use_tiered_exits': True,      # System warstwowy
                'use_trailing_stop': True,     # Trailing stop
                'use_volatility_based': True   # Bazowanie na ATR
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
            'won_long_trades': 0,
            'won_short_trades': 0,
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

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Oblicza Average True Range dla danego symbolu używając danych z Bybit"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < period + 1:
                return 0.02  # fallback 2%
            
            prices = [entry['price'] for entry in self.price_history[symbol]]
            true_ranges = []
            
            for i in range(1, len(prices)):
                high = max(prices[i], prices[i-1])
                low = min(prices[i], prices[i-1])
                true_range = high - low
                true_ranges.append(true_range)
            
            # Weź ostatnie N true ranges
            recent_true_ranges = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
            atr = np.mean(recent_true_ranges) if recent_true_ranges else 0
            
            # Normalizuj do procentów
            current_price = prices[-1] if prices else 1
            atr_percent = atr / current_price if current_price > 0 else 0.02
            
            return max(min(atr_percent, 0.1), 0.005)  # Limit 0.5% - 10%
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR for {symbol}: {e}")
            return 0.02

    def calculate_dynamic_sl(self, symbol: str, entry_price: float, side: str, confidence: float) -> float:
        """Oblicza dynamiczny Stop Loss bazujący na ATR - TAK SAMO JAK W VIRTUAL"""
        profile = self.get_current_profile()
        
        # ✅ TA SAMA LOGIKA CO W VIRTUAL TRADING
        atr_percent = self.calculate_atr(symbol)
        
        if self.active_profile == 'Qwen' and profile.get('use_volatility_based', True):
            if confidence > 0.8:
                sl_multiplier = 1.0 * profile['sl_multiplier']
            elif confidence > 0.6:
                sl_multiplier = 1.2 * profile['sl_multiplier']
            else:
                sl_multiplier = 1.5 * profile['sl_multiplier']
        else:
            sl_multiplier = profile['sl_multiplier']
        
        # Oblicz SL na podstawie ATR
        if side == "LONG":
            stop_loss = entry_price * (1 - atr_percent * sl_multiplier)
        else:
            stop_loss = entry_price * (1 + atr_percent * sl_multiplier)
        
        self.logger.info(f"🎯 DYNAMIC SL CALCULATED: {symbol} {side} - ATR: {atr_percent:.3%}, Multiplier: {sl_multiplier:.1f}, SL: ${stop_loss:.2f}")
        
        return stop_loss

    def place_bybit_order_with_dynamic_sl(self, symbol: str, side: str, quantity: float, price: float, confidence: float) -> Tuple[Optional[str], Optional[float]]:
        """Składa zlecenie na Bybit Z DYNAMICZNYM STOP-LOSSEM (ATR-based)"""
        
        self.logger.info(f"🚀 PLACE_BYBIT_ORDER_WITH_DYNAMIC_SL: {symbol} {side}")
        
        if not self.real_trading:
            order_id = f"virtual_{int(time.time())}"
            self.logger.info(f"🔄 Virtual order with dynamic SL: {order_id}")
            return order_id, None
        
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return None, None
        
        try:
            self.set_leverage(symbol, self.leverage)
            quantity_str = self.format_quantity(symbol, quantity)
            
            # ✅ OBLICZ DYNAMICZNY SL BAZUJĄCY NA ATR (TAK SAMO JAK W VIRTUAL)
            stop_loss = self.calculate_dynamic_sl(symbol, price, side, confidence)
            
            # Składanie zlecenia z DYNAMICZNYM STOP LOSS
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side="Buy" if side == "LONG" else "Sell",
                orderType="Market",
                qty=quantity_str,
                timeInForce="GTC",
                stopLoss=str(stop_loss)  # ✅ DYNAMICZNY SL NA BYBIT
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"✅ DYNAMIC SL ORDER SUCCESS: {symbol} {side} - Entry: ${price:.2f}, SL: ${stop_loss:.2f}")
                return order_id, stop_loss  # ✅ ZWRÓĆ RÓWNIEŻ WYLICZONY SL
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ DYNAMIC SL ORDER FAILED: {error_msg}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"💥 ERROR in place_bybit_order_with_dynamic_sl: {e}")
            return None, None

    def update_bybit_stop_loss(self, symbol: str, stop_loss: float) -> bool:
        """Aktualizuje Stop Loss na Bybit - dla trailing stop"""
        if not self.real_trading:
            return True
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return False

        try:
            response = self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(stop_loss),  # ✅ AKTUALIZACJA SL NA BYBIT
                positionIdx=0
            )
            
            if response['retCode'] == 0:
                self.logger.info(f"✅ BYBIT SL UPDATED: {symbol} - New SL: ${stop_loss:.2f}")
                return True
            else:
                if response['retCode'] == 10001:  # No position found
                    return False
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ BYBIT SL UPDATE FAILED: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error updating Bybit SL: {e}")
            return False

    def calculate_volatility_based_exits(self, symbol: str, entry_price: float, side: str, confidence: float) -> Tuple[float, float]:
        """Oblicza TP/SL bazujące na zmienności (ATR) - DOSTOSOWANE DO BYBIT"""
        profile = self.get_current_profile()
        atr_percent = self.calculate_atr(symbol)
        
        # Domyślne multiplikatory
        if self.active_profile == 'Qwen' and profile.get('use_volatility_based', True):
            if confidence > 0.8:
                tp_multiplier = 2.5 * profile['tp_multiplier']
                sl_multiplier = 1.0 * profile['sl_multiplier']
            elif confidence > 0.6:
                tp_multiplier = 2.0 * profile['tp_multiplier']
                sl_multiplier = 1.2 * profile['sl_multiplier']
            else:
                tp_multiplier = 1.5 * profile['tp_multiplier']
                sl_multiplier = 1.5 * profile['sl_multiplier']
        else:
            tp_multiplier = profile['tp_multiplier']
            sl_multiplier = profile['sl_multiplier']
        
        if side == "LONG":
            take_profit = entry_price * (1 + atr_percent * tp_multiplier)
            stop_loss = entry_price * (1 - atr_percent * sl_multiplier)
        else:
            take_profit = entry_price * (1 - atr_percent * tp_multiplier)
            stop_loss = entry_price * (1 + atr_percent * sl_multiplier)
        
        return take_profit, stop_loss

    def calculate_tiered_exit_plan(self, entry_price: float, side: str, confidence: float) -> List[Dict]:
        """System warstwowych zysków dla agresywnego Qwen - DOSTOSOWANE DO BYBIT"""
        profile = self.get_current_profile()
        
        if self.active_profile == 'Qwen' and profile.get('use_tiered_exits', False):
            if confidence > 0.8:
                tiers = [
                    {'percent': 0.3, 'tp_pct': 0.010},  # 30% pozycji przy 1%
                    {'percent': 0.4, 'tp_pct': 0.018},  # 40% przy 1.8%  
                    {'percent': 0.3, 'tp_pct': 0.025}   # 30% przy 2.5%
                ]
            elif confidence > 0.6:
                tiers = [
                    {'percent': 0.5, 'tp_pct': 0.008},  # 50% przy 0.8%
                    {'percent': 0.5, 'tp_pct': 0.015}   # 50% przy 1.5%
                ]
            else:
                tiers = [
                    {'percent': 0.7, 'tp_pct': 0.006},  # 70% przy 0.6%
                    {'percent': 0.3, 'tp_pct': 0.012}   # 30% przy 1.2%
                ]
            
            # Konwersja na ceny
            partial_exits = []
            for tier in tiers:
                if side == "LONG":
                    tp_price = entry_price * (1 + tier['tp_pct'])
                else:
                    tp_price = entry_price * (1 - tier['tp_pct'])
                partial_exits.append({
                    'price': round(tp_price, 4),
                    'percent': tier['percent']
                })
            
            return partial_exits
        else:
            # Dla innych profili - brak partial exits
            return []

    def check_partial_exits(self, position_id: str, current_price: float) -> bool:
        """Sprawdza warunki partial take profits - DOSTOSOWANE DO BYBIT"""
        position = self.positions[position_id]
        exit_plan = position.get('exit_plan', {})
        
        if not exit_plan.get('partial_exits'):
            return False
        
        for partial_exit in exit_plan['partial_exits']:
            if partial_exit['price'] in position.get('partial_exits_taken', []):
                continue
                
            if position['side'] == "LONG" and current_price >= partial_exit['price']:
                return self.execute_partial_exit(position_id, partial_exit)
            elif position['side'] == "SHORT" and current_price <= partial_exit['price']:
                return self.execute_partial_exit(position_id, partial_exit)
        
        return False

    def execute_partial_exit(self, position_id: str, partial_exit: Dict) -> bool:
        """Wykonuje partial exit z pozycji - DOSTOSOWANE DO BYBIT"""
        position = self.positions[position_id]
        
        # Oblicz ilość do zamknięcia
        close_quantity = position['quantity'] * partial_exit['percent']
        close_quantity_str = self.format_quantity(position['symbol'], close_quantity)
        
        if self.real_trading:
            # Real trading - składamy zlecenie na Bybit
            success = self.close_bybit_position_partial(position['symbol'], position['side'], close_quantity)
            if not success:
                return False
        else:
            # Virtual trading - symulacja
            self.logger.info(f"🟡 VIRTUAL PARTIAL EXIT: {position['symbol']} - {partial_exit['percent']:.0%}")
        
        # Aktualizuj pozycję lokalnie
        position['quantity'] -= close_quantity
        position['margin'] *= (1 - partial_exit['percent'])  # Zmniejsz margin proporcjonalnie
        
        # Oblicz P&L dla partial exit
        current_price = self.get_current_price(position['symbol'])
        if position['side'] == "LONG":
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
        
        realized_pnl = pnl_pct * close_quantity * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        # Zwróć margin i P&L
        returned_margin = position['margin'] * partial_exit['percent']
        
        if not self.real_trading:
            self.virtual_balance += returned_margin + realized_pnl_after_fee
            self.virtual_capital += realized_pnl_after_fee
        
        # Zapisz partial exit
        if 'partial_exits_taken' not in position:
            position['partial_exits_taken'] = []
        position['partial_exits_taken'].append(partial_exit['price'])
        
        self.logger.info(f"🟡 PARTIAL EXIT: {position['symbol']} - {partial_exit['percent']:.0%} @ ${current_price:.4f} | P&L: ${realized_pnl_after_fee:+.2f}")
        
        return True

    def close_bybit_position_partial(self, symbol: str, side: str, quantity: float) -> bool:
        """Zamyka część pozycji na Bybit używając pybit"""
        if not self.real_trading:
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
                self.logger.info(f"✅ Partial position closed on Bybit: {symbol} - Qty: {quantity_str}")
                return True
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ Błąd zamykania części pozycji na Bybit dla {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error closing partial Bybit position: {e}")
            return False

    def update_trailing_stop(self, position_id: str, current_price: float):
        """Aktualizuje trailing stop - ZACHOWUJĄC LOGIKĘ ATR"""
        position = self.positions[position_id]
        exit_plan = position.get('exit_plan', {})
        
        if not exit_plan.get('use_trailing_stop', False):
            return
        
        unrealized_pnl_pct = abs(current_price - position['entry_price']) / position['entry_price']
        trailing_start = exit_plan.get('trailing_start', 0.008)
        
        # Sprawdź czy osiągnięto poziom startu trailing
        if unrealized_pnl_pct >= trailing_start:
            if exit_plan.get('original_sl') is None:
                exit_plan['original_sl'] = exit_plan['stop_loss']
            
            # ✅ OBLICZ TRAILING STEP BAZUJĄCY NA ATR (TAK SAMO JAK W VIRTUAL)
            atr_percent = self.calculate_atr(position['symbol'])
            trailing_step = atr_percent * 0.5  # 50% ATR jako trailing step
            
            if position['side'] == "LONG":
                new_sl = current_price * (1 - trailing_step)
                # Podnieś SL tylko jeśli wyższy niż obecny
                if new_sl > exit_plan['stop_loss']:
                    old_sl = exit_plan['stop_loss']
                    exit_plan['stop_loss'] = new_sl
                    
                    # ✅ AKTUALIZUJ NA BYBIT ZACHOWUJĄC LOGIKĘ ATR
                    if self.real_trading:
                        success = self.update_bybit_stop_loss(position['symbol'], new_sl)
                        if success:
                            self.logger.info(f"📈 ATR TRAILING STOP UPDATED: {position['symbol']} - New SL: ${new_sl:.4f} (ATR-based)")
                        else:
                            self.logger.warning(f"⚠️ ATR trailing stop updated locally but failed on Bybit: {position['symbol']}")
                    else:
                        self.logger.info(f"📈 ATR TRAILING STOP UPDATED (VIRTUAL): {position['symbol']} - New SL: ${new_sl:.4f}")
                    
            else:  # SHORT
                new_sl = current_price * (1 + trailing_step)
                # Obniż SL tylko jeśli niższy niż obecny
                if new_sl < exit_plan['stop_loss']:
                    old_sl = exit_plan['stop_loss']
                    exit_plan['stop_loss'] = new_sl
                    
                    # ✅ AKTUALIZUJ NA BYBIT ZACHOWUJĄC LOGIKĘ ATR
                    if self.real_trading:
                        success = self.update_bybit_stop_loss(position['symbol'], new_sl)
                        if success:
                            self.logger.info(f"📈 ATR TRAILING STOP UPDATED: {position['symbol']} - New SL: ${new_sl:.4f} (ATR-based)")
                        else:
                            self.logger.warning(f"⚠️ ATR trailing stop updated locally but failed on Bybit: {position['symbol']}")
                    else:
                        self.logger.info(f"📈 ATR TRAILING STOP UPDATED (VIRTUAL): {position['symbol']} - New SL: ${new_sl:.4f}")

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
                    symbol='BTCUSDT'
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

    def check_all_conditions_for_eth(self):
        """Sprawdza wszystkie warunki dla ETH"""
        self.logger.info("🔍 CHECKING ALL CONDITIONS FOR ETHUSDT...")
        
        symbol = "ETHUSDT"
        
        # 1. Cena
        price = self.get_current_price(symbol)
        self.logger.info(f"1. PRICE: ${price}" if price else "1. PRICE: ❌ Unavailable")
        
        # 2. Sygnał
        signal, confidence = self.generate_llm_signal(symbol)
        self.logger.info(f"2. SIGNAL: {signal} (Confidence: {confidence:.1%})")
        
        # 3. Częstotliwość tradingu
        should_enter = self.should_enter_trade()
        self.logger.info(f"3. TRADE FREQUENCY: {should_enter}")
        
        # 4. Aktywne pozycje
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        self.logger.info(f"4. ACTIVE POSITIONS: {active_positions}/{self.max_simultaneous_positions}")
        
        # 5. Czy już ma ETH
        has_eth = any(p['symbol'] == symbol and p['status'] == 'ACTIVE' for p in self.positions.values())
        self.logger.info(f"5. HAS ETH POSITION: {has_eth}")
        
        # 6. Balans
        if self.real_trading:
            balance = self.get_account_balance()
        else:
            balance = self.virtual_balance
        self.logger.info(f"6. BALANCE: ${balance:.2f}")
        
        # Podsumowanie
        profile = self.get_current_profile()
        can_open = (price is not None and signal != "HOLD" and confidence > profile['min_confidence'] and 
                    should_enter and active_positions < self.max_simultaneous_positions and 
                    not has_eth)
        
        self.logger.info(f"🎯 CAN OPEN ETH POSITION: {'✅ YES' if can_open else '❌ NO'}")
        
        return can_open
    
    def sync_all_positions_with_bybit(self):
        """POPRAWIONA synchronizacja pozycji z Bybit - UWZGLĘDNIA SL"""
        if not self.real_trading:
            self.logger.info("🔄 SYNC: Virtual mode - no Bybit sync needed")
            return
            
        self.logger.info("🔄 FULL SYNC: Synchronizing all positions with Bybit...")
        
        try:
            # Pobierz wszystkie pozycje z Bybit
            bybit_positions = self.get_bybit_positions()
            self.logger.info(f"📊 BYBIT POSITIONS: Found {len(bybit_positions)} positions on Bybit")
            
            # ✅ DODAJ: Sprawdź aktualne SL dla każdej pozycji na Bybit
            for bybit_pos in bybit_positions:
                try:
                    response = self.session.get_positions(
                        category="linear",
                        symbol=bybit_pos['symbol']
                    )
                    
                    if response['retCode'] == 0 and response['result']['list']:
                        current_bybit_sl = response['result']['list'][0].get('stopLoss')
                        if current_bybit_sl and current_bybit_sl != '':
                            bybit_pos['bybit_stop_loss'] = float(current_bybit_sl)
                            self.logger.info(f"📊 BYBIT SL for {bybit_pos['symbol']}: ${bybit_pos['bybit_stop_loss']:.2f}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not fetch SL from Bybit for {bybit_pos['symbol']}: {e}")
            
            # Tworzymy nowy słownik pozycji
            new_positions = {}
            
            for i, bybit_pos in enumerate(bybit_positions):
                position_id = f"bybit_sync_{bybit_pos['symbol']}_{i}_{int(time.time())}"
                
                current_price = self.get_current_price(bybit_pos['symbol'])
                if not current_price:
                    current_price = bybit_pos.get('mark_price', bybit_pos['entry_price'])
                    self.logger.warning(f"⚠️ Could not get current price for {bybit_pos['symbol']}, using mark price")
                
                # Użyj unrealized P&L bezpośrednio z Bybit
                unrealized_pnl = bybit_pos.get('unrealised_pnl', 0)
                
                # Oblicz exit plan dla zsynchronizowanej pozycji
                exit_plan = self.calculate_llm_exit_plan(bybit_pos['entry_price'], 0.5, bybit_pos['side'])
                
                # ✅ UŻYJ RZECZYWISTEGO SL Z BYBIT JEŚLI JEST DOSTĘPNY
                if 'bybit_stop_loss' in bybit_pos:
                    exit_plan['stop_loss'] = bybit_pos['bybit_stop_loss']
                    self.logger.info(f"🔄 USING ACTUAL BYBIT SL: {bybit_pos['symbol']} - SL: ${exit_plan['stop_loss']:.2f}")
                
                new_positions[position_id] = {
                    'symbol': bybit_pos['symbol'],
                    'side': bybit_pos['side'],
                    'entry_price': bybit_pos['entry_price'],
                    'quantity': bybit_pos['size'],
                    'leverage': bybit_pos['leverage'],
                    'entry_time': bybit_pos.get('created_time', datetime.now()),
                    'status': 'ACTIVE',
                    'order_id': f"bybit_{bybit_pos['symbol']}_{i}",
                    'real_trading': True,
                    'llm_profile': self.active_profile,
                    'confidence': 0.5,
                    'margin': bybit_pos.get('position_margin', bybit_pos['size'] * bybit_pos['entry_price'] / bybit_pos['leverage']),
                    'exit_plan': exit_plan,
                    'liquidation_price': bybit_pos.get('liq_price'),
                    'unrealized_pnl': unrealized_pnl,
                    'current_price': current_price,
                    'partial_exits_taken': [],
                    'bybit_sl_set': True,  # ✅ OZNACZENIE CZY SL JEST USTAWIONY NA BYBIT
                    'sl_calculation_method': 'ATR-based'  # ✅ DODAJ METODĘ OBLICZANIA SL
                }
                
                self.logger.info(f"✅ SYNCED: {bybit_pos['symbol']} {bybit_pos['side']} - Size: {bybit_pos['size']}, Entry: ${bybit_pos['entry_price']}, SL: ${exit_plan['stop_loss']:.2f}")
            
            # Zamień stare pozycje na nowe (żeby nie stracić virtual positions)
            old_positions_count = len(self.positions)
            self.positions = new_positions
            
            self.logger.info(f"🎯 SYNC COMPLETE: {old_positions_count} -> {len(self.positions)} positions synchronized with Bybit")
            
        except Exception as e:
            self.logger.error(f"❌ SYNC ERROR: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")

    def debug_eth_signal(self):
        """Debugowanie sygnału dla ETHUSDT"""
        self.logger.info("🔍 DEBUG ETHUSDT SIGNAL...")
        
        # Sprawdź cenę
        eth_price = self.get_current_price("ETHUSDT")
        self.logger.info(f"💰 ETHUSDT Price: ${eth_price}")
        
        # Sprawdź historię cen
        if "ETHUSDT" in self.price_history:
            history = self.price_history["ETHUSDT"]
            self.logger.info(f"📊 ETH Price History: {len(history)} entries")
            if len(history) >= 2:
                momentum = (history[-1]['price'] - history[-2]['price']) / history[-2]['price']
                self.logger.info(f"📈 ETH Momentum: {momentum:.4%}")
        
        # Generuj sygnał
        signal, confidence = self.generate_llm_signal("ETHUSDT")
        self.logger.info(f"🎯 ETH Signal: {signal}, Confidence: {confidence:.1%}")
        
        # Sprawdź warunki wejścia
        should_enter = self.should_enter_trade()
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        has_eth_position = any(p['symbol'] == "ETHUSDT" and p['status'] == 'ACTIVE' 
                              for p in self.positions.values())
        
        self.logger.info(f"📊 Entry Conditions:")
        self.logger.info(f"   - Should enter: {should_enter}")
        self.logger.info(f"   - Active positions: {active_positions}/{self.max_simultaneous_positions}")
        self.logger.info(f"   - Has ETH position: {has_eth_position}")
        self.logger.info(f"   - Signal not HOLD: {signal != 'HOLD'}")
        self.logger.info(f"   - Confidence > 0.3: {confidence > 0.3}")
        
        return signal, confidence

    def force_open_eth_position(self, side="LONG", confidence=0.8):
        """Wymusza otwarcie pozycji na ETH"""
        self.logger.info(f"🚀 FORCE OPENING ETH POSITION: {side}")
        
        symbol = "ETHUSDT"
        
        # Sprawdź czy już masz pozycję
        existing_position = any(
            p['symbol'] == symbol and p['status'] == 'ACTIVE' 
            for p in self.positions.values()
        )
        
        if existing_position:
            self.logger.warning(f"⚠️ Already have active position for {symbol}")
            return None
        
        # Pobierz cenę
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.error(f"❌ Could not get price for {symbol}")
            return None
        
        # Oblicz wielkość pozycji
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        # Sprawdź balans
        if self.real_trading:
            real_balance = self.get_account_balance()
            available_balance = real_balance if real_balance else self.virtual_balance
        else:
            available_balance = self.virtual_balance
            
        if margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance. Required: ${margin_required:.2f}, Available: ${available_balance:.2f}")
            # Zmniejsz wielkość pozycji
            position_value = available_balance * 0.8  # użyj 80% dostępnego balansu
            quantity = position_value / current_price
            margin_required = position_value / self.leverage
            self.logger.info(f"🔄 Adjusted position size: ${position_value:.2f}")
        
        # Przygotuj dane pozycji z nowym systemem exit plan
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, side)
        
        if side == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        # Składanie zlecenia
        order_id = None
        initial_sl = None
        
        if self.real_trading:
            # REAL TRADING: złoż zlecenie Z DYNAMICZNYM STOP LOSSEM na Bybit
            order_id, initial_sl = self.place_bybit_order_with_dynamic_sl(
                symbol, side, quantity, current_price, confidence
            )
            if not order_id:
                self.logger.error("❌ Failed to place order on Bybit")
                return None
            
            # ✅ NADPISZ SL W EXIT PLAN TYM WYLICZONYM DYNAMICZNIE
            exit_plan['stop_loss'] = initial_sl
            self.logger.info(f"🎯 DYNAMIC SL SET: {symbol} - ATR-based SL: ${initial_sl:.2f}")
        else:
            order_id = f"forced_eth_{int(time.time())}"
        
        position_id = order_id
        
        # Zapisz pozycję
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
            'confidence': confidence,
            'llm_profile': self.active_profile,
            'exit_plan': exit_plan,
            'order_id': order_id,
            'real_trading': self.real_trading,
            'current_price': current_price,
            'partial_exits_taken': [],
            'bybit_sl_set': self.real_trading,  # ✅ OZNACZENIE CZY SL JEST USTAWIONY NA BYBIT
            'sl_calculation_method': 'ATR-based'  # ✅ DODAJ METODĘ OBLICZANIA SL
        }
        
        self.positions[position_id] = position
        
        # Aktualizuj statystyki
        if side == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        self.stats['total_trades'] += 1
        
        if not self.real_trading:
            self.virtual_balance -= margin_required
        
        sl_method = "ATR-based DYNAMIC SL" if self.real_trading else "VIRTUAL SL"
        self.logger.info(f"✅ FORCE OPENED: ETHUSDT {side} @ ${current_price:.2f}")
        self.logger.info(f"   📏 Size: ${position_value:.2f}, Qty: {quantity:.4f}")
        self.logger.info(f"   🛑 {sl_method}: ${exit_plan['stop_loss']:.2f}")
        
        return position_id
    
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

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na podstawie rzeczywistych danych z Bybit API"""
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
        """Sprawdza aktywność wolumenu na podstawie zmienności cen z Bybit API"""
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
        """Generuje sygnał w stylu LLM na podstawie rzeczywistych danych z Bybit API"""
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

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu LLM - ZMODYFIKOWANE DLA QWEN"""
        profile = self.get_current_profile()
        
        base_allocation = {
            'Claude': 0.15,
            'Gemini': 0.25, 
            'GPT': 0.10,
            'Qwen': 0.40  # Zwiększone z 0.30
        }.get(self.active_profile, 0.15)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        sizing_multiplier = {
            'CONSERVATIVE': 0.8,
            'AGGRESSIVE': 1.2,
            'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        real_balance = self.get_account_balance()
        if real_balance is None:
            real_balance = self.virtual_balance
        
        position_value = (real_balance * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = real_balance * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        self.logger.info(f"📏 POSITION SIZE: Value: ${position_value:.2f}, Qty: {quantity:.6f}, Margin: ${margin_required:.2f}, Confidence: {confidence:.1%}")
        
        return quantity, position_value, margin_required

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

    def get_bybit_positions(self) -> List[Dict]:
        """POPRAWIONE pobieranie pozycji z Bybit dla API V5"""
        if not self.real_trading:
            self.logger.info("🔄 Virtual mode - no Bybit positions")
            return []
            
        if not self.session:
            self.logger.error("❌ Brak sesji pybit")
            return []
    
        try:
            self.logger.info("🔍 Fetching ALL positions from Bybit (V5 API)...")
            
            # SPRAWDŹ POZYCJE DLA RÓŻNYCH settleCoin
            settle_coins = ['USDT', 'USDC', 'BTC', 'ETH']  # Najczęstsze stablecoiny
            all_positions = []
            
            for settle_coin in settle_coins:
                try:
                    self.logger.info(f"🔍 Checking positions for settleCoin: {settle_coin}")
                    
                    response = self.session.get_positions(
                        category="linear",
                        settleCoin=settle_coin  # ← WAŻNE: podaj settleCoin zamiast pustego symbol
                    )
                    
                    self.logger.info(f"📨 Bybit API Response for {settle_coin}: {response['retCode']}")
                    
                    if response['retCode'] == 0:
                        result_list = response['result'].get('list', [])
                        self.logger.info(f"📊 Found {len(result_list)} position entries for {settle_coin}")
                        
                        for i, pos in enumerate(result_list):
                            size = float(pos['size'])
                            symbol = pos['symbol']
                            side = 'LONG' if pos['side'] == 'Buy' else 'SHORT'
                            
                            self.logger.info(f"  📍 Position {i}: {symbol} {side} - Size: {size}, AvgPrice: ${pos['avgPrice']}")
                            
                            if size > 0:
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
                                    'mark_price': float(pos['markPrice']) if pos.get('markPrice') else float(pos['avgPrice']),
                                    'settle_coin': settle_coin
                                }
                                
                                all_positions.append(position_data)
                                self.logger.info(f"  ✅ ADDED ACTIVE: {symbol} {side} Size: {size}, Entry: ${position_data['entry_price']}, PnL: ${position_data['unrealised_pnl']:.2f}")
                            else:
                                self.logger.info(f"  ❌ SKIPPED (zero size): {symbol} {side} - Size: {size}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error checking {settle_coin} positions: {e}")
                    continue
            
            self.logger.info(f"✅ Final: {len(all_positions)} ACTIVE positions on Bybit")
            return all_positions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting Bybit positions: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            return []
        
    def get_bybit_unrealized_pnl(self) -> float:
        """POPRAWIONE pobieranie unrealized P&L z Bybit dla API V5"""
        if not self.real_trading:
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
            total_unrealized = 0.0
            settle_coins = ['USDT', 'USDC', 'BTC', 'ETH']
            
            for settle_coin in settle_coins:
                try:
                    response = self.session.get_positions(
                        category="linear",
                        settleCoin=settle_coin  # ← WAŻNE: podaj settleCoin
                    )
                    
                    if response['retCode'] == 0:
                        for pos in response['result']['list']:
                            size = float(pos.get('size', 0))
                            if size > 0:
                                unrealised_pnl = float(pos.get('unrealisedPnl', 0))
                                total_unrealized += unrealised_pnl
                                self.logger.info(f"📊 Position {pos['symbol']} P&L: ${unrealised_pnl:.2f}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting P&L for {settle_coin}: {e}")
                    continue
            
            self.logger.info(f"📊 Total Real Unrealized P&L from Bybit: ${total_unrealized:.2f}")
            return total_unrealized
                
        except Exception as e:
            self.logger.error(f"❌ Error getting unrealized P&L from Bybit: {e}")
            return 0.0

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia - JEDNOLITA LOGIKA DLA REAL I VIRTUAL"""
        profile = self.get_current_profile()
        
        # ✅ ZAWSZE UŻYWAJ VOLATILITY-BASED EXITS DLA QWEN (NIEZALEŻNIE OD REAL/VIRTUAL)
        if self.active_profile == 'Qwen' and profile.get('use_volatility_based', True):
            # Użyj proxy symbol dla obliczenia ATR
            take_profit, stop_loss = self.calculate_volatility_based_exits(
                'BTCUSDT', entry_price, side, confidence
            )
        else:
            # Standardowe obliczenia dla innych profili
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
        
        # Zastosuj multiplikatory profilu
        take_profit = entry_price + (take_profit - entry_price) * profile['tp_multiplier']
        stop_loss = entry_price + (stop_loss - entry_price) * profile['sl_multiplier']
        
        # Oblicz partial exits dla Qwen
        partial_exits = self.calculate_tiered_exit_plan(entry_price, side, confidence)
        
        exit_plan = {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': entry_price * 0.98 if side == "LONG" else entry_price * 1.02,
            'max_holding_hours': random.randint(profile['min_holding_hours'], profile['max_holding_hours']),
            'partial_exits': partial_exits,
            'use_trailing_stop': profile.get('use_trailing_stop', False),
            'trailing_start': 0.008 if self.active_profile == 'Qwen' else 0.012,
            'trailing_step': 0.003 if self.active_profile == 'Qwen' else 0.005,
            'original_sl': None,
            'calculation_method': 'ATR-based' if (self.active_profile == 'Qwen' and profile.get('use_volatility_based', True)) else 'Fixed'
        }
        
        return exit_plan
       
    def should_enter_trade(self) -> bool:
        """Decyduje czy wejść w transakcję wg profilu częstotliwości - ZMODYFIKOWANE"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.2,        # Zmniejszone
            'MEDIUM': 0.3,     # Zmniejszone  
            'HIGH': 0.8        # Zmniejszone
        }.get(profile['trade_frequency'], 0.3)
        
        # DODATKOWY FILTR DLA QWEN - mniej, ale większe pozycje
        if self.active_profile == 'Qwen' and len([p for p in self.positions.values() if p['status'] == 'ACTIVE']) >= 2:
            return False  # Qwen powinien trzymać 1-2 pozycje
        
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwieranie pozycji z DYNAMICZNYM SL (ATR-based)"""
        self.logger.info(f"🔧 ATTEMPTING TO OPEN WITH DYNAMIC SL: {symbol}")
        
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Could not get price for {symbol}")
            return None
            
        signal, confidence = self.generate_llm_signal(symbol)
        
        profile = self.get_current_profile()
        if signal == "HOLD" or confidence < profile['min_confidence']:
            return None
            
        # Sprawdź czy już masz aktywną pozycję
        existing_position = any(
            p['symbol'] == symbol and p['status'] == 'ACTIVE' 
            for p in self.positions.values()
        )
        if existing_position:
            self.logger.info(f"⏸️ Already have active position for {symbol}")
            return None
            
        active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_count >= self.max_simultaneous_positions:
            self.logger.info(f"⏸️ Max positions reached ({active_count}/{self.max_simultaneous_positions})")
            return None
            
        # Oblicz wielkość pozycji
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if self.real_trading:
            real_balance = self.get_account_balance()
            available_balance = real_balance if real_balance else self.virtual_balance
        else:
            available_balance = self.virtual_balance
            
        if margin_required > available_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        # ✅ UŻYJ TEJ SAMEJ LOGIKI EXIT PLAN CO W VIRTUAL
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        # ✅ HYBRYD: Użyj dynamicznego SL dla real trading
        order_id = None
        initial_sl = None
        
        if self.real_trading:
            # REAL TRADING: złoż zlecenie Z DYNAMICZNYM STOP LOSSEM na Bybit
            order_id, initial_sl = self.place_bybit_order_with_dynamic_sl(
                symbol, signal, quantity, current_price, confidence
            )
            if not order_id:
                return None
            
            # ✅ NADPISZ SL W EXIT PLAN TYM WYLICZONYM DYNAMICZNIE
            exit_plan['stop_loss'] = initial_sl
            self.logger.info(f"🎯 DYNAMIC SL SET: {symbol} - ATR-based SL: ${initial_sl:.2f}")
            
        else:
            # VIRTUAL TRADING: zwykłe zlecenie (zachowaj starą logikę)
            order_id = self.place_bybit_order(symbol, signal, quantity, current_price)
            if not order_id:
                return None
        
        position_id = order_id
        
        # Zapisz pozycję
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
            'real_trading': self.real_trading,
            'current_price': current_price,
            'partial_exits_taken': [],
            'bybit_sl_set': self.real_trading,  # ✅ OZNACZENIE CZY SL JEST USTAWIONY NA BYBIT
            'sl_calculation_method': 'ATR-based'  # ✅ DODAJ METODĘ OBLICZANIA SL
        }
        
        self.positions[position_id] = position
        
        # Aktualizuj statystyki
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        self.stats['total_trades'] += 1
        
        if not self.real_trading:
            self.virtual_balance -= margin_required
        
        sl_method = "ATR-based DYNAMIC SL" if self.real_trading else "VIRTUAL SL"
        self.logger.info(f"🎯 OPENED: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   🛑 {sl_method}: ${exit_plan['stop_loss']:.4f}")
        self.logger.info(f"   ⏰ Max holding time: {exit_plan['max_holding_hours']}h")
        
        if exit_plan['partial_exits']:
            self.logger.info(f"   📈 Partial exits configured: {len(exit_plan['partial_exits'])} tiers")
        
        return position_id

    def update_positions_pnl(self):
        """POPRAWIONE aktualizowanie P&L Z NOWYMI FUNKCJAMI"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        # Synchronizuj z Bybit jeśli używamy real trading
        if self.real_trading:
            self.sync_all_positions_with_bybit()  # Używamy poprawionej synchronizacji
            
            # Pobierz unrealized P&L bezpośrednio z Bybit
            total_unrealized = self.get_bybit_unrealized_pnl()
            self.dashboard_data['unrealized_pnl'] = total_unrealized
            
            # Aktualizuj P&L dla każdej pozycji lokalnie
            for position in self.positions.values():
                if position['status'] == 'ACTIVE' and position.get('real_trading', False):
                    # Użyj P&L z Bybit lub oblicz lokalnie
                    current_price = position.get('current_price') or self.get_current_price(position['symbol'])
                    if current_price:
                        if position['side'] == 'LONG':
                            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                            unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                        else:
                            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                            unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                        
                        position['unrealized_pnl'] = unrealized_pnl
                        position['current_price'] = current_price
                        
                        # NOWE: Aktualizuj trailing stop i sprawdź partial exits
                        self.update_trailing_stop(position['symbol'], current_price)
                        self.check_partial_exits(position['symbol'], current_price)
                        
                        total_margin += position.get('margin', 0)
                        total_confidence += position.get('confidence', 0)
                        confidence_count += 1
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
                
                # NOWE: Aktualizuj trailing stop i sprawdź partial exits
                self.update_trailing_stop(position['symbol'], current_price)
                self.check_partial_exits(position['symbol'], current_price)
                
                total_unrealized += unrealized_pnl
                total_margin += position.get('margin', 0)
                total_confidence += position.get('confidence', 0)
                confidence_count += 1
            
            self.dashboard_data['unrealized_pnl'] = total_unrealized
        
        # Użyj rzeczywistego salda konta
        real_balance = self.get_account_balance()
        if real_balance is not None:
            account_value = real_balance + total_unrealized
            available_cash = real_balance
        else:
            account_value = self.virtual_capital + total_unrealized
            available_cash = self.virtual_balance
        
        self.dashboard_data['account_value'] = account_value
        self.dashboard_data['available_cash'] = available_cash
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        if self.virtual_capital > 0:
            self.stats['portfolio_utilization'] = total_margin / self.virtual_capital
        
        self.dashboard_data['last_update'] = datetime.now()

    def debug_api_connection(self):
        """Testuje połączenie z API BEZ składania prawdziwych zleceń"""
        
        self.logger.info("🔧 DEBUG API CONNECTION (SAFE MODE)")
        
        symbol = "BTCUSDT"
        price = self.get_current_price(symbol)
        if price:
            self.logger.info(f"✅ Price OK: {symbol} = ${price}")
        else:
            self.logger.error(f"❌ Price FAILED")
            return False
        
        balance = self.get_account_balance()
        if balance:
            self.logger.info(f"✅ Balance OK: ${balance:.2f}")
        else:
            self.logger.error(f"❌ Balance FAILED")
            return False
        
        leverage_ok = self.set_leverage(symbol, self.leverage)
        if leverage_ok:
            self.logger.info(f"✅ Leverage OK: {self.leverage}x")
        else:
            self.logger.warning(f"⚠️ Leverage may have failed")
        
        # ZMIENIONE: Tylko symulacja zlecenia, nie składaj prawdziwego!
        self.logger.info("🚀 TESTING ORDER PLACEMENT (SIMULATION ONLY)...")
        
        # Symuluj zlecenie bez rzeczywistego składania
        test_quantity = 0.001
        order_value = test_quantity * price
        
        if self.real_trading:
            self.logger.info(f"🎯 SIMULATED ORDER: {symbol} LONG Qty: {test_quantity}, Value: ${order_value:.2f}")
            self.logger.info("✅ ORDER TEST SIMULATION SUCCESS! (No real order placed)")
        else:
            self.logger.info(f"🎯 VIRTUAL ORDER: {symbol} LONG Qty: {test_quantity}")
            self.logger.info("✅ VIRTUAL ORDER TEST SUCCESS!")
        
        return True  # Zawsze zwracaj True w trybie safe

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji - NAPRAWIONA WERSJA"""
        positions_to_close = []
        
        for position_id, position in list(self.positions.items()):  # Użyj list() aby uniknąć modyfikacji podczas iteracji
            if position['status'] != 'ACTIVE':
                continue
                
            try:
                symbol = position['symbol']
                # ✅ ZAWSZE pobierz świeżą cenę dla dokładnych obliczeń
                current_price = self.get_current_price(symbol)
                if not current_price:
                    self.logger.warning(f"⚠️ Could not get current price for {symbol}")
                    continue
                    
                # ✅ ZAPISZ aktualną cenę w pozycji
                position['current_price'] = current_price
                
                exit_reason = None
                exit_plan = position.get('exit_plan', {})
                
                if not exit_plan:
                    self.logger.warning(f"⚠️ No exit plan for {position_id}, creating default")
                    exit_plan = self.calculate_llm_exit_plan(
                        position['entry_price'], 
                        position.get('confidence', 0.5), 
                        position['side']
                    )
                    position['exit_plan'] = exit_plan
                
                # ✅ POPRAWIONE WARUNKI WYJŚCIA - DODAJ DEBUG LOGS
                self.logger.info(f"🔍 CHECKING EXIT: {symbol} {position['side']} | Price: ${current_price:.4f} | TP: ${exit_plan['take_profit']:.4f} | SL: ${exit_plan['stop_loss']:.4f}")
                
                if position['side'] == 'LONG':
                    if current_price >= exit_plan['take_profit']:
                        exit_reason = "TAKE_PROFIT"
                        self.logger.info(f"🎯 TP HIT: {symbol} - Current: ${current_price:.4f} >= TP: ${exit_plan['take_profit']:.4f}")
                    elif current_price <= exit_plan['stop_loss']:
                        exit_reason = "STOP_LOSS"
                        self.logger.info(f"🎯 SL HIT: {symbol} - Current: ${current_price:.4f} <= SL: ${exit_plan['stop_loss']:.4f}")
                else:  # SHORT
                    if current_price <= exit_plan['take_profit']:
                        exit_reason = "TAKE_PROFIT"
                        self.logger.info(f"🎯 TP HIT: {symbol} - Current: ${current_price:.4f} <= TP: ${exit_plan['take_profit']:.4f}")
                    elif current_price >= exit_plan['stop_loss']:
                        exit_reason = "STOP_LOSS" 
                        self.logger.info(f"🎯 SL HIT: {symbol} - Current: ${current_price:.4f} >= SL: ${exit_plan['stop_loss']:.4f}")
                
                # ✅ WARUNEK CZASOWY
                holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
                max_holding = exit_plan.get('max_holding_hours', 6)
                
                if holding_time > max_holding:
                    exit_reason = "TIME_EXPIRED"
                    self.logger.info(f"🕐 TIME EXPIRED: {symbol} - {holding_time:.2f}h > {max_holding}h")
                
                if exit_reason:
                    positions_to_close.append((position_id, exit_reason, current_price))
                    self.logger.info(f"🎯 EXIT CONDITION MET: {symbol} - {exit_reason}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking exit conditions for {position_id}: {e}")
                continue
        
        return positions_to_close

    def debug_xrp_position(self):
        """Debuguje pozycję XRPUSDT i wymusza zamknięcie jeśli warunki są spełnione"""
        self.logger.info("🔧 DEBUG XRP POSITION...")
        
        # Znajdź aktywną pozycję XRPUSDT
        xrp_position = None
        position_id = None
        
        for pid, pos in self.positions.items():
            if pos['symbol'] == 'XRPUSDT' and pos['status'] == 'ACTIVE':
                xrp_position = pos
                position_id = pid
                break
        
        if not xrp_position:
            self.logger.info("❌ No active XRPUSDT position found")
            return
        
        # Pobierz aktualną cenę
        current_price = self.get_current_price('XRPUSDT')
        if not current_price:
            self.logger.error("❌ Could not get XRP price")
            return
        
        self.logger.info(f"📊 XRP POSITION DEBUG:")
        self.logger.info(f"   Side: {xrp_position['side']}")
        self.logger.info(f"   Entry: ${xrp_position['entry_price']:.4f}")
        self.logger.info(f"   Current: ${current_price:.4f}")
        
        exit_plan = xrp_position.get('exit_plan', {})
        if exit_plan:
            self.logger.info(f"   TP: ${exit_plan['take_profit']:.4f}")
            self.logger.info(f"   SL: ${exit_plan['stop_loss']:.4f}")
            
            # Sprawdź warunki ręcznie
            if xrp_position['side'] == 'SHORT':
                tp_distance_pct = ((current_price - exit_plan['take_profit']) / current_price) * 100
                self.logger.info(f"   TP Distance: {tp_distance_pct:.2f}%")
                
                if current_price <= exit_plan['take_profit']:
                    self.logger.info("🎯 MANUAL TP CLOSE: Closing XRP position due to TP condition")
                    self.close_position(position_id, "TAKE_PROFIT", current_price)
                    return True
        
        return False

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję - Z INFORMACJĄ O SL"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        if position.get('real_trading', False):
            success = self.close_bybit_position(position['symbol'], position['side'], position['quantity'])
            if not success:
                self.logger.error(f"❌ Failed to close position on Bybit: {position_id}")
                return
        
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
            'real_trading': position.get('real_trading', False),
            'partial_exits_taken': len(position.get('partial_exits_taken', [])),  # DODANE: liczba partial exits
            'sl_calculation_method': position.get('sl_calculation_method', 'Fixed')  # DODANE: metoda SL
        }
        
        self.trade_history.append(trade_record)
        
        # ✅ POPRAWIONE: Aktualizacja statystyk przy ZAMYKANIU pozycji
        # NIE inkrementujemy total_trades tutaj (już zrobione przy otwarciu)
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
            # ✅ DODANE: Aktualizacja wygranych long/short trades
            if position['side'] == "LONG":
                self.stats['won_long_trades'] += 1
            else:
                self.stats['won_short_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        total_holding = sum((t['exit_time'] - t['entry_time']).total_seconds() 
                          for t in self.trade_history) / 3600
        self.stats['avg_holding_time'] = total_holding / len(self.trade_history) if self.trade_history else 0
        
        position['status'] = 'CLOSED'
        
        # ✅ POPRAWIONE: Aktualizacja net realized P&L
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        margin_return = pnl_pct * self.leverage * 100
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        trading_mode = "REAL" if position.get('real_trading', False) else "VIRTUAL"
        sl_method = position.get('sl_calculation_method', 'Fixed')
        
        # ✅ DODANE: Logowanie statystyk z uwzględnieniem metody SL
        partial_exits_count = len(position.get('partial_exits_taken', []))
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        self.logger.info(f"{pnl_color} {trading_mode} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}% margin) - Reason: {exit_reason} ({sl_method} SL)")
        if partial_exits_count > 0:
            self.logger.info(f"   📈 Partial exits taken: {partial_exits_count}")
        self.logger.info(f"   📊 STATS UPDATE: Total Trades: {self.stats['total_trades']}, Win Rate: {win_rate:.1f}%, Net P&L: ${self.stats['total_pnl']:.2f}")

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

    def check_api_status(self) -> Dict:
        """Sprawdza status połączenia z Bybit API"""
        status = {
            'real_trading': self.real_trading,
            'api_connected': False,
            'balance_available': False,
            'testnet': self.testnet,
            'available_categories': [],
            'message': '',
            'balance': 0
        }
        
        if not self.real_trading:
            status['message'] = '🔄 Tryb wirtualny - brak kluczy API'
            status['balance'] = self.virtual_balance
            return status
        
        try:
            available_categories = self.check_available_categories()
            status['available_categories'] = available_categories
            
            balance = self.get_account_balance()
            
            if balance is not None:
                status['api_connected'] = True
                status['balance_available'] = True
                status['balance'] = balance
                status['message'] = f'✅ Połączono z Bybit API'
            else:
                status['api_connected'] = False
                status['message'] = '❌ Błąd połączenia z Bybit - sprawdź klucze API'
                    
        except Exception as e:
            status['api_connected'] = False
            status['message'] = f'❌ Błąd API: {str(e)}'
        
        return status

    def get_dashboard_data(self):
        """POPRAWIONE przygotowywanie danych dla dashboardu Z INFORMACJĄ O METODZIE SL"""
        self.logger.info("🔄 Generating dashboard data...")
        
        api_status = self.check_api_status()
        
        # ZAWSZE synchronizuj z Bybit przed pobraniem danych
        if self.real_trading:
            self.sync_all_positions_with_bybit()
        
        # Pobierz unrealized P&L
        total_unrealized_pnl = self.get_bybit_unrealized_pnl()
        self.dashboard_data['unrealized_pnl'] = total_unrealized_pnl
        
        # Przygotuj aktywne pozycje
        active_positions = []
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price') or self.get_current_price(position['symbol'])
                if not current_price:
                    continue
                
                unrealized_pnl = position.get('unrealized_pnl', 0)
                
                exit_plan = position.get('exit_plan', self.calculate_llm_exit_plan(
                    position['entry_price'], 
                    position.get('confidence', 0.5), 
                    position['side']
                ))
                
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
                    'llm_profile': position.get('llm_profile', self.active_profile),
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'exit_plan': exit_plan,
                    'tp_distance_pct': round(tp_distance_pct, 2),
                    'sl_distance_pct': round(sl_distance_pct, 2),
                    'real_trading': position.get('real_trading', False),
                    'partial_exits_taken': len(position.get('partial_exits_taken', [])),  # DODANE: partial exits
                    'use_trailing_stop': exit_plan.get('use_trailing_stop', False),  # DODANE: trailing stop info
                    'bybit_sl_set': position.get('bybit_sl_set', False),
                    'sl_type': 'BYBIT' if position.get('bybit_sl_set') else 'VIRTUAL',
                    'sl_calculation': position.get('sl_calculation_method', 'Fixed')  # ✅ DODAJ METODĘ OBLICZANIA
                })
        
        self.logger.info(f"📊 DASHBOARD: {len(active_positions)} active positions to display")
        
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
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'llm_profile': trade['llm_profile'],
                'holding_hours': round(trade['holding_hours'], 2),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'real_trading': trade.get('real_trading', False),
                'partial_exits': trade.get('partial_exits_taken', 0),  # DODANE: partial exits
                'sl_calculation_method': trade.get('sl_calculation_method', 'Fixed')  # DODANE: metoda SL
            })
        
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
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
                'avg_confidence': round(self.dashboard_data['average_confidence'] * 100, 1),
                'won_long_trades': self.stats.get('won_long_trades', 0),
                'won_short_trades': self.stats.get('won_short_trades', 0)
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
        """Główna pętla strategii LLM - NAPRAWIONA"""
        self.logger.info("🚀 STARTING LLM TRADING STRATEGY (FIXED)")
        
        # Debuguj istniejące pozycje
        self.debug_xrp_position()
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 LLM Trading Iteration #{iteration}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia (NAPRAWIONE)
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.logger.info(f"🔔 CLOSING POSITION: {position_id} - {exit_reason}")
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Otwieraj nowe pozycje tylko jeśli mamy miejsce
                active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.assets:
                        try:
                            position_id = self.open_llm_position(symbol)
                            if position_id:
                                time.sleep(2)  # Daj czas między zleceniami
                        except Exception as e:
                            self.logger.error(f"❌ Error opening position for {symbol}: {e}")
                            continue
                
                # 4. Loguj status
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                # 5. Czekaj przed następną iteracją
                wait_time = random.randint(45, 120)
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in LLM trading loop: {e}")
                import traceback
                self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
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
    print("🚀 Starting LLM Trading Bot Server...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 LLM Profiles: Claude, Gemini, GPT, Qwen")
    print("📈 Trading assets: BTC, ETH, SOL, XRP, BNB, DOGE")
    print("💹 Using REAL-TIME prices from Bybit API only")
    print("🎯 Qwen Profile Features: Extended holding periods, Tiered exits, Volatility-based TP/SL")
    print("🛑 REAL TRADING: ATR-based Dynamic SL + Trailing Stop on Bybit")
    app.run(debug=True, host='0.0.0.0', port=5000)
