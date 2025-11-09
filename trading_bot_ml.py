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
import hmac
import hashlib
import base64
from urllib.parse import urlencode # Dodane do obsługi parametrów GET

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
        self.recv_window = 5000 # Czas oczekiwania na odpowiedź w ms (dla V5)
        self.category = 'linear' # Dla kontraktów USDT/USDC futures (Linear Perpetual)

        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        
        if not self.api_key or not self.api_secret:
            logging.warning("⚠️ Brak kluczy API Bybit - bot będzie działał w trybie wirtualnym")
            self.virtual_mode = True
        else:
            self.virtual_mode = False
            
        # Pozostałe inicjalizacje...
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self.assets = ['BTCUSDT', 'ETHUSDT']
        self.max_simultaneous_positions = 2
        self.positions: Dict[str, Dict] = {}
        
        # Wirtualne dane dla dashboardu
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital * 0.7,
            'total_fees': 0.0,
            'net_realized': 0.0,
            'performance': {},
            'open_positions': []
        }
        
    # ====================================================================
    #           BYBIT V5 API - FUNKCJE POMOCNICZE
    # ====================================================================

    def get_server_time(self) -> int:
        """Pobiera czas serwera Bybit w milisekundach."""
        try:
            # Używamy prostego endpointu bez podpisu
            response = requests.get(self.base_url + "/v3/public/time", timeout=5)
            response.raise_for_status()
            data = response.json()
            return int(data['result']['time_now']) * 1000 # Czas w sekundach, mnożymy by uzyskać ms
        except Exception as e:
            self.logger.error(f"❌ Nie udało się pobrać czasu serwera: {e}")
            return int(time.time() * 1000)

    def generate_signature(self, params: Dict[str, any], timestamp: int) -> Dict[str, str]:
        """Generuje nagłówki i podpis HMAC-SHA256 dla Bybit API V5."""
        
        # Ciąg do podpisania
        if params:
            # Dla POST i GET w V5: timestamp + api_key + recv_window + (query_string dla GET / json_string dla POST)
            param_str = json.dumps(params)
        else:
            param_str = ""
        
        param_str_to_sign = str(timestamp) + self.api_key + str(self.recv_window) + param_str
        
        # Podpis (HMAC-SHA256)
        hash_value = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str_to_sign.encode('utf-8'),
            hashlib.sha256
        )
        signature = hash_value.hexdigest()
        
        # Nagłówki
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': str(timestamp),
            'X-BAPI-RECV-WINDOW': str(self.recv_window),
            'Content-Type': 'application/json'
        }
        
        return headers

    def _send_request(self, method: str, path: str, params: Optional[Dict[str, any]] = None) -> Optional[Dict[str, any]]:
        """Wysyła podpisane żądanie do Bybit API V5."""
        if self.virtual_mode:
            self.logger.warning(f"⚠️ Żądanie do Bybit '{path}' pominięte: tryb wirtualny.")
            return None

        url = self.base_url + path
        timestamp = self.get_server_time()
        
        # Dla GET parametry są w URL, ale dla podpisu używamy pustego stringa w V5 (jeśli nie ma parametrów w ciele)
        # BARDZO WAŻNE: W V5 API, jeśli params to GET, query string jest częścią ciągu do podpisania. 
        # Aby uprościć i użyć najczęściej działającej metody: dla POST ciało JSON, dla GET nie ma ciała, params idą jako query string.
        
        if method == "GET":
            # Dla GET, params idą w query string, a w ciągu do podpisu jest query string
            # Najprostszy sposób to użycie `urlencode` i ręczne stworzenie ciągu do podpisu.
            query_string = urlencode(params) if params else ""
            param_str_to_sign = str(timestamp) + self.api_key + str(self.recv_window) + query_string
            
            hash_value = hmac.new(self.api_secret.encode('utf-8'), param_str_to_sign.encode('utf-8'), hashlib.sha256)
            signature = hash_value.hexdigest()
            
            headers = {
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': str(timestamp),
                'X-BAPI-RECV-WINDOW': str(self.recv_window),
                # Brak Content-Type dla GET
            }
            
        elif method == "POST":
            # Dla POST, params idą w ciele JSON, a w ciągu do podpisu jest string JSON
            headers = self.generate_signature(params or {}, timestamp)
        else:
            raise ValueError("Niewspierana metoda HTTP.")

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=params, timeout=10)
            else:
                return None

            response.raise_for_status()
            data = response.json()
            
            if data.get('retCode') != 0:
                self.logger.error(f"❌ Bybit API Error: Code {data.get('retCode')}, Msg: {data.get('retMsg')}, Path: {path}, Params: {params}")
                return None
            
            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Błąd zapytania HTTP do Bybit: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Ogólny błąd przy wysyłaniu żądania: {e}")
            return None
            
    # ====================================================================
    #           LOGIKA TRADINGOWA
    # ====================================================================

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Ustawia leverage dla danego symbolu."""
        self.logger.info(f"⚙️ Ustawianie lewarowania ({leverage}x) dla {symbol}...")
        
        path = "/v5/position/set-leverage"
        
        params = {
            "category": self.category,
            "symbol": symbol,
            "buyLeverage": str(leverage), 
            "sellLeverage": str(leverage),
        }
        
        response = self._send_request("POST", path, params)
        
        if response and response.get('retCode') == 0:
            self.logger.info(f"✅ Pomyślnie ustawiono lewarowanie {leverage}x dla {symbol}.")
            return True
        else:
            # Błąd może być ignorowany, jeśli lewarowanie jest już ustawione.
            # Ważne jest, że możemy kontynuować.
            return True

    def open_llm_position(self, symbol: str) -> Optional[str]:
        """
        Otwiera pozycję Market order na danym symbolu. 
        Wymaga, aby model LLM dostarczył 'direction' ('BUY' lub 'SELL').
        """
        self.logger.info(f"✨ Otwieranie pozycji LLM dla {symbol}...")
        
        if self.virtual_mode:
            # Logika wirtualna pozostaje bez zmian
            position_id = f"VIRTUAL_{int(time.time() * 1000)}"
            self.positions[position_id] = {
                'symbol': symbol, 'side': 'Buy', 'qty': 0.001, 
                'entry_price': 30000, 'status': 'ACTIVE', 'open_time': datetime.now()
            }
            self.logger.info(f"✨ Tryb wirtualny: symulacja otwarcia pozycji. ID: {position_id}")
            return position_id
        
        # Krok 1: Ustawienie lewarowania
        if not self.set_leverage(symbol, self.leverage):
             self.logger.error(f"❌ Niepowodzenie w ustawieniu lewarowania dla {symbol}.")
             return None
        
        # Krok 2: Określenie kierunku i ilości (PRZYKŁAD: W tym miejscu wstawisz logikę LLM)
        # W TYM MIEJSCU WSTAW LOGIKĘ LLM/ML
        direction = "BUY" # PRZYKŁAD: Ustawiamy na Long (Kupno)
        side = "Buy" if direction == "BUY" else "Sell"
        
        # Należy określić poprawną ilość (qty) na podstawie aktualnej ceny rynkowej i ryzyka.
        # W tym przykładzie, używamy stałej, małej ilości (np. 0.001 BTC)
        order_qty = 0.001 
        
        # Krok 3: Wysłanie zlecenia Market
        path = "/v5/order/create"
        params = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(order_qty),
            "isLeverage": 1, 
            "timeInForce": "GTC",
            "positionIdx": 0 # Tryb One-Way (domyślny)
        }
        
        response = self._send_request("POST", path, params)
        
        if response and response.get('retCode') == 0:
            order_id = response['result']['orderId']
            self.logger.info(f"✅ Pomyślnie złożono zlecenie Market {side} dla {symbol}. ID Zlecenia: {order_id}")
            return order_id
        else:
            self.logger.error(f"❌ Nie udało się otworzyć pozycji dla {symbol}. Odpowiedź: {response}")
            return None

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, any]]:
        """Pobiera listę aktywnych pozycji z Bybit."""
        if self.virtual_mode:
            return [p for p in self.positions.values() if p['status'] == 'ACTIVE' and (not symbol or p['symbol'] == symbol)]

        path = "/v5/position/list"
        params = {"category": self.category}
        if symbol:
            params['symbol'] = symbol

        response = self._send_request("GET", path, params)
        
        if response and response.get('retCode') == 0:
            # Zwracamy tylko aktywne pozycje (size > 0)
            active_positions = [
                pos for pos in response['result']['list'] 
                if float(pos.get('size', 0)) > 0
            ]
            return active_positions
        else:
            return []

    def close_position(self, symbol: str) -> bool:
        """Zamyka aktywną pozycję Market order dla danego symbolu."""
        self.logger.info(f"🛑 Próba zamknięcia pozycji dla {symbol}...")

        if self.virtual_mode:
            # Logika wirtualna pozostaje bez zmian
            for pos_id, pos in self.positions.items():
                if pos['symbol'] == symbol and pos['status'] == 'ACTIVE':
                    pos['status'] = 'CLOSED'
                    self.logger.info(f"✅ Pozycja wirtualna dla {symbol} zamknięta.")
                    return True
            return False

        positions = self.get_open_positions(symbol=symbol)
        
        if not positions:
            self.logger.warning(f"⚠️ Brak otwartej pozycji dla symbolu {symbol} do zamknięcia.")
            return False

        for pos in positions:
            position_size = pos['size']
            position_side = pos['side'] # "Buy" dla long, "Sell" dla short
            
            # Strona zamknięcia musi być przeciwna do strony pozycji
            closing_side = "Sell" if position_side == "Buy" else "Buy" 
            
            self.logger.info(f"Zamykanie: Wielkość: {position_size}, Strona pozycji: {position_side}, Strona zamknięcia: {closing_side}")

            path = "/v5/order/create"
            params = {
                "category": self.category,
                "symbol": symbol,
                "side": closing_side, # Strona zamykająca
                "orderType": "Market",
                "qty": position_size, # Wielkość musi być równa rozmiarowi pozycji
                "timeInForce": "IOC", # Immediate Or Cancel
                "positionIdx": 0 
            }
            
            response = self._send_request("POST", path, params)
            
            if response and response.get('retCode') == 0:
                self.logger.info(f"✅ Pomyślnie wysłano zlecenie zamknięcia Market dla {symbol}. ID Zlecenia: {response['result']['orderId']}")
                return True
            else:
                self.logger.error(f"❌ Nie udało się zamknąć pozycji dla {symbol}. Odpowiedź: {response}")
                return False
        
        return False

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
