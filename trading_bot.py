import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading
import random

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_live.log', encoding='utf-8')
    ]
)

class TradingBot:
    def __init__(self, initial_capital=50000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # PARAMETRY Z TWOJEJ STRATEGII
        self.max_simultaneous_positions = 6
        self.position_size_percentage = 0.05  # 5% depozytu na pozycję (MAX z wymagań)
        
        # WIELKOŚCI POZYCJI Z TWOJEJ HISTORII
        self.position_sizes = {
            'BTCUSDT': 0.12,
            'ETHUSDT': 3.2968,
            'BNBUSDT': 7.036,
            'XRPUSDT': 1737.0,
            'DOGEUSDT': 27858.0,
            'SOLUSDT': 20.76,
            'ADAUSDT': 5000.0,
            'DOTUSDT': 200.0
        }
        
        # PRIORYTETOWE KRYPTOWALUTY
        self.priority_symbols = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'BTCUSDT', 'DOGEUSDT']
        
        # Statystyki
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0
        }
        
        # Dashboard
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': 10,
            'average_confidence': 0,
            'account_history': [initial_capital],
            'last_update': datetime.now()
        }
        
        self.logger.info("🎯 TRADING BOT - Based on your exact positions")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")

    def calculate_sma(self, data, window):
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, window):
        return data.ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def get_binance_klines(self, symbol: str, interval: str = '3m', limit: int = 100):
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str):
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame):
        try:
            df['sma_20'] = self.calculate_sma(df['close'], 20)
            df['sma_50'] = self.calculate_sma(df['close'], 50)
            df['ema_12'] = self.calculate_ema(df['close'], 12)
            df['ema_26'] = self.calculate_ema(df['close'], 26)
            df['rsi_14'] = self.calculate_rsi(df['close'], 14)
            
            macd, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            
            df['volume_ma_20'] = self.calculate_sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            
            df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
            df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating advanced indicators: {e}")
            return df

    def generate_signal(self, symbol: str):
        df = self.get_binance_klines(symbol, '3m', 100)
        if df is None or len(df) < 50:
            return "HOLD", 0
        
        df = self.calculate_advanced_indicators(df)
        
        try:
            current_rsi = df['rsi_14'].iloc[-1]
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            ema_12 = df['ema_12'].iloc[-1]
            ema_26 = df['ema_26'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1
            macd_histogram = df['macd_histogram'].iloc[-1]
            price_vs_sma20 = df['price_vs_sma20'].iloc[-1]

            confidence = 0.0
            signal = "HOLD"
            
            long_conditions = 0
            total_conditions = 6
            
            if 30 <= current_rsi <= 65:
                long_conditions += 1
                confidence += 0.15
            
            if volume_ratio > 1.2:
                long_conditions += 1
                confidence += 0.15
            
            if current_price > sma_20:
                long_conditions += 1
                confidence += 0.15
            
            if ema_12 > ema_26:
                long_conditions += 1
                confidence += 0.15
            
            if macd_histogram > 0:
                long_conditions += 1
                confidence += 0.20
            
            if 0.5 <= price_vs_sma20 <= 3.0:
                long_conditions += 1
                confidence += 0.20
            
            if long_conditions >= 4:
                signal = "LONG"
                confidence = min(confidence + (long_conditions - 4) * 0.1, 0.95)
                
                if confidence >= 0.70:
                    self.logger.info(f"🎯 STRONG LONG signal for {symbol} - Confidence: {confidence:.1%}")
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return "HOLD", 0

    def calculate_position_size(self, symbol: str, price: float, confidence: float):
        base_quantity = self.position_sizes.get(symbol, 1000.0)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        position_value = (self.virtual_capital * self.position_size_percentage) * confidence_multiplier
        adjusted_quantity = position_value / price
        
        final_quantity = min(base_quantity, adjusted_quantity)
        
        position_value = final_quantity * price
        margin_required = position_value / self.leverage
        
        return final_quantity, position_value, margin_required

    def calculate_exit_levels(self, entry_price: float):
        take_profit = entry_price * 1.15
        stop_loss = entry_price * 0.95
        invalidation = entry_price * 0.9375
        
        return take_profit, stop_loss, invalidation

    def open_position(self, symbol: str, side: str):
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_signal(symbol)
        if signal != "LONG" or confidence < 0.65:
            return None
        
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}. Required: ${margin_required:.2f}, Available: ${self.virtual_balance:.2f}")
            return None
        
        take_profit, stop_loss, invalidation = self.calculate_exit_levels(current_price)
        liquidation_price = current_price * (1 - 0.9 / self.leverage)
        
        position_id = f"pos_{self.position_id}"
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
            'unrealized_pnl_pct': 0,
            'current_price': current_price,
            'confidence': confidence,
            'exit_plan': {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'invalidation': invalidation
            }
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        self.logger.info(f"🎯 OPEN POSITION: {side} {quantity:.4f} {symbol} @ ${current_price:.2f}")
        self.logger.info(f"   📊 TP: ${take_profit:.2f} | SL: ${stop_loss:.2f} | Margin: ${margin_required:.2f}")
        self.logger.info(f"   💪 Confidence: {confidence:.1%} | Leverage: {self.leverage}X")
        
        return position_id

    def update_positions_pnl(self):
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
            position['unrealized_pnl_pct'] = pnl_pct * 100
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        if total_margin > 0:
            total_position_value = sum(
                p['quantity'] * p['entry_price'] 
                for p in self.positions.values() 
                if p['status'] == 'ACTIVE'
            )
            self.dashboard_data['average_leverage'] = round(total_position_value / (total_margin + 0.001), 1)
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            exit_reason = None
            
            if position['side'] == 'LONG':
                if current_price >= position['exit_plan']['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= position['exit_plan']['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price <= position['exit_plan']['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price <= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
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
            'pnl_pct': pnl_pct * 100,
            'exit_reason': exit_reason,
            'fee': fee,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'confidence': position.get('confidence', 0)
        }
        
        self.trade_history.append(trade_record)
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        self.stats['total_fees'] += fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
            if realized_pnl_after_fee > self.stats['biggest_win']:
                self.stats['biggest_win'] = realized_pnl_after_fee
        else:
            self.stats['losing_trades'] += 1
            if realized_pnl_after_fee < self.stats['biggest_loss']:
                self.stats['biggest_loss'] = realized_pnl_after_fee
        
        position['status'] = 'CLOSED'
        
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        self.dashboard_data['total_fees'] = self.stats['total_fees']
        
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSED: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} ({pnl_pct:+.2%}) - Reason: {exit_reason}")

    def get_performance_metrics(self):
        total_trades = self.stats['total_trades']
        winning_trades = self.stats['winning_trades']
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade_pnl = (self.stats['total_pnl'] / total_trades) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': self.stats['losing_trades'],
            'win_rate': round(win_rate, 1),
            'total_pnl': round(self.stats['total_pnl'], 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'biggest_win': round(self.stats['biggest_win'], 2),
            'biggest_loss': round(self.stats['biggest_loss'], 2),
            'total_fees': round(self.stats['total_fees'], 2)
        }

    def get_dashboard_data(self):
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
        for position in self.positions.values():
            if position['status'] == 'ACTIVE':
                active_positions.append({
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'entry_price': position['entry_price'],
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'margin': position['margin'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'symbol': position['symbol'],
                    'confidence': position.get('confidence', 0)
                })
        
        for symbol in self.priority_symbols:
            signal, confidence = self.generate_signal(symbol)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = round((total_confidence / confidence_count) * 100, 1)
        
        recent_trades = []
        for trade in self.trade_history[-25:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'confidence': trade.get('confidence', 0)
            })
        
        if len(self.dashboard_data['account_history']) > 50:
            self.dashboard_data['account_history'].pop(0)
        self.dashboard_data['account_history'].append(self.dashboard_data['account_value'])
        
        performance = self.get_performance_metrics()
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'total_fees': round(self.dashboard_data['total_fees'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2)
            },
            'performance_metrics': {
                'avg_leverage': self.dashboard_data['average_leverage'],
                'avg_confidence': self.dashboard_data['average_confidence'],
                'biggest_win': performance['biggest_win'],
                'biggest_loss': performance['biggest_loss']
            },
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_strategy(self):
        self.logger.info("🚀 STARTING TRADING STRATEGY...")
        self.logger.info("📊 Based on your exact trading patterns")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 Iteration #{iteration} | {current_time}")
                
                self.update_positions_pnl()
                
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            signal, confidence = self.generate_signal(symbol)
                            
                            if signal == "LONG" and confidence >= 0.65:
                                self.logger.info(f"🎯 STRONG SIGNAL: {symbol} - Confidence: {confidence:.1%}")
                                position_id = self.open_position(symbol, "LONG")
                                if position_id:
                                    time.sleep(2)
                
                performance = self.get_performance_metrics()
                self.logger.info(f"📊 Status: {active_count} active positions | Account: ${self.dashboard_data['account_value']:.2f}")
                
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        self.is_running = True
        self.run_strategy()

    def stop_trading(self):
        self.is_running = False
        self.logger.info("🛑 Trading stopped")

# Globalna instancja bota
trading_bot = TradingBot(initial_capital=50000, leverage=10)
