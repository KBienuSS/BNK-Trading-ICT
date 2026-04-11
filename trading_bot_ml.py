import os
import time
import json
import hmac
import hashlib
import logging
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series,
                    period: int = 20, std_dev: float = 2.0
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()


class SignalEngine:
    LONG_THRESHOLD  = 0.40
    SHORT_THRESHOLD = 0.40

    WEIGHTS = {
        "ema_cross":   0.30,
        "macd":        0.25,
        "rsi":         0.20,
        "bb":          0.15,
        "volume":      0.10,
    }

    @staticmethod
    def score(df: pd.DataFrame) -> Tuple[str, float, Dict]:
        if len(df) < 60:
            return "HOLD", 0.0, {}

        close   = df["close"]
        high    = df["high"]
        low     = df["low"]
        volume  = df["volume"]

        ema20 = ema(close, 20)
        ema50 = ema(close, 50)

        rsi14 = rsi(close, 14)

        macd_line, signal_line, histogram = macd(close)

        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)

        vol_ma20 = volume_ma(volume, 20)

        atr14 = atr(high, low, close, 14)

        c0  = close.iloc[-1]
        c1  = close.iloc[-2]
        e20 = ema20.iloc[-1]
        e50 = ema50.iloc[-1]
        e20_prev = ema20.iloc[-2]
        e50_prev = ema50.iloc[-2]

        r0  = rsi14.iloc[-1]
        r1  = rsi14.iloc[-2]

        m0  = macd_line.iloc[-1]
        s0  = signal_line.iloc[-1]
        m1  = macd_line.iloc[-2]
        s1  = signal_line.iloc[-2]
        h0  = histogram.iloc[-1]
        h1  = histogram.iloc[-2]

        bb_u = bb_upper.iloc[-1]
        bb_m = bb_mid.iloc[-1]
        bb_l = bb_lower.iloc[-1]

        vol0    = volume.iloc[-1]
        vol_avg = vol_ma20.iloc[-1]

        atr_val = atr14.iloc[-1]

        if e20 > e50 and e20_prev <= e50_prev:
            ema_vote = 1.0
        elif e20 < e50 and e20_prev >= e50_prev:
            ema_vote = -1.0
        elif e20 > e50 and c0 > e20:
            ema_vote = 0.7
        elif e20 < e50 and c0 < e20:
            ema_vote = -0.7
        else:
            ema_vote = 0.0

        if h0 > 0 and h1 <= 0:
            macd_vote = 1.0
        elif h0 < 0 and h1 >= 0:
            macd_vote = -1.0
        elif h0 > 0 and h0 > h1:
            macd_vote = 0.5
        elif h0 < 0 and h0 < h1:
            macd_vote = -0.5
        else:
            macd_vote = 0.0

        if r0 < 35 and r0 > r1:
            rsi_vote = 1.0
        elif r0 > 65 and r0 < r1:
            rsi_vote = -1.0
        elif 40 <= r0 <= 60:
            rsi_vote = 0.0
        elif r0 > 60:
            rsi_vote = -0.3
        else:
            rsi_vote = 0.3

        bb_width = (bb_u - bb_l) / bb_m
        if c0 <= bb_l and c1 > bb_l:
            bb_vote = 1.0
        elif c0 >= bb_u and c1 < bb_u:
            bb_vote = -1.0
        elif c0 < bb_m and c0 > bb_l:
            bb_vote = 0.3
        elif c0 > bb_m and c0 < bb_u:
            bb_vote = -0.3
        else:
            bb_vote = 0.0

        if vol_avg and vol_avg > 0:
            vol_ratio = vol0 / vol_avg
        else:
            vol_ratio = 1.0

        if vol_ratio > 1.5:
            price_move = c0 - c1
            if price_move > 0:
                vol_vote = 1.0
            elif price_move < 0:
                vol_vote = -1.0
            else:
                vol_vote = 0.0
        elif vol_ratio > 1.0:
            vol_vote = 0.3 if (c0 > c1) else -0.3
        else:
            vol_vote = 0.0

        w = SignalEngine.WEIGHTS
        composite = (
            w["ema_cross"] * ema_vote
            + w["macd"]    * macd_vote
            + w["rsi"]     * rsi_vote
            + w["bb"]      * bb_vote
            + w["volume"]  * vol_vote
        )
        total_weight = sum(w.values())
        score = composite / total_weight

        if score >= SignalEngine.LONG_THRESHOLD:
            signal = "LONG"
        elif score <= -SignalEngine.SHORT_THRESHOLD:
            signal = "SHORT"
        else:
            signal = "HOLD"

        if signal == "LONG":
            confidence = min(0.95, 0.5 + (score - SignalEngine.LONG_THRESHOLD))
        elif signal == "SHORT":
            confidence = min(0.95, 0.5 + (abs(score) - SignalEngine.SHORT_THRESHOLD))
        else:
            confidence = 0.0

        details = {
            "score":       round(score, 4),
            "ema_vote":    round(ema_vote, 2),
            "macd_vote":   round(macd_vote, 2),
            "rsi_vote":    round(rsi_vote, 2),
            "bb_vote":     round(bb_vote, 2),
            "vol_vote":    round(vol_vote, 2),
            "rsi_value":   round(r0, 1),
            "atr":         round(atr_val, 6),
            "vol_ratio":   round(vol_ratio, 2),
            "ema20":       round(e20, 4),
            "ema50":       round(e50, 4),
            "bb_upper":    round(bb_u, 4),
            "bb_lower":    round(bb_l, 4),
        }

        return signal, round(confidence, 4), details


class LLMTradingBot:
    ASSETS           = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
    KLINE_INTERVAL   = "60"
    KLINE_LIMIT      = 100
    LOOP_INTERVAL    = 300

    RISK_PER_TRADE   = 0.015
    MAX_POSITIONS    = 4
    LEVERAGE         = 10
    MIN_CONFIDENCE   = 0.45

    TP_ATR_MULT      = 3.0
    SL_ATR_MULT      = 1.5
    MAX_HOLD_HOURS   = 48

    FEE_RATE         = 0.00055

    BINANCE_BASE     = "https://api.binance.com/api/v3"

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 initial_capital: float = 10_000,
                 leverage: int = None):

        self.logger = logging.getLogger(__name__)

        self.api_key    = api_key    or os.getenv("BYBIT_API_KEY",    "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")

        if leverage is not None:
            self.LEVERAGE = leverage

        self.real_trading = bool(self.api_key and self.api_secret)
        self.testnet      = False

        self.session: Optional[HTTP] = None
        if self.real_trading and PYBIT_AVAILABLE:
            try:
                self.session = HTTP(
                    testnet    = self.testnet,
                    api_key    = self.api_key,
                    api_secret = self.api_secret,
                )
                self.logger.info("✅ Bybit session initialised")
            except Exception as exc:
                self.logger.error(f"❌ Bybit session error: {exc}")

        self.initial_capital  = initial_capital
        self.virtual_capital  = initial_capital
        self.virtual_balance  = initial_capital

        self.positions:     Dict[str, dict] = {}
        self.trade_history: List[dict]      = []
        self.position_id    = 0
        self.is_running     = False

        self.price_cache:   Dict[str, dict] = {}
        self.kline_cache:   Dict[str, dict] = {}
        self.signal_cache:  Dict[str, dict] = {}

        self.stats = {
            "total_trades":   0,
            "winning_trades": 0,
            "losing_trades":  0,
            "total_pnl":      0.0,
            "total_fees":     0.0,
            "long_trades":    0,
            "short_trades":   0,
            "avg_hold_hours": 0.0,
        }

        self.dashboard_data = {
            "account_value":      initial_capital,
            "available_cash":     initial_capital,
            "total_fees":         0.0,
            "net_realized":       0.0,
            "unrealized_pnl":     0.0,
            "last_update":        datetime.now(),
            "active_profile":     "SwingStrategy",
        }
        self.chart_data = {"labels": [], "values": []}

        self.active_profile = "SwingStrategy"
        self.llm_profiles   = {
            "SwingStrategy": {
                "risk_appetite":    "MEDIUM",
                "confidence_bias":  0.65,
                "short_frequency":  0.35,
                "holding_bias":     "NEUTRAL",
                "trade_frequency":  "MEDIUM",
                "position_sizing":  "CONSERVATIVE",
            }
        }
        self.assets = self.ASSETS

        self.logger.info("🚀 Swing Trading Bot initialised")
        self.logger.info(f"   Assets   : {', '.join(self.assets)}")
        self.logger.info(f"   Risk/trade: {self.RISK_PER_TRADE*100:.1f}%")
        self.logger.info(f"   Leverage  : {self.LEVERAGE}x")
        self.logger.info(f"   Real mode : {self.real_trading}")

    def get_klines(self, symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
        now = datetime.now()
        cached = self.kline_cache.get(symbol)
        if not force and cached:
            age = (now - cached["timestamp"]).total_seconds()
            if age < 240:
                return cached["df"]

        try:
            url    = f"{self.BINANCE_BASE}/klines"
            params = {
                "symbol":   symbol,
                "interval": "1h",
                "limit":    self.KLINE_LIMIT,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_base",
                "taker_quote", "ignore",
            ])
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)

            self.kline_cache[symbol] = {"df": df, "timestamp": now}
            return df

        except Exception as exc:
            self.logger.error(f"❌ klines fetch error for {symbol}: {exc}")
            if cached:
                return cached["df"]
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            resp = requests.get(
                f"{self.BINANCE_BASE}/ticker/price",
                params={"symbol": symbol},
                timeout=8,
            )
            resp.raise_for_status()
            price = float(resp.json()["price"])
            self.price_cache[symbol] = {"price": price, "timestamp": datetime.now()}
            return price
        except Exception:
            cached = self.price_cache.get(symbol)
            if cached:
                age = (datetime.now() - cached["timestamp"]).total_seconds()
                if age < 120:
                    return cached["price"]
            return None

    def get_binance_price(self, symbol: str) -> Optional[float]:
        return self.get_current_price(symbol)

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        now = datetime.now()
        cached = self.signal_cache.get(symbol)
        if cached:
            age = (now - cached["timestamp"]).total_seconds()
            if age < 600:
                return cached["signal"], cached["confidence"]

        df = self.get_klines(symbol)
        if df is None or len(df) < 60:
            self.logger.warning(f"⚠️ Not enough data for {symbol}")
            return "HOLD", 0.0

        signal, confidence, details = SignalEngine.score(df)

        self.signal_cache[symbol] = {
            "signal":     signal,
            "confidence": confidence,
            "details":    details,
            "timestamp":  now,
        }

        price = self.get_current_price(symbol) or 0
        self.logger.info(
            f"📊 {symbol}: {signal} conf={confidence:.2f} score={details.get('score',0):.3f} "
            f"RSI={details.get('rsi_value',0):.1f} volR={details.get('vol_ratio',0):.2f} "
            f"price=${price:.4f}"
        )
        return signal, confidence

    def get_signal_details(self, symbol: str) -> dict:
        return self.signal_cache.get(symbol, {}).get("details", {})

    def calculate_position_size(
        self,
        symbol:     str,
        price:      float,
        confidence: float,
    ) -> Tuple[float, float, float]:
        df = self.get_klines(symbol)

        if df is not None and len(df) >= 20:
            atr_series = atr(df["high"], df["low"], df["close"], 14)
            atr_val    = float(atr_series.iloc[-1])
        else:
            atr_val = price * 0.01

        stop_distance = self.SL_ATR_MULT * atr_val

        equity = self.get_account_balance() or self.virtual_balance

        confidence_scalar = 0.7 + (confidence - self.MIN_CONFIDENCE) * 1.0
        confidence_scalar = max(0.5, min(1.5, confidence_scalar))

        risk_budget = equity * self.RISK_PER_TRADE * confidence_scalar

        if stop_distance <= 0 or price <= 0:
            return 0.0, 0.0, 0.0

        quantity      = risk_budget / stop_distance
        position_value = quantity * price
        margin_required = position_value / self.LEVERAGE

        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(
        self,
        entry_price: float,
        confidence:  float,
        side:        str,
        symbol:      str = "",
    ) -> dict:
        df = self.get_klines(symbol) if symbol else None

        if df is not None and len(df) >= 20:
            atr_series = atr(df["high"], df["low"], df["close"], 14)
            atr_val    = float(atr_series.iloc[-1])
        else:
            atr_val = entry_price * 0.012

        sl_dist = self.SL_ATR_MULT * atr_val
        tp_dist = self.TP_ATR_MULT * atr_val

        if side == "LONG":
            take_profit = entry_price + tp_dist
            stop_loss   = entry_price - sl_dist
            invalidation = entry_price - sl_dist * 1.2
        else:
            take_profit = entry_price - tp_dist
            stop_loss   = entry_price + sl_dist
            invalidation = entry_price + sl_dist * 1.2

        return {
            "take_profit":    round(take_profit, 6),
            "stop_loss":      round(stop_loss, 6),
            "invalidation":   round(invalidation, 6),
            "max_holding_hours": self.MAX_HOLD_HOURS,
            "atr_val":        round(atr_val, 6),
            "rr_ratio":       round(tp_dist / sl_dist, 2),
        }

    def get_account_balance(self) -> Optional[float]:
        if not self.real_trading or not self.session:
            return None
        try:
            resp = self.session.get_wallet_balance(accountType="UNIFIED")
            if resp["retCode"] == 0:
                equity = float(resp["result"]["list"][0]["totalEquity"])
                self.logger.info(f"💰 Bybit equity: ${equity:.2f}")
                return equity
        except Exception as exc:
            self.logger.error(f"❌ get_account_balance: {exc}")
        return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if not self.real_trading or not self.session:
            return True
        try:
            resp = self.session.set_leverage(
                category    = "linear",
                symbol      = symbol,
                buyLeverage = str(leverage),
                sellLeverage= str(leverage),
            )
            if resp["retCode"] in (0, 110043):
                return True
            self.logger.error(f"❌ set_leverage: {resp.get('retMsg')}")
            return False
        except Exception as exc:
            self.logger.error(f"❌ set_leverage exception: {exc}")
            return False

    LOT_RULES: Dict[str, float] = {
        "BTCUSDT":  0.001,
        "ETHUSDT":  0.01,
        "SOLUSDT":  0.1,
        "XRPUSDT":  1.0,
        "BNBUSDT":  0.01,
        "DOGEUSDT": 1.0,
    }
    MIN_QTY: Dict[str, float] = {
        "BTCUSDT":  0.001,
        "ETHUSDT":  0.01,
        "SOLUSDT":  0.1,
        "XRPUSDT":  1.0,
        "BNBUSDT":  0.01,
        "DOGEUSDT": 10.0,
    }

    def format_quantity(self, symbol: str, quantity: float) -> str:
        lot  = self.LOT_RULES.get(symbol, 0.001)
        qty  = round(quantity / lot) * lot
        mq   = self.MIN_QTY.get(symbol, lot)
        qty  = max(qty, mq)
        if lot >= 1:
            return str(int(qty))
        decimals = len(str(lot).rstrip("0").split(".")[-1])
        return f"{qty:.{decimals}f}"

    def place_bybit_order(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float,
    ) -> Optional[str]:
        if not self.real_trading:
            return f"virtual_{int(time.time())}"

        if not self.session:
            self.logger.error("❌ No Bybit session")
            return None

        self.set_leverage(symbol, self.LEVERAGE)
        qty_str = self.format_quantity(symbol, quantity)

        try:
            resp = self.session.place_order(
                category    = "linear",
                symbol      = symbol,
                side        = "Buy" if side == "LONG" else "Sell",
                orderType   = "Market",
                qty         = qty_str,
                timeInForce = "GTC",
            )
            if resp["retCode"] == 0:
                oid = resp["result"]["orderId"]
                self.logger.info(f"✅ Order placed: {symbol} {side} qty={qty_str} id={oid}")
                return oid
            self.logger.error(f"❌ Order failed: {resp.get('retMsg')}")
        except Exception as exc:
            self.logger.error(f"❌ place_bybit_order exception: {exc}")
        return None

    def close_bybit_position(self, symbol: str, side: str, quantity: float) -> bool:
        if not self.real_trading:
            return True
        if not self.session:
            return False
        close_side = "Sell" if side == "LONG" else "Buy"
        qty_str    = self.format_quantity(symbol, quantity)
        try:
            resp = self.session.place_order(
                category    = "linear",
                symbol      = symbol,
                side        = close_side,
                orderType   = "Market",
                qty         = qty_str,
                timeInForce = "GTC",
                reduceOnly  = True,
            )
            if resp["retCode"] == 0:
                self.logger.info(f"✅ Position closed: {symbol}")
                return True
            self.logger.error(f"❌ Close failed: {resp.get('retMsg')}")
        except Exception as exc:
            self.logger.error(f"❌ close_bybit_position exception: {exc}")
        return False

    def open_llm_position(self, symbol: str) -> Optional[str]:
        signal, confidence = self.generate_llm_signal(symbol)

        if signal == "HOLD" or confidence < self.MIN_CONFIDENCE:
            return None

        active_count = sum(1 for p in self.positions.values() if p["status"] == "ACTIVE")
        if active_count >= self.MAX_POSITIONS:
            return None

        if any(
            p["symbol"] == symbol and p["status"] == "ACTIVE"
            for p in self.positions.values()
        ):
            return None

        price = self.get_current_price(symbol)
        if not price:
            return None

        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, price, confidence
        )
        if quantity <= 0 or margin_required <= 0:
            self.logger.warning(f"⚠️ Invalid position size for {symbol}")
            return None

        equity = self.get_account_balance() or self.virtual_balance
        if margin_required > equity * 0.9:
            self.logger.warning(
                f"💰 Insufficient margin for {symbol}: need ${margin_required:.2f}, have ${equity:.2f}"
            )
            return None

        exit_plan = self.calculate_llm_exit_plan(price, confidence, signal, symbol)

        if signal == "LONG":
            liq_price = price * (1 - 0.9 / self.LEVERAGE)
        else:
            liq_price = price * (1 + 0.9 / self.LEVERAGE)

        order_id = self.place_bybit_order(symbol, signal, quantity, price)
        if self.real_trading and not order_id:
            return None

        position_id = f"pos_{self.position_id}"
        self.position_id += 1

        position = {
            "symbol":            symbol,
            "side":              signal,
            "entry_price":       price,
            "quantity":          quantity,
            "leverage":          self.LEVERAGE,
            "margin":            margin_required,
            "liquidation_price": liq_price,
            "entry_time":        datetime.now(),
            "status":            "ACTIVE",
            "unrealized_pnl":    0.0,
            "confidence":        confidence,
            "llm_profile":       "SwingStrategy",
            "exit_plan":         exit_plan,
            "order_id":          order_id,
            "real_trading":      self.real_trading,
            "current_price":     price,
        }
        self.positions[position_id] = position

        if not self.real_trading:
            self.virtual_balance -= margin_required

        if signal == "LONG":
            self.stats["long_trades"] += 1
        else:
            self.stats["short_trades"] += 1

        rr   = exit_plan["rr_ratio"]
        tp_p = (exit_plan["take_profit"] - price) / price * 100
        sl_p = (price - exit_plan["stop_loss"]) / price * 100 if signal == "LONG" else (exit_plan["stop_loss"] - price) / price * 100

        self.logger.info(
            f"🟢 OPEN {signal}: {symbol} @ ${price:.4f} | "
            f"qty={quantity:.4f} val=${position_value:.2f} | "
            f"TP={exit_plan['take_profit']:.4f} ({tp_p:+.2f}%) "
            f"SL={exit_plan['stop_loss']:.4f} ({sl_p:+.2f}%) "
            f"R:R={rr:.1f} conf={confidence:.2f}"
        )
        return position_id

    def open_real_position(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
    ) -> Optional[str]:
        price = self.get_current_price(symbol)
        if not price:
            return None

        position_value  = quantity * price
        margin_required = position_value / self.LEVERAGE

        equity = self.get_account_balance() or self.virtual_balance
        if margin_required > equity:
            self.logger.warning("💰 Insufficient balance for manual open")
            return None

        exit_plan = self.calculate_llm_exit_plan(price, 0.7, side, symbol)
        liq_price = (
            price * (1 - 0.9 / self.LEVERAGE) if side == "LONG"
            else price * (1 + 0.9 / self.LEVERAGE)
        )

        order_id = self.place_bybit_order(symbol, side, quantity, price)
        if self.real_trading and not order_id:
            return None

        position_id = f"manual_{self.position_id}"
        self.position_id += 1

        position = {
            "symbol":            symbol,
            "side":              side,
            "entry_price":       price,
            "quantity":          quantity,
            "leverage":          self.LEVERAGE,
            "margin":            margin_required,
            "liquidation_price": liq_price,
            "entry_time":        datetime.now(),
            "status":            "ACTIVE",
            "unrealized_pnl":    0.0,
            "confidence":        0.7,
            "llm_profile":       f"MANUAL",
            "exit_plan":         exit_plan,
            "order_id":          order_id,
            "real_trading":      self.real_trading,
            "current_price":     price,
            "manual":            True,
        }
        self.positions[position_id] = position

        if not self.real_trading:
            self.virtual_balance -= margin_required

        self.logger.info(f"✅ MANUAL OPEN: {position_id} {symbol} {side} @ ${price:.4f}")
        return position_id

    def close_position(
        self,
        position_id: str,
        exit_reason: str,
        exit_price:  float,
    ):
        position = self.positions[position_id]

        if position["side"] == "LONG":
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"]

        gross_pnl = pnl_pct * position["quantity"] * position["entry_price"]
        fee       = (position["quantity"] * exit_price + position["quantity"] * position["entry_price"]) * self.FEE_RATE
        net_pnl   = gross_pnl - fee

        if position.get("real_trading"):
            self.close_bybit_position(position["symbol"], position["side"], position["quantity"])

        if not self.real_trading:
            self.virtual_balance += position["margin"] + net_pnl
            self.virtual_capital  = self.virtual_balance + sum(
                p["unrealized_pnl"] for p in self.positions.values()
                if p["status"] == "ACTIVE" and p is not position
            )

        hold_h = (datetime.now() - position["entry_time"]).total_seconds() / 3600

        record = {
            "position_id":    position_id,
            "symbol":         position["symbol"],
            "side":           position["side"],
            "entry_price":    position["entry_price"],
            "exit_price":     exit_price,
            "quantity":       position["quantity"],
            "position_value": position["quantity"] * position["entry_price"],
            "realized_pnl":   net_pnl,
            "exit_reason":    exit_reason,
            "llm_profile":    position["llm_profile"],
            "confidence":     position["confidence"],
            "entry_time":     position["entry_time"],
            "exit_time":      datetime.now(),
            "holding_hours":  hold_h,
            "real_trading":   position.get("real_trading", False),
            "rr_ratio":       position["exit_plan"].get("rr_ratio", 0),
        }
        self.trade_history.append(record)

        self.stats["total_trades"]  += 1
        self.stats["total_pnl"]     += net_pnl
        self.stats["total_fees"]    += fee
        if net_pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"]  += 1

        if self.trade_history:
            self.stats["avg_hold_hours"] = (
                sum(t["holding_hours"] for t in self.trade_history)
                / len(self.trade_history)
            )

        position["status"] = "CLOSED"
        self.dashboard_data["net_realized"] = self.stats["total_pnl"]

        icon = "🟢" if net_pnl > 0 else "🔴"
        self.logger.info(
            f"{icon} CLOSE {position['side']}: {position['symbol']} "
            f"@ ${exit_price:.4f} | PnL=${net_pnl:+.2f} ({pnl_pct*100:+.2f}%) "
            f"| reason={exit_reason} hold={hold_h:.1f}h"
        )

    def update_positions_pnl(self):
        total_unrealized = 0.0
        total_margin     = 0.0
        confidence_sum   = 0.0
        conf_n           = 0

        for position in self.positions.values():
            if position["status"] != "ACTIVE":
                continue
            price = self.get_current_price(position["symbol"])
            if not price:
                continue

            if position["side"] == "LONG":
                pnl = (price - position["entry_price"]) * position["quantity"]
            else:
                pnl = (position["entry_price"] - price) * position["quantity"]

            position["unrealized_pnl"] = pnl
            position["current_price"]  = price

            total_unrealized += pnl
            total_margin     += position["margin"]
            confidence_sum   += position["confidence"]
            conf_n           += 1

        self.dashboard_data["unrealized_pnl"] = total_unrealized

        equity = self.get_account_balance()
        if equity is not None:
            self.dashboard_data["account_value"]  = equity + total_unrealized
            self.dashboard_data["available_cash"] = equity
            self.virtual_balance = equity
        else:
            self.dashboard_data["account_value"]  = self.virtual_capital + total_unrealized
            self.dashboard_data["available_cash"] = self.virtual_balance

        if conf_n > 0:
            self.dashboard_data["average_confidence"] = confidence_sum / conf_n

        equity_val = self.get_account_balance() or self.virtual_balance
        if equity_val > 0:
            self.stats["portfolio_utilization"] = total_margin / equity_val

        self.dashboard_data["last_update"] = datetime.now()

    def check_exit_conditions(self) -> List[Tuple[str, str, float]]:
        to_close = []

        for pid, position in self.positions.items():
            if position["status"] != "ACTIVE":
                continue

            price = position.get("current_price") or self.get_current_price(position["symbol"])
            if not price:
                continue

            ep   = position["exit_plan"]
            side = position["side"]
            reason = None

            if side == "LONG":
                if price >= ep["take_profit"]:
                    reason = "TAKE_PROFIT"
                elif price <= ep["stop_loss"]:
                    reason = "STOP_LOSS"
                elif price <= ep["invalidation"]:
                    reason = "INVALIDATION"
                elif price <= position["liquidation_price"]:
                    reason = "LIQUIDATION"
            else:
                if price <= ep["take_profit"]:
                    reason = "TAKE_PROFIT"
                elif price >= ep["stop_loss"]:
                    reason = "STOP_LOSS"
                elif price >= ep["invalidation"]:
                    reason = "INVALIDATION"
                elif price >= position["liquidation_price"]:
                    reason = "LIQUIDATION"

            hold_h = (datetime.now() - position["entry_time"]).total_seconds() / 3600
            if hold_h > ep.get("max_holding_hours", self.MAX_HOLD_HOURS):
                reason = "TIME_EXPIRED"

            if not reason:
                cached_sig = self.signal_cache.get(position["symbol"])
                if cached_sig:
                    sig = cached_sig["signal"]
                    if (side == "LONG" and sig == "SHORT") or (side == "SHORT" and sig == "LONG"):
                        reason = "SIGNAL_REVERSAL"

            if reason:
                to_close.append((pid, reason, price))

        return to_close

    def run_llm_trading_strategy(self):
        self.logger.info("🚀 Swing Trading loop started")
        iteration = 0

        while self.is_running:
            iteration += 1
            self.logger.info(f"\n── Iteration #{iteration} ─────────────────────────")

            try:
                self.update_positions_pnl()

                for pid, reason, price in self.check_exit_conditions():
                    self.close_position(pid, reason, price)

                active_symbols = {
                    p["symbol"]
                    for p in self.positions.values()
                    if p["status"] == "ACTIVE"
                }
                active_count = len(active_symbols)

                if active_count < self.MAX_POSITIONS:
                    for symbol in self.assets:
                        if symbol not in active_symbols:
                            self.open_llm_position(symbol)
                            time.sleep(0.5)

                equity = self.dashboard_data["account_value"]
                ret_pct = (equity - self.initial_capital) / self.initial_capital * 100
                self.logger.info(
                    f"📊 Equity=${equity:.2f} ({ret_pct:+.2f}%) | "
                    f"Positions={active_count}/{self.MAX_POSITIONS} | "
                    f"Trades={self.stats['total_trades']}"
                )

            except Exception as exc:
                self.logger.error(f"❌ Loop error: {exc}", exc_info=True)

            for _ in range(self.LOOP_INTERVAL):
                if not self.is_running:
                    break
                time.sleep(1)

        self.logger.info("🛑 Swing Trading loop stopped")

    def start_trading(self):
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("▶️  Trading bot started")

    def stop_trading(self):
        self.is_running = False
        self.logger.info("⏹️  Trading bot stopping…")

    def get_dashboard_data(self) -> dict:
        active_positions = []
        total_unrealized = 0.0

        for pid, position in self.positions.items():
            if position["status"] != "ACTIVE":
                continue

            price = position.get("current_price") or self.get_current_price(position["symbol"])
            if not price:
                continue

            if position["side"] == "LONG":
                pnl     = (price - position["entry_price"]) * position["quantity"]
                tp_dist = (position["exit_plan"]["take_profit"] - price) / price * 100
                sl_dist = (price - position["exit_plan"]["stop_loss"])  / price * 100
            else:
                pnl     = (position["entry_price"] - price) * position["quantity"]
                tp_dist = (price - position["exit_plan"]["take_profit"]) / price * 100
                sl_dist = (position["exit_plan"]["stop_loss"] - price)  / price * 100

            active_positions.append({
                "position_id":    pid,
                "symbol":         position["symbol"],
                "side":           position["side"],
                "entry_price":    position["entry_price"],
                "current_price":  price,
                "quantity":       position["quantity"],
                "leverage":       position["leverage"],
                "margin":         position["margin"],
                "unrealized_pnl": pnl,
                "confidence":     position["confidence"],
                "llm_profile":    position["llm_profile"],
                "entry_time":     position["entry_time"].strftime("%H:%M:%S"),
                "exit_plan":      position["exit_plan"],
                "tp_distance_pct": tp_dist,
                "sl_distance_pct": sl_dist,
                "real_trading":   position.get("real_trading", False),
            })
            total_unrealized += pnl

        confidence_levels = {}
        for symbol in self.assets:
            _, conf = self.generate_llm_signal(symbol)
            confidence_levels[symbol] = round(conf * 100, 1)

        recent_trades = []
        for trade in sorted(self.trade_history, key=lambda x: x["exit_time"], reverse=True):
            recent_trades.append({
                "symbol":         trade["symbol"],
                "side":           trade["side"],
                "entry_price":    trade["entry_price"],
                "exit_price":     trade["exit_price"],
                "quantity":       trade["quantity"],
                "position_value": trade["position_value"],
                "realized_pnl":   trade["realized_pnl"],
                "exit_reason":    trade["exit_reason"],
                "llm_profile":    trade["llm_profile"],
                "confidence":     trade["confidence"],
                "holding_hours":  round(trade["holding_hours"], 2),
                "entry_time":     trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time":      trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "real_trading":   trade.get("real_trading", False),
                "rr_ratio":       trade.get("rr_ratio", 0),
            })

        total_trades = self.stats["total_trades"]
        win_rate     = (self.stats["winning_trades"] / total_trades * 100) if total_trades else 0
        ret_pct      = (self.dashboard_data["account_value"] - self.initial_capital) / self.initial_capital * 100

        return {
            "account_summary": {
                "total_value":    round(self.dashboard_data["account_value"], 2),
                "available_cash": round(self.dashboard_data["available_cash"], 2),
                "net_realized":   round(self.dashboard_data["net_realized"], 2),
                "unrealized_pnl": round(self.dashboard_data["unrealized_pnl"], 2),
                "real_trading":   self.real_trading,
            },
            "performance_metrics": {
                "total_return_pct":     round(ret_pct, 2),
                "win_rate":             round(win_rate, 1),
                "total_trades":         total_trades,
                "long_trades":          self.stats["long_trades"],
                "short_trades":         self.stats["short_trades"],
                "avg_holding_hours":    round(self.stats["avg_hold_hours"], 2),
                "portfolio_utilization": round(self.stats.get("portfolio_utilization", 0) * 100, 1),
                "portfolio_diversity":  round(self._portfolio_diversity() * 100, 1),
                "avg_confidence":       round(self.dashboard_data.get("average_confidence", 0) * 100, 1),
                "total_fees":           round(self.stats["total_fees"], 2),
            },
            "llm_config": {
                "active_profile":     self.active_profile,
                "available_profiles": list(self.llm_profiles.keys()),
                "max_positions":      self.MAX_POSITIONS,
                "leverage":           self.LEVERAGE,
                "real_trading":       self.real_trading,
                "strategy":           "Multi-Indicator Swing",
                "risk_per_trade_pct": self.RISK_PER_TRADE * 100,
                "tp_atr_mult":        self.TP_ATR_MULT,
                "sl_atr_mult":        self.SL_ATR_MULT,
                "min_confidence":     self.MIN_CONFIDENCE,
            },
            "confidence_levels":   confidence_levels,
            "active_positions":    active_positions,
            "recent_trades":       recent_trades,
            "total_unrealized_pnl": total_unrealized,
            "last_update":         self.dashboard_data["last_update"].isoformat(),
        }

    def _portfolio_diversity(self) -> float:
        active = [p for p in self.positions.values() if p["status"] == "ACTIVE"]
        if not active:
            return 0.0
        total = sum(p["margin"] for p in active)
        if total == 0:
            return 0.0
        hhi = sum((p["margin"] / total) ** 2 for p in active)
        return 1 - hhi

    def set_active_profile(self, profile_name: str) -> bool:
        self.active_profile = profile_name
        self.dashboard_data["active_profile"] = profile_name
        return True

    def get_current_profile(self) -> dict:
        return self.llm_profiles.get(self.active_profile, list(self.llm_profiles.values())[0])

    def save_chart_data(self, data: dict) -> bool:
        self.chart_data = data
        return True

    def load_chart_data(self) -> dict:
        return self.chart_data

    def check_api_status(self) -> dict:
        equity = self.get_account_balance()
        return {
            "real_trading":      self.real_trading,
            "api_connected":     equity is not None,
            "balance":           equity,
            "balance_available": equity is not None,
            "message":           f"Balance: ${equity:.2f}" if equity else "Virtual mode",
        }


app = Flask(__name__)
CORS(app)

trading_bot = LLMTradingBot(initial_capital=10_000, leverage=10)


@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("index.html")


@app.route("/api/trading-data")
def get_trading_data():
    try:
        return jsonify(trading_bot.get_dashboard_data())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/bot-status")
def get_bot_status():
    return jsonify({"status": "running" if trading_bot.is_running else "stopped"})


@app.route("/api/start-bot", methods=["POST"])
def start_bot():
    try:
        trading_bot.start_trading()
        return jsonify({"status": "Bot started"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stop-bot", methods=["POST"])
def stop_bot():
    try:
        trading_bot.stop_trading()
        return jsonify({"status": "Bot stopped"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/change-profile", methods=["POST"])
def change_profile():
    try:
        data = request.get_json() or {}
        trading_bot.set_active_profile(data.get("profile", "SwingStrategy"))
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/force-update", methods=["POST"])
def force_update():
    try:
        trading_bot.update_positions_pnl()
        return jsonify({"status": "updated"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/open-position", methods=["POST"])
def open_position():
    try:
        data     = request.get_json() or {}
        symbol   = data.get("symbol")
        side     = data.get("side")
        quantity = float(data.get("quantity", 0))
        if not symbol or not side or quantity <= 0:
            return jsonify({"error": "symbol, side and quantity required"}), 400
        pid = trading_bot.open_real_position(symbol, side, quantity)
        if pid:
            return jsonify({"status": "ok", "position_id": pid})
        return jsonify({"error": "Failed to open position"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/save-chart-data", methods=["POST"])
def save_chart_data():
    try:
        trading_bot.save_chart_data(request.get_json() or {})
        return jsonify({"status": "success"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/load-chart-data")
def load_chart_data():
    try:
        return jsonify({"status": "success", "chartData": trading_bot.load_chart_data()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/api-status")
def api_status():
    return jsonify(trading_bot.check_api_status())


@app.route("/api/signal-details/<symbol>")
def signal_details(symbol):
    try:
        sig, conf = trading_bot.generate_llm_signal(symbol)
        details   = trading_bot.get_signal_details(symbol)
        return jsonify({"symbol": symbol, "signal": sig, "confidence": conf, "indicators": details})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("🚀  Swing Trading Bot — starting on http://localhost:5000")
    print(f"    Strategy : Multi-Indicator (EMA×, MACD, RSI, BB, Volume)")
    print(f"    Timeframe: 1h candles (Binance)")
    print(f"    Risk     : {trading_bot.RISK_PER_TRADE*100:.1f}% equity per trade")
    print(f"    Leverage : {trading_bot.LEVERAGE}x")
    print(f"    TP/SL    : {trading_bot.TP_ATR_MULT}×ATR / {trading_bot.SL_ATR_MULT}×ATR")
    print(f"    Real mode: {trading_bot.real_trading}")
    app.run(debug=True, host="0.0.0.0", port=5000)
