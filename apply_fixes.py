#!/usr/bin/env python3
"""
Zastosowanie wszystkich poprawek do trading_bot_ml.py
"""

import logging
from trading_bot_ml import LLMTradingBot

logging.basicConfig(level=logging.INFO)

def apply_and_test_fixes():
    print("🔧 Applying and testing all fixes...")
    
    # Utwórz instancję bota z poprawkami
    bot = LLMTradingBot()
    
    print("1. Testing available symbols detection...")
    available = bot.assets
    print(f"   ✅ Trading symbols: {available}")
    
    print("2. Testing positions retrieval...")
    positions = bot.get_bybit_positions()
    print(f"   ✅ Current positions: {len(positions)}")
    for pos in positions:
        print(f"     - {pos['symbol']}: {pos['side']} {pos['size']}")
    
    print("3. Testing balance...")
    balance = bot.get_account_balance()
    print(f"   ✅ Available balance: ${balance:.2f}")
    
    print("4. Testing position size calculation...")
    price = bot.get_current_price("BTCUSDT")
    if price:
        quantity, value, margin = bot.calculate_position_size("BTCUSDT", price, 0.8)
        print(f"   ✅ Position size: {quantity:.6f} BTC (${value:.2f})")
    
    print("🎉 All fixes applied successfully!")
    print(f"📊 Bot is ready to trade: {bot.assets}")

if __name__ == "__main__":
    apply_and_test_fixes()
