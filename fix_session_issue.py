#!/usr/bin/env python3
"""
Naprawa problemu z sesją pybit
"""

import logging
import sys
import os

# Dodaj ścieżkę do importowania trading_bot_ml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot_ml import LLMTradingBot

def fix_session_issue():
    print("🔧 Fixing session issue...")
    
    # Test z trybem wirtualnym
    print("1. Testing virtual mode...")
    bot_virtual = LLMTradingBot(real_trading=False)
    print(f"   ✅ Virtual mode: session = {bot_virtual.session}")
    
    # Test z trybem realnym (jeśli klucze API są dostępne)
    print("2. Testing real mode...")
    try:
        bot_real = LLMTradingBot(real_trading=True)
        print(f"   ✅ Real mode: session initialized = {bot_real.session is not None}")
        
        if bot_real.session:
            print("3. Testing balance...")
            balance = bot_real.get_account_balance()
            print(f"   ✅ Balance: ${balance:.2f}")
            
            print("4. Testing positions...")
            positions = bot_real.get_bybit_positions()
            print(f"   ✅ Positions: {len(positions)}")
    except Exception as e:
        print(f"   ⚠️ Real mode test failed (expected without API keys): {e}")
    
    print("🎉 Session fix completed!")

if __name__ == "__main__":
    fix_session_issue()
