from trading_bot_ml import LLMTradingBot
import logging

logging.basicConfig(level=logging.INFO)

print("🚀 Testing bot directly...")
try:
    bot = LLMTradingBot()
    print("✅ Bot created successfully")
    
    # Test ceny
    price = bot.get_current_price("BTCUSDT")
    print(f"💰 BTC Price: ${price}")
    
    # Test salda
    balance = bot.get_account_balance()
    print(f"💵 Balance: ${balance}")
    
    # Test pozycji
    print("🎯 Testing position opening...")
    position_id = bot.open_llm_position("BTCUSDT")
    print(f"📦 Position result: {position_id}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
