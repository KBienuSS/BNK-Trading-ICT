from trading_bot_ml import LLMTradingBot
import logging

logging.basicConfig(level=logging.INFO)

def test_pybit():
    print("🚀 Testing bot with PYBIT...")
    try:
        bot = LLMTradingBot()
        print("✅ Bot created successfully")
        
        # Test inicjalizacji pybit
        if hasattr(bot, 'session') and bot.session:
            print("✅ Pybit session initialized")
        else:
            print("❌ Pybit session NOT initialized")
            return
        
        # Test ceny
        price = bot.get_current_price("BTCUSDT")
        print(f"💰 BTC Price: ${price}")
        
        # Test salda przez pybit
        balance = bot.get_account_balance_pybit()
        print(f"💵 PYBIT Balance: ${balance}")
        
        # Test ustawienia dźwigni
        leverage_ok = bot.set_leverage_pybit("BTCUSDT", 10)
        print(f"🔧 Leverage setting: {leverage_ok}")
        
        # Test pozycji
        print("🎯 Testing position opening with PYBIT...")
        position_id = bot.open_llm_position("BTCUSDT")
        print(f"📦 Position result: {position_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pybit()
