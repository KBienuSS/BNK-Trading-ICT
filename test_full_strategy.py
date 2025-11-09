# test_full_strategy.py
from trading_bot_ml import LLMTradingBot
import logging
import time

logging.basicConfig(level=logging.INFO)

def test_full_strategy():
    print("🎯 Testing full LLM strategy...")
    try:
        bot = LLMTradingBot()
        
        # Start strategii
        bot.start_trading()
        print("✅ Strategy started")
        
        # Działaj przez 2 minuty
        print("⏳ Running for 2 minutes...")
        time.sleep(120)
        
        # Zatrzymaj strategię
        bot.stop_trading()
        print("🛑 Strategy stopped")
        
        # Pokaż podsumowanie
        dashboard = bot.get_dashboard_data()
        print(f"📊 Final account value: ${dashboard['account_summary']['total_value']:.2f}")
        print(f"📈 Total trades: {dashboard['performance_metrics']['total_trades']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_strategy()
