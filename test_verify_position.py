from trading_bot_ml import LLMTradingBot
import logging
import time

logging.basicConfig(level=logging.INFO)

def test_verification():
    print("🔍 Testing position verification...")
    try:
        bot = LLMTradingBot()
        
        # Test weryfikacji ostatniej pozycji
        order_id = "967f04f5-974a-4f09-994b-4a4391e1261e"
        symbol = "BTCUSDT"
        
        print("⏳ Waiting 5 seconds for position to settle...")
        time.sleep(5)
        
        verified = bot.verify_position_opened(order_id, symbol)
        print(f"✅ Position verified: {verified}")
        
        # Pobierz aktualne pozycje
        positions = bot.get_bybit_positions_pybit()
        print(f"📊 All active positions: {len(positions)}")
        for pos in positions:
            print(f"  - {pos['symbol']}: {pos['side']} {pos['size']} @ ${pos['entry_price']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_verification()
