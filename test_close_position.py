from trading_bot_ml import LLMTradingBot
import logging

logging.basicConfig(level=logging.INFO)

def test_close_position():
    print("🔒 Testing position closing...")
    try:
        bot = LLMTradingBot()
        
        # Pobierz aktywne pozycje
        positions = bot.get_bybit_positions_pybit()
        print(f"📊 Found {len(positions)} active positions")
        
        for pos in positions:
            if pos['symbol'] == 'BTCUSDT':
                print(f"🚀 Closing BTCUSDT position...")
                success = bot.close_bybit_position_pybit(
                    pos['symbol'], 
                    pos['side'], 
                    pos['size']
                )
                print(f"✅ Close result: {success}")
                break
        else:
            print("❌ No BTCUSDT position found to close")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_close_position()
