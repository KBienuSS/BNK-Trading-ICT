# app.py - CAŁKOWICIE NOWY PLIK BEZ DUPLIKATÓW
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from trading_bot_ml import LLMTradingBot
import os
import logging

# Konfiguracja logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja Flask
app = Flask(__name__)
CORS(app)

# Globalna inicjalizacja bota
try:
    bot = LLMTradingBot(
        api_key=os.getenv('BYBIT_API_KEY'),
        api_secret=os.getenv('BYBIT_API_SECRET'),
        initial_capital=10000,
        leverage=10
    )
    logger.info("✅ Bot initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize bot: {e}")
    bot = None

# ========== ENDPOINTY ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard')
def get_dashboard():
    """Endpoint dla dashboardu"""
    try:
        if bot is None:
            return jsonify({"error": "Bot not initialized"}), 500
        data = bot.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Rozpoczyna trading"""
    try:
        if bot is None:
            return jsonify({"status": "error", "message": "Bot not initialized"}), 500
        bot.start_trading()
        return jsonify({"status": "success", "message": "Trading started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop-trading', methods=['POST'])
def stop_trading():
    """Zatrzymuje trading"""
    try:
        if bot is None:
            return jsonify({"status": "error", "message": "Bot not initialized"}), 500
        bot.stop_trading()
        return jsonify({"status": "success", "message": "Trading stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/switch-profile', methods=['POST'])
def switch_profile():
    """Zmienia profil LLM"""
    try:
        if bot is None:
            return jsonify({"status": "error", "message": "Bot not initialized"}), 500
        profile = request.json.get('profile')
        if bot.set_active_profile(profile):
            return jsonify({"status": "success", "message": f"Profile switched to {profile}"})
        else:
            return jsonify({"status": "error", "message": "Invalid profile"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check-permissions', methods=['GET'])
def check_permissions():
    """Sprawdza uprawnienia API"""
    try:
        if bot is None:
            return jsonify({
                "status": "error", 
                "message": "Bot not initialized - check API keys"
            }), 500
            
        api_status = bot.check_api_status()
        
        # Test zlecenia z minimalną kwotą
        test_symbol = "BTCUSDT"
        test_price = bot.get_current_price(test_symbol)
        
        if test_price:
            # Sprawdź czy możemy ustawić dźwignię
            leverage_set = bot.set_leverage(test_symbol, bot.leverage)
            
            return jsonify({
                "status": "success",
                "api_status": api_status,
                "price_check": f"✅ ${test_price}",
                "leverage_set": leverage_set,
                "real_trading": bot.real_trading,
                "message": f"API Status: {api_status['message']}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Cannot fetch price data"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error checking permissions: {str(e)}"
        }), 500

@app.route('/api/debug-api', methods=['GET'])
def debug_api():
    """Debuguje połączenie API"""
    try:
        if bot is None:
            return jsonify({"status": "error", "message": "Bot not initialized"}), 500
            
        # Test podstawowego requesta
        test_symbol = "BTCUSDT"
        
        # 1. Test publicznego API (ceny)
        price = bot.get_current_price(test_symbol)
        
        # 2. Test prywatnego API (saldo)
        api_status = bot.check_api_status()
        
        # 3. Test ustawienia dźwigni
        leverage_set = bot.set_leverage(test_symbol, bot.leverage)
        
        return jsonify({
            "status": "success",
            "public_api_working": price is not None,
            "price": price,
            "private_api_working": api_status['api_connected'],
            "balance_available": api_status['balance_available'],
            "leverage_set": leverage_set,
            "api_status": api_status
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/test-order', methods=['POST'])
def test_order():
    """Testuje składanie zlecenia na Bybit"""
    try:
        if bot is None:
            return jsonify({"status": "error", "message": "Bot not initialized"}), 500
            
        symbol = request.json.get('symbol', 'BTCUSDT')
        
        # Pobierz cenę
        price = bot.get_current_price(symbol)
        if not price:
            return jsonify({
                "status": "error", 
                "message": f"Cannot get price for {symbol}"
            }), 500
        
        # Oblicz minimalną ilość
        min_quantity = 5 / price  # $5 minimalne zlecenie
        quantity = max(min_quantity, 0.001)  # co najmniej 0.001
        
        bot.logger.info(f"🧪 TEST ORDER: {symbol} Qty: {quantity:.6f} Price: ${price}")
        
        # Spróbuj złożyć zlecenie
        order_id = bot.place_bybit_order(symbol, "LONG", quantity, price)
        
        if order_id:
            return jsonify({
                "status": "success",
                "message": f"Test order placed for {symbol}",
                "order_id": order_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": price
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Failed to place test order for {symbol}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error in test order: {str(e)}"
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
