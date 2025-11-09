from flask import Flask, render_template, jsonify, request
import threading
import time
import json
import logging
from datetime import datetime, timedelta
import random
import requests
import os
from dotenv import load_dotenv  # Dodaj tę linię

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Zmieniam import na LLMTradingBot zamiast MLTradingBot
from trading_bot_ml import LLMTradingBot

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global bot instances
trading_bot = None
llm_trading_bot = None  # Zmieniam nazwę na llm_trading_bot
bot_status = "stopped"

class TradingData:
    def __init__(self):
        self.account_value = 50000
        self.available_cash = 35000
        self.total_fees = 124.50
        self.net_realized = 1567.89
        
    def get_trading_data(self):
        return {
            'account_summary': {
                'total_value': self.account_value,
                'available_cash': self.available_cash,
                'total_fees': self.total_fees,
                'net_realized': self.net_realized
            },
            'performance_metrics': {
                'avg_leverage': 8.5,
                'avg_confidence': 76.2,
                'biggest_win': 1245.67,
                'biggest_loss': -567.89
            },
            'active_positions': [],
            'recent_trades': [],
            'total_unrealized_pnl': 0
        }

trading_data = TradingData()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trading-data')
def get_trading_data():
    if llm_trading_bot and bot_status == "running":
        return jsonify(llm_trading_bot.get_dashboard_data())
    else:
        # Użyj danych z aktywnej instancji bota, nawet jeśli zatrzymana
        if llm_trading_bot:
            return jsonify(llm_trading_bot.get_dashboard_data())
        else:
            return jsonify(trading_data.get_trading_data())

@app.route('/api/bot-status')
def get_bot_status():
    return jsonify({'status': bot_status})

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    global bot_status, llm_trading_bot  # Zmieniam na llm_trading_bot
    try:
        if bot_status != "running":
            # Start LLM bot
            llm_trading_bot = LLMTradingBot()  # Używamy LLMTradingBot
            
            bot_thread = threading.Thread(target=run_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            bot_status = "running"
            return jsonify({'status': 'LLM Bot started successfully'})
        else:
            return jsonify({'status': 'Bot is already running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    global bot_status
    try:
        if bot_status == "running":
            if llm_trading_bot:  # Zmieniam na llm_trading_bot
                llm_trading_bot.stop_trading()
            
            bot_status = "stopped"
            return jsonify({'status': 'Bot stopped successfully'})
        else:
            return jsonify({'status': 'Bot is not running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-order', methods=['POST'])
def test_order():
    """Endpoint do bezpośredniego testowania składania zleceń"""
    symbol = "BTCUSDT"
    
    # Testowe parametry
    test_quantity = 0.001  # Minimalna ilość BTC
    test_price = 104000.0
    test_side = "LONG"
    
    bot.logger.info("🧪 TEST ORDER - Starting direct API test")
    
    # 1. Test pobierania ceny
    price = bot.get_current_price(symbol)
    if not price:
        return jsonify({"status": "failed", "message": "Could not get price"})
    
    # 2. Test składania zlecenia
    order_id = bot.place_bybit_order(symbol, test_side, test_quantity, price)
    
    if order_id:
        return jsonify({"status": "success", "order_id": order_id})
    else:
        return jsonify({"status": "failed", "message": "Order placement failed"})

@app.route('/api/change-profile', methods=['POST'])
def change_profile():
    """Nowy endpoint do zmiany profilu LLM"""
    try:
        data = request.get_json()
        profile_name = data.get('profile')
        
        if llm_trading_bot and llm_trading_bot.set_active_profile(profile_name):
            return jsonify({'status': f'Profile changed to {profile_name}'})
        else:
            return jsonify({'status': 'Invalid profile name or bot not running'})
    except Exception as e:
        return jsonify({'status': f'Error changing profile: {str(e)}'})

@app.route('/api/debug-api')
def debug_api():
    """Debuguje połączenie API"""
    if not llm_trading_bot:
        return jsonify({'error': 'Bot not initialized'})
    
    try:
        # Sprawdź dostępne kategorie
        categories = llm_trading_bot.check_available_categories()
        
        # Sprawdź ceny dla wszystkich symboli
        price_check = {}
        for symbol in llm_trading_bot.assets:
            price = llm_trading_bot.get_current_price(symbol)
            price_check[symbol] = price if price else "NO PRICE"
        
        # Sprawdź status API
        api_status = llm_trading_bot.check_api_status()
        
        return jsonify({
            'api_status': api_status,
            'price_check': price_check,
            'categories_available': categories
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
        
@app.route('/api/force-update')
def force_update():
    try:
        if llm_trading_bot:  # Zmieniam na llm_trading_bot
            llm_trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/api-status')
def api_status():
    """Check API connection status"""
    try:
        if llm_trading_bot:
            # Użyj funkcji check_api_status() z bota
            api_status = llm_trading_bot.check_api_status()
            return jsonify(api_status)
        else:
            return jsonify({
                'real_trading': False,
                'api_connected': False,
                'message': 'Bot not initialized'
            })
        
    except Exception as e:
        return jsonify({
            'real_trading': False,
            'api_connected': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    """Zapisuje dane wykresu"""
    try:
        data = request.get_json()
        if llm_trading_bot and hasattr(llm_trading_bot, 'save_chart_data'):
            if llm_trading_bot.save_chart_data(data):
                return jsonify({'status': 'success'})
        
        return jsonify({'error': 'Failed to save chart data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-conditions')
def debug_conditions():
    """Debuguje warunki wejścia"""
    if not llm_trading_bot:
        return jsonify({'error': 'Bot not initialized'})
    
    results = {}
    for symbol in llm_trading_bot.assets:
        current_price = llm_trading_bot.get_current_price(symbol)
        signal, confidence = llm_trading_bot.generate_llm_signal(symbol)
        should_enter = llm_trading_bot.should_enter_trade()
        active_count = sum(1 for p in llm_trading_bot.positions.values() if p['status'] == 'ACTIVE')
        
        results[symbol] = {
            'price': current_price,
            'signal': signal,
            'confidence': confidence,
            'should_enter': should_enter,
            'active_positions': active_count,
            'max_positions': llm_trading_bot.max_simultaneous_positions,
            'can_enter': (
                should_enter and 
                signal != "HOLD" and 
                confidence >= 0.3 and
                active_count < llm_trading_bot.max_simultaneous_positions
            )
        }
    
    return jsonify(results)

@app.route('/api/debug-position-size/<symbol>')
def debug_position_size(symbol):
    """Debuguje kalkulację wielkości pozycji"""
    if not llm_trading_bot:
        return jsonify({'error': 'Bot not initialized'})
    
    current_price = llm_trading_bot.get_current_price(symbol)
    if not current_price:
        return jsonify({'error': f'Could not get price for {symbol}'})
    
    quantity, position_value, margin_required = llm_trading_bot.calculate_position_size(
        symbol, current_price, 0.95
    )
    
    api_status = llm_trading_bot.check_api_status()
    available_balance = api_status['balance'] if api_status['balance_available'] else llm_trading_bot.virtual_balance
    
    return jsonify({
        'symbol': symbol,
        'current_price': current_price,
        'quantity': quantity,
        'position_value': position_value,
        'margin_required': margin_required,
        'available_balance': available_balance,
        'sufficient_balance': margin_required <= available_balance,
        'min_order_value': quantity * current_price,
        'min_order_met': (quantity * current_price) >= 5
    })

@app.route('/api/force-open-position/<symbol>', methods=['POST'])
def force_open_position(symbol):
    """Wymusza otwarcie pozycji dla testów"""
    if not llm_trading_bot:
        app.logger.error(f"❌ Bot not initialized for symbol {symbol}")
        return jsonify({'error': 'Bot not initialized'})
    
    try:
        # DODAJ LOGI DIAGNOSTYCZNE
        app.logger.info(f"🎯 MANUAL FORCE OPEN POSITION REQUEST for {symbol}")
        app.logger.info(f"🔧 Real Trading: {llm_trading_bot.real_trading}")
        app.logger.info(f"🔧 Bot Running: {llm_trading_bot.is_running}")
        
        # Pobierz aktualną cenę
        current_price = llm_trading_bot.get_current_price(symbol)
        app.logger.info(f"💰 Current price for {symbol}: ${current_price}")
        
        if not current_price:
            return jsonify({
                'status': 'failed', 
                'message': f'Could not get price for {symbol}'
            })
        
        # Sprawdź aktywne pozycje
        active_positions = sum(1 for p in llm_trading_bot.positions.values() if p['status'] == 'ACTIVE')
        app.logger.info(f"📊 Active positions: {active_positions}/{llm_trading_bot.max_simultaneous_positions}")
        
        if active_positions >= llm_trading_bot.max_simultaneous_positions:
            return jsonify({
                'status': 'failed',
                'message': f'Max positions reached: {active_positions}/{llm_trading_bot.max_simultaneous_positions}'
            })
        
        # Wywołaj funkcję bota
        app.logger.info(f"🚀 Calling llm_trading_bot.open_llm_position('{symbol}')")
        position_id = llm_trading_bot.open_llm_position(symbol)
        
        app.logger.info(f"📨 Result from open_llm_position: {position_id}")
        
        if position_id:
            return jsonify({
                'status': 'success',
                'position_id': position_id,
                'message': f'Position opened for {symbol}'
            })
        else:
            return jsonify({
                'status': 'failed', 
                'message': f'open_llm_position returned None for {symbol}'
            })
            
    except Exception as e:
        app.logger.error(f"💥 Error in force-open-position: {e}")
        import traceback
        app.logger.error(f"💥 Stack trace: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-permissions', methods=['GET'])
def check_permissions():
    """Sprawdza uprawnienia konta"""
    try:
        # Sprawdź informacje o koncie
        endpoint = "/v5/account/info"
        data = bot.bybit_request('GET', endpoint, {}, private=True)
        
        if data:
            return jsonify({
                "status": "success", 
                "account_info": data,
                "message": "Account permissions retrieved"
            })
        else:
            return jsonify({
                "status": "failed",
                "message": "Could not retrieve account info - check API permissions"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error checking permissions: {str(e)}"
        })
        
def run_bot():
    """Run the trading bot in a separate thread"""
    global llm_trading_bot  # Zmieniam na llm_trading_bot
    try:
        logging.info("🧠 Starting LLM Trading Bot thread...")
        if llm_trading_bot:
            llm_trading_bot.start_trading()
        else:
            logging.error("❌ llm_trading_bot is None - cannot start trading")
    except Exception as e:
        logging.error(f"❌ Error running LLM bot: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
