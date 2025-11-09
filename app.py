from flask import Flask, render_template, jsonify, request
import threading
import time
import json
import logging
from datetime import datetime, timedelta
import random
import requests

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
    if llm_trading_bot and bot_status == "running":  # Zmieniam na llm_trading_bot
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
        # Sprawdź czy API keys są skonfigurowane
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        real_trading = bool(api_key and api_secret)
        api_connected = False
        
        if real_trading:
            # Spróbuj połączyć się z API Bybit
            try:
                # Tutaj dodaj kod testujący połączenie z Bybit
                # Na razie ustawiamy na False dla demo
                api_connected = False
            except:
                api_connected = False
        
        return jsonify({
            'real_trading': real_trading,
            'api_connected': api_connected,
            'message': 'Virtual demo mode' if not real_trading else 'Real trading mode'
        })
        
    except Exception as e:
        return jsonify({
            'real_trading': False,
            'api_connected': False,
            'message': f'Error: {str(e)}'
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
