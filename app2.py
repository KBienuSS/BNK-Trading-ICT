from flask import Flask, render_template, jsonify, request
import threading
import time
import json
import logging
from datetime import datetime, timedelta
import random
import requests
from trading_bot_ict import ICTTradingBot  # Import nowego bota ICT

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global bot instances
ict_trading_bot = None
bot_status = "stopped"

class TradingData:
    def __init__(self):
        self.account_value = 8000
        self.available_cash = 8000
        self.total_fees = 0
        self.net_realized = 0
        
    def get_trading_data(self):
        return {
            'account_summary': {
                'total_value': self.account_value,
                'available_cash': self.available_cash,
                'total_fees': self.total_fees,
                'net_realized': self.net_realized
            },
            'performance_metrics': {
                'avg_leverage': 5,
                'portfolio_utilization': 0,
                'ict_trades': 0,
                'win_rate': 0,
                'total_trades': 0,
                'biggest_win': 0,
                'biggest_loss': 0,
                'average_rr': 0
            },
            'trading_session': {
                'current': 'CLOSED',
                'sessions': {
                    'asian_range': {'start': 0, 'end': 6},
                    'london_open': {'start': 7, 'end': 10},
                    'new_york_open': {'start': 13, 'end': 16},
                    'enabled': True
                }
            },
            'confidence_levels': {
                'BTCUSDT': 0, 'ETHUSDT': 0, 'SOLUSDT': 0, 
                'BNBUSDT': 0, 'XRPUSDT': 0
            },
            'active_positions': [],
            'recent_trades': [],
            'total_unrealized_pnl': 0,
            'strategy_profile': {
                'risk_reward': "1:3",
                'min_confidence': "75%",
                'max_positions': 3,
                'trading_sessions': True
            },
            'last_update': datetime.now().isoformat()
        }

trading_data = TradingData()

@app.route('/')
def index():
    return render_template('index2.html')  # Inny template

@app.route('/api/trading-data')
def get_trading_data():
    if ict_trading_bot and bot_status == "running":
        return jsonify(ict_trading_bot.get_dashboard_data())
    else:
        return jsonify(trading_data.get_trading_data())

@app.route('/api/bot-status')
def get_bot_status():
    return jsonify({'status': bot_status})

@app.route('/api/start-bot')
def start_bot():
    global bot_status, ict_trading_bot
    try:
        if bot_status != "running":
            # Start ICT bot
            ict_trading_bot = ICTTradingBot(initial_capital=8000, leverage=5)
            
            bot_thread = threading.Thread(target=run_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            bot_status = "running"
            return jsonify({'status': 'ICT Bot started successfully'})
        else:
            return jsonify({'status': 'ICT Bot is already running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot')
def stop_bot():
    global bot_status
    try:
        if bot_status == "running":
            if ict_trading_bot:
                ict_trading_bot.stop_trading()
            
            bot_status = "stopped"
            return jsonify({'status': 'ICT Bot stopped successfully'})
        else:
            return jsonify({'status': 'ICT Bot is not running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-update')
def force_update():
    try:
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_bot():
    """Run the ICT trading bot in a separate thread"""
    global ict_trading_bot
    try:
        logging.info("🤖 Starting ICT Trading Bot thread...")
        if ict_trading_bot:
            ict_trading_bot.start_trading()
        else:
            logging.error("❌ ict_trading_bot is None - cannot start trading")
    except Exception as e:
        logging.error(f"❌ Error running ICT bot: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
