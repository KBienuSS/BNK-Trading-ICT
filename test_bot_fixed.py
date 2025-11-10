#!/usr/bin/env python3
"""
Test poprawionego bota
"""

import requests
import json

def test_bot_start():
    print("🧪 Testing fixed bot startup...")
    
    # Test z trybem wirtualnym
    payload = {
        "real_trading": False,
        "initial_capital": 5000,
        "leverage": 5
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/start-bot",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        result = response.json()
        print(f"📋 Response: {json.dumps(result, indent=2)}")
        
        if result['status'] == 'success':
            print("✅ Bot started successfully!")
            
            # Sprawdź status
            status_response = requests.get("http://localhost:5000/api/bot-status")
            status = status_response.json()
            print(f"📊 Bot status: {json.dumps(status, indent=2)}")
            
        else:
            print(f"❌ Bot startup failed: {result['message']}")
            
    except Exception as e:
        print(f"💥 Test failed: {e}")

if __name__ == "__main__":
    test_bot_start()
