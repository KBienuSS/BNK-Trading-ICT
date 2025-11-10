#!/usr/bin/env python3
"""
Test końcowy po zastosowaniu wszystkich poprawek
"""

import logging
import requests
import json
import time

def test_bot_start():
    print("🧪 Testing bot startup...")
    
    # Test z trybem wirtualnym (bez API keys)
    payload = {
        "real_trading": False
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
            print("✅ Bot started successfully in virtual mode!")
            
            # Sprawdź status bota
            time.sleep(2)
            status_response = requests.get("http://localhost:5000/api/bot-status")
            status = status_response.json()
            print(f"📊 Bot status: {json.dumps(status, indent=2)}")
            
        else:
            print(f"❌ Bot startup failed: {result['message']}")
            
    except Exception as e:
        print(f"💥 Test failed: {e}")

if __name__ == "__main__":
    test_bot_start()
