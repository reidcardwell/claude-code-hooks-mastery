#!/usr/bin/env python3
"""Test ElevenLabs API key validity"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_api_key():
    """Test if API key is valid"""
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"✅ API key loaded: {api_key[:8]}...{api_key[-4:]}")
    
    # Test API key with simple user info request
    headers = {
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get('https://api.elevenlabs.io/v1/user', headers=headers)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API key is valid!")
            user_info = response.json()
            print(f"User subscription: {user_info.get('subscription', {}).get('tier', 'unknown')}")
            return True
        else:
            print(f"❌ API key validation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return False

if __name__ == "__main__":
    test_api_key()