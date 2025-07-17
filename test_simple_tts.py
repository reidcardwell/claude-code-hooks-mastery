#!/usr/bin/env python3

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('ELEVENLABS_API_KEY')
print(f"API key loaded: {api_key is not None}")

if api_key:
    # Test direct API call
    voice_id = "6HWqrqOzDfj3UnywjJoZ"  # Britney voice
    text = "Testing local voice now"
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }
    
    print(f"Making API call to: {url}")
    response = requests.post(url, json=data, headers=headers)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ TTS API call successful")
        # Save audio file
        with open("test_voice.mp3", "wb") as f:
            f.write(response.content)
        print("Audio saved to test_voice.mp3")
        
        # Try to play it
        try:
            import subprocess
            subprocess.run(["afplay", "test_voice.mp3"], check=True)
            print("✅ Audio played successfully")
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
    else:
        print(f"❌ API call failed: {response.text}")
else:
    print("❌ No API key found")