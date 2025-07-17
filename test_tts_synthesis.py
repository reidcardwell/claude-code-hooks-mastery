#!/usr/bin/env python3
"""Test ElevenLabs TTS synthesis with limited API key"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_tts_synthesis():
    """Test TTS synthesis with API key"""
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"✅ API key loaded: {api_key[:8]}...{api_key[-4:]}")
    
    # Test TTS synthesis with Britney voice
    britney_voice_id = "6HWqrqOzDfj3UnywjJoZ"
    
    headers = {
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'text': 'Testing Britney voice TTS synthesis',
        'model_id': 'eleven_monolingual_v1',
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }
    
    try:
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{britney_voice_id}',
            headers=headers,
            json=data
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ TTS synthesis successful!")
            print(f"Audio data length: {len(response.content)} bytes")
            
            # Test if we can save and play the audio
            with open('test_audio.mp3', 'wb') as f:
                f.write(response.content)
            print("✅ Audio saved to test_audio.mp3")
            
            # Try to play with system audio
            os.system('afplay test_audio.mp3')
            print("✅ Audio playback attempted")
            
            return True
        else:
            print(f"❌ TTS synthesis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS synthesis failed: {e}")
        return False

if __name__ == "__main__":
    test_tts_synthesis()