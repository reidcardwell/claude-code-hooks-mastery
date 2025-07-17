#!/usr/bin/env python3
"""Direct TTS test to verify voice ID and API key functionality"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the hooks directory to Python path
sys.path.insert(0, '/Users/reidcardwell/projects/claudecode/claude-code-hooks-mastery/.claude/hooks')

from utils.tts.elevenlabs_client import TTSClient

def test_direct_tts():
    """Test TTS with direct API call"""
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"✅ API key loaded: {api_key[:8]}...")
    
    # Test with Britney voice ID
    britney_voice_id = "6HWqrqOzDfj3UnywjJoZ"
    client = TTSClient(api_key=api_key)
    
    print(f"🎤 Testing with Britney voice ID: {britney_voice_id}")
    
    try:
        # Test synthesis
        audio_data = client.synthesize_speech(
            text="Testing Britney voice direct TTS",
            voice_id=britney_voice_id,
            model="eleven_monolingual_v1"
        )
        
        if audio_data:
            print("✅ Audio synthesis successful!")
            
            # Test playback
            success = client.play_audio(audio_data)
            if success:
                print("✅ Audio playback successful!")
                return True
            else:
                print("❌ Audio playback failed")
                return False
        else:
            print("❌ Audio synthesis failed")
            return False
            
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_tts()
    sys.exit(0 if success else 1)