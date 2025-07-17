#!/usr/bin/env python3
"""Direct test of TTS voice ID to verify Britney voice is working"""

import sys
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path('.claude/hooks')))

# Test the TTS client directly
from utils.tts.elevenlabs_client import TTSClient
import os

# Get API key
api_key = os.getenv('ELEVENLABS_API_KEY')
if not api_key:
    print("No ELEVENLABS_API_KEY found")
    exit(1)

# Test with correct Britney voice ID
client = TTSClient(api_key=api_key)
print("Testing Britney voice with ID: 6HWqrqOzDfj3UnywjJoZ")
client.speak_text("This is a test of Britney's voice from the TTS client", voice_id="6HWqrqOzDfj3UnywjJoZ")
print("TTS test completed!")