#!/usr/bin/env python3
"""Test response TTS directly"""

import sys
import json
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path('.claude/hooks')))

# Test the response TTS system directly
from response_tts import ResponseTTSMonitor

# Create a test monitor
monitor = ResponseTTSMonitor({})
config = monitor.load_tts_config()

print("TTS Config loaded:")
print(json.dumps(config, indent=2))

# Test if voice ID is correct
print(f"\nVoice ID: {config.get('default_voice_id')}")

# Test the voice if we have an API key
if config.get('api_key'):
    print("Testing TTS with loaded config...")
    try:
        monitor_with_config = ResponseTTSMonitor(config)
        monitor_with_config.speak_response("This is a test of the response TTS system")
        print("TTS test completed successfully!")
    except Exception as e:
        print(f"TTS test failed: {e}")
else:
    print("No API key found in config")