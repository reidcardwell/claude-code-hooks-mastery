#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

# Add the hooks directory to Python path
hooks_dir = Path(__file__).parent / ".claude" / "hooks"
sys.path.insert(0, str(hooks_dir))

# Test 1: Check if tts.json exists and is readable
tts_config_path = Path(__file__).parent / ".claude" / "tts.json"
print(f"1. TTS Config Path: {tts_config_path}")
print(f"   Exists: {tts_config_path.exists()}")

if tts_config_path.exists():
    try:
        with open(tts_config_path, 'r') as f:
            config = json.load(f)
        print(f"   Config loaded successfully: {config.get('enabled', 'unknown')}")
        print(f"   Voice ID: {config.get('default_voice_id', 'unknown')}")
    except Exception as e:
        print(f"   Error loading config: {e}")

# Test 2: Check API key loading
print(f"\n2. API Key Check:")
print(f"   ELEVENLABS_API_KEY in os.environ: {'ELEVENLABS_API_KEY' in os.environ}")
if 'ELEVENLABS_API_KEY' in os.environ:
    key = os.environ['ELEVENLABS_API_KEY']
    print(f"   API key length: {len(key)} characters")
    print(f"   API key prefix: {key[:10]}...")

# Test 3: Try to import and test TTS components
print(f"\n3. TTS Component Import Test:")
try:
    from utils.tts.elevenlabs_tts import text_to_speech_elevenlabs
    print("   ✅ elevenlabs_tts imported successfully")
    
    # Test basic TTS functionality
    result = text_to_speech_elevenlabs("Testing local voice", "6HWqrqOzDfj3UnywjJoZ")
    print(f"   TTS result: {result}")
    
except Exception as e:
    print(f"   ❌ Import/TTS error: {e}")

# Test 4: Check if hook can load configuration
print(f"\n4. Hook Configuration Loading Test:")
try:
    # Load dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✅ dotenv loaded")
    except ImportError:
        print("   ⚠️  dotenv not available")
    
    # Test configuration loading from hook
    from post_tool_use import load_tts_config
    config = load_tts_config()
    print(f"   ✅ Hook config loaded: enabled={config.get('enabled', False)}")
    print(f"   Voice ID: {config.get('default_voice_id', 'unknown')}")
    
except Exception as e:
    print(f"   ❌ Hook config error: {e}")

print(f"\n5. Current working directory: {os.getcwd()}")
print(f"   Python path: {sys.path[:3]}...")