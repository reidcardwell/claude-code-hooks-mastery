#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

# Test if the hook is being called at all
test_data = {
    "session_id": "test-session-direct",
    "transcript_path": "/tmp/test.jsonl",
    "hook_event_name": "PostToolUse",
    "tool_name": "Bash",
    "tool_input": {
        "command": "echo 'Hello World'",
        "description": "Test command"
    },
    "tool_response": {
        "stdout": "Hello World",
        "stderr": "",
        "interrupted": False,
        "isImage": False
    }
}

print("Testing PostToolUse hook directly...")
hook_path = Path.cwd() / ".claude" / "hooks" / "post_tool_use.py"

# Run the hook with test data
try:
    result = subprocess.run([
        "uv", "run", str(hook_path)
    ], input=json.dumps(test_data), text=True, capture_output=True, timeout=10)
    
    print(f"Hook exit code: {result.returncode}")
    print(f"Hook stdout: {result.stdout}")
    print(f"Hook stderr: {result.stderr}")
    
    # Check logs
    print("\nTTS Debug Log (last 5 lines):")
    try:
        with open("logs/tts_debug.log", "r") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(line.strip())
    except FileNotFoundError:
        print("No TTS debug log found")
    
except subprocess.TimeoutExpired:
    print("Hook timed out")
except Exception as e:
    print(f"Error: {e}")