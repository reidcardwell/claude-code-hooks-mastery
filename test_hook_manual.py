#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

# Test data to simulate a PostToolUse hook event
test_data = {
    "session_id": "test-session",
    "transcript_path": "/tmp/test.jsonl",
    "hook_event_name": "PostToolUse",
    "tool_name": "Bash",
    "tool_input": {
        "command": "echo 'test command'",
        "description": "Test command for TTS"
    },
    "tool_response": {
        "stdout": "Command completed successfully",
        "stderr": "",
        "interrupted": False,
        "isImage": False
    }
}

# Write test data to temporary file
test_file = Path("/tmp/test_hook_data.json")
with open(test_file, 'w') as f:
    json.dump(test_data, f)

print("Testing PostToolUse hook manually...")
print(f"Test data: {test_data}")

# Run the hook manually
hook_path = Path.cwd() / ".claude" / "hooks" / "post_tool_use.py"
print(f"Running hook: {hook_path}")

try:
    # Run the hook with test data
    result = subprocess.run([
        "uv", "run", str(hook_path)
    ], input=json.dumps(test_data), text=True, capture_output=True, timeout=30)
    
    print(f"Hook exit code: {result.returncode}")
    print(f"Hook stdout: {result.stdout}")
    print(f"Hook stderr: {result.stderr}")
    
    if result.returncode == 0:
        print("✅ Hook executed successfully")
    else:
        print("❌ Hook failed")
        
except subprocess.TimeoutExpired:
    print("❌ Hook timed out")
except Exception as e:
    print(f"❌ Error running hook: {e}")

# Clean up
test_file.unlink(missing_ok=True)