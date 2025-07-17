#!/usr/bin/env python3
"""Test the post_tool_use hook with sample data"""

import json
import subprocess
import tempfile
import os

# Sample tool usage data
sample_data = {
    "tool_name": "Bash",
    "tool_input": {
        "command": "echo 'Hello world'",
        "description": "Test command"
    },
    "tool_response": {
        "output": "Hello world",
        "error": None
    }
}

def test_hook():
    """Test the hook with sample data"""
    # Create temporary file with sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        temp_file = f.name
    
    try:
        # Run the hook with sample data
        result = subprocess.run([
            'python', 
            '/Users/reidcardwell/projects/claudecode/claude-code-hooks-mastery/.claude/hooks/post_tool_use.py'
        ], 
        input=json.dumps(sample_data),
        text=True,
        capture_output=True
        )
        
        print(f"Hook exit code: {result.returncode}")
        print(f"Hook stdout: {result.stdout}")
        print(f"Hook stderr: {result.stderr}")
        
        return result.returncode == 0
        
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    success = test_hook()
    print(f"Test {'PASSED' if success else 'FAILED'}")