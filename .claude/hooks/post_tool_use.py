#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

import json
import os
import sys
import time
from pathlib import Path

# TTS Components - Conditional imports to handle missing components gracefully
try:
    from utils.tts.text_processor import extract_text_from_response, word_count, strip_ansi_codes
    from utils.tts.tool_filters import (
        ToolFilterRegistry, 
        get_global_registry,
        BashFilter,
        GitFilter, 
        FileOperationFilter,
        SearchFilter
    )
    from utils.tts.elevenlabs_client import TTSClient
    TTS_AVAILABLE = True
except ImportError as e:
    TTS_AVAILABLE = False
    # Fallback functions for when TTS components aren't available
    def extract_text_from_response(response):
        return str(response) if response else ""
    
    def word_count(text):
        return len(text.split()) if text else 0
    
    def strip_ansi_codes(text):
        return text

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Ensure log directory exists
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'post_tool_use.json'
        
        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []
        
        # Append new data
        log_data.append(input_data)
        
        # Write back to file with formatting
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        sys.exit(0)
        
    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)

if __name__ == '__main__':
    main()