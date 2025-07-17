#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["requests>=2.31.0", "python-dotenv"]
# ///

import json
import os
import sys
import time
import subprocess
import requests
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# TTS Components - Conditional imports to handle missing components gracefully
try:
    from utils.tts.text_processor import extract_text_from_response, word_count, strip_ansi_codes
    from utils.tts.tool_filters import (
        ToolFilterRegistry, 
        get_global_registry,
        initialize_global_registry,
        BashFilter,
        GitFilter, 
        FileOperationFilter,
        SearchFilter
    )
    TTS_AVAILABLE = True
except ImportError as e:
    TTS_AVAILABLE = False
    # Log import error for debugging
    debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
    with open(debug_log_path, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp}: TTS import error: {str(e)}\n")
    
    # Fallback functions for when TTS components aren't available
    def extract_text_from_response(response):
        return str(response) if response else ""
    
    def word_count(text):
        return len(text.split()) if text else 0
    
    def strip_ansi_codes(text):
        return text

def simple_tts_speak(text, voice_id="6HWqrqOzDfj3UnywjJoZ"):
    """
    Simple TTS synthesis and playback that bypasses complex validation.
    
    Args:
        text: Text to speak
        voice_id: ElevenLabs voice ID (defaults to Britney)
    
    Returns:
        bool: True if successful, False otherwise
    """
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        return False
    
    headers = {
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'text': text,
        'model_id': 'eleven_monolingual_v1',
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }
    
    try:
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            # Save audio temporarily
            audio_file = '/tmp/hook_tts_output.mp3'
            with open(audio_file, 'wb') as f:
                f.write(response.content)
            
            # Play with afplay (macOS)
            result = subprocess.run(['afplay', audio_file], capture_output=True)
            
            # Clean up
            try:
                os.remove(audio_file)
            except:
                pass
            
            return result.returncode == 0
        else:
            return False
            
    except Exception:
        return False

def load_tts_config():
    """Load TTS configuration from .claude/tts.json or environment variables."""
    config = {}
    
    # Try to load from TTS configuration file
    try:
        tts_config_path = Path.cwd() / '.claude' / 'tts.json'
        if tts_config_path.exists():
            with open(tts_config_path, 'r') as f:
                config = json.load(f)
    except Exception:
        pass
    
    # Default configuration values
    defaults = {
        'enabled': True,
        'api_key': os.getenv('ELEVENLABS_API_KEY', ''),
        'default_voice_id': '6HWqrqOzDfj3UnywjJoZ',  # Britney
        'word_count_threshold': 1,  # Minimum words to trigger TTS
        'max_word_count': 200,  # Maximum words to process
        'timeout': 30,
        'max_retries': 3,
        'retry_delay': 1.0,
        'rate_limit_requests': 100,
        'rate_limit_window': 60,
        'enable_caching': True,
        'cache_duration': 3600,
        'excluded_tools': ['Read', 'Grep', 'LS', 'TodoRead'],
        'filter_settings': {
            'bash': {'speak_success': True, 'speak_errors': True},
            'git': {'speak_success': True, 'speak_errors': True},
            'file_operation': {'speak_write_operations': True, 'speak_read_operations': False},
            'search': {'speak_search_results': False, 'speak_errors': True}
        }
    }
    
    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def should_process_tool_for_tts(tool_name, tool_input, tool_response, tts_config):
    """
    Determine if a tool use should trigger TTS processing with enhanced filtering.
    
    Args:
        tool_name: Name of the tool that was used
        tool_input: Input data passed to the tool
        tool_response: Response from the tool
        tts_config: TTS configuration dictionary
    
    Returns:
        bool: True if TTS should be triggered, False otherwise
    """
    # Debug logging
    debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
    with open(debug_log_path, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp}: Checking TTS for tool '{tool_name}'\n")
    
    # First check: Excluded tools in configuration
    excluded_tools = tts_config.get('excluded_tools', [])
    if tool_name in excluded_tools:
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Tool '{tool_name}' is excluded by configuration\n")
        return False
    
    # Second check: Tool-specific filtering via registry
    if TTS_AVAILABLE:
        try:
            # Initialize registry with TTS configuration path
            tts_config_path = str(Path.cwd() / '.claude' / 'tts.json')
            initialize_global_registry(tts_config_path)
            
            registry = get_global_registry()
            
            # Get tool-specific filter
            tool_filter = registry.get_filter(tool_name)
            if tool_filter:
                # Use tool-specific should_speak logic
                result = tool_filter.should_speak(tool_name, tool_response)
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Tool filter result for '{tool_name}': {result}\n")
                return result
            else:
                # No specific filter found, use registry fallback
                result = registry.should_speak(tool_name, tool_response)
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Registry fallback result for '{tool_name}': {result}\n")
                return result
            
        except Exception as e:
            # Log filter error but continue with fallback
            with open(debug_log_path, 'a') as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp}: Filter registry error for '{tool_name}': {str(e)}\n")
    
    # Third check: Basic fallback logic for when filters unavailable
    # Skip tools that typically produce verbose output
    verbose_tools = {'Read', 'Grep', 'LS', 'TodoRead', 'Glob'}
    if tool_name in verbose_tools:
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Tool '{tool_name}' is verbose, skipping\n")
        return False
    
    # Process tools that indicate completion or success
    completion_tools = {'Edit', 'Write', 'Bash', 'MultiEdit', 'TodoWrite'}
    if tool_name in completion_tools:
        # Check if the operation was successful
        if hasattr(tool_response, 'get'):
            # For structured responses, check for error indicators
            if tool_response.get('error') or tool_response.get('stderr'):
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Tool '{tool_name}' had errors, skipping\n")
                return False
        
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Tool '{tool_name}' is completion tool, processing\n")
        return True
    
    # Default: Allow processing for unknown tools
    with open(debug_log_path, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp}: Tool '{tool_name}' using default allow\n")
    return True

def extract_tts_text(tool_name, tool_input, tool_response, tts_config):
    """
    Extract appropriate text for TTS from tool usage data with enhanced filtering.
    
    Args:
        tool_name: Name of the tool that was used
        tool_input: Input data passed to the tool
        tool_response: Response from the tool
        tts_config: TTS configuration dictionary
    
    Returns:
        str: Text to be spoken, or empty string if no text should be spoken
    """
    if not TTS_AVAILABLE:
        return ""
    
    try:
        # Use text processor to extract relevant text
        response_text = extract_text_from_response(tool_response)
        
        # Strip ANSI codes and clean up text
        clean_text = strip_ansi_codes(response_text)
        
        # First check: Word count verification
        word_count_val = word_count(clean_text)
        min_words = tts_config.get('word_count_threshold', 1)
        max_words = tts_config.get('max_word_count', 200)
        
        if word_count_val < min_words or word_count_val > max_words:
            return ""
        
        # Second check: Tool-specific filtering using filter registry
        if TTS_AVAILABLE:
            try:
                # Get tool filter registry
                tts_config_path = str(Path.cwd() / '.claude' / 'tts.json')
                initialize_global_registry(tts_config_path)
                registry = get_global_registry()
                
                # Get tool-specific filter
                tool_filter = registry.get_filter(tool_name)
                if tool_filter:
                    # Apply tool-specific should_speak logic
                    if not tool_filter.should_speak(tool_name, tool_response):
                        return ""
                    
                    # Generate tool-specific custom message
                    custom_message = tool_filter.get_custom_message(tool_name, tool_response)
                    if custom_message:
                        return custom_message
                
            except Exception as e:
                # Log filter error but continue with fallback
                debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Filter error for '{tool_name}': {str(e)}\n")
        
        # Fallback: Generate contextual message based on tool and success
        return generate_contextual_message(tool_name, tool_input, tool_response, clean_text)
        
    except Exception as e:
        # Log extraction error and fallback to simple success message
        debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Text extraction error for '{tool_name}': {str(e)}\n")
        return "Task completed successfully."

def generate_contextual_message(tool_name, tool_input, tool_response, extracted_text):
    """
    Generate a contextual message based on the tool used and operation result.
    
    Args:
        tool_name: Name of the tool that was used
        tool_input: Input data passed to the tool
        tool_response: Response from the tool
        extracted_text: Extracted text from the response
    
    Returns:
        str: Contextual message for TTS
    """
    # Tool-specific message generation
    if tool_name == 'Edit':
        return "File edited successfully."
    elif tool_name == 'Write':
        return "File written successfully."
    elif tool_name == 'MultiEdit':
        return "Multiple edits completed successfully."
    elif tool_name == 'Bash':
        # For bash commands, provide more context
        command = tool_input.get('command', '') if isinstance(tool_input, dict) else str(tool_input)
        if 'git' in command:
            return "Git operation completed."
        elif 'npm' in command or 'yarn' in command:
            return "Package operation completed."
        elif 'test' in command:
            return "Test execution completed."
        else:
            return "Command executed successfully."
    elif tool_name == 'TodoWrite':
        return "Task list updated."
    else:
        # Generic success message
        return "Operation completed successfully."

def process_tts_for_tool_use(input_data, tts_config):
    """
    Process tool usage data for TTS output.
    
    Args:
        input_data: JSON data from the tool usage event
        tts_config: TTS configuration dictionary
    """
    try:
        # Extract tool information
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        tool_response = input_data.get('tool_response', {})
        
        # Check if this tool usage should trigger TTS
        should_process = should_process_tool_for_tts(tool_name, tool_input, tool_response, tts_config)
        debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Should process '{tool_name}': {should_process}\n")
        
        if not should_process:
            return
        
        # Extract text for TTS
        tts_text = extract_tts_text(tool_name, tool_input, tool_response, tts_config)
        
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: TTS text extracted: '{tts_text}'\n")
        
        if not tts_text:
            return
        
        # Use simple TTS synthesis
        if tts_config.get('api_key') or os.getenv('ELEVENLABS_API_KEY'):
            try:
                # Speak the text using configured voice
                voice_id = tts_config.get('default_voice_id', '6HWqrqOzDfj3UnywjJoZ')
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Attempting TTS with voice {voice_id}: '{tts_text}'\n")
                
                result = simple_tts_speak(tts_text, voice_id=voice_id)
                
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: TTS result: {result}\n")
                
            except Exception as e:
                # Log TTS errors but don't fail the hook
                debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: TTS error: {str(e)}\n")
    
    except Exception as e:
        # Silently handle any errors to avoid breaking the hook
        pass

def start_response_tts_monitoring(input_data, tts_config):
    """
    Start response TTS monitoring for the session.
    
    Args:
        input_data: JSON data from the tool usage event
        tts_config: TTS configuration dictionary
    """
    try:
        transcript_path = input_data.get('transcript_path')
        if not transcript_path:
            return
        
        # Check if monitoring is already running for this session
        session_id = input_data.get('session_id')
        monitor_flag_path = Path.cwd() / 'logs' / f'response_tts_monitor_{session_id}.flag'
        
        if monitor_flag_path.exists():
            return  # Already monitoring this session
        
        # Create flag file to prevent multiple monitors
        monitor_flag_path.touch()
        
        # Start response TTS monitor as background process
        import subprocess
        import threading
        
        def run_monitor():
            try:
                # Prepare monitor input
                monitor_input = {
                    'transcript_path': transcript_path,
                    'session_id': session_id
                }
                
                # Run response TTS monitor
                subprocess.run([
                    'uv', 'run', 
                    str(Path.cwd() / '.claude' / 'hooks' / 'response_tts.py')
                ], input=json.dumps(monitor_input), text=True, timeout=None)
                
            except Exception as e:
                # Log error
                debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
                with open(debug_log_path, 'a') as f:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp}: Response TTS monitor error: {str(e)}\n")
            finally:
                # Clean up flag file
                if monitor_flag_path.exists():
                    monitor_flag_path.unlink()
        
        # Start monitor in background thread
        monitor_thread = threading.Thread(target=run_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    except Exception as e:
        # Silently handle any errors to avoid breaking the hook
        pass

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Load TTS configuration
        tts_config = load_tts_config()
        
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
        
        # TTS processing integration point
        debug_log_path = Path.cwd() / 'logs' / 'tts_debug.log'
        with open(debug_log_path, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: PostToolUse hook executed. TTS_AVAILABLE={TTS_AVAILABLE}, enabled={tts_config.get('enabled', True)}\n")
        
        if TTS_AVAILABLE and tts_config.get('enabled', True):
            # Debug logging
            with open(debug_log_path, 'a') as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp}: PostToolUse TTS triggered for {input_data.get('tool_name', 'unknown')}\n")
            
            process_tts_for_tool_use(input_data, tts_config)
            
            # Start response TTS monitoring if enabled (temporarily disabled to fix timeout)
            # if tts_config.get('enable_response_tts', True):
            #     start_response_tts_monitoring(input_data, tts_config)
        
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