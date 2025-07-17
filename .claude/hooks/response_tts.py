#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["python-dotenv"]
# ///

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue
import signal

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# TTS Components
try:
    from utils.tts.elevenlabs_client import TTSClient
    from utils.tts.text_processor import extract_text_from_response, word_count, strip_ansi_codes
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class ResponseTTSMonitor:
    """Monitor Claude responses from transcript files and provide TTS."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tts_client = None
        self.transcript_path = None
        self.last_position = 0
        self.running = False
        self.monitor_thread = None
        self.response_queue = queue.Queue()
        
        # Initialize TTS client if available
        if TTS_AVAILABLE and config.get('api_key'):
            try:
                self.tts_client = TTSClient(api_key=config['api_key'])
            except Exception as e:
                print(f"Failed to initialize TTS client: {e}")
    
    def load_tts_config(self) -> Dict[str, Any]:
        """Load TTS configuration from .claude/tts.json."""
        config = {}
        
        try:
            tts_config_path = Path.cwd() / '.claude' / 'tts.json'
            if tts_config_path.exists():
                with open(tts_config_path, 'r') as f:
                    config = json.load(f)
        except Exception:
            pass
        
        # Defaults
        defaults = {
            'enabled': True,
            'api_key': os.getenv('ELEVENLABS_API_KEY', ''),
            'default_voice_id': '6HWqrqOzDfj3UnywjJoZ',
            'min_word_count': 3,
            'max_word_count': 150,
            'response_delay': 1.0,  # Delay before speaking response
            'filter_tool_responses': True,
            'filter_code_blocks': True,
            'filter_file_paths': True,
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def should_speak_response(self, response_text: str) -> bool:
        """Determine if a response should be spoken."""
        if not response_text or not response_text.strip():
            return False
        
        # Check word count
        words = word_count(response_text)
        if words < self.config.get('min_word_count', 3):
            return False
        if words > self.config.get('max_word_count', 150):
            return False
        
        # Filter out tool responses if configured
        if self.config.get('filter_tool_responses', True):
            tool_indicators = [
                '```',  # Code blocks
                'function_calls',  # Function calls
                'antml:invoke',  # Tool invocations
                'file_path',  # File operations
                'command',  # Bash commands
            ]
            
            for indicator in tool_indicators:
                if indicator in response_text:
                    return False
        
        # Filter out file paths if configured
        if self.config.get('filter_file_paths', True):
            if response_text.startswith('/') or '/' in response_text:
                # Likely a file path
                return False
        
        return True
    
    def process_response_text(self, response_text: str) -> str:
        """Process and clean response text for TTS."""
        # Strip ANSI codes
        clean_text = strip_ansi_codes(response_text)
        
        # Remove markdown formatting
        clean_text = clean_text.replace('**', '').replace('*', '')
        clean_text = clean_text.replace('`', '')
        
        # Remove excessive whitespace
        clean_text = ' '.join(clean_text.split())
        
        # Truncate if too long
        max_words = self.config.get('max_word_count', 150)
        words = clean_text.split()
        if len(words) > max_words:
            clean_text = ' '.join(words[:max_words]) + '...'
        
        return clean_text
    
    def speak_response(self, response_text: str):
        """Speak a response using TTS."""
        if not self.tts_client:
            return
        
        try:
            voice_id = self.config.get('default_voice_id', '6HWqrqOzDfj3UnywjJoZ')
            self.tts_client.speak_text(response_text, voice_id=voice_id)
        except Exception as e:
            # Log error but don't fail
            print(f"TTS error: {e}")
    
    def parse_transcript_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line from the transcript file."""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def extract_assistant_response(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract assistant response text from a transcript entry."""
        try:
            if entry.get('type') == 'assistant':
                content = entry.get('content', '')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle structured content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return ' '.join(text_parts)
        except Exception:
            pass
        return None
    
    def monitor_transcript(self):
        """Monitor transcript file for new responses."""
        while self.running:
            try:
                if not self.transcript_path or not Path(self.transcript_path).exists():
                    time.sleep(1)
                    continue
                
                with open(self.transcript_path, 'r') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                    
                    if new_content:
                        lines = new_content.split('\n')
                        for line in lines:
                            if line.strip():
                                entry = self.parse_transcript_line(line)
                                if entry:
                                    response_text = self.extract_assistant_response(entry)
                                    if response_text and self.should_speak_response(response_text):
                                        clean_text = self.process_response_text(response_text)
                                        self.response_queue.put(clean_text)
                        
                        self.last_position = f.tell()
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(1)
    
    def process_response_queue(self):
        """Process queued responses for TTS."""
        while self.running:
            try:
                response_text = self.response_queue.get(timeout=1)
                
                # Add delay before speaking
                time.sleep(self.config.get('response_delay', 1.0))
                
                # Speak the response
                self.speak_response(response_text)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Response processing error: {e}")
    
    def start_monitoring(self, transcript_path: str):
        """Start monitoring a transcript file."""
        self.transcript_path = transcript_path
        self.running = True
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self.monitor_transcript)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start response processing thread
        self.response_thread = threading.Thread(target=self.process_response_queue)
        self.response_thread.daemon = True
        self.response_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

def main():
    """Main entry point for response TTS monitoring."""
    try:
        # Load configuration
        monitor = ResponseTTSMonitor({})
        config = monitor.load_tts_config()
        
        if not config.get('enabled', True):
            return
        
        # Read input data
        input_data = json.load(sys.stdin)
        transcript_path = input_data.get('transcript_path')
        
        if not transcript_path:
            return
        
        # Initialize monitor with config
        monitor = ResponseTTSMonitor(config)
        
        # Start monitoring
        monitor.start_monitoring(transcript_path)
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            monitor.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    
    except Exception as e:
        print(f"Response TTS error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()