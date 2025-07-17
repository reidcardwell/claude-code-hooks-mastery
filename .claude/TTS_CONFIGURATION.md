# TTS Configuration Guide

This document explains the Text-to-Speech (TTS) configuration system for Claude Code hooks.

## Configuration File Location

TTS settings are now stored in:
```
.claude/tts.json
```

**Note:** TTS settings should NOT be placed in `.claude/settings.json` as this is not a valid Claude Code configuration section.

## Configuration Structure

The `tts.json` file should contain the following structure:

```json
{
  "enabled": true,
  "default_voice_id": "6HWqrqOzDfj3UnywjJoZ",
  "word_count_threshold": 5,
  "max_word_count": 100,
  "excluded_tools": ["Read", "Grep", "LS", "TodoRead", "Glob"],
  "enable_response_tts": true,
  "response_tts": {
    "min_word_count": 3,
    "max_word_count": 150,
    "response_delay": 1.0,
    "filter_tool_responses": true,
    "filter_code_blocks": true,
    "filter_file_paths": true
  },
  "filter_settings": {
    "bash": {
      "speak_success": true,
      "speak_errors": true
    },
    "git": {
      "speak_success": true,
      "speak_errors": true
    },
    "file_operation": {
      "speak_write_operations": true,
      "speak_read_operations": false
    },
    "search": {
      "speak_search_results": false,
      "speak_errors": true
    }
  }
}
```

## Configuration Fields

### General Settings
- `enabled`: Enable/disable TTS functionality
- `default_voice_id`: ElevenLabs voice ID to use for TTS
- `word_count_threshold`: Minimum words needed to trigger TTS
- `max_word_count`: Maximum words to process through TTS
- `excluded_tools`: List of tools to exclude from TTS processing

### Response TTS Settings
- `enable_response_tts`: Enable TTS for Claude responses
- `response_tts.min_word_count`: Minimum words in response to trigger TTS
- `response_tts.max_word_count`: Maximum words in response to process
- `response_tts.response_delay`: Delay before speaking response (seconds)
- `response_tts.filter_tool_responses`: Filter out tool response content
- `response_tts.filter_code_blocks`: Filter out code blocks from responses
- `response_tts.filter_file_paths`: Filter out file paths from responses

### Tool-Specific Filter Settings
- `filter_settings.bash`: Configuration for bash command TTS
- `filter_settings.git`: Configuration for git command TTS
- `filter_settings.file_operation`: Configuration for file operation TTS
- `filter_settings.search`: Configuration for search tool TTS

## Environment Variables

TTS also requires the following environment variable:
```bash
ELEVENLABS_API_KEY=your_api_key_here
```

## Migration from settings.json

If you previously had TTS settings in `.claude/settings.json`, they should be removed from there and moved to `.claude/tts.json`. The `tts_settings` section is not a valid Claude Code configuration section.

### Before (Invalid)
```json
{
  "permissions": { ... },
  "hooks": { ... },
  "tts_settings": {
    "enabled": true,
    ...
  }
}
```

### After (Valid)
`.claude/settings.json`:
```json
{
  "permissions": { ... },
  "hooks": { ... }
}
```

`.claude/tts.json`:
```json
{
  "enabled": true,
  "default_voice_id": "6HWqrqOzDfj3UnywjJoZ",
  ...
}
```

## Testing Configuration

You can test your configuration by running:
```bash
python test_tts_config.py
```

This will verify that:
1. `tts.json` exists and has the correct structure
2. `settings.json` no longer contains `tts_settings`
3. The hooks can successfully load the configuration

## Troubleshooting

If TTS is not working:
1. Check that `.claude/tts.json` exists
2. Verify the JSON structure is valid
3. Ensure `ELEVENLABS_API_KEY` is set
4. Check that `enabled` is set to `true`
5. Run the test script to verify configuration loading