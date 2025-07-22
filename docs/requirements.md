# Voice Command Results Improvement Plan

## Overview
Enhance the Claude Code hooks system to automatically speak command results using text-to-speech when the output is concise (≤20 words). This provides immediate audio feedback for short command outputs while avoiding verbose responses.

## Current Architecture Analysis

### Existing Components
- **TTS Script**: `.claude/hooks/utils/tts/elevenlabs_tts.py`
  - Uses ElevenLabs Turbo v2.5 model
  - Accepts text as command line argument
  - Handles API authentication and error management
  - Supports multiple voice options (David, Cornelius, Britney)

- **Hook System**: `.claude/hooks/post_tool_use.py`
  - Receives JSON data from Claude Code after tool execution
  - Logs tool usage data including tool_response content
  - Currently only performs logging, no processing

### Data Structure
PostToolUse hook receives:
```json
{
  "session_id": "...",
  "transcript_path": "...",
  "hook_event_name": "PostToolUse",
  "tool_name": "Read|Bash|Write|Edit|...",
  "tool_input": {...},
  "tool_response": {
    "type": "text|file|...",
    "content": "...",
    "file": {...}
  }
}
```

## Proposed Solution

### 1. Enhanced PostToolUse Hook
**File**: `.claude/hooks/post_tool_use.py`

**New Functionality**:
- Extract meaningful content from tool responses
- Count words in the response
- Trigger TTS for responses ≤20 words
- Handle different response types appropriately
- Maintain existing logging functionality

**Implementation Strategy**:
```python
def extract_response_text(tool_response):
    """Extract readable text from various tool response types"""
    if tool_response.get('type') == 'text':
        return tool_response.get('content', '')
    elif tool_response.get('type') == 'file':
        # For file responses, use a brief summary
        file_info = tool_response.get('file', {})
        return f"File {file_info.get('filePath', 'unknown')} read successfully"
    # Add more type handlers as needed
    return ""

def should_speak_result(text, tool_name):
    """Determine if result should be spoken based on word count and tool type"""
    if not text.strip():
        return False
    
    words = text.split()
    if len(words) > 20:
        return False
    
    # Skip certain tools that typically have verbose output
    skip_tools = ['Read', 'Grep', 'LS']  # Configurable
    if tool_name in skip_tools:
        return False
    
    return True

def speak_result(text):
    """Invoke TTS script to speak the result"""
    try:
        subprocess.run([
            'uv', 'run', 
            '.claude/hooks/utils/tts/elevenlabs_tts.py',
            text
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Fail silently to avoid interrupting workflow
        pass
```

### 2. Word Counting Algorithm
**Criteria for Speaking**:
- Text length ≤20 words
- Non-empty content
- Exclude certain verbose tools (configurable)
- Handle multiline content appropriately

**Word Count Logic**:
```python
def count_words(text):
    """Count words in text, handling multiline and special characters"""
    # Remove excessive whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and filter empty strings
    words = [word for word in cleaned.split() if word]
    return len(words)
```

### 3. Tool-Specific Handling
**Bash Commands**:
- Speak short command outputs (exit codes, brief confirmations)
- Skip verbose outputs like file listings, long logs

**File Operations**:
- For Read: Skip content, optionally speak "File loaded"
- For Write/Edit: Speak confirmation messages
- For LS: Skip directory listings

**Other Tools**:
- Grep: Skip matches, optionally speak match count
- Git commands: Speak brief status updates

### 4. Configuration Options
**Settings Integration**:
Add configuration to `.claude/settings.json`:
```json
{
  "tts_settings": {
    "enabled": true,
    "max_words": 20,
    "voice_id": "6sFKzaJr574YWVu4UuJF",
    "skip_tools": ["Read", "Grep", "LS"],
    "speak_confirmations": true
  }
}
```

### 5. Error Handling & Fallbacks
**Robust Implementation**:
- TTS failures should not interrupt Claude Code workflow
- Graceful degradation when ElevenLabs API is unavailable
- Fallback to system TTS if configured
- Comprehensive logging for debugging

**Error Scenarios**:
- API key missing/invalid
- Network connectivity issues
- Audio device unavailable
- Malformed tool response data

### 6. Performance Considerations
**Optimization**:
- Asynchronous TTS execution to avoid blocking
- Cache voice selections to reduce API calls
- Intelligent content filtering to reduce processing

**Resource Management**:
- Limit concurrent TTS requests
- Handle API rate limiting gracefully
- Memory-efficient text processing

## Implementation Phases

### Phase 1: Core Integration
1. Modify `post_tool_use.py` to include TTS functionality
2. Implement word counting and content extraction
3. Add basic tool filtering
4. Test with common command outputs

### Phase 2: Enhanced Filtering
1. Implement tool-specific handling logic
2. Add configuration options to settings.json
3. Create content summarization for verbose tools
4. Handle edge cases and error conditions

### Phase 3: Advanced Features
1. Add voice selection options
2. Implement custom vocabulary/pronunciation
3. Add user preferences for different tool types
4. Create audio feedback for different types of results

### Phase 4: Polish & Optimization
1. Performance optimization
2. Comprehensive error handling
3. User documentation
4. Testing across different environments

## Testing Strategy

### Unit Tests
- Word counting accuracy
- Content extraction from various tool responses
- Configuration parsing
- Error handling scenarios

### Integration Tests
- Hook system integration
- TTS script execution
- Settings configuration
- Real-world command scenarios

### User Acceptance Tests
- Voice quality and clarity
- Appropriate content filtering
- Non-intrusive operation
- Configuration flexibility

## Success Metrics
- Commands with ≤20 word outputs are spoken within 2 seconds
- No interruption to normal Claude Code workflow
- <1% failure rate for TTS operations
- User satisfaction with voice feedback timing and content

## Risks & Mitigations

### Technical Risks
- **API Rate Limiting**: Implement request queuing and backoff
- **Audio Device Conflicts**: Graceful fallback to silent operation
- **Performance Impact**: Asynchronous processing and optimization

### User Experience Risks
- **Unwanted Audio**: Comprehensive configuration options
- **Timing Issues**: Careful synchronization with tool completion
- **Content Appropriateness**: Intelligent filtering and summarization

## Future Enhancements
- Multi-language support
- Custom voice training
- Integration with system notifications
- Visual feedback synchronization
- Command result categorization
- Smart content summarization for longer outputs

## Conclusion
This enhancement will provide immediate audio feedback for short command results while maintaining the existing Claude Code workflow. The implementation prioritizes reliability, performance, and user control over the voice feedback experience.