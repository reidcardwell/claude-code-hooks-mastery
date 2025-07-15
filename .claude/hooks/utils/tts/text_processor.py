#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

"""
Text Processing Module for TTS Enhancement
Handles text extraction from various tool response formats
"""

import json
import re
from typing import Any, Optional, Union


def extract_text_from_response(response: Any) -> Optional[str]:
    """
    Extract text content from various tool response formats.
    
    Handles:
    - String responses directly
    - Dictionary responses (checking common keys)
    - List responses (concatenating items)
    - Other types with safe fallbacks
    
    Args:
        response: Tool response in any format
        
    Returns:
        Extracted text content or None if no text found
    """
    if response is None:
        return None
    
    # Handle string responses directly
    if isinstance(response, str):
        return response if response.strip() else None
    
    # Handle dictionary responses
    if isinstance(response, dict):
        return _extract_from_dict(response)
    
    # Handle list responses
    if isinstance(response, list):
        return _extract_from_list(response)
    
    # Handle other types (int, float, bool, etc.)
    try:
        text = str(response)
        return text if text.strip() else None
    except Exception:
        return None


def _extract_from_dict(data: dict) -> Optional[str]:
    """
    Extract text from dictionary response by checking common keys.
    
    Args:
        data: Dictionary response
        
    Returns:
        Extracted text or None
    """
    # Common keys to check for text content, in order of preference
    text_keys = [
        'content', 'text', 'message', 'output', 'result', 'data',
        'description', 'body', 'value', 'response', 'stdout', 'stderr'
    ]
    
    # Check for common text keys
    for key in text_keys:
        if key in data:
            value = data[key]
            if isinstance(value, str) and value.strip():
                return value
            elif isinstance(value, (dict, list)):
                # Recursively extract from nested structures
                nested_text = extract_text_from_response(value)
                if nested_text:
                    return nested_text
    
    # If no common keys found, try to extract from all string values
    text_parts = []
    for key, value in data.items():
        if isinstance(value, str) and value.strip():
            text_parts.append(value)
        elif isinstance(value, (dict, list)):
            nested_text = extract_text_from_response(value)
            if nested_text:
                text_parts.append(nested_text)
    
    if text_parts:
        return ' '.join(text_parts)
    
    return None


def _extract_from_list(data: list) -> Optional[str]:
    """
    Extract text from list response by concatenating items.
    
    Args:
        data: List response
        
    Returns:
        Concatenated text or None
    """
    if not data:
        return None
    
    text_parts = []
    for item in data:
        if isinstance(item, str) and item.strip():
            text_parts.append(item)
        elif isinstance(item, (dict, list)):
            # Recursively extract from nested structures
            nested_text = extract_text_from_response(item)
            if nested_text:
                text_parts.append(nested_text)
        else:
            # Convert other types to string
            try:
                text = str(item)
                if text.strip():
                    text_parts.append(text)
            except Exception:
                continue
    
    if text_parts:
        return ' '.join(text_parts)
    
    return None


def strip_ansi_codes(text: Optional[str]) -> Optional[str]:
    """
    Remove ANSI escape sequences from text to clean terminal output.
    
    This function removes:
    - Color codes (\\x1b[...m)
    - Cursor movement codes (\\x1b[...H, \\x1b[...A, etc.)
    - Clear screen codes (\\x1b[...J, \\x1b[...K)
    - Other ANSI control sequences
    
    Args:
        text: Input text that may contain ANSI escape sequences
        
    Returns:
        Clean text with ANSI codes removed, or None if input is None
    """
    if not text:
        return text
    
    # Comprehensive ANSI escape sequence pattern
    # Matches sequences starting with ESC [ followed by parameters and command letter
    ansi_pattern = r'\x1b\[[0-9;]*[a-zA-Z]'
    
    # Also match other common ANSI sequences
    # ESC followed by single character commands
    ansi_single_pattern = r'\x1b[a-zA-Z]'
    
    # Remove ANSI escape sequences
    clean_text = re.sub(ansi_pattern, '', text)
    clean_text = re.sub(ansi_single_pattern, '', clean_text)
    
    # Also remove other common terminal control sequences
    # Bell character, backspace, form feed, etc.
    control_chars_pattern = r'[\x00-\x08\x0B-\x1F\x7F]'
    clean_text = re.sub(control_chars_pattern, '', clean_text)
    
    return clean_text


def word_count(text: Optional[str]) -> int:
    """
    Count words in text using regex pattern r'\\S+' for accurate counting.
    
    This approach correctly handles:
    - Multiline content
    - Special characters and punctuation
    - Various whitespace types (spaces, tabs, newlines)
    - Technical terms, URLs, and symbols
    
    Args:
        text: Input text to count words in
        
    Returns:
        Integer count of words found (0 for None/empty text)
    """
    if not text:
        return 0
    
    # Use regex pattern to match sequences of non-whitespace characters
    # This pattern treats any sequence of non-whitespace as a "word"
    word_pattern = r'\S+'
    
    # Find all matches and return count
    words = re.findall(word_pattern, text)
    return len(words)


if __name__ == "__main__":
    # Test the functions with some examples
    test_cases = [
        "Simple string response",
        {"content": "Dict with content key"},
        {"message": "Dict with message key"},
        {"text": "Dict with text key", "other": "ignored"},
        ["List", "with", "multiple", "items"],
        [{"content": "Nested dict"}, "and string"],
        {"nested": {"content": "Deep nested content"}},
        None,
        "",
        123,
        {"empty": ""},
        []
    ]
    
    print("Testing extract_text_from_response:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        result = extract_text_from_response(test_case)
        print(f"{i:2d}. Input: {test_case}")
        print(f"    Output: {result}")
        print()
    
    # Test ANSI code stripping function
    ansi_test_cases = [
        "Normal text without ANSI codes",
        "\x1b[31mRed text\x1b[0m",
        "\x1b[1;32mBold green text\x1b[0m",
        "\x1b[33mYellow\x1b[0m and \x1b[34mblue\x1b[0m text",
        "\x1b[2J\x1b[H Clear screen and home cursor",
        "\x1b[1A\x1b[2K Move up and clear line",
        "Mixed \x1b[31mcolored\x1b[0m text with \x1b[1mbold\x1b[0m formatting",
        "\x1b[38;5;196mHigh color code\x1b[0m",
        "Control chars: \x07\x08\x0c\x7f",
        "",
        None,
        "No ANSI codes here",
    ]
    
    print("\nTesting strip_ansi_codes:")
    print("=" * 50)
    
    for i, test_text in enumerate(ansi_test_cases, 1):
        result = strip_ansi_codes(test_text)
        print(f"{i:2d}. Input: {repr(test_text)}")
        print(f"    Output: {repr(result)}")
        print()
    
    # Test word counting function
    word_count_tests = [
        "Simple text",
        "Multiple words in a sentence",
        "Text with\nnewlines and\ttabs",
        "Special@characters#and$symbols%",
        "URLs like https://example.com/path",
        "Code like: function() { return true; }",
        "Numbers 123 and 456.789",
        "Mixed: file.txt, path/to/dir, user@email.com",
        "",
        None,
        "   \n\t   ",  # Only whitespace
        "a",  # Single character
        "a b c d e f g h i j k l m n o p q r s t u",  # 21 words
        "a b c d e f g h i j k l m n o p q r s t"  # 20 words
    ]
    
    print("\nTesting word_count:")
    print("=" * 50)
    
    for i, test_text in enumerate(word_count_tests, 1):
        count = word_count(test_text)
        print(f"{i:2d}. Input: {repr(test_text)}")
        print(f"    Word count: {count}")
        print()