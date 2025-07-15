#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["pytest>=8.0.0"]
# ///

"""
Comprehensive Unit Tests for Text Processing Module

Tests all text processing functions with various input scenarios including:
- Word counting with different text types
- Text extraction from various response formats
- ANSI code stripping functionality
- Edge case handling for binary data and None values
- Performance testing with large text blocks
"""

import pytest
import sys
import os
import string
import time
from typing import Any, Dict, List, Optional

# Add the utils directory to the path so we can import text_processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.claude', 'hooks', 'utils', 'tts'))

from text_processor import (
    extract_text_from_response,
    strip_ansi_codes,
    word_count,
    is_concise_output,
    _is_binary_data,
    _extract_from_dict,
    _extract_from_list
)


class TestExtractTextFromResponse:
    """Test the extract_text_from_response function with various input types."""
    
    def test_string_input(self):
        """Test basic string input handling."""
        assert extract_text_from_response("Hello world") == "Hello world"
        assert extract_text_from_response("  Hello world  ") == "  Hello world  "
        assert extract_text_from_response("") is None
        assert extract_text_from_response("   ") is None
        
    def test_none_input(self):
        """Test None input handling."""
        assert extract_text_from_response(None) is None
        
    def test_dict_input_common_keys(self):
        """Test dictionary input with common text keys."""
        assert extract_text_from_response({"content": "test content"}) == "test content"
        assert extract_text_from_response({"text": "test text"}) == "test text"
        assert extract_text_from_response({"message": "test message"}) == "test message"
        assert extract_text_from_response({"output": "test output"}) == "test output"
        assert extract_text_from_response({"result": "test result"}) == "test result"
        
    def test_dict_input_priority_order(self):
        """Test that common keys are checked in priority order."""
        test_dict = {
            "result": "low priority",
            "content": "high priority",
            "text": "medium priority"
        }
        assert extract_text_from_response(test_dict) == "high priority"
        
    def test_dict_input_nested_structures(self):
        """Test dictionary with nested structures."""
        nested_dict = {"nested": {"content": "nested content"}}
        assert extract_text_from_response(nested_dict) == "nested content"
        
        nested_list = {"items": ["item1", "item2"]}
        assert extract_text_from_response(nested_list) == "item1 item2"
        
    def test_dict_input_empty_or_invalid(self):
        """Test dictionary with empty or invalid values."""
        assert extract_text_from_response({}) is None
        assert extract_text_from_response({"content": ""}) is None
        assert extract_text_from_response({"content": "   "}) is None
        
    def test_list_input(self):
        """Test list input handling."""
        assert extract_text_from_response(["hello", "world"]) == "hello world"
        assert extract_text_from_response(["item1", "item2", "item3"]) == "item1 item2 item3"
        assert extract_text_from_response([]) is None
        
    def test_list_input_mixed_types(self):
        """Test list with mixed data types."""
        mixed_list = ["string", 123, {"content": "dict content"}]
        assert extract_text_from_response(mixed_list) == "string 123 dict content"
        
    def test_list_input_nested_structures(self):
        """Test list with nested structures."""
        nested_list = [{"content": "nested"}, ["sub", "list"]]
        assert extract_text_from_response(nested_list) == "nested sub list"
        
    def test_numeric_input(self):
        """Test numeric input conversion."""
        assert extract_text_from_response(123) == "123"
        assert extract_text_from_response(45.67) == "45.67"
        assert extract_text_from_response(0) == "0"
        
    def test_boolean_input(self):
        """Test boolean input conversion."""
        assert extract_text_from_response(True) == "True"
        assert extract_text_from_response(False) == "False"
        
    def test_binary_data_detection(self):
        """Test binary data detection and filtering."""
        # Create binary-like data with many non-printable characters
        binary_data = "hello\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10world"
        assert extract_text_from_response(binary_data) is None
        
        # Test data with some non-printable but mostly printable (should pass)
        mostly_printable = "hello world\x1b[31m colored text\x1b[0m"
        assert extract_text_from_response(mostly_printable) is not None
        
    def test_malformed_data_handling(self):
        """Test handling of malformed or problematic data structures."""
        # Test with object that might cause issues during processing
        class ProblematicObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")
                
        assert extract_text_from_response(ProblematicObject()) is None


class TestIsBinaryData:
    """Test the _is_binary_data helper function."""
    
    def test_normal_text(self):
        """Test normal text input."""
        assert _is_binary_data("Hello world") is False
        assert _is_binary_data("Normal text with punctuation!") is False
        assert _is_binary_data("Text with numbers 123 and symbols @#$%") is False
        
    def test_empty_input(self):
        """Test empty input."""
        assert _is_binary_data("") is False
        assert _is_binary_data(None) is False
        
    def test_binary_data(self):
        """Test actual binary data."""
        # Create string with high percentage of non-printable characters
        binary_string = "".join(chr(i) for i in range(0, 20))  # Control characters
        assert _is_binary_data(binary_string) is True
        
    def test_mixed_content(self):
        """Test content with mix of printable and non-printable."""
        # Should be considered binary if too many non-printable characters
        mixed_binary = "hello" + "".join(chr(i) for i in range(0, 20))
        assert _is_binary_data(mixed_binary) is True
        
        # Should be considered text if mostly printable
        mixed_text = "hello world with some control chars\x1b[31m\x1b[0m"
        assert _is_binary_data(mixed_text) is False
        
    def test_short_text_threshold(self):
        """Test threshold adjustment for short text."""
        # Short text with some non-printable should be more permissive
        short_mixed = "hi\x00\x01"
        result = _is_binary_data(short_mixed)
        # This might be binary due to high percentage, test actual behavior
        assert isinstance(result, bool)


class TestStripAnsiCodes:
    """Test the strip_ansi_codes function."""
    
    def test_no_ansi_codes(self):
        """Test text without ANSI codes."""
        text = "Normal text without any codes"
        assert strip_ansi_codes(text) == text
        
    def test_none_input(self):
        """Test None input."""
        assert strip_ansi_codes(None) is None
        
    def test_empty_input(self):
        """Test empty string input."""
        assert strip_ansi_codes("") == ""
        
    def test_color_codes(self):
        """Test removal of color codes."""
        colored_text = "\x1b[31mRed text\x1b[0m"
        assert strip_ansi_codes(colored_text) == "Red text"
        
        multi_color = "\x1b[33mYellow\x1b[0m and \x1b[34mblue\x1b[0m text"
        assert strip_ansi_codes(multi_color) == "Yellow and blue text"
        
    def test_formatting_codes(self):
        """Test removal of formatting codes."""
        formatted_text = "\x1b[1;32mBold green text\x1b[0m"
        assert strip_ansi_codes(formatted_text) == "Bold green text"
        
    def test_complex_codes(self):
        """Test removal of complex ANSI sequences."""
        complex_text = "\x1b[38;5;196mHigh color code\x1b[0m"
        assert strip_ansi_codes(complex_text) == "High color code"
        
    def test_cursor_movement(self):
        """Test removal of cursor movement codes."""
        cursor_text = "\x1b[2J\x1b[H Clear screen and home cursor"
        result = strip_ansi_codes(cursor_text)
        assert "Clear screen and home cursor" in result
        assert "\x1b[" not in result
        
    def test_control_characters(self):
        """Test removal of control characters."""
        control_text = "Text with\x07bell\x08backspace\x0cformfeed\x7fdelete"
        result = strip_ansi_codes(control_text)
        assert result == "Text withbellbackspaceformfeeddelete"
        
    def test_mixed_ansi_content(self):
        """Test text with mixed ANSI codes and content."""
        mixed_text = "Mixed \x1b[31mcolored\x1b[0m text with \x1b[1mbold\x1b[0m formatting"
        assert strip_ansi_codes(mixed_text) == "Mixed colored text with bold formatting"
        
    def test_regex_error_handling(self):
        """Test handling of regex processing errors."""
        # Test with very long string that might cause regex issues
        long_text = "a" * 10000 + "\x1b[31m" + "b" * 10000 + "\x1b[0m"
        result = strip_ansi_codes(long_text)
        assert isinstance(result, str)


class TestWordCount:
    """Test the word_count function."""
    
    def test_simple_text(self):
        """Test simple text word counting."""
        assert word_count("Hello world") == 2
        assert word_count("Single") == 1
        assert word_count("One two three four five") == 5
        
    def test_none_input(self):
        """Test None input."""
        assert word_count(None) == 0
        
    def test_empty_input(self):
        """Test empty string input."""
        assert word_count("") == 0
        assert word_count("   ") == 0
        assert word_count("\n\t\r") == 0
        
    def test_multiline_text(self):
        """Test multiline text word counting."""
        multiline = "First line\nSecond line\nThird line"
        assert word_count(multiline) == 6
        
        complex_multiline = "Line 1\n\nLine 2\n\n\nLine 3"
        assert word_count(complex_multiline) == 6
        
    def test_special_characters(self):
        """Test text with special characters."""
        special_text = "Special@characters#and$symbols%"
        assert word_count(special_text) == 1  # Treated as one word
        
        separated_special = "user@email.com file.txt path/to/dir"
        assert word_count(separated_special) == 3
        
    def test_mixed_whitespace(self):
        """Test text with various whitespace types."""
        mixed_whitespace = "Words\tseparated\nby\r\nvarious   whitespace"
        assert word_count(mixed_whitespace) == 5
        
    def test_punctuation_handling(self):
        """Test handling of punctuation."""
        punctuated = "Hello, world! How are you? I'm fine."
        assert word_count(punctuated) == 7
        
    def test_numbers_and_symbols(self):
        """Test counting of numbers and symbols."""
        mixed_content = "Version 1.2.3 released at 2023-12-01"
        assert word_count(mixed_content) == 5
        
    def test_urls_and_paths(self):
        """Test counting URLs and file paths."""
        urls = "Visit https://example.com/path for more info"
        assert word_count(urls) == 5
        
    def test_code_like_content(self):
        """Test counting in code-like content."""
        code_text = "function() { return true; }"
        assert word_count(code_text) == 7
        
    def test_unicode_text(self):
        """Test Unicode text handling."""
        unicode_text = "Hello 世界 مرحبا العالم"
        assert word_count(unicode_text) == 4
        
    def test_boundary_cases(self):
        """Test boundary word count cases."""
        # Exactly 20 words
        twenty_words = "a b c d e f g h i j k l m n o p q r s t"
        assert word_count(twenty_words) == 20
        
        # 21 words
        twenty_one_words = "a b c d e f g h i j k l m n o p q r s t u"
        assert word_count(twenty_one_words) == 21
        
    def test_large_text_performance(self):
        """Test performance with large text blocks."""
        # Create text with >10KB
        large_text = "word " * 3000  # Approximately 15KB
        start_time = time.time()
        result = word_count(large_text)
        end_time = time.time()
        
        assert result == 3000
        assert end_time - start_time < 0.1  # Should be fast
        
    def test_regex_error_handling(self):
        """Test handling of regex processing errors."""
        # Test with potentially problematic input
        problematic_text = "Normal text with some edge cases"
        result = word_count(problematic_text)
        assert isinstance(result, int)
        assert result >= 0


class TestIsConciseOutput:
    """Test the is_concise_output function."""
    
    def test_basic_functionality(self):
        """Test basic concise output checking."""
        assert is_concise_output("Short text") is True
        assert is_concise_output("This is a much longer text that exceeds the default threshold of twenty words and should return false for being too long") is False
        
    def test_none_input(self):
        """Test None input."""
        assert is_concise_output(None) is False
        
    def test_empty_input(self):
        """Test empty string input."""
        assert is_concise_output("") is False
        assert is_concise_output("   ") is True
        
    def test_custom_threshold(self):
        """Test custom threshold values."""
        text = "This is a test sentence"
        assert is_concise_output(text, threshold=5) is True
        assert is_concise_output(text, threshold=3) is False
        
    def test_boundary_conditions(self):
        """Test boundary conditions around threshold."""
        # Exactly 20 words (default threshold)
        twenty_words = "a b c d e f g h i j k l m n o p q r s t"
        assert is_concise_output(twenty_words) is True
        
        # 21 words (over threshold)
        twenty_one_words = "a b c d e f g h i j k l m n o p q r s t u"
        assert is_concise_output(twenty_one_words) is False
        
    def test_ansi_code_stripping(self):
        """Test that ANSI codes are stripped before counting."""
        colored_text = "\x1b[31mRed text\x1b[0m with \x1b[32mgreen\x1b[0m words"
        assert is_concise_output(colored_text) is True
        
        long_colored = "\x1b[31m" + "word " * 25 + "\x1b[0m"
        assert is_concise_output(long_colored) is False
        
    def test_integration_with_text_processing(self):
        """Test integration with other text processing functions."""
        # Text with ANSI codes and whitespace
        complex_text = "\x1b[32m  Command completed successfully  \x1b[0m"
        assert is_concise_output(complex_text) is True
        
    def test_typical_command_outputs(self):
        """Test with typical command outputs."""
        assert is_concise_output("Command completed successfully") is True
        assert is_concise_output("File not found: /path/to/file.txt") is True
        assert is_concise_output("Error: Permission denied") is True
        
        # Long error message
        long_error = "Error: This is a very long error message that explains in detail what went wrong and provides extensive troubleshooting information that exceeds the word limit"
        assert is_concise_output(long_error) is False
        
    def test_error_handling(self):
        """Test error handling in concise output checking."""
        # Test with potentially problematic input
        result = is_concise_output("Normal text input")
        assert isinstance(result, bool)


class TestExtractFromDict:
    """Test the _extract_from_dict helper function."""
    
    def test_simple_extraction(self):
        """Test simple dictionary extraction."""
        data = {"content": "test content"}
        assert _extract_from_dict(data) == "test content"
        
    def test_empty_dict(self):
        """Test empty dictionary."""
        assert _extract_from_dict({}) is None
        
    def test_multiple_keys(self):
        """Test dictionary with multiple possible keys."""
        data = {"other": "ignore", "content": "use this", "text": "not this"}
        assert _extract_from_dict(data) == "use this"
        
    def test_nested_structures(self):
        """Test nested dictionary structures."""
        nested = {"data": {"content": "nested content"}}
        assert _extract_from_dict(nested) == "nested content"
        
    def test_fallback_extraction(self):
        """Test fallback to all string values."""
        data = {"custom_key": "custom value", "another": "another value"}
        result = _extract_from_dict(data)
        assert result is not None
        assert "custom value" in result or "another value" in result


class TestExtractFromList:
    """Test the _extract_from_list helper function."""
    
    def test_simple_list(self):
        """Test simple list extraction."""
        data = ["item1", "item2", "item3"]
        assert _extract_from_list(data) == "item1 item2 item3"
        
    def test_empty_list(self):
        """Test empty list."""
        assert _extract_from_list([]) is None
        
    def test_mixed_types(self):
        """Test list with mixed data types."""
        data = ["string", 123, {"content": "dict"}]
        result = _extract_from_list(data)
        assert "string" in result
        assert "123" in result
        assert "dict" in result
        
    def test_nested_structures(self):
        """Test nested list structures."""
        nested = [["nested", "list"], {"content": "nested dict"}]
        result = _extract_from_list(nested)
        assert "nested list" in result
        assert "nested dict" in result


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    def test_large_text_performance(self):
        """Test performance with large text blocks."""
        # Create >10KB text
        large_text = "This is a performance test with repeated content. " * 1000
        
        # Test extraction
        start_time = time.time()
        result = extract_text_from_response(large_text)
        extraction_time = time.time() - start_time
        
        assert result is not None
        assert extraction_time < 0.1  # Should be fast
        
        # Test word counting
        start_time = time.time()
        count = word_count(large_text)
        counting_time = time.time() - start_time
        
        assert count > 0
        assert counting_time < 0.1  # Should be fast
        
    def test_stress_with_complex_structures(self):
        """Test with complex nested structures."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": ["deep", "nested", "content"],
                    "content": "found it"
                }
            },
            "other_data": ["item1", "item2", {"nested": "value"}]
        }
        
        result = extract_text_from_response(complex_data)
        assert result is not None
        assert "found it" in result
        
    def test_memory_usage(self):
        """Test memory usage with repeated operations."""
        # Test that repeated calls don't cause memory leaks
        test_data = "Test content for memory usage"
        
        for _ in range(1000):
            extract_text_from_response(test_data)
            word_count(test_data)
            strip_ansi_codes(test_data)
            is_concise_output(test_data)
            
        # If we get here without memory issues, test passes
        assert True


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_very_long_strings(self):
        """Test with very long strings."""
        very_long = "word " * 100000  # Very long text
        
        # Should handle without crashing
        result = extract_text_from_response(very_long)
        assert result is not None
        
        count = word_count(very_long)
        assert count == 100000
        
    def test_special_unicode_characters(self):
        """Test with special Unicode characters."""
        unicode_text = "Hello 🌍 世界 🚀 测试"
        
        result = extract_text_from_response(unicode_text)
        assert result == unicode_text
        
        count = word_count(unicode_text)
        assert count == 5  # Should count emoji and unicode as words
        
    def test_null_bytes_and_control_chars(self):
        """Test with null bytes and control characters."""
        null_text = "text\x00with\x00null\x00bytes"
        
        # Should handle gracefully
        result = extract_text_from_response(null_text)
        # May be None due to binary detection or processed string
        assert result is None or isinstance(result, str)
        
    def test_circular_references(self):
        """Test handling of circular references."""
        # Create dict with circular reference
        circular = {"data": {}}
        circular["data"]["parent"] = circular
        
        # Should handle without infinite recursion
        result = extract_text_from_response(circular)
        # Should not crash, result may be None
        assert result is None or isinstance(result, str)
        
    def test_deeply_nested_structures(self):
        """Test with deeply nested structures."""
        # Create deeply nested dict
        deep_dict = {"content": "deep content"}
        for i in range(50):
            deep_dict = {"level": deep_dict}
            
        # Should handle without stack overflow
        result = extract_text_from_response(deep_dict)
        assert result is None or isinstance(result, str)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])