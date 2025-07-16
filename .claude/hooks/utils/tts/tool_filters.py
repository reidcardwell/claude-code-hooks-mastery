#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

"""
Tool-Specific Content Filters for TTS Enhancement

This module provides an abstract base class and concrete implementations for filtering
Claude Code tool outputs to determine TTS eligibility and generate custom messages.

The filtering system enables intelligent TTS behavior by:
- Determining which tool outputs should trigger text-to-speech
- Generating tool-specific custom messages for better user experience
- Providing extensible architecture for adding new tool filters
- Supporting user-level configuration and customization

Usage:
    from tool_filters import ToolFilter, ToolFilterRegistry
    
    registry = ToolFilterRegistry()
    filter_instance = registry.get_filter("Bash")
    
    if filter_instance and filter_instance.should_speak("Bash", response_data):
        message = filter_instance.get_custom_message("Bash", response_data)
        # Queue message for TTS
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
import json
import re


class ToolFilter(ABC):
    """
    Abstract base class for tool-specific content filters.
    
    This class defines the interface that all tool filters must implement
    to provide intelligent TTS behavior for different Claude Code tools.
    
    Subclasses should implement:
    - should_speak(): Determine if tool output should trigger TTS
    - get_custom_message(): Generate tool-specific TTS messages
    
    The filtering system supports:
    - Tool-specific logic for TTS eligibility
    - Custom message generation for better user experience
    - Response parsing utilities for common data formats
    - Error handling and graceful degradation
    """
    
    @abstractmethod
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Determine if a tool's output should trigger text-to-speech.
        
        This method implements the core filtering logic for each tool type.
        It should return True if the tool output is suitable for TTS based on:
        - Tool-specific criteria (e.g., exit codes, file operations)
        - Response content analysis
        - User preferences and configuration
        
        Args:
            tool_name: Name of the Claude Code tool that generated the response
            response_data: Raw response data from the tool execution
            
        Returns:
            bool: True if the response should trigger TTS, False otherwise
            
        Example:
            >>> filter_instance.should_speak("Bash", {"exit_code": 0, "output": "Success"})
            True
            >>> filter_instance.should_speak("Read", {"content": "file contents"})
            False
        """
        pass
    
    @abstractmethod
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Generate a custom TTS message for the tool output.
        
        This method creates tool-specific messages that are more suitable for
        speech synthesis than raw tool output. It should:
        - Extract relevant information from response_data
        - Format it into a concise, speech-friendly message
        - Return None if no custom message is appropriate
        
        Args:
            tool_name: Name of the Claude Code tool that generated the response
            response_data: Raw response data from the tool execution
            
        Returns:
            Optional[str]: Custom message for TTS, or None to use default processing
            
        Example:
            >>> filter_instance.get_custom_message("Bash", {"exit_code": 0})
            "Command completed successfully"
            >>> filter_instance.get_custom_message("Git", {"operation": "commit"})
            "Changes committed successfully"
        """
        pass
    
    def get_tool_name(self) -> str:
        """
        Get the tool name this filter is designed to handle.
        
        By default, this extracts the tool name from the class name by removing
        the 'Filter' suffix. Subclasses can override this for custom behavior.
        
        Returns:
            str: The tool name this filter handles
            
        Example:
            >>> BashFilter().get_tool_name()
            "Bash"
            >>> GitFilter().get_tool_name()
            "Git"
        """
        class_name = self.__class__.__name__
        if class_name.endswith('Filter'):
            return class_name[:-6]  # Remove 'Filter' suffix
        return class_name
    
    def _parse_response_data(self, response_data: Any) -> Dict[str, Any]:
        """
        Parse and normalize response data into a consistent format.
        
        This utility method handles common response data formats and converts
        them into a standardized dictionary structure for easier processing.
        
        Args:
            response_data: Raw response data in various formats
            
        Returns:
            Dict[str, Any]: Normalized response data
            
        Example:
            >>> filter_instance._parse_response_data("simple string")
            {"content": "simple string", "type": "string"}
            >>> filter_instance._parse_response_data({"exit_code": 0})
            {"exit_code": 0, "type": "dict"}
        """
        if response_data is None:
            return {"content": None, "type": "none"}
        
        if isinstance(response_data, str):
            return {"content": response_data, "type": "string"}
        
        if isinstance(response_data, dict):
            # Already a dictionary, add type info
            result = response_data.copy()
            result["type"] = "dict"
            return result
        
        if isinstance(response_data, list):
            return {"content": response_data, "type": "list"}
        
        # For other types, convert to string
        return {"content": str(response_data), "type": "other"}
    
    def _extract_text_content(self, response_data: Any) -> Optional[str]:
        """
        Extract text content from response data for analysis.
        
        This utility method provides a convenient way to extract readable text
        from various response formats. It's useful for filters that need to
        analyze the actual content of tool outputs.
        
        Args:
            response_data: Raw response data
            
        Returns:
            Optional[str]: Extracted text content, or None if no text found
            
        Example:
            >>> filter_instance._extract_text_content({"output": "Hello world"})
            "Hello world"
            >>> filter_instance._extract_text_content({"exit_code": 0})
            None
        """
        parsed = self._parse_response_data(response_data)
        
        if parsed["type"] == "string":
            return parsed["content"]
        
        if parsed["type"] == "dict":
            # Check common keys for text content
            text_keys = ["content", "output", "stdout", "message", "text", "result"]
            for key in text_keys:
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key]
        
        if parsed["type"] == "list":
            # Try to join list items as text
            try:
                content = parsed["content"]
                if all(isinstance(item, str) for item in content):
                    return " ".join(content)
            except (TypeError, AttributeError):
                pass
        
        return None
    
    def _is_error_response(self, response_data: Any) -> bool:
        """
        Check if the response indicates an error condition.
        
        This utility method provides a common way to detect error conditions
        across different tool types. It checks for common error indicators.
        
        Args:
            response_data: Raw response data
            
        Returns:
            bool: True if the response indicates an error
            
        Example:
            >>> filter_instance._is_error_response({"exit_code": 1})
            True
            >>> filter_instance._is_error_response({"exit_code": 0})
            False
        """
        parsed = self._parse_response_data(response_data)
        
        if parsed["type"] == "dict":
            # Check for exit code
            if "exit_code" in parsed:
                return parsed["exit_code"] != 0
            
            # Check for error flags
            if "error" in parsed:
                return bool(parsed["error"])
            
            # Check for success flags
            if "success" in parsed:
                return not bool(parsed["success"])
        
        return False
    
    def _format_message(self, template: str, **kwargs) -> str:
        """
        Format a message template with provided variables.
        
        This utility method provides safe string formatting for message templates
        with error handling and fallback behavior.
        
        Args:
            template: Message template string with format placeholders
            **kwargs: Variables to substitute in the template
            
        Returns:
            str: Formatted message
            
        Example:
            >>> filter_instance._format_message("Command {action} {status}", 
            ...                                action="completed", status="successfully")
            "Command completed successfully"
        """
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            # If formatting fails, return template as-is
            return template


class DefaultToolFilter(ToolFilter):
    """
    Default implementation of ToolFilter for tools without specific filters.
    
    This filter provides basic behavior for tools that don't have specialized
    filtering logic. It uses simple heuristics to determine TTS eligibility
    and falls back to standard text processing.
    """
    
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Default implementation: speak unless the response is empty or None.
        
        This provides a safe default for tools without specific filtering logic.
        """
        if response_data is None:
            return False
        
        # Check if there's meaningful content
        text_content = self._extract_text_content(response_data)
        return text_content is not None and text_content.strip() != ""
    
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Default implementation: return None to use standard text processing.
        
        This allows tools without specific message generation to fall back
        to the standard text extraction and processing pipeline.
        """
        return None


if __name__ == "__main__":
    # Test the base class implementation
    print("Testing ToolFilter base class...")
    
    # Test default filter
    default_filter = DefaultToolFilter()
    print(f"Default filter tool name: {default_filter.get_tool_name()}")
    
    # Test response parsing
    test_cases = [
        "simple string",
        {"exit_code": 0, "output": "Success"},
        {"exit_code": 1, "error": "Failed"},
        ["item1", "item2", "item3"],
        None,
        123
    ]
    
    print("\nTesting response parsing:")
    for i, test_case in enumerate(test_cases, 1):
        parsed = default_filter._parse_response_data(test_case)
        text_content = default_filter._extract_text_content(test_case)
        is_error = default_filter._is_error_response(test_case)
        should_speak = default_filter.should_speak("Test", test_case)
        
        print(f"{i}. Input: {test_case}")
        print(f"   Parsed: {parsed}")
        print(f"   Text content: {text_content}")
        print(f"   Is error: {is_error}")
        print(f"   Should speak: {should_speak}")
        print()
    
    # Test message formatting
    print("Testing message formatting:")
    template = "Command {action} {status}"
    formatted = default_filter._format_message(template, action="completed", status="successfully")
    print(f"Template: {template}")
    print(f"Formatted: {formatted}")
    
    # Test with missing variables
    try:
        formatted_incomplete = default_filter._format_message(template, action="completed")
        print(f"Incomplete formatting: {formatted_incomplete}")
    except Exception as e:
        print(f"Error handling: {e}")
    
    print("\nToolFilter base class implementation complete!")