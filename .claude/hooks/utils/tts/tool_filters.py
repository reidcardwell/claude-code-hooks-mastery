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


class BashFilter(ToolFilter):
    """
    Filter for Bash command outputs with intelligent exit code handling.
    
    This filter provides specialized behavior for Bash tool responses by:
    - Analyzing exit codes to determine TTS eligibility
    - Generating appropriate messages for success/failure scenarios
    - Filtering out common silent commands that don't need TTS
    - Providing context-aware error messages
    """
    
    # Commands that should remain silent even on success
    SILENT_COMMANDS = {
        'cd', 'export', 'alias', 'unalias', 'set', 'unset', 'source', '.',
        'ulimit', 'umask', 'history', 'fc', 'jobs', 'bg', 'fg', 'disown',
        'suspend', 'times', 'type', 'which', 'command', 'builtin', 'enable',
        'pwd', 'dirs', 'pushd', 'popd', 'readonly', 'declare', 'typeset',
        'local', 'eval', 'exec', 'shift', 'getopts', 'wait', 'trap',
        'shopt', 'complete', 'compgen', 'bind', 'hash', 'help'
    }
    
    def __init__(self):
        """Initialize BashFilter with default configuration."""
        super().__init__()
        self.speak_success = True  # Can be configured via settings
        self.speak_errors = True   # Can be configured via settings
    
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Determine if Bash command output should trigger TTS.
        
        Logic:
        - Always speak on errors (exit_code != 0)
        - For success (exit_code == 0), speak only if:
          * Command is not in SILENT_COMMANDS list
          * Command produced meaningful output
          * speak_success is enabled
        
        Args:
            tool_name: Should be "Bash"
            response_data: Bash command response data
            
        Returns:
            bool: True if output should trigger TTS
        """
        if not self.speak_errors and not self.speak_success:
            return False
        
        parsed = self._parse_response_data(response_data)
        
        # Get exit code (default to 0 if not present)
        exit_code = parsed.get("exit_code", 0)
        
        # Always speak on errors if enabled
        if exit_code != 0:
            return self.speak_errors
        
        # For successful commands, check if we should speak
        if not self.speak_success:
            return False
        
        # Check if command should be silent
        if self._is_silent_command(parsed):
            return False
        
        # Check if there's meaningful output to speak
        return self._has_meaningful_output(parsed)
    
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Generate custom TTS message for Bash command output.
        
        Creates context-aware messages based on:
        - Exit code and error conditions
        - Command type and expected output
        - Actual command output content
        
        Args:
            tool_name: Should be "Bash"
            response_data: Bash command response data
            
        Returns:
            Optional[str]: Custom message for TTS, or None for default processing
        """
        parsed = self._parse_response_data(response_data)
        exit_code = parsed.get("exit_code", 0)
        
        # Handle error cases
        if exit_code != 0:
            return self._generate_error_message(parsed, exit_code)
        
        # Handle success cases
        return self._generate_success_message(parsed)
    
    def _is_silent_command(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if the command should remain silent even on success.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            bool: True if command should be silent
        """
        # Try to extract command from various possible fields
        command = self._extract_command(parsed_data)
        
        if not command:
            return False
        
        # Get the base command (first word)
        base_command = command.split()[0] if command.split() else ""
        
        # Remove path prefixes to get actual command name
        base_command = base_command.split('/')[-1]
        
        return base_command in self.SILENT_COMMANDS
    
    def _extract_command(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract the actual command that was executed.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: The command that was executed, or None if not found
        """
        # Check various fields where command might be stored
        command_fields = ["command", "cmd", "description", "input"]
        
        for field in command_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                return parsed_data[field].strip()
        
        return None
    
    def _has_meaningful_output(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if the command produced meaningful output worth speaking.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            bool: True if output is meaningful
        """
        # Extract text content
        text_content = self._extract_text_content(parsed_data)
        
        if not text_content or not text_content.strip():
            return False
        
        # Check if output is just whitespace or very short
        cleaned_output = text_content.strip()
        if len(cleaned_output) < 3:
            return False
        
        # Check for common meaningless outputs
        meaningless_patterns = [
            r'^\s*$',                    # Empty or whitespace only
            r'^\.+$',                    # Just dots
            r'^\s*ok\s*$',              # Just "ok"
            r'^\s*done\s*$',            # Just "done"
            r'^\s*[0-9]+\s*$',          # Just numbers
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, cleaned_output, re.IGNORECASE):
                return False
        
        return True
    
    def _generate_error_message(self, parsed_data: Dict[str, Any], exit_code: int) -> str:
        """
        Generate appropriate error message based on exit code and context.
        
        Args:
            parsed_data: Normalized response data
            exit_code: Command exit code
            
        Returns:
            str: Error message for TTS
        """
        # Get command for context
        command = self._extract_command(parsed_data)
        command_name = command.split()[0].split('/')[-1] if command else "Command"
        
        # Common exit code meanings
        exit_code_messages = {
            1: "Command failed",
            2: "Command usage error",
            126: "Command not executable",
            127: "Command not found",
            128: "Invalid exit argument",
            130: "Command interrupted",
            255: "Command exit status out of range"
        }
        
        # Get specific message for exit code
        if exit_code in exit_code_messages:
            base_message = exit_code_messages[exit_code]
        else:
            base_message = f"Command failed with exit code {exit_code}"
        
        # Try to include error output if available
        error_output = self._extract_error_output(parsed_data)
        if error_output and len(error_output) < 100:  # Keep it concise
            return f"{base_message}: {error_output}"
        
        return base_message
    
    def _generate_success_message(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate appropriate success message for completed commands.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Success message for TTS, or None for default processing
        """
        # Get command for context
        command = self._extract_command(parsed_data)
        
        if not command:
            return "Command completed successfully"
        
        # Get base command name
        base_command = command.split()[0].split('/')[-1]
        
        # Context-aware success messages for common commands
        success_messages = {
            'mkdir': "Directory created successfully",
            'rmdir': "Directory removed successfully",
            'rm': "File removed successfully",
            'cp': "File copied successfully",
            'mv': "File moved successfully",
            'ln': "Link created successfully",
            'chmod': "Permissions changed successfully",
            'chown': "Ownership changed successfully",
            'tar': "Archive operation completed",
            'gzip': "File compressed successfully",
            'gunzip': "File decompressed successfully",
            'wget': "Download completed successfully",
            'curl': "Request completed successfully",
            'ssh': "SSH connection established",
            'scp': "File transfer completed successfully",
            'rsync': "Synchronization completed successfully",
            'make': "Build completed successfully",
            'pip': "Package operation completed successfully",
            'npm': "Package operation completed successfully",
            'yarn': "Package operation completed successfully",
            'docker': "Docker operation completed successfully",
            'git': "Git operation completed successfully",
        }
        
        # Return specific message if available
        if base_command in success_messages:
            return success_messages[base_command]
        
        # Default success message
        return "Command completed successfully"
    
    def _extract_error_output(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract error output from response data.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Error output text, or None if not found
        """
        # Check various fields where error output might be stored
        error_fields = ["stderr", "error", "error_output", "error_message"]
        
        for field in error_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                error_text = parsed_data[field].strip()
                if error_text:
                    return error_text
        
        return None


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
    # Test both base class and BashFilter implementation
    print("Testing ToolFilter base class and BashFilter...")
    
    # Test default filter
    default_filter = DefaultToolFilter()
    print(f"Default filter tool name: {default_filter.get_tool_name()}")
    
    # Test BashFilter
    bash_filter = BashFilter()
    print(f"Bash filter tool name: {bash_filter.get_tool_name()}")
    
    # Test cases for BashFilter
    bash_test_cases = [
        # Success cases
        {"exit_code": 0, "command": "ls -la", "output": "total 64\ndrwxr-xr-x  3 user  staff   96 Jul 16 08:00 .\n"},
        {"exit_code": 0, "command": "mkdir test_dir", "output": ""},
        {"exit_code": 0, "command": "cd /tmp", "output": ""},
        {"exit_code": 0, "command": "pwd", "output": "/tmp"},
        {"exit_code": 0, "command": "echo 'Hello World'", "output": "Hello World"},
        
        # Error cases
        {"exit_code": 1, "command": "ls /nonexistent", "stderr": "ls: /nonexistent: No such file or directory"},
        {"exit_code": 127, "command": "nonexistentcommand", "stderr": "nonexistentcommand: command not found"},
        {"exit_code": 2, "command": "grep", "stderr": "usage: grep pattern file"},
        
        # Edge cases
        {"exit_code": 0, "command": "make", "output": "gcc -o program program.c\nBuild successful"},
        {"exit_code": 0, "command": "/usr/bin/git status", "output": "On branch main\nnothing to commit"},
        None,
        "simple string output",
        {"exit_code": 0, "output": "ok"},
        {"exit_code": 0, "output": "done"},
    ]
    
    print("\nTesting BashFilter with various scenarios:")
    for i, test_case in enumerate(bash_test_cases, 1):
        should_speak = bash_filter.should_speak("Bash", test_case)
        custom_message = bash_filter.get_custom_message("Bash", test_case)
        
        print(f"{i}. Input: {test_case}")
        print(f"   Should speak: {should_speak}")
        print(f"   Custom message: {custom_message}")
        print()
    
    # Test silent commands detection
    print("Testing silent commands detection:")
    silent_test_cases = [
        {"exit_code": 0, "command": "cd /home/user", "output": ""},
        {"exit_code": 0, "command": "export PATH=/usr/bin:$PATH", "output": ""},
        {"exit_code": 0, "command": "source ~/.bashrc", "output": ""},
        {"exit_code": 0, "command": "alias ll='ls -la'", "output": ""},
        {"exit_code": 0, "command": "history", "output": "1  ls\n2  cd /tmp\n3  pwd"},
    ]
    
    for i, test_case in enumerate(silent_test_cases, 1):
        should_speak = bash_filter.should_speak("Bash", test_case)
        print(f"{i}. Command: {test_case.get('command', 'N/A')} -> Should speak: {should_speak}")
    
    # Test error message generation
    print("\nTesting error message generation:")
    error_test_cases = [
        {"exit_code": 1, "command": "ls /nonexistent", "stderr": "ls: /nonexistent: No such file or directory"},
        {"exit_code": 127, "command": "badcommand", "stderr": "badcommand: command not found"},
        {"exit_code": 2, "command": "grep", "stderr": "usage: grep pattern file"},
        {"exit_code": 130, "command": "sleep 100", "stderr": "Interrupted"},
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        error_message = bash_filter._generate_error_message(bash_filter._parse_response_data(test_case), test_case["exit_code"])
        print(f"{i}. Exit code {test_case['exit_code']}: {error_message}")
    
    # Test success message generation
    print("\nTesting success message generation:")
    success_test_cases = [
        {"exit_code": 0, "command": "mkdir test_dir"},
        {"exit_code": 0, "command": "rm test_file.txt"},
        {"exit_code": 0, "command": "cp source.txt dest.txt"},
        {"exit_code": 0, "command": "npm install package"},
        {"exit_code": 0, "command": "git commit -m 'message'"},
        {"exit_code": 0, "command": "customcommand --flag"},
    ]
    
    for i, test_case in enumerate(success_test_cases, 1):
        success_message = bash_filter._generate_success_message(bash_filter._parse_response_data(test_case))
        print(f"{i}. Command: {test_case['command']} -> {success_message}")
    
    print("\nBashFilter implementation testing complete!")