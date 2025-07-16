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


class GitFilter(ToolFilter):
    """
    Filter for Git command outputs with concise status messages.
    
    This filter provides specialized behavior for Git tool responses by:
    - Detecting Git operations from command patterns
    - Generating concise status messages for common Git operations
    - Parsing Git output for relevant information (branch names, file counts)
    - Suppressing verbose Git output while preserving important status updates
    """
    
    def __init__(self):
        """Initialize GitFilter with default configuration."""
        super().__init__()
        self.speak_success = True   # Can be configured via settings
        self.speak_errors = True    # Can be configured via settings
        self.speak_info = True      # Can be configured via settings
    
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Determine if Git command output should trigger TTS.
        
        Logic:
        - Always speak on errors (exit_code != 0)
        - For success, speak on important operations (commit, push, pull, merge)
        - Suppress verbose output but preserve status updates
        - Consider output length and meaningfulness
        
        Args:
            tool_name: Should be "Git" or similar
            response_data: Git command response data
            
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
        
        # Check if this is an important Git operation
        if self._is_important_git_operation(parsed):
            return True
        
        # Check if there's meaningful status information
        return self._has_meaningful_git_output(parsed)
    
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Generate custom TTS message for Git command output.
        
        Creates concise messages for common Git operations:
        - Commit: "Changes committed successfully"
        - Push: "Pushed to remote"
        - Pull: "Repository updated"
        - Status: Brief status summary
        
        Args:
            tool_name: Should be "Git" or similar
            response_data: Git command response data
            
        Returns:
            Optional[str]: Custom message for TTS, or None for default processing
        """
        parsed = self._parse_response_data(response_data)
        exit_code = parsed.get("exit_code", 0)
        
        # Handle error cases
        if exit_code != 0:
            return self._generate_git_error_message(parsed, exit_code)
        
        # Handle success cases
        return self._generate_git_success_message(parsed)
    
    def _is_important_git_operation(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if the Git command is an important operation worth speaking about.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            bool: True if this is an important Git operation
        """
        # Try to extract command from various possible fields
        command = self._extract_command(parsed_data)
        
        if not command:
            return False
        
        # Parse the Git command
        git_operation = self._parse_git_command(command)
        
        # Important operations that should always be spoken
        important_operations = {
            'commit', 'push', 'pull', 'merge', 'rebase', 'clone', 'fetch',
            'checkout', 'branch', 'tag', 'stash', 'reset', 'revert'
        }
        
        return git_operation in important_operations
    
    def _has_meaningful_git_output(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if Git output contains meaningful information worth speaking.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            bool: True if output is meaningful
        """
        # Extract text content
        text_content = self._extract_text_content(parsed_data)
        
        if not text_content or not text_content.strip():
            return False
        
        # Check for meaningful Git status patterns
        meaningful_patterns = [
            r'modified:',           # File modifications
            r'deleted:',            # File deletions
            r'new file:',           # New files
            r'renamed:',            # File renames
            r'Changes to be committed:', # Staged changes
            r'Changes not staged',  # Unstaged changes
            r'Untracked files:',    # Untracked files
            r'Your branch is',      # Branch status
            r'Fast-forward',        # Fast-forward merges
            r'Merge made by',       # Merge commits
            r'Conflict',            # Merge conflicts
            r'Already up to date',  # Already up to date
            r'files? changed',      # Summary statistics
            r'insertions?\(\+\)',   # Addition statistics
            r'deletions?\(\-\)',    # Deletion statistics
        ]
        
        for pattern in meaningful_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                return True
        
        # Check for short, concise messages
        if len(text_content.strip()) < 100:
            return True
        
        return False
    
    def _extract_command(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract the Git command that was executed.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: The Git command, or None if not found
        """
        # Check various fields where command might be stored
        command_fields = ["command", "cmd", "description", "input"]
        
        for field in command_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                return parsed_data[field].strip()
        
        return None
    
    def _parse_git_command(self, command: str) -> Optional[str]:
        """
        Parse Git command to extract the operation type.
        
        Args:
            command: Full Git command string
            
        Returns:
            Optional[str]: Git operation (commit, push, pull, etc.) or None
        """
        if not command:
            return None
        
        # Remove 'git' prefix and get the operation
        parts = command.split()
        if len(parts) < 2:
            return None
        
        # Handle cases like 'git commit', '/usr/bin/git status', etc.
        git_index = -1
        for i, part in enumerate(parts):
            if part.endswith('git') or part == 'git':
                git_index = i
                break
        
        if git_index == -1 or git_index + 1 >= len(parts):
            return None
        
        return parts[git_index + 1].lower()
    
    def _generate_git_error_message(self, parsed_data: Dict[str, Any], exit_code: int) -> str:
        """
        Generate appropriate error message for Git command failures.
        
        Args:
            parsed_data: Normalized response data
            exit_code: Git command exit code
            
        Returns:
            str: Error message for TTS
        """
        # Get command for context
        command = self._extract_command(parsed_data)
        operation = self._parse_git_command(command) if command else "operation"
        
        # Common Git exit codes and their meanings
        git_error_messages = {
            1: f"Git {operation} failed",
            128: f"Git {operation} failed - repository error",
            129: f"Git {operation} failed - invalid command",
            130: f"Git {operation} interrupted by user",
        }
        
        # Get specific message for exit code
        if exit_code in git_error_messages:
            base_message = git_error_messages[exit_code]
        else:
            base_message = f"Git {operation} failed with exit code {exit_code}"
        
        # Try to include specific error information
        error_output = self._extract_error_output(parsed_data)
        if error_output:
            # Extract meaningful error information
            if "not a git repository" in error_output.lower():
                return "Not a Git repository"
            elif "permission denied" in error_output.lower():
                return f"Git {operation} failed - permission denied"
            elif "authentication failed" in error_output.lower():
                return f"Git {operation} failed - authentication failed"
            elif "connection refused" in error_output.lower():
                return f"Git {operation} failed - connection refused"
            elif "merge conflict" in error_output.lower():
                return f"Git {operation} failed - merge conflict"
            elif "nothing to commit" in error_output.lower():
                return "Nothing to commit"
            elif len(error_output) < 80:  # Keep it concise
                return f"Git {operation} failed: {error_output}"
        
        return base_message
    
    def _generate_git_success_message(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate appropriate success message for Git operations.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Success message for TTS, or None for default processing
        """
        # Get command for context
        command = self._extract_command(parsed_data)
        operation = self._parse_git_command(command) if command else None
        
        if not operation:
            return "Git operation completed successfully"
        
        # Context-aware success messages for Git operations
        git_success_messages = {
            'commit': self._generate_commit_message(parsed_data),
            'push': self._generate_push_message(parsed_data),
            'pull': self._generate_pull_message(parsed_data),
            'merge': self._generate_merge_message(parsed_data),
            'checkout': self._generate_checkout_message(parsed_data),
            'branch': self._generate_branch_message(parsed_data),
            'clone': "Repository cloned successfully",
            'fetch': "Fetched changes from remote",
            'rebase': "Rebase completed successfully",
            'stash': "Changes stashed successfully",
            'tag': "Tag created successfully",
            'reset': "Reset completed successfully",
            'revert': "Revert completed successfully",
            'add': "Files staged successfully",
            'rm': "Files removed successfully",
            'mv': "Files moved successfully",
        }
        
        # Return specific message if available
        if operation in git_success_messages:
            message = git_success_messages[operation]
            if callable(message):
                return message
            return message
        
        # Default success message
        return f"Git {operation} completed successfully"
    
    def _generate_commit_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for commit operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            # Look for commit hash and file counts
            if re.search(r'\[[\w\s]+\s+[a-f0-9]+\]', text_content):
                # Extract file change information
                files_changed = re.search(r'(\d+) files? changed', text_content)
                if files_changed:
                    count = files_changed.group(1)
                    return f"Changes committed - {count} files modified"
        
        return "Changes committed successfully"
    
    def _generate_push_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for push operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            # Look for branch information
            branch_match = re.search(r'(\w+)\s*->\s*(\w+)', text_content)
            if branch_match:
                branch = branch_match.group(1)
                return f"Pushed {branch} to remote"
        
        return "Pushed to remote successfully"
    
    def _generate_pull_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for pull operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            if "Already up to date" in text_content:
                return "Repository already up to date"
            elif "Fast-forward" in text_content:
                return "Repository updated via fast-forward"
            elif re.search(r'(\d+) files? changed', text_content):
                return "Repository updated with changes"
        
        return "Repository updated successfully"
    
    def _generate_merge_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for merge operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            if "Fast-forward" in text_content:
                return "Merge completed via fast-forward"
            elif "Merge made by" in text_content:
                return "Merge commit created successfully"
        
        return "Merge completed successfully"
    
    def _generate_checkout_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for checkout operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            # Look for branch switching
            branch_match = re.search(r"Switched to branch '(\w+)'", text_content)
            if branch_match:
                branch = branch_match.group(1)
                return f"Switched to branch {branch}"
            
            # Look for new branch creation
            new_branch_match = re.search(r"Switched to a new branch '(\w+)'", text_content)
            if new_branch_match:
                branch = new_branch_match.group(1)
                return f"Created and switched to branch {branch}"
        
        return "Checkout completed successfully"
    
    def _generate_branch_message(self, parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for branch operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content:
            # Look for branch creation
            if re.search(r'created', text_content, re.IGNORECASE):
                return "Branch created successfully"
            elif re.search(r'deleted', text_content, re.IGNORECASE):
                return "Branch deleted successfully"
        
        return "Branch operation completed successfully"
    
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


class FileOperationFilter(ToolFilter):
    """
    Filter for file operation tools (Read, Write, Edit, MultiEdit) with appropriate confirmations.
    
    This filter provides specialized behavior for file operation tools by:
    - Generating confirmations for Write/Edit operations like "File updated: [filename]"
    - Excluding Read operations from TTS unless file not found
    - Extracting filenames from tool parameters
    - Handling error cases with appropriate messages
    """
    
    # File operation tools this filter handles
    FILE_OPERATION_TOOLS = {
        'Read', 'Write', 'Edit', 'MultiEdit', 'NotebookRead', 'NotebookEdit'
    }
    
    # Tools that should be excluded from TTS by default
    EXCLUDED_TOOLS = {
        'Read', 'NotebookRead'  # Read operations are typically silent
    }
    
    def __init__(self):
        """Initialize FileOperationFilter with default configuration."""
        super().__init__()
        self.speak_write_operations = True   # Can be configured via settings
        self.speak_read_operations = False   # Can be configured via settings
        self.speak_errors = True             # Can be configured via settings
    
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Determine if file operation output should trigger TTS.
        
        Logic:
        - Always speak on errors (file not found, permission denied)
        - For Write/Edit operations, speak on success if enabled
        - For Read operations, only speak on errors (file not found)
        - Consider file operation type and result
        
        Args:
            tool_name: File operation tool name (Read, Write, Edit, etc.)
            response_data: File operation response data
            
        Returns:
            bool: True if output should trigger TTS
        """
        if not self.speak_errors and not self.speak_write_operations and not self.speak_read_operations:
            return False
        
        parsed = self._parse_response_data(response_data)
        
        # Check if this is an error response
        if self._is_error_response(response_data):
            return self.speak_errors
        
        # Handle read operations
        if tool_name in {'Read', 'NotebookRead'}:
            return self.speak_read_operations
        
        # Handle write operations
        if tool_name in {'Write', 'Edit', 'MultiEdit', 'NotebookEdit'}:
            return self.speak_write_operations
        
        # Default to speaking for other file operations
        return True
    
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Generate custom TTS message for file operation output.
        
        Creates file operation-specific messages:
        - Write: "File updated: [filename]"
        - Edit: "File modified: [filename]"
        - MultiEdit: "Files updated: [count] files"
        - Read errors: "File not found: [filename]"
        
        Args:
            tool_name: File operation tool name
            response_data: File operation response data
            
        Returns:
            Optional[str]: Custom message for TTS, or None for default processing
        """
        parsed = self._parse_response_data(response_data)
        
        # Handle error cases
        if self._is_error_response(response_data):
            return self._generate_file_error_message(tool_name, parsed)
        
        # Handle success cases
        return self._generate_file_success_message(tool_name, parsed)
    
    def _generate_file_error_message(self, tool_name: str, parsed_data: Dict[str, Any]) -> str:
        """
        Generate appropriate error message for file operation failures.
        
        Args:
            tool_name: File operation tool name
            parsed_data: Normalized response data
            
        Returns:
            str: Error message for TTS
        """
        # Extract filename from various possible sources
        filename = self._extract_filename(parsed_data)
        
        # Extract error information
        error_text = self._extract_error_output(parsed_data)
        
        # Generate context-aware error messages
        if error_text:
            error_lower = error_text.lower()
            
            # File not found errors
            if "not found" in error_lower or "no such file" in error_lower:
                if filename:
                    return f"File not found: {filename}"
                return "File not found"
            
            # Permission errors
            elif "permission denied" in error_lower or "access denied" in error_lower:
                if filename:
                    return f"Permission denied: {filename}"
                return "Permission denied"
            
            # Directory errors
            elif "is a directory" in error_lower:
                if filename:
                    return f"Target is a directory: {filename}"
                return "Target is a directory"
            
            # Read-only errors
            elif "read-only" in error_lower or "readonly" in error_lower:
                if filename:
                    return f"File is read-only: {filename}"
                return "File is read-only"
            
            # Generic error with filename
            elif filename and len(error_text) < 100:
                return f"Error with {filename}: {error_text}"
        
        # Default error messages by tool type
        error_messages = {
            'Read': f"Failed to read file{': ' + filename if filename else ''}",
            'Write': f"Failed to write file{': ' + filename if filename else ''}",
            'Edit': f"Failed to edit file{': ' + filename if filename else ''}",
            'MultiEdit': "Failed to edit files",
            'NotebookRead': f"Failed to read notebook{': ' + filename if filename else ''}",
            'NotebookEdit': f"Failed to edit notebook{': ' + filename if filename else ''}",
        }
        
        return error_messages.get(tool_name, f"File operation failed{': ' + filename if filename else ''}")
    
    def _generate_file_success_message(self, tool_name: str, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate appropriate success message for file operations.
        
        Args:
            tool_name: File operation tool name
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Success message for TTS, or None for default processing
        """
        # Extract filename from various possible sources
        filename = self._extract_filename(parsed_data)
        
        # Generate context-aware success messages
        success_messages = {
            'Write': f"File created{': ' + filename if filename else ''}",
            'Edit': f"File updated{': ' + filename if filename else ''}",
            'MultiEdit': self._generate_multiedit_message(parsed_data),
            'NotebookEdit': f"Notebook updated{': ' + filename if filename else ''}",
        }
        
        # For Read operations, we typically don't speak on success
        # unless specifically configured to do so
        if tool_name in {'Read', 'NotebookRead'}:
            return None
        
        return success_messages.get(tool_name, f"File operation completed{': ' + filename if filename else ''}")
    
    def _generate_multiedit_message(self, parsed_data: Dict[str, Any]) -> str:
        """
        Generate specific message for MultiEdit operations.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            str: MultiEdit-specific message
        """
        # Try to extract number of edits or files affected
        edit_count = self._extract_edit_count(parsed_data)
        filename = self._extract_filename(parsed_data)
        
        if edit_count and edit_count > 1:
            return f"Applied {edit_count} edits{' to ' + filename if filename else ''}"
        elif filename:
            return f"File updated: {filename}"
        else:
            return "Multiple edits applied successfully"
    
    def _extract_filename(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract filename from response data.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Filename if found, None otherwise
        """
        # Check various fields where filename might be stored
        filename_fields = [
            "file_path", "filepath", "filename", "file", "path", "target",
            "notebook_path", "notebookPath", "input_file", "output_file"
        ]
        
        for field in filename_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                filepath = parsed_data[field].strip()
                if filepath:
                    # Extract just the filename from the path
                    return filepath.split('/')[-1] if '/' in filepath else filepath
        
        # Try to extract from error messages or content
        import re
        # Check multiple text sources
        text_sources = [
            self._extract_text_content(parsed_data),
            self._extract_error_output(parsed_data),
        ]
        
        for text_content in text_sources:
            if text_content:
                # Look for common file path patterns in error messages
                file_patterns = [
                    r"(?:file|path|target)(?:\s+is\s+|\s*:\s*)['\"]?([^\s'\"]+)['\"]?",
                    r"['\"]([^'\"]+\.[a-zA-Z0-9]+)['\"]",  # Quoted filenames with extensions
                    r"(?:^|\s)([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)(?:\s|$)",  # Filenames with extensions
                    r"(?:not found|missing|denied|error).*?:\s*([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)",  # Error messages with filenames
                    r":\s*([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)",  # Simple colon-separated filenames
                ]
                
                for pattern in file_patterns:
                    match = re.search(pattern, text_content, re.IGNORECASE)
                    if match:
                        filepath = match.group(1)
                        return filepath.split('/')[-1] if '/' in filepath else filepath
        
        return None
    
    def _extract_edit_count(self, parsed_data: Dict[str, Any]) -> Optional[int]:
        """
        Extract the number of edits from MultiEdit response data.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[int]: Number of edits if found, None otherwise
        """
        # Check for edit count in various fields
        count_fields = ["edit_count", "edits", "changes", "modifications"]
        
        for field in count_fields:
            if field in parsed_data and isinstance(parsed_data[field], int):
                return parsed_data[field]
        
        # Try to extract from edits array
        if "edits" in parsed_data and isinstance(parsed_data["edits"], list):
            return len(parsed_data["edits"])
        
        # Try to extract from text content
        text_content = self._extract_text_content(parsed_data)
        if text_content:
            import re
            # Look for patterns like "Applied 3 edits" or "3 changes made"
            count_patterns = [
                r"(?:applied|made|completed)\s+(\d+)\s+(?:edits?|changes?|modifications?)",
                r"(\d+)\s+(?:edits?|changes?|modifications?)\s+(?:applied|made|completed)",
                r"(\d+)\s+(?:files?|edits?)\s+(?:updated|modified|changed)",
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        
        return None
    
    def _extract_error_output(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract error output from response data.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Error output text, or None if not found
        """
        # Check various fields where error output might be stored
        error_fields = ["error", "error_message", "stderr", "exception", "message"]
        
        for field in error_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                error_text = parsed_data[field].strip()
                if error_text:
                    return error_text
        
        # Check if the main content indicates an error
        text_content = self._extract_text_content(parsed_data)
        if text_content:
            # Look for common error indicators
            error_indicators = [
                "error:", "failed:", "exception:", "not found", "permission denied",
                "access denied", "no such file", "is a directory", "read-only"
            ]
            
            text_lower = text_content.lower()
            for indicator in error_indicators:
                if indicator in text_lower:
                    return text_content
        
        return None
    
    def _is_error_response(self, response_data: Any) -> bool:
        """
        Check if the response indicates an error condition.
        
        Enhanced for file operations to detect common file-related errors.
        
        Args:
            response_data: Raw response data
            
        Returns:
            bool: True if the response indicates an error
        """
        # Use base class error detection first
        if super()._is_error_response(response_data):
            return True
        
        # File operation specific error detection
        text_content = self._extract_text_content(response_data)
        if text_content:
            error_patterns = [
                r"file not found",
                r"no such file",
                r"permission denied",
                r"access denied",
                r"is a directory",
                r"read-only",
                r"readonly",
                r"cannot (read|write|access)",
                r"failed to (read|write|open|save)",
                r"error (reading|writing|opening|saving)",
                r"unable to (read|write|access|open|save)",
            ]
            
            text_lower = text_content.lower()
            for pattern in error_patterns:
                if re.search(pattern, text_lower):
                    return True
        
        return False


class SearchFilter(ToolFilter):
    """
    Filter for search tools (Grep and LS) with result count summaries.
    
    This filter provides specialized behavior for search tools by:
    - Filtering Grep and LS tools by default (returns False from should_speak)
    - Providing option to enable with result summaries like "Found X matches in Y files" for Grep
    - Generating "Directory contains X items" for LS
    - Parsing search results to extract counts
    - Handling empty results appropriately
    """
    
    # Search tools this filter handles
    SEARCH_TOOLS = {
        'Grep', 'LS', 'Find', 'Locate', 'Which', 'Whereis'
    }
    
    # Tools that should be excluded from TTS by default
    DEFAULT_EXCLUDED_TOOLS = {
        'Grep', 'LS'  # Search operations are typically silent
    }
    
    def __init__(self):
        """Initialize SearchFilter with default configuration."""
        super().__init__()
        self.speak_search_results = False    # Can be configured via settings
        self.speak_empty_results = False     # Can be configured via settings
        self.speak_errors = True             # Can be configured via settings
        self.enable_result_summaries = True  # Can be configured via settings
    
    def should_speak(self, tool_name: str, response_data: Any) -> bool:
        """
        Determine if search tool output should trigger TTS.
        
        Logic:
        - Always speak on errors (file not found, permission denied)
        - For search results, speak only if enabled and has meaningful results
        - By default, Grep and LS are excluded from TTS
        - Consider result count and meaningfulness
        
        Args:
            tool_name: Search tool name (Grep, LS, Find, etc.)
            response_data: Search tool response data
            
        Returns:
            bool: True if output should trigger TTS
        """
        if not self.speak_errors and not self.speak_search_results:
            return False
        
        parsed = self._parse_response_data(response_data)
        
        # Check if this is an error response
        if self._is_error_response(response_data):
            return self.speak_errors
        
        # Check if tool is in default excluded list
        if tool_name in self.DEFAULT_EXCLUDED_TOOLS and not self.speak_search_results:
            return False
        
        # For search results, check if we should speak
        if not self.speak_search_results:
            return False
        
        # Check if there are meaningful search results
        if self._has_meaningful_search_results(tool_name, parsed):
            return True
        
        # Check if we should speak empty results
        if self._has_empty_results(tool_name, parsed):
            return self.speak_empty_results
        
        return False
    
    def get_custom_message(self, tool_name: str, response_data: Any) -> Optional[str]:
        """
        Generate custom TTS message for search tool output.
        
        Creates search-specific messages:
        - Grep: "Found X matches in Y files" or "No matches found"
        - LS: "Directory contains X items" or "Directory is empty"
        - Find: "Found X items matching criteria"
        - Error cases: "Search failed: [error message]"
        
        Args:
            tool_name: Search tool name
            response_data: Search tool response data
            
        Returns:
            Optional[str]: Custom message for TTS, or None for default processing
        """
        parsed = self._parse_response_data(response_data)
        
        # Handle error cases
        if self._is_error_response(response_data):
            return self._generate_search_error_message(tool_name, parsed)
        
        # Handle search results
        if self.enable_result_summaries:
            return self._generate_search_result_message(tool_name, parsed)
        
        # Fallback to None for default processing
        return None
    
    def _has_meaningful_search_results(self, tool_name: str, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if search results contain meaningful information worth speaking.
        
        Args:
            tool_name: Search tool name
            parsed_data: Normalized response data
            
        Returns:
            bool: True if results are meaningful
        """
        # Extract result count
        result_count = self._extract_result_count(tool_name, parsed_data)
        
        # If we have a count, check if it's meaningful
        if result_count is not None:
            return result_count > 0
        
        # Check for text content that indicates results
        text_content = self._extract_text_content(parsed_data)
        if not text_content:
            return False
        
        # Tool-specific result detection
        if tool_name == 'Grep':
            return self._has_grep_results(text_content)
        elif tool_name == 'LS':
            return self._has_ls_results(text_content)
        elif tool_name in {'Find', 'Locate'}:
            return self._has_find_results(text_content)
        
        # Default: if there's content, it's meaningful
        return len(text_content.strip()) > 0
    
    def _has_empty_results(self, tool_name: str, parsed_data: Dict[str, Any]) -> bool:
        """
        Check if search results are empty.
        
        Args:
            tool_name: Search tool name
            parsed_data: Normalized response data
            
        Returns:
            bool: True if results are empty
        """
        # Extract result count
        result_count = self._extract_result_count(tool_name, parsed_data)
        
        # If we have a count, check if it's zero
        if result_count is not None:
            return result_count == 0
        
        # Check for empty text content
        text_content = self._extract_text_content(parsed_data)
        if not text_content or not text_content.strip():
            return True
        
        # Tool-specific empty result detection
        if tool_name == 'Grep':
            return not self._has_grep_results(text_content)
        elif tool_name == 'LS':
            return not self._has_ls_results(text_content)
        elif tool_name in {'Find', 'Locate'}:
            return not self._has_find_results(text_content)
        
        return False
    
    def _has_grep_results(self, text_content: str) -> bool:
        """Check if grep output contains actual matches."""
        if not text_content.strip():
            return False
        
        # Grep with no matches typically returns empty output
        # Grep with matches returns the matching lines
        lines = text_content.strip().split('\n')
        
        # Filter out empty lines and grep status messages
        result_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('grep:') and not line.startswith('Binary file'):
                result_lines.append(line)
        
        return len(result_lines) > 0
    
    def _has_ls_results(self, text_content: str) -> bool:
        """Check if ls output contains directory entries."""
        if not text_content.strip():
            return False
        
        # LS with no entries typically returns empty output
        # LS with entries returns file/directory listings
        lines = text_content.strip().split('\n')
        
        # Filter out empty lines and ls status messages
        result_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('ls:') and not line.startswith('total'):
                result_lines.append(line)
        
        return len(result_lines) > 0
    
    def _has_find_results(self, text_content: str) -> bool:
        """Check if find/locate output contains search results."""
        if not text_content.strip():
            return False
        
        # Find with no matches typically returns empty output
        # Find with matches returns the matching paths
        lines = text_content.strip().split('\n')
        
        # Filter out empty lines and find status messages
        result_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('find:') and not line.startswith('locate:'):
                result_lines.append(line)
        
        return len(result_lines) > 0
    
    def _extract_result_count(self, tool_name: str, parsed_data: Dict[str, Any]) -> Optional[int]:
        """
        Extract result count from search tool response data.
        
        Args:
            tool_name: Search tool name
            parsed_data: Normalized response data
            
        Returns:
            Optional[int]: Number of results if found, None otherwise
        """
        # Check for explicit count fields
        count_fields = ["count", "result_count", "matches", "items", "files", "total"]
        
        for field in count_fields:
            if field in parsed_data and isinstance(parsed_data[field], int):
                return parsed_data[field]
        
        # Try to extract count from text content
        text_content = self._extract_text_content(parsed_data)
        if text_content is not None:  # Allow empty strings
            # Count lines for basic result counting
            lines = text_content.strip().split('\n') if text_content.strip() else []
            
            # Tool-specific counting
            if tool_name == 'Grep':
                return self._count_grep_results(lines)
            elif tool_name == 'LS':
                return self._count_ls_results(lines)
            elif tool_name in {'Find', 'Locate'}:
                return self._count_find_results(lines)
        
        return None
    
    def _count_grep_results(self, lines: list) -> int:
        """Count actual grep result lines."""
        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('grep:') and not line.startswith('Binary file'):
                count += 1
        return count
    
    def _count_ls_results(self, lines: list) -> int:
        """Count actual ls result lines."""
        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('ls:') and not line.startswith('total'):
                count += 1
        return count
    
    def _count_find_results(self, lines: list) -> int:
        """Count actual find/locate result lines."""
        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('find:') and not line.startswith('locate:'):
                count += 1
        return count
    
    def _generate_search_error_message(self, tool_name: str, parsed_data: Dict[str, Any]) -> str:
        """
        Generate appropriate error message for search tool failures.
        
        Args:
            tool_name: Search tool name
            parsed_data: Normalized response data
            
        Returns:
            str: Error message for TTS
        """
        # Extract error information
        error_text = self._extract_error_output(parsed_data)
        
        # Generate context-aware error messages
        if error_text:
            error_lower = error_text.lower()
            
            # Common search tool errors
            if "no such file or directory" in error_lower:
                return f"{tool_name} failed: Path not found"
            elif "permission denied" in error_lower:
                return f"{tool_name} failed: Permission denied"
            elif "not found" in error_lower:
                return f"{tool_name} failed: Not found"
            elif "invalid" in error_lower:
                return f"{tool_name} failed: Invalid search criteria"
            elif "too many arguments" in error_lower:
                return f"{tool_name} failed: Too many arguments"
            elif len(error_text) < 50:  # Keep it concise
                return f"{tool_name} failed: {error_text}"
        
        # Default error messages by tool type
        error_messages = {
            'Grep': "Search failed",
            'LS': "Directory listing failed",
            'Find': "Find operation failed",
            'Locate': "Locate operation failed",
            'Which': "Which command failed",
            'Whereis': "Whereis command failed",
        }
        
        return error_messages.get(tool_name, f"{tool_name} operation failed")
    
    def _generate_search_result_message(self, tool_name: str, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate appropriate result message for search operations.
        
        Args:
            tool_name: Search tool name
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Result message for TTS, or None for default processing
        """
        # Extract result count
        result_count = self._extract_result_count(tool_name, parsed_data)
        
        # Generate tool-specific messages
        if tool_name == 'Grep':
            return self._generate_grep_message(result_count, parsed_data)
        elif tool_name == 'LS':
            return self._generate_ls_message(result_count, parsed_data)
        elif tool_name in {'Find', 'Locate'}:
            return self._generate_find_message(tool_name, result_count, parsed_data)
        elif tool_name in {'Which', 'Whereis'}:
            return self._generate_which_message(tool_name, result_count, parsed_data)
        
        # Default message
        if result_count is not None:
            if result_count == 0:
                return f"No results found"
            elif result_count == 1:
                return f"Found 1 result"
            else:
                return f"Found {result_count} results"
        
        return None
    
    def _generate_grep_message(self, result_count: Optional[int], parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for grep operations."""
        if result_count is not None:
            if result_count == 0:
                return "No matches found"
            elif result_count == 1:
                return "Found 1 match"
            else:
                # Try to extract file count for more detailed message
                file_count = self._extract_file_count(parsed_data)
                if file_count is not None and file_count > 1:
                    return f"Found {result_count} matches in {file_count} files"
                else:
                    return f"Found {result_count} matches"
        
        # Fallback based on content analysis
        text_content = self._extract_text_content(parsed_data)
        if text_content and self._has_grep_results(text_content):
            return "Found matches"
        else:
            return "No matches found"
    
    def _generate_ls_message(self, result_count: Optional[int], parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for ls operations."""
        if result_count is not None:
            if result_count == 0:
                return "Directory is empty"
            elif result_count == 1:
                return "Directory contains 1 item"
            else:
                return f"Directory contains {result_count} items"
        
        # Fallback based on content analysis
        text_content = self._extract_text_content(parsed_data)
        if text_content and self._has_ls_results(text_content):
            return "Directory listing available"
        else:
            return "Directory is empty"
    
    def _generate_find_message(self, tool_name: str, result_count: Optional[int], parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for find/locate operations."""
        if result_count is not None:
            if result_count == 0:
                return "No files found"
            elif result_count == 1:
                return "Found 1 file"
            else:
                return f"Found {result_count} files"
        
        # Fallback based on content analysis
        text_content = self._extract_text_content(parsed_data)
        if text_content and self._has_find_results(text_content):
            return "Found matching files"
        else:
            return "No files found"
    
    def _generate_which_message(self, tool_name: str, result_count: Optional[int], parsed_data: Dict[str, Any]) -> str:
        """Generate specific message for which/whereis operations."""
        text_content = self._extract_text_content(parsed_data)
        
        if text_content and text_content.strip():
            if tool_name == 'Which':
                return "Command found"
            else:  # Whereis
                return "Location found"
        else:
            if tool_name == 'Which':
                return "Command not found"
            else:  # Whereis
                return "Location not found"
    
    def _extract_file_count(self, parsed_data: Dict[str, Any]) -> Optional[int]:
        """
        Extract file count from search results (e.g., for grep matches across files).
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[int]: Number of files if found, None otherwise
        """
        # Check for explicit file count fields
        file_fields = ["file_count", "files", "num_files"]
        
        for field in file_fields:
            if field in parsed_data and isinstance(parsed_data[field], int):
                return parsed_data[field]
        
        # Try to extract from text content
        text_content = self._extract_text_content(parsed_data)
        if text_content:
            # For grep output, count unique filenames
            lines = text_content.strip().split('\n')
            filenames = set()
            
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    # Extract filename from "filename:match" format
                    filename = line.split(':', 1)[0]
                    if filename and not filename.startswith('grep:'):
                        filenames.add(filename)
            
            return len(filenames) if filenames else None
        
        return None
    
    def _extract_error_output(self, parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract error output from response data.
        
        Args:
            parsed_data: Normalized response data
            
        Returns:
            Optional[str]: Error output text, or None if not found
        """
        # Check various fields where error output might be stored
        error_fields = ["error", "error_message", "stderr", "exception", "message"]
        
        for field in error_fields:
            if field in parsed_data and isinstance(parsed_data[field], str):
                error_text = parsed_data[field].strip()
                if error_text:
                    return error_text
        
        # Check if the main content indicates an error
        text_content = self._extract_text_content(parsed_data)
        if text_content:
            # Look for common error indicators
            error_indicators = [
                "error:", "failed:", "exception:", "not found", "permission denied",
                "no such file", "invalid", "cannot", "unable to"
            ]
            
            text_lower = text_content.lower()
            for indicator in error_indicators:
                if indicator in text_lower:
                    return text_content
        
        return None
    
    def _is_error_response(self, response_data: Any) -> bool:
        """
        Check if the response indicates an error condition.
        
        Enhanced for search operations to detect common search-related errors.
        
        Args:
            response_data: Raw response data
            
        Returns:
            bool: True if the response indicates an error
        """
        # Use base class error detection first
        if super()._is_error_response(response_data):
            return True
        
        # Search operation specific error detection
        # Check both text content and error output
        text_sources = [
            self._extract_text_content(response_data),
            self._extract_error_output(self._parse_response_data(response_data))
        ]
        
        for text_content in text_sources:
            if text_content:
                import re
                error_patterns = [
                    r"no such file or directory",
                    r"permission denied",
                    r"not found",
                    r"invalid (option|argument|pattern)",
                    r"too many arguments",
                    r"cannot (access|read|open)",
                    r"failed to",
                    r"error:",
                    r"usage:",  # Usage messages indicate incorrect usage
                ]
                
                text_lower = text_content.lower()
                for pattern in error_patterns:
                    if re.search(pattern, text_lower):
                        return True
        
        return False


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
    
    # Test GitFilter
    git_filter = GitFilter()
    print(f"Git filter tool name: {git_filter.get_tool_name()}")
    
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
    
    # Test GitFilter
    print("\n" + "="*50)
    print("Testing GitFilter implementation...")
    
    # Test cases for GitFilter
    git_test_cases = [
        # Success cases - Important operations
        {"exit_code": 0, "command": "git commit -m 'Initial commit'", "output": "[main 1234567] Initial commit\n 1 file changed, 10 insertions(+)\n create mode 100644 file.txt"},
        {"exit_code": 0, "command": "git push origin main", "output": "To github.com:user/repo.git\n   abc123..def456  main -> main"},
        {"exit_code": 0, "command": "git pull origin main", "output": "From github.com:user/repo\n * branch            main       -> FETCH_HEAD\nAlready up to date."},
        {"exit_code": 0, "command": "git checkout -b feature-branch", "output": "Switched to a new branch 'feature-branch'"},
        {"exit_code": 0, "command": "git checkout main", "output": "Switched to branch 'main'"},
        {"exit_code": 0, "command": "git merge feature-branch", "output": "Merge made by the 'recursive' strategy.\n file.txt | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)"},
        {"exit_code": 0, "command": "git clone https://github.com/user/repo.git", "output": "Cloning into 'repo'...\nremote: Enumerating objects: 100, done."},
        
        # Status operations with meaningful output
        {"exit_code": 0, "command": "git status", "output": "On branch main\nChanges to be committed:\n  (use \"git restore --staged <file>...\" to unstage)\n        modified:   file.txt"},
        {"exit_code": 0, "command": "git status", "output": "On branch main\nUntracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n        new_file.txt"},
        {"exit_code": 0, "command": "git status", "output": "On branch main\nnothing to commit, working tree clean"},
        
        # Less important operations
        {"exit_code": 0, "command": "git log --oneline", "output": "abc123 Latest commit\ndef456 Previous commit"},
        {"exit_code": 0, "command": "git diff", "output": "diff --git a/file.txt b/file.txt\nindex 1234567..abcdefg 100644\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old line\n+new line"},
        
        # Error cases
        {"exit_code": 128, "command": "git commit", "stderr": "fatal: not a git repository (or any of the parent directories): .git"},
        {"exit_code": 1, "command": "git push", "stderr": "fatal: Authentication failed for 'https://github.com/user/repo.git/'"},
        {"exit_code": 1, "command": "git merge branch", "stderr": "error: merge conflict in file.txt"},
        {"exit_code": 1, "command": "git commit", "stderr": "nothing to commit, working tree clean"},
        
        # Edge cases
        None,
        "simple string output",
        {"exit_code": 0, "output": ""},
        {"exit_code": 0, "command": "git --version", "output": "git version 2.34.1"},
    ]
    
    print("\nTesting GitFilter with various scenarios:")
    for i, test_case in enumerate(git_test_cases, 1):
        should_speak = git_filter.should_speak("Git", test_case)
        custom_message = git_filter.get_custom_message("Git", test_case)
        
        print(f"{i}. Input: {test_case}")
        print(f"   Should speak: {should_speak}")
        print(f"   Custom message: {custom_message}")
        print()
    
    # Test Git command parsing
    print("Testing Git command parsing:")
    command_parsing_tests = [
        {"command": "git commit -m 'message'", "expected": "commit"},
        {"command": "/usr/bin/git push origin main", "expected": "push"},
        {"command": "git status --porcelain", "expected": "status"},
        {"command": "git checkout -b feature", "expected": "checkout"},
        {"command": "git log --oneline", "expected": "log"},
        {"command": "git", "expected": None},
        {"command": "not-git command", "expected": "command"},
    ]
    
    for i, test_case in enumerate(command_parsing_tests, 1):
        parsed_data = git_filter._parse_response_data(test_case)
        operation = git_filter._parse_git_command(test_case["command"])
        expected = test_case["expected"]
        
        print(f"{i}. Command: '{test_case['command']}' -> Parsed: {operation} (Expected: {expected})")
        assert operation == expected, f"Expected {expected}, got {operation}"
    
    # Test important operation detection
    print("\nTesting important Git operation detection:")
    important_operations_tests = [
        {"command": "git commit -m 'test'", "expected": True},
        {"command": "git push origin main", "expected": True},
        {"command": "git pull", "expected": True},
        {"command": "git merge branch", "expected": True},
        {"command": "git log --oneline", "expected": False},
        {"command": "git diff", "expected": False},
        {"command": "git show", "expected": False},
    ]
    
    for i, test_case in enumerate(important_operations_tests, 1):
        parsed_data = git_filter._parse_response_data(test_case)
        is_important = git_filter._is_important_git_operation(parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. Command: '{test_case['command']}' -> Important: {is_important} (Expected: {expected})")
        assert is_important == expected, f"Expected {expected}, got {is_important}"
    
    # Test meaningful output detection
    print("\nTesting meaningful Git output detection:")
    meaningful_output_tests = [
        {"output": "modified:   file.txt", "expected": True},
        {"output": "Changes to be committed:", "expected": True},
        {"output": "Untracked files:", "expected": True},
        {"output": "Your branch is ahead of 'origin/main' by 1 commit.", "expected": True},
        {"output": "Fast-forward", "expected": True},
        {"output": "Already up to date", "expected": True},
        {"output": "3 files changed, 15 insertions(+), 2 deletions(-)", "expected": True},
        {"output": "On branch main", "expected": True},  # Short message
        {"output": "very long verbose output that goes on and on with lots of details that probably shouldn't be spoken aloud because it's too much information", "expected": False},
        {"output": "", "expected": False},
    ]
    
    for i, test_case in enumerate(meaningful_output_tests, 1):
        parsed_data = git_filter._parse_response_data({"output": test_case["output"]})
        is_meaningful = git_filter._has_meaningful_git_output(parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. Output: '{test_case['output'][:50]}...' -> Meaningful: {is_meaningful} (Expected: {expected})")
        assert is_meaningful == expected, f"Expected {expected}, got {is_meaningful}"
    
    # Test specific message generation
    print("\nTesting specific Git message generation:")
    specific_message_tests = [
        {"command": "git commit -m 'test'", "output": "[main 1234567] test commit\n 2 files changed, 5 insertions(+)", "expected_contains": "Changes committed"},
        {"command": "git push origin main", "output": "main -> main", "expected_contains": "Pushed main to remote"},
        {"command": "git pull", "output": "Already up to date", "expected_contains": "already up to date"},
        {"command": "git checkout main", "output": "Switched to branch 'main'", "expected_contains": "Switched to branch main"},
        {"command": "git merge feature", "output": "Fast-forward", "expected_contains": "fast-forward"},
    ]
    
    for i, test_case in enumerate(specific_message_tests, 1):
        parsed_data = git_filter._parse_response_data({"command": test_case["command"], "output": test_case["output"]})
        message = git_filter._generate_git_success_message(parsed_data)
        expected_contains = test_case["expected_contains"]
        
        print(f"{i}. Command: '{test_case['command']}' -> Message: '{message}'")
        assert expected_contains.lower() in message.lower(), f"Expected message to contain '{expected_contains}', got '{message}'"
    
    print("\nGitFilter implementation testing complete!")
    
    # Test FileOperationFilter
    print("\n" + "="*50)
    print("Testing FileOperationFilter implementation...")
    
    # Test FileOperationFilter
    file_filter = FileOperationFilter()
    print(f"FileOperation filter tool name: {file_filter.get_tool_name()}")
    
    # Test cases for FileOperationFilter
    file_test_cases = [
        # Write operations - Success
        {"file_path": "/path/to/file.txt", "content": "Hello World"},
        {"filepath": "config.json", "status": "success"},
        {"filename": "script.py", "operation": "write"},
        
        # Edit operations - Success  
        {"file_path": "/path/to/existing.txt", "old_string": "old", "new_string": "new"},
        {"filepath": "main.py", "edits": [{"old": "foo", "new": "bar"}]},
        
        # MultiEdit operations - Success
        {"file_path": "app.js", "edits": [{"old": "a", "new": "b"}, {"old": "c", "new": "d"}]},
        {"filepath": "utils.py", "edit_count": 5},
        
        # Read operations - Success (should be silent)
        {"file_path": "/path/to/read.txt", "content": "File contents here"},
        {"filepath": "data.json", "output": "{\n  \"key\": \"value\"\n}"},
        
        # Error cases
        {"file_path": "/nonexistent/file.txt", "error": "File not found"},
        {"filepath": "protected.txt", "error": "Permission denied"},
        {"filename": "directory", "error": "Target is a directory"},
        {"file_path": "readonly.txt", "error": "File is read-only"},
        {"filepath": "failed.txt", "error": "Failed to read file: I/O error"},
        
        # Error with stderr
        {"file_path": "missing.txt", "stderr": "No such file or directory: missing.txt"},
        {"filepath": "denied.txt", "stderr": "Permission denied: Access is denied"},
        
        # Notebook operations
        {"notebook_path": "/path/to/notebook.ipynb", "cell_id": "123"},
        {"notebookPath": "analysis.ipynb", "error": "Failed to read notebook"},
        
        # Edge cases
        None,
        "simple string response",
        {"status": "success"},
        {"file_path": "", "content": "empty path"},
        {"filepath": "file.txt", "content": ""},
    ]
    
    print("\nTesting FileOperationFilter with various scenarios:")
    
    # Test with different tool names
    tool_names = ["Read", "Write", "Edit", "MultiEdit", "NotebookRead", "NotebookEdit"]
    
    for tool_name in tool_names:
        print(f"\n--- Testing {tool_name} tool ---")
        
        for i, test_case in enumerate(file_test_cases[:8], 1):  # Test first 8 cases for each tool
            should_speak = file_filter.should_speak(tool_name, test_case)
            custom_message = file_filter.get_custom_message(tool_name, test_case)
            
            print(f"{i}. {tool_name} Input: {test_case}")
            print(f"   Should speak: {should_speak}")
            print(f"   Custom message: {custom_message}")
            print()
    
    # Test error detection
    print("\nTesting file operation error detection:")
    error_test_cases = [
        {"file_path": "test.txt", "error": "File not found"},
        {"filepath": "test.txt", "error": "Permission denied"},
        {"filename": "test.txt", "error": "Target is a directory"},
        {"file_path": "test.txt", "error": "File is read-only"},
        {"filepath": "test.txt", "stderr": "No such file or directory"},
        {"file_path": "test.txt", "content": "error: unable to read file"},
        {"filepath": "test.txt", "content": "failed to write file"},
        {"file_path": "test.txt", "content": "success"},
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        is_error = file_filter._is_error_response(test_case)
        print(f"{i}. Input: {test_case} -> Is error: {is_error}")
    
    # Test filename extraction
    print("\nTesting filename extraction:")
    filename_test_cases = [
        {"file_path": "/path/to/file.txt", "expected": "file.txt"},
        {"filepath": "config.json", "expected": "config.json"},
        {"filename": "script.py", "expected": "script.py"},
        {"notebook_path": "/notebooks/analysis.ipynb", "expected": "analysis.ipynb"},
        {"path": "data/results.csv", "expected": "results.csv"},
        {"error": "File not found: missing.txt", "expected": "missing.txt"},
        {"content": "Error reading 'test.py': permission denied", "expected": "test.py"},
        {"output": "Successfully wrote to output.log", "expected": "output.log"},
        {"file_path": "", "expected": None},
        {"content": "no file mentioned", "expected": None},
    ]
    
    for i, test_case in enumerate(filename_test_cases, 1):
        parsed_data = file_filter._parse_response_data(test_case)
        extracted = file_filter._extract_filename(parsed_data)
        expected = test_case.get("expected")
        
        print(f"{i}. Input: {test_case} -> Extracted: {extracted} (Expected: {expected})")
        if expected is not None:
            assert extracted == expected, f"Expected {expected}, got {extracted}"
    
    # Test edit count extraction
    print("\nTesting edit count extraction:")
    edit_count_test_cases = [
        {"edit_count": 3, "expected": 3},
        {"edits": [{"old": "a", "new": "b"}, {"old": "c", "new": "d"}], "expected": 2},
        {"changes": 5, "expected": 5},
        {"content": "Applied 4 edits successfully", "expected": 4},
        {"output": "3 changes made to the file", "expected": 3},
        {"content": "2 files updated", "expected": 2},
        {"content": "Single edit applied", "expected": None},
        {"edits": [], "expected": 0},
        {"content": "no numbers here", "expected": None},
    ]
    
    for i, test_case in enumerate(edit_count_test_cases, 1):
        parsed_data = file_filter._parse_response_data(test_case)
        extracted = file_filter._extract_edit_count(parsed_data)
        expected = test_case.get("expected")
        
        print(f"{i}. Input: {test_case} -> Extracted: {extracted} (Expected: {expected})")
        if expected is not None:
            assert extracted == expected, f"Expected {expected}, got {extracted}"
    
    # Test specific error message generation
    print("\nTesting file operation error message generation:")
    file_error_test_cases = [
        {"tool": "Read", "file_path": "missing.txt", "error": "File not found", "expected_contains": "File not found: missing.txt"},
        {"tool": "Write", "filepath": "protected.txt", "error": "Permission denied", "expected_contains": "Permission denied: protected.txt"},
        {"tool": "Edit", "filename": "directory", "error": "Target is a directory", "expected_contains": "Target is a directory: directory"},
        {"tool": "MultiEdit", "error": "Read-only filesystem", "expected_contains": "read-only"},
        {"tool": "NotebookRead", "notebook_path": "test.ipynb", "error": "No such file", "expected_contains": "File not found: test.ipynb"},
    ]
    
    for i, test_case in enumerate(file_error_test_cases, 1):
        parsed_data = file_filter._parse_response_data(test_case)
        message = file_filter._generate_file_error_message(test_case["tool"], parsed_data)
        expected_contains = test_case["expected_contains"]
        
        print(f"{i}. Tool: {test_case['tool']} -> Message: '{message}'")
        assert expected_contains.lower() in message.lower(), f"Expected message to contain '{expected_contains}', got '{message}'"
    
    # Test specific success message generation
    print("\nTesting file operation success message generation:")
    file_success_test_cases = [
        {"tool": "Write", "file_path": "new.txt", "expected_contains": "File created: new.txt"},
        {"tool": "Edit", "filepath": "existing.py", "expected_contains": "File updated: existing.py"},
        {"tool": "MultiEdit", "file_path": "app.js", "edit_count": 3, "expected_contains": "Applied 3 edits"},
        {"tool": "NotebookEdit", "notebook_path": "analysis.ipynb", "expected_contains": "Notebook updated: analysis.ipynb"},
        {"tool": "Read", "file_path": "data.txt", "expected": None},  # Should be None for Read
    ]
    
    for i, test_case in enumerate(file_success_test_cases, 1):
        parsed_data = file_filter._parse_response_data(test_case)
        message = file_filter._generate_file_success_message(test_case["tool"], parsed_data)
        
        print(f"{i}. Tool: {test_case['tool']} -> Message: '{message}'")
        
        if "expected" in test_case:
            expected = test_case["expected"]
            assert message == expected, f"Expected {expected}, got {message}"
        elif "expected_contains" in test_case:
            expected_contains = test_case["expected_contains"]
            assert expected_contains.lower() in message.lower(), f"Expected message to contain '{expected_contains}', got '{message}'"
    
    print("\nFileOperationFilter implementation testing complete!")
    
    # Test SearchFilter
    print("\n" + "="*50)
    print("Testing SearchFilter implementation...")
    
    # Test SearchFilter
    search_filter = SearchFilter()
    print(f"Search filter tool name: {search_filter.get_tool_name()}")
    
    # Test cases for SearchFilter
    search_test_cases = [
        # Grep operations - Results
        {"output": "file1.txt:match1\nfile2.txt:match2\nfile1.txt:match3", "tool": "Grep"},
        {"content": "main.py:def function\nutils.py:def helper", "tool": "Grep"},
        {"result": "config.json:key=value", "tool": "Grep"},
        
        # Grep operations - No results
        {"output": "", "tool": "Grep"},
        {"content": "grep: pattern not found", "tool": "Grep"},
        
        # LS operations - Results
        {"output": "file1.txt\nfile2.py\ndirectory/", "tool": "LS"},
        {"content": "total 64\n-rw-r--r-- 1 user staff 1234 file.txt\ndrwxr-xr-x 2 user staff 68 dir/", "tool": "LS"},
        {"result": "README.md\nsrc/\ntests/", "tool": "LS"},
        
        # LS operations - Empty directory
        {"output": "", "tool": "LS"},
        {"content": "total 0", "tool": "LS"},
        
        # Find operations - Results
        {"output": "/path/to/file1.txt\n/path/to/file2.py", "tool": "Find"},
        {"content": "./src/main.py\n./tests/test.py", "tool": "Find"},
        
        # Find operations - No results
        {"output": "", "tool": "Find"},
        {"content": "find: no matches found", "tool": "Find"},
        
        # Which operations
        {"output": "/usr/bin/python", "tool": "Which"},
        {"output": "", "tool": "Which"},
        
        # Error cases
        {"error": "Permission denied", "tool": "Grep"},
        {"stderr": "No such file or directory", "tool": "LS"},
        {"error": "Invalid search pattern", "tool": "Find"},
        
        # Edge cases
        None,
        "simple string response",
        {"status": "success"},
        {"output": "Binary file matches", "tool": "Grep"},
    ]
    
    print("\nTesting SearchFilter with various scenarios:")
    
    # Test with different search tools
    search_tools = ["Grep", "LS", "Find", "Locate", "Which", "Whereis"]
    
    for tool_name in search_tools:
        print(f"\n--- Testing {tool_name} tool ---")
        
        # Test a subset of cases for each tool
        relevant_cases = [case for case in search_test_cases[:8] if case and case.get("tool") == tool_name]
        if not relevant_cases:
            relevant_cases = search_test_cases[:3]  # Use first 3 generic cases
        
        for i, test_case in enumerate(relevant_cases, 1):
            should_speak = search_filter.should_speak(tool_name, test_case)
            custom_message = search_filter.get_custom_message(tool_name, test_case)
            
            print(f"{i}. {tool_name} Input: {test_case}")
            print(f"   Should speak: {should_speak}")
            print(f"   Custom message: {custom_message}")
            print()
    
    # Test result count extraction
    print("\nTesting search result count extraction:")
    count_test_cases = [
        {"tool": "Grep", "output": "file1.txt:match1\nfile2.txt:match2\nfile1.txt:match3", "expected": 3},
        {"tool": "LS", "output": "file1.txt\nfile2.py\ndirectory/", "expected": 3},
        {"tool": "Find", "output": "/path/file1\n/path/file2", "expected": 2},
        {"tool": "Grep", "output": "", "expected": 0},
        {"tool": "LS", "output": "total 0", "expected": 0},
        {"tool": "Find", "output": "", "expected": 0},
        {"tool": "Grep", "count": 5, "expected": 5},
        {"tool": "LS", "items": 10, "expected": 10},
        {"tool": "Find", "result_count": 7, "expected": 7},
    ]
    
    for i, test_case in enumerate(count_test_cases, 1):
        tool_name = test_case["tool"]
        parsed_data = search_filter._parse_response_data(test_case)
        extracted = search_filter._extract_result_count(tool_name, parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. {tool_name} Input: {test_case} -> Extracted: {extracted} (Expected: {expected})")
        assert extracted == expected, f"Expected {expected}, got {extracted}"
    
    # Test meaningful results detection
    print("\nTesting meaningful search results detection:")
    meaningful_test_cases = [
        {"tool": "Grep", "output": "file.txt:match", "expected": True},
        {"tool": "Grep", "output": "", "expected": False},
        {"tool": "Grep", "output": "grep: pattern not found", "expected": False},
        {"tool": "LS", "output": "file1.txt\nfile2.py", "expected": True},
        {"tool": "LS", "output": "", "expected": False},
        {"tool": "LS", "output": "total 0", "expected": False},
        {"tool": "Find", "output": "/path/file.txt", "expected": True},
        {"tool": "Find", "output": "", "expected": False},
        {"tool": "Which", "output": "/usr/bin/python", "expected": True},
        {"tool": "Which", "output": "", "expected": False},
    ]
    
    for i, test_case in enumerate(meaningful_test_cases, 1):
        tool_name = test_case["tool"]
        parsed_data = search_filter._parse_response_data(test_case)
        is_meaningful = search_filter._has_meaningful_search_results(tool_name, parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. {tool_name} '{test_case.get('output', '')[:30]}...' -> Meaningful: {is_meaningful} (Expected: {expected})")
        assert is_meaningful == expected, f"Expected {expected}, got {is_meaningful}"
    
    # Test empty results detection
    print("\nTesting empty search results detection:")
    empty_test_cases = [
        {"tool": "Grep", "output": "file.txt:match", "expected": False},
        {"tool": "Grep", "output": "", "expected": True},
        {"tool": "LS", "output": "file1.txt", "expected": False},
        {"tool": "LS", "output": "", "expected": True},
        {"tool": "Find", "output": "/path/file.txt", "expected": False},
        {"tool": "Find", "output": "", "expected": True},
        {"tool": "Which", "output": "/usr/bin/python", "expected": False},
        {"tool": "Which", "output": "", "expected": True},
    ]
    
    for i, test_case in enumerate(empty_test_cases, 1):
        tool_name = test_case["tool"]
        parsed_data = search_filter._parse_response_data(test_case)
        is_empty = search_filter._has_empty_results(tool_name, parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. {tool_name} '{test_case.get('output', '')[:30]}...' -> Empty: {is_empty} (Expected: {expected})")
        assert is_empty == expected, f"Expected {expected}, got {is_empty}"
    
    # Test error detection
    print("\nTesting search operation error detection:")
    search_error_test_cases = [
        {"tool": "Grep", "error": "Permission denied", "expected": True},
        {"tool": "LS", "stderr": "No such file or directory", "expected": True},
        {"tool": "Find", "content": "find: invalid option", "expected": True},
        {"tool": "Grep", "output": "file.txt:match", "expected": False},
        {"tool": "LS", "output": "file1.txt", "expected": False},
        {"tool": "Find", "output": "/path/file.txt", "expected": False},
        {"tool": "Which", "content": "usage: which command", "expected": True},
    ]
    
    for i, test_case in enumerate(search_error_test_cases, 1):
        tool_name = test_case["tool"]
        is_error = search_filter._is_error_response(test_case)
        expected = test_case["expected"]
        
        print(f"{i}. {tool_name} Input: {test_case} -> Is error: {is_error} (Expected: {expected})")
        assert is_error == expected, f"Expected {expected}, got {is_error}"
    
    # Test specific message generation
    print("\nTesting search-specific message generation:")
    message_test_cases = [
        {"tool": "Grep", "output": "file1.txt:match1\nfile2.txt:match2", "expected_contains": "Found 2 matches"},
        {"tool": "Grep", "output": "", "expected_contains": "No matches found"},
        {"tool": "LS", "output": "file1.txt\nfile2.py\ndir/", "expected_contains": "Directory contains 3 items"},
        {"tool": "LS", "output": "", "expected_contains": "Directory is empty"},
        {"tool": "Find", "output": "/path/file1\n/path/file2", "expected_contains": "Found 2 files"},
        {"tool": "Find", "output": "", "expected_contains": "No files found"},
        {"tool": "Which", "output": "/usr/bin/python", "expected_contains": "Command found"},
        {"tool": "Which", "output": "", "expected_contains": "Command not found"},
        {"tool": "Whereis", "output": "/usr/bin/python", "expected_contains": "Location found"},
        {"tool": "Whereis", "output": "", "expected_contains": "Location not found"},
    ]
    
    for i, test_case in enumerate(message_test_cases, 1):
        tool_name = test_case["tool"]
        parsed_data = search_filter._parse_response_data(test_case)
        message = search_filter._generate_search_result_message(tool_name, parsed_data)
        expected_contains = test_case["expected_contains"]
        
        print(f"{i}. {tool_name} -> Message: '{message}'")
        assert expected_contains.lower() in message.lower(), f"Expected message to contain '{expected_contains}', got '{message}'"
    
    # Test file count extraction for grep
    print("\nTesting file count extraction for grep:")
    file_count_test_cases = [
        {"output": "file1.txt:match1\nfile2.txt:match2\nfile1.txt:match3", "expected": 2},
        {"output": "single.txt:match", "expected": 1},
        {"output": "file1.txt:match1\nfile2.txt:match2\nfile3.txt:match3", "expected": 3},
        {"output": "", "expected": None},
        {"output": "match without colon", "expected": None},
        {"files": 5, "expected": 5},
    ]
    
    for i, test_case in enumerate(file_count_test_cases, 1):
        parsed_data = search_filter._parse_response_data(test_case)
        extracted = search_filter._extract_file_count(parsed_data)
        expected = test_case["expected"]
        
        print(f"{i}. Input: {test_case} -> Extracted: {extracted} (Expected: {expected})")
        if expected is not None:
            assert extracted == expected, f"Expected {expected}, got {extracted}"
    
    # Test error message generation
    print("\nTesting search error message generation:")
    search_error_message_test_cases = [
        {"tool": "Grep", "error": "Permission denied", "expected_contains": "Permission denied"},
        {"tool": "LS", "error": "No such file or directory", "expected_contains": "Path not found"},
        {"tool": "Find", "error": "Invalid search pattern", "expected_contains": "Invalid search criteria"},
        {"tool": "Locate", "error": "Database not found", "expected_contains": "Locate failed"},
        {"tool": "Which", "error": "Command not found", "expected_contains": "Not found"},
    ]
    
    for i, test_case in enumerate(search_error_message_test_cases, 1):
        tool_name = test_case["tool"]
        parsed_data = search_filter._parse_response_data(test_case)
        message = search_filter._generate_search_error_message(tool_name, parsed_data)
        expected_contains = test_case["expected_contains"]
        
        print(f"{i}. {tool_name} -> Error message: '{message}'")
        assert expected_contains.lower() in message.lower(), f"Expected message to contain '{expected_contains}', got '{message}'"
    
    print("\nSearchFilter implementation testing complete!")