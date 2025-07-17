#!/usr/bin/env python3
"""
Configuration Management System for TTS Settings

This module provides the TTSConfig class for managing TTS configuration
with proper type hints, defaults, and validation support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple


@dataclass
class TTSConfig:
    """
    Core TTS configuration class with all required properties and type annotations.
    
    This class manages TTS settings including voice configuration, word limits,
    tool exclusions, and audio playback settings.
    """
    
    # Core TTS Settings
    enabled: bool = True
    max_words: int = 20
    voice_id: str = "6sFKzaJr574YWVu4UuJF"
    skip_tools: List[str] = field(default_factory=lambda: [
        "Read", "Grep", "LS", "TodoRead", "Glob", "git", "Bash"
    ])
    speak_confirmations: bool = True
    
    # Audio Settings
    volume: float = 1.0  # Range: 0.0-1.0
    speed: float = 1.0   # Range: 0.5-2.0
    
    # Word Count Settings
    word_count_threshold: int = 5
    max_word_count: int = 100
    
    # Response TTS Settings
    enable_response_tts: bool = True
    response_tts_min_words: int = 5
    response_tts_max_words: int = 150
    response_delay: float = 1.0
    
    # Filter Settings
    filter_tool_responses: bool = True
    filter_code_blocks: bool = True
    filter_file_paths: bool = True
    
    # Advanced Filter Settings
    filter_settings: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "bash": {
            "speak_success": True,
            "speak_errors": True
        },
        "git": {
            "speak_success": False,
            "speak_errors": True
        },
        "file_operation": {
            "speak_write_operations": True,
            "speak_read_operations": False
        },
        "search": {
            "speak_search_results": False,
            "speak_errors": True
        }
    })
    
    def __post_init__(self):
        """
        Post-initialization validation and normalization.
        Called automatically after dataclass initialization.
        """
        # Ensure volume is within valid range
        if self.volume < 0.0:
            self.volume = 0.0
        elif self.volume > 1.0:
            self.volume = 1.0
            
        # Ensure speed is within valid range
        if self.speed < 0.5:
            self.speed = 0.5
        elif self.speed > 2.0:
            self.speed = 2.0
            
        # Ensure positive word counts
        if self.max_words <= 0:
            self.max_words = 20
        if self.word_count_threshold <= 0:
            self.word_count_threshold = 5
        if self.max_word_count <= 0:
            self.max_word_count = 100
            
        # Ensure voice_id is not empty
        if not self.voice_id or not self.voice_id.strip():
            self.voice_id = "6sFKzaJr574YWVu4UuJF"
            
        # Ensure skip_tools is a list
        if not isinstance(self.skip_tools, list):
            self.skip_tools = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TTSConfig to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing all configuration values
        """
        return {
            "enabled": self.enabled,
            "max_words": self.max_words,
            "voice_id": self.voice_id,
            "skip_tools": self.skip_tools.copy(),
            "speak_confirmations": self.speak_confirmations,
            "volume": self.volume,
            "speed": self.speed,
            "word_count_threshold": self.word_count_threshold,
            "max_word_count": self.max_word_count,
            "enable_response_tts": self.enable_response_tts,
            "response_tts_min_words": self.response_tts_min_words,
            "response_tts_max_words": self.response_tts_max_words,
            "response_delay": self.response_delay,
            "filter_tool_responses": self.filter_tool_responses,
            "filter_code_blocks": self.filter_code_blocks,
            "filter_file_paths": self.filter_file_paths,
            "filter_settings": self.filter_settings.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TTSConfig':
        """
        Create TTSConfig instance from dictionary.
        
        Args:
            data: Dictionary containing configuration values
            
        Returns:
            TTSConfig: New TTSConfig instance with provided values
        """
        # Create instance with defaults, then update with provided data
        config = cls()
        
        # Update only the fields that are present in the data
        for field_name, field_value in data.items():
            if hasattr(config, field_name):
                setattr(config, field_name, field_value)
        
        return config
    
    def update(self, **kwargs) -> None:
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Re-run post-init validation
        self.__post_init__()
    
    def copy(self) -> 'TTSConfig':
        """
        Create a deep copy of the configuration.
        
        Returns:
            TTSConfig: New TTSConfig instance with same values
        """
        return TTSConfig.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        """String representation of TTSConfig for debugging."""
        return f"TTSConfig(enabled={self.enabled}, voice_id='{self.voice_id}', max_words={self.max_words})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"TTS Configuration: {status}, Voice: {self.voice_id}, Max Words: {self.max_words}"
    
    # Validation Methods
    
    def validate_volume(self) -> Tuple[bool, str]:
        """
        Validate volume setting (0.0-1.0 range).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.volume, (int, float)):
            return False, "Volume must be a number"
        
        if self.volume < 0.0:
            return False, "Volume must be 0.0 or higher"
        
        if self.volume > 1.0:
            return False, "Volume must be 1.0 or lower"
        
        return True, ""
    
    def validate_speed(self) -> Tuple[bool, str]:
        """
        Validate speed setting (0.5-2.0 range).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.speed, (int, float)):
            return False, "Speed must be a number"
        
        if self.speed < 0.5:
            return False, "Speed must be 0.5 or higher"
        
        if self.speed > 2.0:
            return False, "Speed must be 2.0 or lower"
        
        return True, ""
    
    def validate_voice_id(self) -> Tuple[bool, str]:
        """
        Validate voice_id setting (non-empty string).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.voice_id, str):
            return False, "Voice ID must be a string"
        
        if not self.voice_id.strip():
            return False, "Voice ID cannot be empty"
        
        # Check for reasonable length (ElevenLabs voice IDs are typically 20 characters)
        if len(self.voice_id.strip()) < 5:
            return False, "Voice ID appears to be too short"
        
        if len(self.voice_id.strip()) > 50:
            return False, "Voice ID appears to be too long"
        
        return True, ""
    
    def validate_skip_tools(self) -> Tuple[bool, str]:
        """
        Validate skip_tools setting (list of strings).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.skip_tools, list):
            return False, "Skip tools must be a list"
        
        for i, tool in enumerate(self.skip_tools):
            if not isinstance(tool, str):
                return False, f"Skip tools item {i} must be a string, got {type(tool).__name__}"
            
            if not tool.strip():
                return False, f"Skip tools item {i} cannot be empty"
        
        return True, ""
    
    def validate_max_words(self) -> Tuple[bool, str]:
        """
        Validate max_words setting (positive integer).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.max_words, int):
            return False, "Max words must be an integer"
        
        if self.max_words <= 0:
            return False, "Max words must be greater than 0"
        
        if self.max_words > 10000:
            return False, "Max words must be 10000 or lower"
        
        return True, ""
    
    def validate_word_count_threshold(self) -> Tuple[bool, str]:
        """
        Validate word_count_threshold setting (positive integer).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.word_count_threshold, int):
            return False, "Word count threshold must be an integer"
        
        if self.word_count_threshold <= 0:
            return False, "Word count threshold must be greater than 0"
        
        if self.word_count_threshold > 1000:
            return False, "Word count threshold must be 1000 or lower"
        
        return True, ""
    
    def validate_max_word_count(self) -> Tuple[bool, str]:
        """
        Validate max_word_count setting (positive integer).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.max_word_count, int):
            return False, "Max word count must be an integer"
        
        if self.max_word_count <= 0:
            return False, "Max word count must be greater than 0"
        
        if self.max_word_count > 10000:
            return False, "Max word count must be 10000 or lower"
        
        return True, ""
    
    def validate_response_delay(self) -> Tuple[bool, str]:
        """
        Validate response_delay setting (non-negative float).
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.response_delay, (int, float)):
            return False, "Response delay must be a number"
        
        if self.response_delay < 0:
            return False, "Response delay must be 0 or higher"
        
        if self.response_delay > 30:
            return False, "Response delay must be 30 seconds or lower"
        
        return True, ""
    
    def validate_filter_settings(self) -> Tuple[bool, str]:
        """
        Validate filter_settings structure.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(self.filter_settings, dict):
            return False, "Filter settings must be a dictionary"
        
        for tool_name, tool_settings in self.filter_settings.items():
            if not isinstance(tool_name, str):
                return False, f"Filter setting key must be a string, got {type(tool_name).__name__}"
            
            if not isinstance(tool_settings, dict):
                return False, f"Filter settings for '{tool_name}' must be a dictionary"
            
            for setting_name, setting_value in tool_settings.items():
                if not isinstance(setting_name, str):
                    return False, f"Filter setting name for '{tool_name}' must be a string"
                
                if not isinstance(setting_value, bool):
                    return False, f"Filter setting '{setting_name}' for '{tool_name}' must be a boolean"
        
        return True, ""
    
    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Run all validators and return combined results.
        
        Returns:
            Tuple[bool, List[str]]: (all_valid, list_of_error_messages)
        """
        errors = []
        
        # Run all validation methods
        validators = [
            self.validate_volume,
            self.validate_speed,
            self.validate_voice_id,
            self.validate_skip_tools,
            self.validate_max_words,
            self.validate_word_count_threshold,
            self.validate_max_word_count,
            self.validate_response_delay,
            self.validate_filter_settings
        ]
        
        for validator in validators:
            is_valid, error_msg = validator()
            if not is_valid:
                errors.append(error_msg)
        
        # Additional cross-field validation
        if self.word_count_threshold > self.max_word_count:
            errors.append("Word count threshold cannot be greater than max word count")
        
        if self.response_tts_min_words > self.response_tts_max_words:
            errors.append("Response TTS min words cannot be greater than max words")
        
        return len(errors) == 0, errors
    
    def is_valid(self) -> bool:
        """
        Check if the configuration is valid.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        is_valid, _ = self.validate_all()
        return is_valid


# Default configuration instance
DEFAULT_CONFIG = TTSConfig()