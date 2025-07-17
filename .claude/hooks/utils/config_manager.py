#!/usr/bin/env python3
"""
Configuration Management System for TTS Settings

This module provides the TTSConfig class for managing TTS configuration
with proper type hints, defaults, and validation support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


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


# Default configuration instance
DEFAULT_CONFIG = TTSConfig()