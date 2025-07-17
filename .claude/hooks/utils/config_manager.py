#!/usr/bin/env python3
"""
Configuration Management System for TTS Settings

This module provides the TTSConfig class for managing TTS configuration
with proper type hints, defaults, and validation support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
import json
import os
import fcntl
import time
import threading
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from threading import Lock, Event


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
    
    # Settings.json Integration Methods
    
    @classmethod
    def load_from_settings(cls, settings_path: Optional[str] = None) -> 'TTSConfig':
        """
        Load TTS configuration from .claude/settings.json file.
        
        Args:
            settings_path: Optional path to settings.json file. If None, uses default location.
            
        Returns:
            TTSConfig: Configuration instance loaded from settings.json
        """
        if settings_path is None:
            settings_path = Path.cwd() / '.claude' / 'settings.json'
        else:
            settings_path = Path(settings_path)
        
        # Start with default configuration
        config = cls()
        
        # Check if settings file exists
        if not settings_path.exists():
            return config
        
        try:
            # Read settings file with file locking
            with open(settings_path, 'r', encoding='utf-8') as f:
                # Apply shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    settings_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Extract TTS configuration section
            tts_config = settings_data.get('tts', {})
            
            # Update configuration with settings data
            if tts_config:
                for field_name, field_value in tts_config.items():
                    if hasattr(config, field_name):
                        setattr(config, field_name, field_value)
                
                # Re-run validation after loading
                config.__post_init__()
            
            return config
            
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Return default configuration if file is corrupted or unreadable
            return config
    
    def save_to_settings(self, settings_path: Optional[str] = None) -> bool:
        """
        Save TTS configuration to .claude/settings.json file while preserving other settings.
        
        Args:
            settings_path: Optional path to settings.json file. If None, uses default location.
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        if settings_path is None:
            settings_path = Path.cwd() / '.claude' / 'settings.json'
        else:
            settings_path = Path(settings_path)
        
        # Ensure directory exists
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration before saving
        if not self.is_valid():
            return False
        
        try:
            # Read existing settings or create new structure
            existing_settings = {}
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    # Apply shared lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        existing_settings = json.load(f)
                    except json.JSONDecodeError:
                        existing_settings = {}
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Update TTS section while preserving other settings
            existing_settings['tts'] = self.to_dict()
            
            # Write back to file with exclusive lock
            with open(settings_path, 'w', encoding='utf-8') as f:
                # Apply exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(existing_settings, f, indent=2, ensure_ascii=False)
                    f.write('\n')  # Add trailing newline
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            return True
            
        except (IOError, OSError, json.JSONEncodeError) as e:
            return False
    
    @classmethod
    def load_from_settings_with_fallback(cls, settings_path: Optional[str] = None, 
                                       fallback_path: Optional[str] = None) -> 'TTSConfig':
        """
        Load TTS configuration with fallback to .claude/tts.json if settings.json doesn't contain TTS config.
        
        Args:
            settings_path: Optional path to settings.json file
            fallback_path: Optional path to fallback tts.json file
            
        Returns:
            TTSConfig: Configuration instance loaded from settings.json or fallback
        """
        # Try to load from settings.json first
        config = cls.load_from_settings(settings_path)
        
        # Check if we got default config (indicating no TTS section in settings.json)
        if settings_path is None:
            settings_path = Path.cwd() / '.claude' / 'settings.json'
        else:
            settings_path = Path(settings_path)
        
        # Check if settings.json exists and has TTS section
        has_tts_section = False
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        settings_data = json.load(f)
                        has_tts_section = 'tts' in settings_data
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (json.JSONDecodeError, IOError, OSError):
                pass
        
        # If no TTS section found, try fallback
        if not has_tts_section:
            if fallback_path is None:
                fallback_path = Path.cwd() / '.claude' / 'tts.json'
            else:
                fallback_path = Path(fallback_path)
            
            if fallback_path.exists():
                try:
                    with open(fallback_path, 'r', encoding='utf-8') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            fallback_data = json.load(f)
                            config = cls.from_dict(fallback_data)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (json.JSONDecodeError, IOError, OSError):
                    pass
        
        return config
    
    def migrate_to_settings(self, settings_path: Optional[str] = None, 
                          source_path: Optional[str] = None) -> bool:
        """
        Migrate TTS configuration from standalone tts.json to settings.json.
        
        Args:
            settings_path: Optional path to settings.json file
            source_path: Optional path to source tts.json file
            
        Returns:
            bool: True if migration was successful, False otherwise
        """
        if source_path is None:
            source_path = Path.cwd() / '.claude' / 'tts.json'
        else:
            source_path = Path(source_path)
        
        # Load configuration from source file
        if source_path.exists():
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        source_data = json.load(f)
                        # Update self with source data
                        for field_name, field_value in source_data.items():
                            if hasattr(self, field_name):
                                setattr(self, field_name, field_value)
                        
                        # Re-run validation after loading
                        self.__post_init__()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (json.JSONDecodeError, IOError, OSError):
                return False
        
        # Save to settings.json
        return self.save_to_settings(settings_path)
    
    # Default merging and backward compatibility methods
    
    def merge_with_defaults(self, user_config: Dict[str, Any], 
                           apply_compatibility: bool = True) -> 'TTSConfig':
        """
        Merge user configuration with default values, handling missing properties
        and backward compatibility.
        
        Args:
            user_config: User's configuration dictionary
            apply_compatibility: Whether to apply backward compatibility transformations
            
        Returns:
            TTSConfig: New configuration instance with merged settings
        """
        # Start with current instance as base (containing defaults)
        merged_config = self.copy()
        
        # Apply compatibility transformations if requested
        if apply_compatibility:
            user_config = self._apply_compatibility_layer(user_config)
        else:
            # Make a copy to avoid modifying the original
            user_config = user_config.copy()
        
        # Merge user settings, preserving user values while filling gaps with defaults
        for field_name, field_value in user_config.items():
            if hasattr(merged_config, field_name):
                # Special handling for nested dictionaries (like filter_settings)
                if field_name == 'filter_settings' and isinstance(field_value, dict):
                    merged_filter_settings = {}
                    
                    # Start with existing defaults
                    for tool_name, tool_settings in merged_config.filter_settings.items():
                        merged_filter_settings[tool_name] = tool_settings.copy()
                    
                    # Merge each tool's settings from user config
                    for tool_name, tool_settings in field_value.items():
                        if tool_name in merged_filter_settings:
                            # Merge tool-specific settings
                            merged_filter_settings[tool_name].update(tool_settings)
                        else:
                            # Add new tool settings
                            merged_filter_settings[tool_name] = tool_settings.copy()
                    
                    merged_config.filter_settings = merged_filter_settings
                elif field_name == 'skip_tools' and isinstance(field_value, list):
                    # Preserve user's skip_tools list
                    merged_config.skip_tools = field_value.copy()
                else:
                    # Direct assignment for other fields
                    setattr(merged_config, field_name, field_value)
        
        # Re-run validation after merging
        merged_config.__post_init__()
        
        return merged_config
    
    def _apply_compatibility_layer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply backward compatibility transformations to configuration.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Dict[str, Any]: Configuration with compatibility transformations applied
        """
        # Create a copy to avoid modifying original
        compat_config = config.copy()
        
        # Detect and handle deprecated property names
        deprecated_mappings = {
            'wordLimit': 'max_words',
            'voiceId': 'voice_id',
            'skipTools': 'skip_tools',
            'speakConfirmations': 'speak_confirmations',
            'wordCountThreshold': 'word_count_threshold',
            'maxWordCount': 'max_word_count',
            'enableResponseTts': 'enable_response_tts',
            'responseTtsMinWords': 'response_tts_min_words',
            'responseTtsMaxWords': 'response_tts_max_words',
            'responseDelay': 'response_delay',
            'filterToolResponses': 'filter_tool_responses',
            'filterCodeBlocks': 'filter_code_blocks',
            'filterFilePaths': 'filter_file_paths',
            'filterSettings': 'filter_settings'
        }
        
        # Apply deprecated property mappings
        for old_name, new_name in deprecated_mappings.items():
            if old_name in compat_config:
                compat_config[new_name] = compat_config.pop(old_name)
        
        # Handle version-specific compatibility
        config_version = compat_config.get('version', '1.0')
        
        if config_version == '1.0':
            # Version 1.0 compatibility: Convert boolean skip_tools to list
            if 'skip_tools' in compat_config:
                skip_tools = compat_config['skip_tools']
                if isinstance(skip_tools, bool):
                    # Convert boolean to default list or empty list
                    compat_config['skip_tools'] = [
                        "Read", "Grep", "LS", "TodoRead", "Glob", "git", "Bash"
                    ] if skip_tools else []
                elif isinstance(skip_tools, str):
                    # Convert single string to list
                    compat_config['skip_tools'] = [skip_tools]
        
        if config_version in ['1.0', '1.1']:
            # Version 1.1 compatibility: Handle old filter settings format
            if 'filter_settings' in compat_config:
                old_filter = compat_config['filter_settings']
                if isinstance(old_filter, dict):
                    # Convert old flat format to new nested format
                    new_filter = {}
                    for key, value in old_filter.items():
                        if key.startswith('bash_'):
                            tool_name = 'bash'
                            setting_name = key[5:]  # Remove 'bash_' prefix
                        elif key.startswith('git_'):
                            tool_name = 'git'
                            setting_name = key[4:]  # Remove 'git_' prefix
                        elif key.startswith('file_'):
                            tool_name = 'file_operation'
                            setting_name = key[5:]  # Remove 'file_' prefix
                        elif key.startswith('search_'):
                            tool_name = 'search'
                            setting_name = key[7:]  # Remove 'search_' prefix
                        else:
                            continue
                        
                        if tool_name not in new_filter:
                            new_filter[tool_name] = {}
                        new_filter[tool_name][setting_name] = value
                    
                    compat_config['filter_settings'] = new_filter
        
        # Remove version field from config (not part of TTSConfig)
        compat_config.pop('version', None)
        
        return compat_config
    
    @classmethod
    def load_with_defaults(cls, settings_path: Optional[str] = None) -> 'TTSConfig':
        """
        Load configuration from settings.json with full default merging and compatibility.
        
        Args:
            settings_path: Optional path to settings.json file
            
        Returns:
            TTSConfig: Configuration instance with merged defaults and compatibility applied
        """
        if settings_path is None:
            settings_path = Path.cwd() / '.claude' / 'settings.json'
        else:
            settings_path = Path(settings_path)
        
        # Start with default configuration
        default_config = cls()
        
        # Check if settings file exists
        if not settings_path.exists():
            return default_config
        
        try:
            # Read settings file with file locking
            with open(settings_path, 'r', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    settings_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Extract TTS configuration section
            tts_config = settings_data.get('tts', {})
            
            # Merge with defaults using compatibility layer
            if tts_config:
                return default_config.merge_with_defaults(tts_config, apply_compatibility=True)
            else:
                return default_config
                
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Return default configuration if file is corrupted or unreadable
            return default_config
    
    def upgrade_config_format(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgrade configuration from older format to current format.
        
        Args:
            config_dict: Configuration in older format
            
        Returns:
            Dict[str, Any]: Configuration in current format
        """
        # Apply compatibility layer to upgrade format
        upgraded_config = self._apply_compatibility_layer(config_dict)
        
        # Add current version marker
        upgraded_config['version'] = '2.0'
        
        return upgraded_config
    
    def get_compatibility_info(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about compatibility issues and upgrades needed.
        
        Args:
            config_dict: Configuration dictionary to analyze
            
        Returns:
            Dict[str, Any]: Information about compatibility status and needed upgrades
        """
        info = {
            'version': config_dict.get('version', '1.0'),
            'deprecated_properties': [],
            'format_changes': [],
            'upgrade_needed': False
        }
        
        # Check for deprecated property names
        deprecated_mappings = {
            'wordLimit': 'max_words',
            'voiceId': 'voice_id',
            'skipTools': 'skip_tools',
            'speakConfirmations': 'speak_confirmations',
            'wordCountThreshold': 'word_count_threshold',
            'maxWordCount': 'max_word_count',
            'enableResponseTts': 'enable_response_tts',
            'responseTtsMinWords': 'response_tts_min_words',
            'responseTtsMaxWords': 'response_tts_max_words',
            'responseDelay': 'response_delay',
            'filterToolResponses': 'filter_tool_responses',
            'filterCodeBlocks': 'filter_code_blocks',
            'filterFilePaths': 'filter_file_paths',
            'filterSettings': 'filter_settings'
        }
        
        for old_name, new_name in deprecated_mappings.items():
            if old_name in config_dict:
                info['deprecated_properties'].append({
                    'old_name': old_name,
                    'new_name': new_name,
                    'value': config_dict[old_name]
                })
                info['upgrade_needed'] = True
        
        # Check for format changes
        version = info['version']
        if version == '1.0':
            if 'skip_tools' in config_dict or 'skipTools' in config_dict:
                skip_tools = config_dict.get('skip_tools', config_dict.get('skipTools'))
                if isinstance(skip_tools, bool):
                    info['format_changes'].append({
                        'property': 'skip_tools',
                        'old_format': 'boolean',
                        'new_format': 'list of strings',
                        'description': 'skip_tools changed from boolean to list of tool names'
                    })
                    info['upgrade_needed'] = True
        
        if version in ['1.0', '1.1']:
            if 'filter_settings' in config_dict or 'filterSettings' in config_dict:
                filter_settings = config_dict.get('filter_settings', config_dict.get('filterSettings'))
                if isinstance(filter_settings, dict):
                    # Check if it uses old flat format
                    has_old_format = any(
                        key.startswith(('bash_', 'git_', 'file_', 'search_'))
                        for key in filter_settings.keys()
                    )
                    if has_old_format:
                        info['format_changes'].append({
                            'property': 'filter_settings',
                            'old_format': 'flat key-value pairs',
                            'new_format': 'nested tool-specific settings',
                            'description': 'filter_settings changed to nested structure'
                        })
                        info['upgrade_needed'] = True
        
        return info
    
    # Hot-reload functionality
    
    def __init_hot_reload(self):
        """Initialize hot-reload infrastructure if not already initialized."""
        if not hasattr(self, '_hot_reload_active'):
            self._hot_reload_active = False
            self._hot_reload_thread = None
            self._hot_reload_stop_event = Event()
            self._hot_reload_lock = Lock()
            self._hot_reload_callbacks = []
            self._hot_reload_last_modified = 0
            self._hot_reload_debounce_time = 0.5  # 500ms debounce
            self._hot_reload_settings_path = None
    
    def start_hot_reload(self, settings_path: Optional[str] = None, 
                        debounce_time: float = 0.5) -> bool:
        """
        Start hot-reload monitoring for settings.json file changes.
        
        Args:
            settings_path: Optional path to settings.json file
            debounce_time: Time in seconds to debounce file changes
            
        Returns:
            bool: True if hot-reload started successfully, False otherwise
        """
        self.__init_hot_reload()
        
        if settings_path is None:
            settings_path = str(Path.cwd() / '.claude' / 'settings.json')
        
        with self._hot_reload_lock:
            if self._hot_reload_active:
                return False  # Already active
            
            self._hot_reload_settings_path = settings_path
            self._hot_reload_debounce_time = debounce_time
            self._hot_reload_stop_event.clear()
            
            # Start monitoring thread
            self._hot_reload_thread = threading.Thread(
                target=self._hot_reload_monitor,
                daemon=True
            )
            self._hot_reload_thread.start()
            self._hot_reload_active = True
            
            return True
    
    def stop_hot_reload(self) -> bool:
        """
        Stop hot-reload monitoring.
        
        Returns:
            bool: True if hot-reload stopped successfully, False otherwise
        """
        if not hasattr(self, '_hot_reload_active') or not self._hot_reload_active:
            return False
        
        with self._hot_reload_lock:
            if not self._hot_reload_active:
                return False
            
            # Signal stop
            self._hot_reload_stop_event.set()
            self._hot_reload_active = False
            
            # Wait for thread to finish
            if self._hot_reload_thread and self._hot_reload_thread.is_alive():
                self._hot_reload_thread.join(timeout=2.0)
            
            return True
    
    def add_reload_callback(self, callback: Callable[['TTSConfig'], None]) -> bool:
        """
        Add a callback to be called when configuration is reloaded.
        
        Args:
            callback: Function to call with new configuration
            
        Returns:
            bool: True if callback was added successfully
        """
        self.__init_hot_reload()
        
        if not callable(callback):
            return False
        
        with self._hot_reload_lock:
            if callback not in self._hot_reload_callbacks:
                self._hot_reload_callbacks.append(callback)
                return True
            return False
    
    def remove_reload_callback(self, callback: Callable[['TTSConfig'], None]) -> bool:
        """
        Remove a reload callback.
        
        Args:
            callback: Function to remove
            
        Returns:
            bool: True if callback was removed successfully
        """
        if not hasattr(self, '_hot_reload_callbacks'):
            return False
        
        with self._hot_reload_lock:
            if callback in self._hot_reload_callbacks:
                self._hot_reload_callbacks.remove(callback)
                return True
            return False
    
    def reload_config(self, settings_path: Optional[str] = None) -> bool:
        """
        Manually reload configuration from settings.json file.
        
        Args:
            settings_path: Optional path to settings.json file
            
        Returns:
            bool: True if reload was successful, False otherwise
        """
        try:
            # Load new configuration
            new_config = self.load_from_settings(settings_path)
            
            # Validate new configuration
            if not new_config.is_valid():
                return False
            
            # Apply new configuration to current instance
            with self._hot_reload_lock if hasattr(self, '_hot_reload_lock') else threading.Lock():
                # Update all fields
                for field_name, field_value in new_config.to_dict().items():
                    if hasattr(self, field_name):
                        setattr(self, field_name, field_value)
                
                # Re-run validation
                self.__post_init__()
                
                # Notify callbacks
                if hasattr(self, '_hot_reload_callbacks'):
                    for callback in self._hot_reload_callbacks:
                        try:
                            callback(self)
                        except Exception:
                            # Ignore callback errors to prevent reload failure
                            pass
            
            return True
            
        except Exception:
            return False
    
    def _hot_reload_monitor(self):
        """Internal method to monitor settings file for changes."""
        settings_path = Path(self._hot_reload_settings_path)
        
        # Get initial modification time
        if settings_path.exists():
            self._hot_reload_last_modified = settings_path.stat().st_mtime
        
        while not self._hot_reload_stop_event.is_set():
            try:
                # Check if file exists and get modification time
                if settings_path.exists():
                    current_modified = settings_path.stat().st_mtime
                    
                    # Check if file was modified
                    if current_modified > self._hot_reload_last_modified:
                        # Wait for debounce period
                        if self._hot_reload_stop_event.wait(self._hot_reload_debounce_time):
                            break  # Stop event was set during debounce
                        
                        # Check if file was modified again during debounce
                        if settings_path.exists():
                            final_modified = settings_path.stat().st_mtime
                            if final_modified == current_modified:
                                # File is stable, reload configuration
                                self.reload_config(str(settings_path))
                                self._hot_reload_last_modified = final_modified
                            else:
                                # File changed during debounce, will check again in next loop
                                pass
                
                # Wait before next check (poll every 100ms)
                if self._hot_reload_stop_event.wait(0.1):
                    break
                    
            except (OSError, IOError):
                # Handle file access errors gracefully
                if self._hot_reload_stop_event.wait(1.0):
                    break
    
    def is_hot_reload_active(self) -> bool:
        """
        Check if hot-reload is currently active.
        
        Returns:
            bool: True if hot-reload is active, False otherwise
        """
        return hasattr(self, '_hot_reload_active') and self._hot_reload_active
    
    def get_hot_reload_status(self) -> Dict[str, Any]:
        """
        Get detailed hot-reload status information.
        
        Returns:
            Dict[str, Any]: Status information including active state, settings path, etc.
        """
        if not hasattr(self, '_hot_reload_active'):
            return {
                'active': False,
                'settings_path': None,
                'debounce_time': 0,
                'callback_count': 0,
                'last_modified': 0
            }
        
        return {
            'active': self._hot_reload_active,
            'settings_path': self._hot_reload_settings_path,
            'debounce_time': self._hot_reload_debounce_time,
            'callback_count': len(self._hot_reload_callbacks),
            'last_modified': self._hot_reload_last_modified
        }


# Default configuration instance
DEFAULT_CONFIG = TTSConfig()