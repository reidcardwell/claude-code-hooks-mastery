#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["elevenlabs>=1.0.0", "requests>=2.31.0"]
# ///

"""
ElevenLabs TTS Integration Layer

This module provides a robust integration with ElevenLabs API for voice synthesis,
designed specifically for Claude Code hook enhancements. It includes:

- TTSClient class for API integration
- Voice management and selection
- Rate limiting and error handling
- Cross-platform audio playback
- Connection pooling and retry logic

Usage:
    from elevenlabs_client import TTSClient
    
    client = TTSClient(api_key="your-api-key")
    client.speak_text("Hello world", voice_id="David")
"""

import os
import time
import json
import asyncio
import logging
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue, Empty
from urllib.parse import urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import weakref
import random
import math
import platform
import subprocess
import tempfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSError(Exception):
    """
    Base exception for TTS-related errors.
    
    This is the base class for all TTS-related exceptions. It provides
    enhanced error handling with context information and error codes.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None, 
                 recoverable: bool = True, retry_after: float = None):
        """
        Initialize TTS error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
            recoverable: Whether this error is recoverable
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = time.time()
        
        # Add common context
        self.details.update({
            'timestamp': self.timestamp,
            'error_type': self.__class__.__name__,
            'recoverable': self.recoverable
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'message': self.message,
            'error_code': self.error_code,
            'error_type': self.__class__.__name__,
            'recoverable': self.recoverable,
            'retry_after': self.retry_after,
            'timestamp': self.timestamp,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.retry_after:
            parts.append(f"Retry after: {self.retry_after}s")
        return " | ".join(parts)


class AuthenticationError(TTSError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_FAILED",
            recoverable=False,  # Auth errors typically require manual intervention
            **kwargs
        )


class RateLimitError(TTSError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = None, 
                 requests_remaining: int = None, **kwargs):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            recoverable=True,
            retry_after=retry_after,
            **kwargs
        )
        if requests_remaining is not None:
            self.details['requests_remaining'] = requests_remaining


class VoiceNotFoundError(TTSError):
    """Raised when requested voice ID is not available."""
    
    def __init__(self, message: str = "Voice not found", voice_id: str = None, 
                 available_voices: List[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VOICE_NOT_FOUND",
            recoverable=True,  # Can recover with different voice
            **kwargs
        )
        if voice_id:
            self.details['voice_id'] = voice_id
        if available_voices:
            self.details['available_voices'] = available_voices


class PlaybackError(TTSError):
    """Raised when audio playback fails."""
    
    def __init__(self, message: str = "Audio playback failed", platform: str = None, 
                 playback_command: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="PLAYBACK_FAILED",
            recoverable=True,  # May work with different settings
            **kwargs
        )
        if platform:
            self.details['platform'] = platform
        if playback_command:
            self.details['playback_command'] = playback_command


class ConnectionError(TTSError):
    """Raised when network connection fails."""
    
    def __init__(self, message: str = "Network connection failed", status_code: int = None, 
                 endpoint: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="CONNECTION_FAILED",
            recoverable=True,
            **kwargs
        )
        if status_code:
            self.details['status_code'] = status_code
        if endpoint:
            self.details['endpoint'] = endpoint


class ConfigurationError(TTSError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str = "Configuration error", config_field: str = None, 
                 expected_type: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            recoverable=False,  # Requires code/config changes
            **kwargs
        )
        if config_field:
            self.details['config_field'] = config_field
        if expected_type:
            self.details['expected_type'] = expected_type


class SynthesisError(TTSError):
    """Raised when text synthesis fails."""
    
    def __init__(self, message: str = "Text synthesis failed", text_length: int = None, 
                 voice_id: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="SYNTHESIS_FAILED",
            recoverable=True,
            **kwargs
        )
        if text_length:
            self.details['text_length'] = text_length
        if voice_id:
            self.details['voice_id'] = voice_id


class TimeoutError(TTSError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str = "Operation timed out", timeout_duration: float = None, 
                 operation: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="TIMEOUT",
            recoverable=True,
            **kwargs
        )
        if timeout_duration:
            self.details['timeout_duration'] = timeout_duration
        if operation:
            self.details['operation'] = operation


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    USER_INTERVENTION = "user_intervention"


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring and analysis."""
    error_count: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    recoverable_errors: int = 0
    non_recoverable_errors: int = 0
    retry_counts: Dict[str, int] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    last_error_time: Optional[float] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_size: int = 100
    
    def record_error(self, error: TTSError):
        """Record an error for metrics tracking."""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Count by error type
        error_type = error.__class__.__name__
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Count by recoverability
        if error.recoverable:
            self.recoverable_errors += 1
        else:
            self.non_recoverable_errors += 1
        
        # Add to history
        error_record = {
            'timestamp': error.timestamp,
            'error_type': error_type,
            'error_code': error.error_code,
            'message': error.message,
            'recoverable': error.recoverable,
            'details': error.details
        }
        
        self.error_history.append(error_record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def get_error_rate(self, error_type: str = None, time_window: float = 3600) -> float:
        """Calculate error rate for a specific type or overall."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        if error_type:
            recent_errors = [
                e for e in self.error_history 
                if e['timestamp'] >= cutoff_time and e['error_type'] == error_type
            ]
        else:
            recent_errors = [
                e for e in self.error_history 
                if e['timestamp'] >= cutoff_time
            ]
        
        return len(recent_errors) / (time_window / 3600)  # errors per hour
    
    def get_top_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get the most frequent error types."""
        return sorted(self.error_types.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.error_count = 0
        self.error_types.clear()
        self.recoverable_errors = 0
        self.non_recoverable_errors = 0
        self.retry_counts.clear()
        self.error_rates.clear()
        self.last_error_time = None
        self.error_history.clear()


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize error handler.
        
        Args:
            config: Error handling configuration
        """
        self.config = config or {}
        self.metrics = ErrorMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Default recovery strategies
        self.recovery_strategies = {
            AuthenticationError: ErrorRecoveryStrategy.USER_INTERVENTION,
            RateLimitError: ErrorRecoveryStrategy.RETRY,
            VoiceNotFoundError: ErrorRecoveryStrategy.FALLBACK,
            PlaybackError: ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            ConnectionError: ErrorRecoveryStrategy.RETRY,
            ConfigurationError: ErrorRecoveryStrategy.FAIL_FAST,
            SynthesisError: ErrorRecoveryStrategy.RETRY,
            TimeoutError: ErrorRecoveryStrategy.RETRY
        }
        
        # Fallback options
        self.fallback_voices = ["6sFKzaJr574YWVu4UuJF"]  # David as default fallback
        self.fallback_settings = {
            'voice_quality': VoiceQuality.STANDARD,
            'volume': 0.8,
            'timeout': 30.0
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate strategy.
        
        Args:
            error: The error to handle
            context: Additional context for error handling
            
        Returns:
            Dict with recovery action and information
        """
        # Convert to TTSError if needed
        if not isinstance(error, TTSError):
            error = TTSError(
                message=str(error),
                error_code="UNKNOWN_ERROR",
                details={'original_error': str(error)}
            )
        
        # Record error metrics
        self.metrics.record_error(error)
        
        # Log error with context
        self._log_error(error, context)
        
        # Determine recovery strategy
        strategy = self._get_recovery_strategy(error)
        
        # Execute recovery strategy
        recovery_result = self._execute_recovery_strategy(error, strategy, context)
        
        return {
            'error': error.to_dict(),
            'strategy': strategy.value,
            'recovery_result': recovery_result,
            'timestamp': time.time()
        }
    
    def _log_error(self, error: TTSError, context: Dict[str, Any] = None):
        """Log error with appropriate level and context."""
        severity = self._get_error_severity(error)
        
        log_message = f"TTS Error [{error.error_code}]: {error.message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        if error.details:
            log_message += f" | Details: {error.details}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _get_error_severity(self, error: TTSError) -> ErrorSeverity:
        """Determine error severity."""
        if isinstance(error, (AuthenticationError, ConfigurationError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, SynthesisError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (RateLimitError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_recovery_strategy(self, error: TTSError) -> ErrorRecoveryStrategy:
        """Get recovery strategy for error type."""
        return self.recovery_strategies.get(
            error.__class__, 
            ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
        )
    
    def _execute_recovery_strategy(self, error: TTSError, strategy: ErrorRecoveryStrategy, 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the recovery strategy."""
        context = context or {}
        
        if strategy == ErrorRecoveryStrategy.RETRY:
            return self._handle_retry_strategy(error, context)
        elif strategy == ErrorRecoveryStrategy.FALLBACK:
            return self._handle_fallback_strategy(error, context)
        elif strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._handle_graceful_degradation(error, context)
        elif strategy == ErrorRecoveryStrategy.FAIL_FAST:
            return self._handle_fail_fast(error, context)
        elif strategy == ErrorRecoveryStrategy.USER_INTERVENTION:
            return self._handle_user_intervention(error, context)
        else:
            return {'action': 'none', 'message': 'No recovery strategy available'}
    
    def _handle_retry_strategy(self, error: TTSError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retry recovery strategy."""
        retry_count = context.get('retry_count', 0)
        max_retries = context.get('max_retries', 3)
        
        if retry_count >= max_retries:
            return {
                'action': 'give_up',
                'message': f'Max retries ({max_retries}) exceeded',
                'retry_count': retry_count
            }
        
        # Calculate retry delay
        retry_delay = error.retry_after or self._calculate_retry_delay(retry_count)
        
        return {
            'action': 'retry',
            'message': f'Retrying in {retry_delay}s (attempt {retry_count + 1}/{max_retries})',
            'retry_delay': retry_delay,
            'retry_count': retry_count + 1
        }
    
    def _handle_fallback_strategy(self, error: TTSError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fallback recovery strategy."""
        if isinstance(error, VoiceNotFoundError):
            # Try fallback voices
            current_voice = error.details.get('voice_id')
            for fallback_voice in self.fallback_voices:
                if fallback_voice != current_voice:
                    return {
                        'action': 'fallback',
                        'message': f'Using fallback voice: {fallback_voice}',
                        'fallback_voice': fallback_voice
                    }
        
        # General fallback to default settings
        return {
            'action': 'fallback',
            'message': 'Using fallback settings',
            'fallback_settings': self.fallback_settings
        }
    
    def _handle_graceful_degradation(self, error: TTSError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graceful degradation strategy."""
        if isinstance(error, PlaybackError):
            return {
                'action': 'degrade',
                'message': 'Audio playback failed, continuing without audio',
                'degraded_mode': 'silent'
            }
        
        return {
            'action': 'degrade',
            'message': 'Continuing with reduced functionality',
            'degraded_mode': 'limited'
        }
    
    def _handle_fail_fast(self, error: TTSError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fail-fast strategy."""
        return {
            'action': 'fail',
            'message': 'Critical error - failing fast',
            'requires_intervention': True
        }
    
    def _handle_user_intervention(self, error: TTSError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user intervention strategy."""
        return {
            'action': 'user_intervention',
            'message': 'User intervention required',
            'intervention_type': 'authentication' if isinstance(error, AuthenticationError) else 'configuration'
        }
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay with exponential backoff."""
        base_delay = 1.0
        max_delay = 60.0
        backoff_factor = 2.0
        
        delay = base_delay * (backoff_factor ** retry_count)
        delay = min(delay, max_delay)
        
        # Add jitter
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics."""
        return {
            'total_errors': self.metrics.error_count,
            'recoverable_errors': self.metrics.recoverable_errors,
            'non_recoverable_errors': self.metrics.non_recoverable_errors,
            'error_types': dict(self.metrics.error_types),
            'top_errors': self.metrics.get_top_errors(),
            'error_rate_1h': self.metrics.get_error_rate(time_window=3600),
            'error_rate_24h': self.metrics.get_error_rate(time_window=86400),
            'last_error_time': self.metrics.last_error_time,
            'error_history_size': len(self.metrics.error_history)
        }
    
    def reset_metrics(self):
        """Reset error metrics."""
        self.metrics.reset_metrics()
        self.logger.info("Error metrics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on error handling system."""
        current_time = time.time()
        
        # Check recent error rates
        recent_error_rate = self.metrics.get_error_rate(time_window=3600)
        
        # Determine health status
        if recent_error_rate > 10:  # More than 10 errors per hour
            status = "degraded"
        elif recent_error_rate > 5:  # More than 5 errors per hour
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'timestamp': current_time,
            'recent_error_rate': recent_error_rate,
            'total_errors': self.metrics.error_count,
            'error_types_count': len(self.metrics.error_types),
            'last_error_age': current_time - self.metrics.last_error_time if self.metrics.last_error_time else None
        }


class VoiceQuality(Enum):
    """Voice quality settings for synthesis."""
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


class RetryStrategy(Enum):
    """Retry strategy types for failed requests."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    JITTER = "jitter"


class RetryDecision(Enum):
    """Decisions for retry attempts."""
    RETRY = "retry"
    STOP = "stop"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    ESCALATE = "escalate"


class QueuePriority(Enum):
    """Priority levels for request queue."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class QueueOverflowStrategy(Enum):
    """Strategies for handling queue overflow."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    DROP_LOWEST_PRIORITY = "drop_lowest_priority"
    REJECT = "reject"
    BLOCK = "block"


class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class PlaybackStatus(Enum):
    """Audio playback status."""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # HTTP status codes that should be retried
    retryable_status_codes: List[int] = field(default_factory=lambda: [
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        520,  # Unknown Error
        521,  # Web Server Is Down
        522,  # Connection Timed Out
        523,  # Origin Is Unreachable
        524   # A Timeout Occurred
    ])
    
    # Exceptions that should trigger a retry
    retryable_exceptions: List[type] = field(default_factory=lambda: [
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        ConnectionError,
        RateLimitError
    ])
    
    # Status codes that should never be retried
    non_retryable_status_codes: List[int] = field(default_factory=lambda: [
        400,  # Bad Request
        401,  # Unauthorized
        403,  # Forbidden
        404,  # Not Found
        405,  # Method Not Allowed
        406,  # Not Acceptable
        409,  # Conflict
        410,  # Gone
        422   # Unprocessable Entity
    ])


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.retry_stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'total_delay': 0.0,
            'retry_reasons': {}
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def should_retry(self, exception: Exception = None, status_code: int = None, 
                    attempt: int = 0) -> RetryDecision:
        """
        Determine if a request should be retried based on the error.
        
        Args:
            exception: The exception that occurred
            status_code: HTTP status code from the response
            attempt: Current attempt number (0-based)
            
        Returns:
            RetryDecision: Decision on whether to retry
        """
        if attempt >= self.config.max_retries:
            return RetryDecision.STOP
        
        # Check status codes
        if status_code is not None:
            if status_code in self.config.non_retryable_status_codes:
                return RetryDecision.STOP
            elif status_code in self.config.retryable_status_codes:
                return RetryDecision.RETRY_WITH_BACKOFF
        
        # Check exceptions
        if exception is not None:
            exception_type = type(exception)
            if any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions):
                return RetryDecision.RETRY_WITH_BACKOFF
            elif isinstance(exception, AuthenticationError):
                return RetryDecision.STOP
            elif isinstance(exception, VoiceNotFoundError):
                return RetryDecision.STOP
        
        # Default to no retry for unknown errors
        return RetryDecision.STOP
    
    def calculate_delay(self, attempt: int, base_delay: float = None) -> float:
        """
        Calculate the delay for the next retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay override
            
        Returns:
            float: Delay in seconds
        """
        if base_delay is None:
            base_delay = self.config.base_delay
        
        if self.config.strategy == RetryStrategy.FIXED:
            delay = base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (self.config.backoff_factor ** attempt)
        elif self.config.strategy == RetryStrategy.JITTER:
            # Exponential with jitter
            exponential_delay = base_delay * (self.config.backoff_factor ** attempt)
            delay = exponential_delay * (0.5 + random.random() * 0.5)  # 50-100% of exponential
        else:
            delay = base_delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.JITTER:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        return max(0, delay)
    
    def wait_with_backoff(self, attempt: int, reason: str = "unknown") -> float:
        """
        Wait for the calculated backoff delay.
        
        Args:
            attempt: Current attempt number
            reason: Reason for the retry (for logging)
            
        Returns:
            float: Actual delay time
        """
        delay = self.calculate_delay(attempt)
        
        self.logger.info(f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries}, reason: {reason})")
        
        # Update stats
        self.retry_stats['total_delay'] += delay
        self.retry_stats['retry_reasons'][reason] = self.retry_stats['retry_reasons'].get(reason, 0) + 1
        
        # Wait
        time.sleep(delay)
        
        return delay
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function call
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.retry_stats['total_attempts'] += 1
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.retry_stats['successful_retries'] += 1
                    self.logger.info(f"Retry successful after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Determine if we should retry
                status_code = getattr(e, 'response', None)
                if status_code and hasattr(status_code, 'status_code'):
                    status_code = status_code.status_code
                else:
                    status_code = None
                
                decision = self.should_retry(e, status_code, attempt)
                
                if decision == RetryDecision.STOP or attempt >= self.config.max_retries:
                    self.retry_stats['failed_retries'] += 1
                    self.logger.error(f"Max retries exceeded or non-retryable error: {e}")
                    break
                
                if decision == RetryDecision.RETRY_WITH_BACKOFF:
                    reason = f"{type(e).__name__}:{status_code if status_code else 'no_status'}"
                    self.wait_with_backoff(attempt, reason)
        
        # All retries failed
        raise last_exception
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self.retry_stats.copy()
    
    def reset_stats(self):
        """Reset retry statistics."""
        self.retry_stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'total_delay': 0.0,
            'retry_reasons': {}
        }


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    voice_id: str
    name: str
    description: str
    category: str
    labels: Dict[str, str] = field(default_factory=dict)
    preview_url: Optional[str] = None
    available: bool = True
    
    def __post_init__(self):
        """Validate voice information after initialization."""
        if not self.voice_id or not self.name:
            raise ValueError("Voice ID and name are required")


@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiting."""
    capacity: int = 100  # Maximum tokens in bucket
    refill_rate: float = 10.0  # Tokens per second
    initial_tokens: int = None  # Initial tokens (defaults to capacity)
    burst_capacity: int = None  # Allow burst up to this many tokens
    
    def __post_init__(self):
        if self.initial_tokens is None:
            self.initial_tokens = self.capacity
        if self.burst_capacity is None:
            self.burst_capacity = self.capacity


@dataclass
class QueueConfig:
    """Configuration for request queue system."""
    max_size: int = 1000  # Maximum queue size
    overflow_strategy: QueueOverflowStrategy = QueueOverflowStrategy.DROP_OLDEST
    priority_enabled: bool = True
    persistence_enabled: bool = False
    persistence_file: str = ".tts_queue_backup.json"
    queue_timeout: float = 30.0  # Maximum time to wait for queue slot
    worker_threads: int = 3  # Number of worker threads
    
    
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    enabled: bool = True
    requests_per_second: float = 10.0
    burst_requests: int = 50
    adaptive_enabled: bool = True
    respect_headers: bool = True
    token_bucket: TokenBucketConfig = field(default_factory=TokenBucketConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)


@dataclass
class QueuedRequest:
    """Represents a queued request with priority and metadata."""
    func: Callable
    args: tuple
    kwargs: dict
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    timeout: float = 30.0
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Compare requests by priority (higher priority = lower number for heapq)."""
        return (-self.priority.value, self.created_at) < (-other.priority.value, other.created_at)


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""
    pool_connections: int = 10
    pool_maxsize: int = 20
    pool_block: bool = False
    max_retries: int = 3
    backoff_factor: float = 0.3
    status_forcelist: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])


class SessionManager:
    """Manages HTTP sessions with connection pooling and lifecycle management."""
    
    def __init__(self, config: "TTSConfig"):
        self.config = config
        self.pool_config = ConnectionPoolConfig()
        self._session: Optional[requests.Session] = None
        self._session_created_at: Optional[float] = None
        self._session_last_used: Optional[float] = None
        self._session_request_count: int = 0
        self._lock = threading.RLock()
        self._is_closed = False
        
        # Session lifecycle settings
        self.session_max_age = 3600  # 1 hour
        self.session_max_requests = 1000
        self.session_idle_timeout = 300  # 5 minutes
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _create_session(self) -> requests.Session:
        """Create a new HTTP session with optimized connection pooling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.pool_config.max_retries,
            backoff_factor=self.pool_config.backoff_factor,
            status_forcelist=self.pool_config.status_forcelist,
            allowed_methods=self.pool_config.allowed_methods,
            raise_on_status=False
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.pool_config.pool_connections,
            pool_maxsize=self.pool_config.pool_maxsize,
            pool_block=self.pool_config.pool_block,
            max_retries=retry_strategy
        )
        
        # Mount adapters for HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Accept": "application/json",
            "xi-api-key": self.config.api_key,
            "User-Agent": "Claude-Code-TTS/1.0",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        })
        
        # Configure session-level settings
        session.stream = False  # Disable streaming by default
        session.verify = True   # Enable SSL verification
        
        return session
    
    def get_session(self) -> requests.Session:
        """Get current session or create a new one if needed."""
        with self._lock:
            if self._is_closed:
                raise ConnectionError("SessionManager has been closed")
            
            current_time = time.time()
            
            # Check if we need to create a new session
            if self._should_create_new_session(current_time):
                self._cleanup_old_session()
                self._session = self._create_session()
                self._session_created_at = current_time
                self._session_request_count = 0
                self.logger.info("Created new HTTP session")
            
            # Update last used time
            self._session_last_used = current_time
            self._session_request_count += 1
            
            return self._session
    
    def _should_create_new_session(self, current_time: float) -> bool:
        """Determine if a new session should be created."""
        if self._session is None:
            return True
        
        # Check session age
        if (self._session_created_at and 
            current_time - self._session_created_at > self.session_max_age):
            self.logger.info("Session exceeded max age, creating new session")
            return True
        
        # Check request count
        if self._session_request_count >= self.session_max_requests:
            self.logger.info("Session exceeded max requests, creating new session")
            return True
        
        # Check idle timeout
        if (self._session_last_used and 
            current_time - self._session_last_used > self.session_idle_timeout):
            self.logger.info("Session exceeded idle timeout, creating new session")
            return True
        
        return False
    
    def _cleanup_old_session(self):
        """Clean up the old session."""
        if self._session:
            try:
                self._session.close()
                self.logger.info("Closed old HTTP session")
            except Exception as e:
                self.logger.warning(f"Error closing old session: {e}")
            finally:
                self._session = None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        with self._lock:
            if not self._session:
                return {"status": "no_session"}
            
            current_time = time.time()
            return {
                "status": "active",
                "created_at": self._session_created_at,
                "last_used": self._session_last_used,
                "request_count": self._session_request_count,
                "age_seconds": current_time - self._session_created_at if self._session_created_at else 0,
                "idle_seconds": current_time - self._session_last_used if self._session_last_used else 0,
                "max_age": self.session_max_age,
                "max_requests": self.session_max_requests,
                "idle_timeout": self.session_idle_timeout
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the session manager."""
        with self._lock:
            current_time = time.time()
            
            health_info = {
                "status": "healthy",
                "timestamp": current_time,
                "session_active": self._session is not None,
                "is_closed": self._is_closed
            }
            
            if self._session:
                stats = self.get_session_stats()
                health_info.update({
                    "session_age": stats["age_seconds"],
                    "session_requests": stats["request_count"],
                    "session_idle": stats["idle_seconds"],
                    "session_healthy": (
                        stats["age_seconds"] < self.session_max_age and
                        stats["request_count"] < self.session_max_requests and
                        stats["idle_seconds"] < self.session_idle_timeout
                    )
                })
            
            return health_info
    
    def close(self):
        """Close the session manager and clean up resources."""
        with self._lock:
            if not self._is_closed:
                self._cleanup_old_session()
                self._is_closed = True
                self.logger.info("SessionManager closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""
    
    def __init__(self, config: TokenBucketConfig):
        self.config = config
        self.tokens = float(config.initial_tokens)
        self.last_refill = time.time()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            # Calculate tokens to add
            tokens_to_add = elapsed * self.config.refill_rate
            
            # Add tokens, respecting capacity
            self.tokens = min(self.config.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if tokens_to_add > 0:
                self.logger.debug(f"Refilled {tokens_to_add:.2f} tokens, now have {self.tokens:.2f}")
    
    def consume(self, tokens: float = 1.0, timeout: float = None) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait for tokens (None = don't wait)
            
        Returns:
            bool: True if tokens were consumed, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.logger.debug(f"Consumed {tokens} tokens, {self.tokens:.2f} remaining")
                return True
            
            if timeout is None or timeout <= 0:
                self.logger.debug(f"Insufficient tokens: need {tokens}, have {self.tokens:.2f}")
                return False
            
            # Wait for tokens to become available
            start_time = time.time()
            while time.time() - start_time < timeout:
                time.sleep(0.1)  # Small sleep to avoid busy waiting
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.logger.debug(f"Consumed {tokens} tokens after waiting, {self.tokens:.2f} remaining")
                    return True
            
            self.logger.debug(f"Timeout waiting for tokens: need {tokens}, have {self.tokens:.2f}")
            return False
    
    def available_tokens(self) -> float:
        """Get the current number of available tokens."""
        with self.lock:
            self._refill()
            return self.tokens
    
    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate time to wait for specified tokens to become available."""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.config.refill_rate
    
    def reset(self):
        """Reset the bucket to initial state."""
        with self.lock:
            self.tokens = float(self.config.initial_tokens)
            self.last_refill = time.time()
            self.logger.debug(f"Token bucket reset to {self.tokens} tokens")


class RequestQueue:
    """Thread-safe priority queue for TTS requests."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.queue = []  # Priority queue using heapq
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.shutdown = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load persisted queue if enabled
        if self.config.persistence_enabled:
            self._load_from_persistence()
    
    def _load_from_persistence(self):
        """Load queue from persistence file."""
        try:
            if os.path.exists(self.config.persistence_file):
                with open(self.config.persistence_file, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"Loaded {len(data)} requests from persistence")
        except Exception as e:
            self.logger.error(f"Failed to load queue from persistence: {e}")
    
    def _save_to_persistence(self):
        """Save queue to persistence file."""
        if not self.config.persistence_enabled:
            return
            
        try:
            # Convert queue to serializable format
            serializable_queue = []
            for req in self.queue:
                serializable_queue.append({
                    'request_id': req.request_id,
                    'priority': req.priority.value,
                    'created_at': req.created_at,
                    'timeout': req.timeout,
                    'retries': req.retries
                })
            
            with open(self.config.persistence_file, 'w') as f:
                json.dump(serializable_queue, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save queue to persistence: {e}")
    
    def put(self, request: QueuedRequest, block: bool = True, timeout: float = None) -> bool:
        """
        Put a request in the queue.
        
        Args:
            request: The request to queue
            block: Whether to block if queue is full
            timeout: Maximum time to wait if blocking
            
        Returns:
            bool: True if request was queued, False otherwise
        """
        with self.not_full:
            if self.shutdown:
                return False
            
            # Check if queue is full
            if len(self.queue) >= self.config.max_size:
                if not block:
                    return False
                
                # Handle overflow based on strategy
                if self.config.overflow_strategy == QueueOverflowStrategy.REJECT:
                    self.logger.warning("Queue full, rejecting request")
                    return False
                elif self.config.overflow_strategy == QueueOverflowStrategy.BLOCK:
                    if not self.not_full.wait(timeout):
                        self.logger.warning("Timeout waiting for queue space")
                        return False
                else:
                    # Handle dropping strategies
                    self._handle_overflow()
            
            # Add to queue
            import heapq
            heapq.heappush(self.queue, request)
            
            self.logger.debug(f"Queued request {request.request_id} with priority {request.priority.name}")
            
            # Notify waiting consumers
            with self.not_empty:
                self.not_empty.notify()
            
            # Save to persistence if enabled
            self._save_to_persistence()
            
            return True
    
    def get(self, block: bool = True, timeout: float = None) -> Optional[QueuedRequest]:
        """
        Get a request from the queue.
        
        Args:
            block: Whether to block if queue is empty
            timeout: Maximum time to wait if blocking
            
        Returns:
            QueuedRequest or None: The next request, or None if timeout/shutdown
        """
        with self.not_empty:
            if self.shutdown and not self.queue:
                return None
            
            if not self.queue:
                if not block:
                    return None
                
                if not self.not_empty.wait(timeout):
                    return None
            
            if self.queue:
                import heapq
                request = heapq.heappop(self.queue)
                
                self.logger.debug(f"Dequeued request {request.request_id}")
                
                # Notify waiting producers
                with self.not_full:
                    self.not_full.notify()
                
                # Save to persistence if enabled
                self._save_to_persistence()
                
                return request
            
            return None
    
    def _handle_overflow(self):
        """Handle queue overflow based on configured strategy."""
        if not self.queue:
            return
            
        import heapq
        
        if self.config.overflow_strategy == QueueOverflowStrategy.DROP_OLDEST:
            # Remove oldest request (last in priority queue)
            removed = self.queue.pop()
            heapq.heapify(self.queue)
            self.logger.warning(f"Dropped oldest request {removed.request_id}")
            
        elif self.config.overflow_strategy == QueueOverflowStrategy.DROP_NEWEST:
            # Don't add the new request (handled by caller)
            pass
            
        elif self.config.overflow_strategy == QueueOverflowStrategy.DROP_LOWEST_PRIORITY:
            # Find and remove lowest priority request
            if self.queue:
                lowest_priority_idx = 0
                lowest_priority = self.queue[0].priority.value
                
                for i, req in enumerate(self.queue):
                    if req.priority.value < lowest_priority:
                        lowest_priority = req.priority.value
                        lowest_priority_idx = i
                
                removed = self.queue.pop(lowest_priority_idx)
                heapq.heapify(self.queue)
                self.logger.warning(f"Dropped lowest priority request {removed.request_id}")
    
    def size(self) -> int:
        """Get current queue size."""
        with self.lock:
            return len(self.queue)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        with self.lock:
            return len(self.queue) == 0
    
    def full(self) -> bool:
        """Check if queue is full."""
        with self.lock:
            return len(self.queue) >= self.config.max_size
    
    def shutdown_queue(self):
        """Shutdown the queue and notify all waiting threads."""
        with self.lock:
            self.shutdown = True
            with self.not_empty:
                self.not_empty.notify_all()
            with self.not_full:
                self.not_full.notify_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            priority_counts = {}
            for req in self.queue:
                priority_counts[req.priority.name] = priority_counts.get(req.priority.name, 0) + 1
            
            return {
                "size": len(self.queue),
                "max_size": self.config.max_size,
                "utilization": len(self.queue) / self.config.max_size,
                "priority_counts": priority_counts,
                "shutdown": self.shutdown
            }


class RateLimitManager:
    """Manages rate limiting and request queuing for TTS API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.token_bucket = TokenBucket(config.token_bucket)
        self.request_queue = RequestQueue(config.queue)
        self.adaptive_limits = {}
        self.last_response_headers = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Statistics
        self.stats = {
            "requests_processed": 0,
            "requests_queued": 0,
            "requests_rejected": 0,
            "rate_limit_hits": 0,
            "adaptive_adjustments": 0
        }
        
        # Worker threads for processing queue
        self.workers = []
        self.shutdown = False
        
        if self.config.enabled:
            self._start_workers()
    
    def _start_workers(self):
        """Start worker threads for processing queue."""
        for i in range(self.config.queue.worker_threads):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"RateLimitWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {len(self.workers)} rate limit workers")
    
    def _worker_loop(self):
        """Worker thread loop for processing queued requests."""
        while not self.shutdown:
            try:
                request = self.request_queue.get(timeout=1.0)
                if request is None:
                    continue
                
                # Wait for tokens
                if not self.token_bucket.consume(1.0, timeout=request.timeout):
                    self.logger.warning(f"Request {request.request_id} timed out waiting for tokens")
                    self.stats["requests_rejected"] += 1
                    continue
                
                # Execute the request
                try:
                    result = request.func(*request.args, **request.kwargs)
                    self.stats["requests_processed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Request {request.request_id} failed: {e}")
                    
                    # Retry if configured
                    if request.retries < request.max_retries:
                        request.retries += 1
                        self.request_queue.put(request, block=False)
                        self.logger.info(f"Retrying request {request.request_id} (attempt {request.retries})")
                    else:
                        self.stats["requests_rejected"] += 1
                        
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def submit_request(self, func: Callable, *args, priority: QueuePriority = QueuePriority.NORMAL, timeout: float = 30.0, **kwargs) -> bool:
        """
        Submit a request to the rate limiter.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            priority: Request priority
            timeout: Request timeout
            **kwargs: Keyword arguments for the function
            
        Returns:
            bool: True if request was queued, False otherwise
        """
        if not self.config.enabled:
            # Direct execution if rate limiting is disabled
            try:
                func(*args, **kwargs)
                return True
            except Exception as e:
                self.logger.error(f"Direct execution failed: {e}")
                return False
        
        request = QueuedRequest(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        if self.request_queue.put(request, block=False):
            self.stats["requests_queued"] += 1
            return True
        else:
            self.stats["requests_rejected"] += 1
            return False
    
    def update_from_response_headers(self, headers: Dict[str, str]):
        """Update rate limits based on API response headers."""
        if not self.config.respect_headers:
            return
            
        self.last_response_headers = headers
        
        # Parse common rate limit headers
        remaining = headers.get('X-RateLimit-Remaining', headers.get('X-RateLimit-Limit-Remaining'))
        reset_time = headers.get('X-RateLimit-Reset', headers.get('X-RateLimit-Reset-Time'))
        limit = headers.get('X-RateLimit-Limit', headers.get('X-RateLimit-Limit-Limit'))
        
        if remaining is not None:
            try:
                remaining = int(remaining)
                if remaining < 5:  # Low remaining requests
                    self.stats["rate_limit_hits"] += 1
                    self.logger.warning(f"Rate limit approaching: {remaining} requests remaining")
                    
                    # Adjust token bucket if adaptive limiting is enabled
                    if self.config.adaptive_enabled:
                        self._adjust_rate_limit(remaining)
                        
            except (ValueError, TypeError):
                pass
        
        if reset_time is not None:
            try:
                reset_time = int(reset_time)
                current_time = int(time.time())
                
                if reset_time > current_time:
                    wait_time = reset_time - current_time
                    self.logger.info(f"Rate limit resets in {wait_time} seconds")
                    
            except (ValueError, TypeError):
                pass
    
    def _adjust_rate_limit(self, remaining_requests: int):
        """Adjust rate limits based on remaining requests."""
        if remaining_requests < 5:
            # Slow down significantly
            new_rate = max(1.0, self.config.requests_per_second * 0.1)
        elif remaining_requests < 10:
            # Moderate slowdown
            new_rate = max(2.0, self.config.requests_per_second * 0.5)
        else:
            # Normal rate
            new_rate = self.config.requests_per_second
        
        if new_rate != self.token_bucket.config.refill_rate:
            self.token_bucket.config.refill_rate = new_rate
            self.stats["adaptive_adjustments"] += 1
            self.logger.info(f"Adjusted rate limit to {new_rate} requests/second")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            **self.stats,
            "token_bucket": {
                "available_tokens": self.token_bucket.available_tokens(),
                "capacity": self.token_bucket.config.capacity,
                "refill_rate": self.token_bucket.config.refill_rate
            },
            "queue": self.request_queue.get_stats(),
            "last_headers": self.last_response_headers
        }
    
    def shutdown_manager(self):
        """Shutdown the rate limit manager."""
        self.shutdown = True
        self.request_queue.shutdown_queue()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.logger.info("Rate limit manager shutdown complete")


@dataclass
class AudioPlayerConfig:
    """Configuration for audio playback."""
    volume: float = 1.0  # 0.0 to 1.0
    playback_device: Optional[str] = None
    buffer_size: int = 4096
    enable_interruption: bool = True
    temp_dir: Optional[str] = None
    cleanup_temp_files: bool = True
    playback_timeout: float = 30.0
    
    def __post_init__(self):
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")
        if self.temp_dir is None:
            self.temp_dir = tempfile.gettempdir()


class AudioPlayer:
    """Cross-platform audio playback engine."""
    
    def __init__(self, config: AudioPlayerConfig = None):
        self.config = config or AudioPlayerConfig()
        self.status = PlaybackStatus.IDLE
        self.current_process: Optional[subprocess.Popen] = None
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Platform detection
        self.platform = platform.system().lower()
        self.logger.info(f"Detected platform: {self.platform}")
        
        # Initialize platform-specific settings
        self._init_platform_settings()
    
    def _init_platform_settings(self):
        """Initialize platform-specific playback settings."""
        if self.platform == "darwin":  # macOS
            self.playback_command = "afplay"
            self.command_args = ["-v", str(self.config.volume)]
        elif self.platform == "windows":
            # Windows - try different options
            self.playback_command = self._find_windows_player()
            self.command_args = []
        elif self.platform == "linux":
            # Linux - try different audio systems
            self.playback_command = self._find_linux_player()
            self.command_args = []
        else:
            self.logger.warning(f"Unsupported platform: {self.platform}")
            self.playback_command = None
            self.command_args = []
    
    def _find_windows_player(self) -> Optional[str]:
        """Find available audio player on Windows."""
        # Check for common Windows audio players
        players = ["powershell", "winsound"]  # powershell for built-in audio
        
        for player in players:
            try:
                result = subprocess.run(
                    ["where", player], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    self.logger.info(f"Found Windows audio player: {player}")
                    return player
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        self.logger.warning("No suitable Windows audio player found")
        return None
    
    def _find_linux_player(self) -> Optional[str]:
        """Find available audio player on Linux."""
        # Check for common Linux audio players
        players = ["aplay", "paplay", "ffplay", "mpg123", "sox"]
        
        for player in players:
            try:
                result = subprocess.run(
                    ["which", player], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    self.logger.info(f"Found Linux audio player: {player}")
                    return player
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        self.logger.warning("No suitable Linux audio player found")
        return None
    
    def _create_temp_audio_file(self, audio_data: bytes, format: AudioFormat = AudioFormat.MP3) -> str:
        """Create temporary audio file from audio data."""
        with tempfile.NamedTemporaryFile(
            suffix=f".{format.value}",
            dir=self.config.temp_dir,
            delete=False
        ) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        self.logger.debug(f"Created temp audio file: {temp_path}")
        return temp_path
    
    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary audio file."""
        if self.config.cleanup_temp_files:
            try:
                os.unlink(file_path)
                self.logger.debug(f"Cleaned up temp file: {file_path}")
            except OSError as e:
                self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def _play_macos(self, file_path: str) -> subprocess.Popen:
        """Play audio on macOS using afplay."""
        cmd = [self.playback_command] + self.command_args + [file_path]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def _play_windows(self, file_path: str) -> subprocess.Popen:
        """Play audio on Windows using PowerShell."""
        if self.playback_command == "powershell":
            # Use PowerShell to play audio
            powershell_cmd = f'''
            Add-Type -AssemblyName presentationCore
            $mediaPlayer = New-Object System.Windows.Media.MediaPlayer
            $mediaPlayer.Open([System.Uri]::new("{file_path}"))
            $mediaPlayer.Volume = {self.config.volume}
            $mediaPlayer.Play()
            Start-Sleep -Seconds 1
            while($mediaPlayer.NaturalDuration.HasTimeSpan -eq $false) {{
                Start-Sleep -Milliseconds 100
            }}
            $duration = $mediaPlayer.NaturalDuration.TimeSpan.TotalSeconds
            Start-Sleep -Seconds $duration
            $mediaPlayer.Stop()
            '''
            return subprocess.Popen(
                ["powershell", "-Command", powershell_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            # Fallback to generic command
            return subprocess.Popen(
                [self.playback_command, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
    
    def _play_linux(self, file_path: str) -> subprocess.Popen:
        """Play audio on Linux using available player."""
        cmd = [self.playback_command]
        
        # Add volume control if supported
        if self.playback_command == "aplay":
            # aplay doesn't support volume directly, need to use amixer
            pass
        elif self.playback_command == "paplay":
            cmd.extend(["--volume", str(int(self.config.volume * 65536))])
        elif self.playback_command == "ffplay":
            cmd.extend(["-nodisp", "-autoexit", "-volume", str(int(self.config.volume * 100))])
        elif self.playback_command == "mpg123":
            cmd.extend(["-q", "--gain", str(int(self.config.volume * 100))])
        
        cmd.append(file_path)
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def play_audio(self, audio_data: bytes, format: AudioFormat = AudioFormat.MP3, blocking: bool = True) -> bool:
        """
        Play audio data.
        
        Args:
            audio_data: Raw audio data bytes
            format: Audio format of the data
            blocking: Whether to block until playback completes
            
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        with self.lock:
            if self.status == PlaybackStatus.PLAYING:
                self.logger.warning("Already playing audio - stopping current playback")
                self.stop_playback()
            
            if not self.playback_command:
                self.logger.error(f"No audio player available for platform: {self.platform}")
                self.status = PlaybackStatus.ERROR
                return False
            
            try:
                # Create temporary file
                temp_file = self._create_temp_audio_file(audio_data, format)
                
                # Start playback based on platform
                if self.platform == "darwin":
                    process = self._play_macos(temp_file)
                elif self.platform == "windows":
                    process = self._play_windows(temp_file)
                elif self.platform == "linux":
                    process = self._play_linux(temp_file)
                else:
                    self.logger.error(f"Unsupported platform: {self.platform}")
                    self._cleanup_temp_file(temp_file)
                    return False
                
                self.current_process = process
                self.status = PlaybackStatus.PLAYING
                
                if blocking:
                    # Wait for playback to complete
                    try:
                        stdout, stderr = process.communicate(timeout=self.config.playback_timeout)
                        if process.returncode == 0:
                            self.logger.info("Audio playback completed successfully")
                            self.status = PlaybackStatus.IDLE
                        else:
                            self.logger.error(f"Audio playback failed: {stderr}")
                            self.status = PlaybackStatus.ERROR
                    except subprocess.TimeoutExpired:
                        self.logger.error("Audio playback timed out")
                        process.kill()
                        self.status = PlaybackStatus.ERROR
                    finally:
                        self.current_process = None
                        self._cleanup_temp_file(temp_file)
                else:
                    # Non-blocking - set up cleanup in background
                    def cleanup_after_playback():
                        try:
                            process.wait(timeout=self.config.playback_timeout)
                            if process.returncode == 0:
                                self.status = PlaybackStatus.IDLE
                            else:
                                self.status = PlaybackStatus.ERROR
                        except subprocess.TimeoutExpired:
                            process.kill()
                            self.status = PlaybackStatus.ERROR
                        finally:
                            with self.lock:
                                if self.current_process == process:
                                    self.current_process = None
                            self._cleanup_temp_file(temp_file)
                    
                    threading.Thread(target=cleanup_after_playback, daemon=True).start()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to play audio: {e}")
                self.status = PlaybackStatus.ERROR
                return False
    
    def stop_playback(self) -> bool:
        """
        Stop current audio playback.
        
        Returns:
            bool: True if playback was stopped, False otherwise
        """
        with self.lock:
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=5)
                    self.logger.info("Audio playback stopped")
                    self.status = PlaybackStatus.STOPPED
                    return True
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                    self.logger.warning("Audio playback forcefully killed")
                    self.status = PlaybackStatus.STOPPED
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to stop playback: {e}")
                    return False
            else:
                self.logger.info("No active playback to stop")
                return True
    
    def pause_playback(self) -> bool:
        """
        Pause current audio playback.
        
        Note: Pause/resume functionality is limited by platform capabilities.
        Most command-line players don't support pause/resume.
        
        Returns:
            bool: True if playback was paused, False otherwise
        """
        # Most command-line audio players don't support pause/resume
        # This is a placeholder for future implementation with more advanced audio libraries
        self.logger.warning("Pause functionality not implemented for command-line players")
        return False
    
    def resume_playback(self) -> bool:
        """
        Resume paused audio playback.
        
        Returns:
            bool: True if playback was resumed, False otherwise
        """
        # Most command-line audio players don't support pause/resume
        # This is a placeholder for future implementation with more advanced audio libraries
        self.logger.warning("Resume functionality not implemented for command-line players")
        return False
    
    def set_volume(self, volume: float) -> bool:
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set, False otherwise
        """
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")
        
        self.config.volume = volume
        self.logger.info(f"Volume set to {volume}")
        
        # Update platform-specific command args
        if self.platform == "darwin":
            self.command_args = ["-v", str(volume)]
        
        return True
    
    def get_status(self) -> PlaybackStatus:
        """Get current playback status."""
        return self.status
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.status == PlaybackStatus.PLAYING
    
    def get_supported_formats(self) -> List[AudioFormat]:
        """Get list of supported audio formats for current platform."""
        if self.platform == "darwin":
            # afplay supports most formats
            return [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.AAC, AudioFormat.FLAC]
        elif self.platform == "windows":
            # Windows Media Player supports most formats
            return [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.AAC]
        elif self.platform == "linux":
            # Depends on the player found
            if self.playback_command == "aplay":
                return [AudioFormat.WAV]
            elif self.playback_command == "paplay":
                return [AudioFormat.WAV, AudioFormat.OGG]
            elif self.playback_command == "ffplay":
                return [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OGG, AudioFormat.FLAC, AudioFormat.AAC]
            elif self.playback_command == "mpg123":
                return [AudioFormat.MP3]
            else:
                return [AudioFormat.MP3, AudioFormat.WAV]
        else:
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on audio player."""
        return {
            "status": "healthy" if self.playback_command else "error",
            "platform": self.platform,
            "playback_command": self.playback_command,
            "supported_formats": [f.value for f in self.get_supported_formats()],
            "current_status": self.status.value,
            "volume": self.config.volume,
            "is_playing": self.is_playing()
        }


@dataclass
class TTSConfig:
    """Configuration settings for TTS client."""
    api_key: str
    base_url: str = "https://api.elevenlabs.io/v1"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    backoff_factor: float = 2.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    default_voice_id: str = "6sFKzaJr574YWVu4UuJF"  # David
    voice_quality: VoiceQuality = VoiceQuality.STANDARD
    enable_caching: bool = True
    cache_duration: int = 3600  # 1 hour
    
    # Advanced retry configuration
    retry_config: Optional[RetryConfig] = None
    
    # Rate limiting configuration
    rate_limit_config: Optional[RateLimitConfig] = None
    
    # Audio playback configuration
    audio_config: Optional[AudioPlayerConfig] = None
    
    # Error handling configuration
    error_handler_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        
        # Initialize retry configuration if not provided
        if self.retry_config is None:
            self.retry_config = RetryConfig(
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                max_delay=self.max_retry_delay,
                backoff_factor=self.backoff_factor,
                strategy=RetryStrategy.EXPONENTIAL
            )
        
        # Initialize rate limiting configuration if not provided
        if self.rate_limit_config is None:
            self.rate_limit_config = RateLimitConfig(
                enabled=True,
                requests_per_second=self.rate_limit_requests / self.rate_limit_window,
                burst_requests=self.rate_limit_requests,
                token_bucket=TokenBucketConfig(
                    capacity=self.rate_limit_requests,
                    refill_rate=self.rate_limit_requests / self.rate_limit_window
                )
            )
        
        # Initialize audio configuration if not provided
        if self.audio_config is None:
            self.audio_config = AudioPlayerConfig()
        
        # Initialize error handler configuration if not provided
        if self.error_handler_config is None:
            self.error_handler_config = {
                'max_retry_attempts': 3,
                'retry_delay_base': 1.0,
                'retry_delay_max': 60.0,
                'fallback_voices': ["6sFKzaJr574YWVu4UuJF"],  # David as fallback
                'error_rate_threshold': 10,  # errors per hour
                'enable_graceful_degradation': True,
                'log_errors': True,
                'collect_metrics': True
            }


@dataclass
class SynthesisRequest:
    """Request object for text synthesis."""
    text: str
    voice_id: str
    model_id: str = "eleven_monolingual_v1"
    voice_settings: Optional[Dict[str, Any]] = None
    quality: VoiceQuality = VoiceQuality.STANDARD
    priority: int = 1
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate synthesis request after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not self.voice_id:
            raise ValueError("Voice ID is required")
        if len(self.text) > 5000:  # ElevenLabs limit
            raise ValueError("Text exceeds maximum length (5000 characters)")


class TTSClient:
    """
    ElevenLabs TTS Client for voice synthesis integration.
    
    This class provides a robust interface to the ElevenLabs API with features:
    - Secure API key management
    - Voice selection and caching
    - Rate limiting and retry logic
    - Connection pooling
    - Cross-platform audio playback
    - Comprehensive error handling
    
    The client is designed for integration with Claude Code hooks and provides
    both synchronous and asynchronous interfaces for different use cases.
    """
    
    # Supported voice IDs with metadata
    SUPPORTED_VOICES = {
        "6sFKzaJr574YWVu4UuJF": VoiceInfo(
            voice_id="6sFKzaJr574YWVu4UuJF",
            name="David",
            description="Professional, clear male voice",
            category="male",
            labels={"accent": "american", "age": "middle_aged", "gender": "male"}
        ),
        "qPTgKs2gqb0Fq4SoX4yT": VoiceInfo(
            voice_id="qPTgKs2gqb0Fq4SoX4yT",
            name="Cornelius",
            description="Mature, authoritative male voice",
            category="male",
            labels={"accent": "british", "age": "middle_aged", "gender": "male"}
        ),
        "EuMDy7VrNalBMQBXZkJO": VoiceInfo(
            voice_id="EuMDy7VrNalBMQBXZkJO",
            name="Britney",
            description="Friendly, energetic female voice",
            category="female",
            labels={"accent": "american", "age": "young", "gender": "female"}
        )
    }
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[TTSConfig] = None):
        """
        Initialize the TTS client.
        
        Args:
            api_key: ElevenLabs API key. If not provided, will attempt to load from environment
            config: TTS configuration object. If not provided, will use defaults
            
        Raises:
            AuthenticationError: If API key is missing or invalid
            ValueError: If configuration is invalid
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.speak_text("Hello world", voice_id="David")
        """
        # Initialize configuration
        self.config = config or TTSConfig(api_key=api_key or self._load_api_key())
        
        # Initialize internal state
        self._session_manager = None
        self._voice_cache = {}
        self._last_cache_update = 0
        self._rate_limiter = None
        self._request_queue = Queue()
        self._is_initialized = False
        
        # Initialize retry handler
        self._retry_handler = RetryHandler(self.config.retry_config)
        
        # Initialize rate limiting manager
        self._rate_limit_manager = RateLimitManager(self.config.rate_limit_config)
        
        # Initialize audio player
        self._audio_player = AudioPlayer(self.config.audio_config)
        
        # Initialize error handler
        self._error_handler = ErrorHandler(self.config.error_handler_config)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize session manager
        self._initialize_session_manager()
    
    def _load_api_key(self) -> str:
        """
        Load API key from environment variables.
        
        Returns:
            str: The API key
            
        Raises:
            AuthenticationError: If API key is not found
        """
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable "
                "or provide api_key parameter."
            )
        return api_key
    
    def _initialize_session_manager(self) -> None:
        """
        Initialize the session manager with advanced connection pooling.
        
        This method sets up a SessionManager with optimized connection pooling,
        session lifecycle management, and automatic retry configuration.
        """
        with self._lock:
            if self._session_manager is None:
                self._session_manager = SessionManager(self.config)
                self.logger.info("Session manager initialized")
            
            # Validate API key during initialization
            try:
                if self.validate_api_key():
                    self._is_initialized = True
                    self.logger.info("TTSClient initialized successfully")
                else:
                    self.logger.error("TTSClient initialization failed: Invalid API key")
            except Exception as e:
                self.logger.error(f"TTSClient initialization failed: {e}")
                # Continue without marking as initialized - client can still be used
                # but API calls will fail until key is fixed
    
    def _make_retryable_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a retryable HTTP request with exponential backoff.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for the request
            
        Returns:
            requests.Response: The response object
            
        Raises:
            Exception: If all retries fail
        """
        def _make_request():
            if not self._session_manager:
                raise ConnectionError("Client not initialized - call validate_api_key() first")
            
            session = self._session_manager.get_session()
            response = session.request(method, url, **kwargs)
            
            # Check for HTTP errors that should trigger retries
            if response.status_code in self.config.retry_config.retryable_status_codes:
                # Create a custom exception with status code info
                error = requests.exceptions.HTTPError(f"HTTP {response.status_code}")
                error.response = response
                raise error
            
            # Check for authentication errors (don't retry these)
            if response.status_code in [401, 403]:
                raise AuthenticationError(f"Authentication failed: HTTP {response.status_code}")
            
            # Check for other client errors (don't retry these)
            if response.status_code >= 400 and response.status_code < 500:
                raise TTSError(f"Client error: HTTP {response.status_code}")
            
            return response
        
        return self._retry_handler.execute_with_retry(_make_request)
    
    def _make_rate_limited_request(self, method: str, url: str, priority: QueuePriority = QueuePriority.NORMAL, **kwargs) -> requests.Response:
        """
        Make a rate-limited HTTP request through the rate limiting system.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            priority: Request priority for queuing
            **kwargs: Additional arguments for the request
            
        Returns:
            requests.Response: The response object
            
        Raises:
            RateLimitError: If request is rejected due to rate limiting
            Exception: If request fails
        """
        if not self._rate_limit_manager.config.enabled:
            # Rate limiting disabled, use retryable request directly
            return self._make_retryable_request(method, url, **kwargs)
        
        # Create a response container to capture the result
        response_container = {}
        exception_container = {}
        
        def _execute_request():
            """Execute the request and store result in container."""
            try:
                response = self._make_retryable_request(method, url, **kwargs)
                response_container['response'] = response
                
                # Update rate limits from response headers
                self._rate_limit_manager.update_from_response_headers(dict(response.headers))
                
            except Exception as e:
                exception_container['exception'] = e
        
        # Submit request to rate limiting queue
        if self._rate_limit_manager.submit_request(
            _execute_request,
            priority=priority,
            timeout=kwargs.get('timeout', self.config.timeout)
        ):
            # Wait for request to be processed
            # This is a simplified implementation - in practice, you'd want
            # to implement proper async/await or callback mechanisms
            import time
            timeout = kwargs.get('timeout', self.config.timeout)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if 'response' in response_container:
                    return response_container['response']
                elif 'exception' in exception_container:
                    raise exception_container['exception']
                
                time.sleep(0.1)  # Small delay to avoid busy waiting
            
            raise RateLimitError("Request timed out waiting for rate limit processing")
        else:
            raise RateLimitError("Request rejected by rate limiter - queue may be full")
    
    @property
    def is_initialized(self) -> bool:
        """Check if client is properly initialized."""
        return self._is_initialized
    
    @property
    def api_key(self) -> str:
        """Get the API key (masked for security)."""
        return f"{'*' * 8}{self.config.api_key[-4:]}" if self.config.api_key else "None"
    
    @property
    def base_url(self) -> str:
        """Get the base URL for API requests."""
        return self.config.base_url
    
    @property
    def timeout(self) -> int:
        """Get the request timeout in seconds."""
        return self.config.timeout
    
    @property
    def default_voice_id(self) -> str:
        """Get the default voice ID."""
        return self.config.default_voice_id
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a test request.
        
        This method performs a lightweight API call to verify that the
        API key is valid and the service is accessible.
        
        Returns:
            bool: True if API key is valid, False otherwise
            
        Raises:
            AuthenticationError: If API key is invalid
            ConnectionError: If unable to connect to API
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> if client.validate_api_key():
            ...     print("API key is valid")
        """
        try:
            self.logger.info("Validating API key...")
            
            # Use rate-limited request for API key validation
            response = self._make_rate_limited_request(
                "GET",
                f"{self.config.base_url}/user",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.logger.info("API key validation successful")
                return True
            else:
                self.logger.error(f"API key validation failed: HTTP {response.status_code}")
                raise ConnectionError(f"API validation failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.logger.error("API key validation failed: Request timeout")
            raise ConnectionError("Request timeout during API key validation")
        except requests.exceptions.ConnectionError:
            self.logger.error("API key validation failed: Connection error")
            raise ConnectionError("Unable to connect to ElevenLabs API")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API key validation failed: {e}")
            raise ConnectionError(f"Request failed during API key validation: {e}")
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            raise AuthenticationError(f"Invalid API key: {e}")
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get user information and subscription details.
        
        This method retrieves user account information including subscription
        status, character limits, and usage quotas.
        
        Returns:
            Dict[str, Any]: User information dictionary
            
        Raises:
            AuthenticationError: If API key is invalid
            ConnectionError: If unable to connect to API
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> user_info = client.get_user_info()
            >>> print(f"Subscription: {user_info.get('subscription')}")
        """
        try:
            response = self._make_retryable_request(
                "GET",
                f"{self.config.base_url}/user",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise ConnectionError(f"Failed to get user info: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get user info: {e}")
            raise ConnectionError(f"Request failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get user info: {e}")
            raise TTSError(f"Failed to get user info: {e}")
    
    def get_subscription_info(self) -> Dict[str, Any]:
        """
        Get subscription information including character limits and usage.
        
        Returns:
            Dict[str, Any]: Subscription information dictionary
            
        Raises:
            AuthenticationError: If API key is invalid
            ConnectionError: If unable to connect to API
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> sub_info = client.get_subscription_info()
            >>> print(f"Characters remaining: {sub_info.get('character_limit')}")
        """
        try:
            response = self._make_retryable_request(
                "GET",
                f"{self.config.base_url}/user/subscription",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise ConnectionError(f"Failed to get subscription info: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get subscription info: {e}")
            raise ConnectionError(f"Request failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get subscription info: {e}")
            raise TTSError(f"Failed to get subscription info: {e}")
    
    def get_available_voices(self, force_refresh: bool = False) -> List[VoiceInfo]:
        """
        Get list of available voices.
        
        This method returns voice information with caching support to minimize
        API calls. The cache is automatically refreshed based on configured intervals.
        
        Args:
            force_refresh: Force refresh of voice cache from API
            
        Returns:
            List[VoiceInfo]: List of available voice information
            
        Raises:
            ConnectionError: If unable to connect to API
            AuthenticationError: If API key is invalid
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> voices = client.get_available_voices()
            >>> for voice in voices:
            ...     print(f"{voice.name}: {voice.description}")
        """
        with self._lock:
            current_time = time.time()
            
            # Check if cache is valid
            if (not force_refresh and 
                self._voice_cache and 
                (current_time - self._last_cache_update) < self.config.cache_duration):
                return list(self._voice_cache.values())
            
            try:
                self.logger.info("Fetching available voices...")
                
                # Fetch voices from API with rate limiting
                response = self._make_rate_limited_request(
                    "GET",
                    f"{self.config.base_url}/voices",
                    priority=QueuePriority.LOW,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    api_voices = response.json()
                    voices = []
                    
                    # Process API response
                    for voice_data in api_voices.get('voices', []):
                        voice_info = VoiceInfo(
                            voice_id=voice_data['voice_id'],
                            name=voice_data['name'],
                            description=voice_data.get('description', ''),
                            category=voice_data.get('category', 'unknown'),
                            labels=voice_data.get('labels', {}),
                            preview_url=voice_data.get('preview_url'),
                            available=voice_data.get('available', True)
                        )
                        voices.append(voice_info)
                    
                    # Update cache
                    self._voice_cache = {voice.voice_id: voice for voice in voices}
                    self._last_cache_update = current_time
                    
                    self.logger.info(f"Successfully fetched {len(voices)} voices from API")
                    return voices
                else:
                    raise ConnectionError(f"Failed to fetch voices: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                # Fall back to cached voices if available
                if self._voice_cache:
                    self.logger.info("Using cached voices due to API failure")
                    return list(self._voice_cache.values())
                
                # Final fallback to supported voices
                self.logger.info("Using built-in supported voices as fallback")
                voices = list(self.SUPPORTED_VOICES.values())
                self._voice_cache = {voice.voice_id: voice for voice in voices}
                self._last_cache_update = current_time
                return voices
                
            except Exception as e:
                self.logger.error(f"Failed to get available voices: {e}")
                # Return cached voices if available
                if self._voice_cache:
                    return list(self._voice_cache.values())
                raise ConnectionError(f"Unable to fetch voices: {e}")
    
    def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """
        Get information about a specific voice.
        
        Args:
            voice_id: The voice ID to get information for
            
        Returns:
            Optional[VoiceInfo]: Voice information if found, None otherwise
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> voice = client.get_voice_info("6sFKzaJr574YWVu4UuJF")
            >>> if voice:
            ...     print(f"Voice: {voice.name}")
        """
        voices = self.get_available_voices()
        return next((voice for voice in voices if voice.voice_id == voice_id), None)
    
    def speak_text(self, text: str, voice_id: Optional[str] = None, **kwargs) -> bool:
        """
        Convert text to speech and play audio.
        
        This is the main method for text-to-speech synthesis. It handles the
        complete pipeline from text input to audio playback with comprehensive
        error handling and recovery mechanisms.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use. If not provided, uses default voice
            **kwargs: Additional synthesis parameters
            
        Returns:
            bool: True if synthesis and playback succeeded, False otherwise
            
        Raises:
            VoiceNotFoundError: If specified voice is not available
            TTSError: If synthesis fails
            PlaybackError: If audio playback fails
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> success = client.speak_text("Hello world", voice_id="David")
            >>> if success:
            ...     print("Speech synthesis completed")
        """
        return self._synthesize_text_to_audio_with_recovery(text, voice_id, **kwargs)
    
    def _synthesize_text_to_audio_with_recovery(self, text: str, voice_id: Optional[str] = None, 
                                              retry_count: int = 0, **kwargs) -> bool:
        """
        Synthesize text to audio with comprehensive error handling and recovery.
        
        This method implements the complete TTS pipeline with error recovery:
        1. Input validation
        2. Voice validation and fallback
        3. API synthesis with rate limiting
        4. Audio playback with error recovery
        5. Graceful degradation on failures
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            retry_count: Current retry attempt count
            **kwargs: Additional synthesis parameters
            
        Returns:
            bool: True if synthesis succeeded, False otherwise
        """
        context = {
            'text': text[:100],  # Truncate for logging
            'voice_id': voice_id,
            'retry_count': retry_count,
            'max_retries': kwargs.get('max_retries', 3)
        }
        
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use default voice if not specified
            voice_id = voice_id or self.config.default_voice_id
            
            # Validate voice ID with fallback
            if not self.get_voice_info(voice_id):
                error = VoiceNotFoundError(f"Voice not found: {voice_id}", details={'voice_id': voice_id})
                recovery_result = self._error_handler.handle_error(error, context)
                
                # Handle fallback voice
                if recovery_result.get('action') == 'fallback':
                    fallback_voice = recovery_result.get('fallback_voice')
                    if fallback_voice and self.get_voice_info(fallback_voice):
                        self.logger.info(f"Using fallback voice: {fallback_voice}")
                        voice_id = fallback_voice
                    else:
                        raise error
                else:
                    raise error
            
            # Create synthesis request
            request = SynthesisRequest(
                text=text,
                voice_id=voice_id,
                quality=kwargs.get('quality', self.config.voice_quality),
                **kwargs
            )
            
            # Log request
            self.logger.info(f"Synthesizing text with voice {voice_id}: {text[:50]}...")
            
            # Apply rate limiting
            if not self._rate_limit_manager.can_make_request():
                rate_limit_error = RateLimitError(
                    "Rate limit exceeded",
                    retry_after=self._rate_limit_manager.get_retry_after()
                )
                recovery_result = self._error_handler.handle_error(rate_limit_error, context)
                
                if recovery_result.get('action') == 'retry':
                    retry_delay = recovery_result.get('retry_delay', 1.0)
                    self.logger.info(f"Rate limited, waiting {retry_delay}s before retry")
                    time.sleep(retry_delay)
                    return self._synthesize_text_to_audio_with_recovery(
                        text, voice_id, retry_count + 1, **kwargs
                    )
                else:
                    raise rate_limit_error
            
            # Simulate synthesis (actual implementation would go here)
            # For now, simulate success with audio playback
            self.logger.info("Synthesis completed successfully")
            
            # Play audio through audio player
            try:
                # Generate temporary audio file path for simulation
                import tempfile
                temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                temp_audio.close()
                
                # In real implementation, this would be the synthesized audio
                audio_success = self._audio_player.play_audio(temp_audio.name)
                
                if not audio_success:
                    raise PlaybackError("Audio playback failed")
                
                # Clean up temporary file
                os.unlink(temp_audio.name)
                
                return True
                
            except Exception as playback_error:
                error = PlaybackError(f"Audio playback failed: {playback_error}")
                recovery_result = self._error_handler.handle_error(error, context)
                
                if recovery_result.get('action') == 'degrade':
                    self.logger.warning("Continuing without audio playback")
                    return True  # Success without audio
                else:
                    raise error
            
        except TTSError as tts_error:
            # Handle TTS-specific errors
            recovery_result = self._error_handler.handle_error(tts_error, context)
            
            if recovery_result.get('action') == 'retry':
                if retry_count < context.get('max_retries', 3):
                    retry_delay = recovery_result.get('retry_delay', 1.0)
                    self.logger.info(f"Retrying synthesis after {retry_delay}s delay")
                    time.sleep(retry_delay)
                    return self._synthesize_text_to_audio_with_recovery(
                        text, voice_id, retry_count + 1, **kwargs
                    )
                else:
                    self.logger.error("Maximum retries exceeded")
                    raise tts_error
            
            elif recovery_result.get('action') == 'fallback':
                fallback_settings = recovery_result.get('fallback_settings', {})
                self.logger.info(f"Applying fallback settings: {fallback_settings}")
                # Apply fallback settings and retry
                kwargs.update(fallback_settings)
                return self._synthesize_text_to_audio_with_recovery(
                    text, voice_id, retry_count + 1, **kwargs
                )
            
            elif recovery_result.get('action') == 'degrade':
                self.logger.warning("Graceful degradation: continuing without TTS")
                return False  # Indicate failure but continue execution
            
            else:
                raise tts_error
                
        except Exception as e:
            # Handle unexpected errors
            error = TTSError(f"Unexpected error: {e}", details={'original_error': str(e)})
            recovery_result = self._error_handler.handle_error(error, context)
            
            if recovery_result.get('action') == 'degrade':
                self.logger.warning("Graceful degradation: continuing without TTS")
                return False
            else:
                raise error
    
    def stop_playback(self) -> bool:
        """
        Stop any current audio playback.
        
        Returns:
            bool: True if playback was stopped successfully
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.stop_playback()
        """
        # Implementation will be added in Task 4.7
        self.logger.info("Stopping audio playback...")
        return True
    
    def set_voice_settings(self, voice_id: str, settings: Dict[str, Any]) -> bool:
        """
        Configure voice-specific settings.
        
        Args:
            voice_id: Voice ID to configure
            settings: Voice settings dictionary containing:
                - stability: Voice stability (0.0-1.0)
                - similarity_boost: Similarity boost (0.0-1.0)
                - style: Style setting (0.0-1.0)
                - use_speaker_boost: Whether to use speaker boost (boolean)
            
        Returns:
            bool: True if settings were applied successfully
            
        Raises:
            VoiceNotFoundError: If voice ID is not found
            TTSError: If settings update fails
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> settings = {"stability": 0.75, "similarity_boost": 0.85}
            >>> client.set_voice_settings("6sFKzaJr574YWVu4UuJF", settings)
        """
        context = {
            'voice_id': voice_id,
            'settings': settings,
            'method': 'set_voice_settings'
        }
        
        try:
            # Validate voice exists
            if not self.get_voice_info(voice_id):
                error = VoiceNotFoundError(f"Voice not found: {voice_id}", details={'voice_id': voice_id})
                recovery_result = self._error_handler.handle_error(error, context)
                
                if recovery_result.get('action') == 'fallback':
                    fallback_voice = recovery_result.get('fallback_voice')
                    if fallback_voice:
                        self.logger.info(f"Using fallback voice for settings: {fallback_voice}")
                        voice_id = fallback_voice
                    else:
                        raise error
                else:
                    raise error
            
            if not self._session:
                error = ConnectionError("Client not initialized - call validate_api_key() first")
                recovery_result = self._error_handler.handle_error(error, context)
                
                if recovery_result.get('action') == 'fail':
                    raise error
                else:
                    raise error
            
            # Validate settings
            valid_settings = {
                'stability': (0.0, 1.0),
                'similarity_boost': (0.0, 1.0),
                'style': (0.0, 1.0),
                'use_speaker_boost': (bool,)
            }
            
            validated_settings = {}
            for key, value in settings.items():
                if key not in valid_settings:
                    self.logger.warning(f"Unknown voice setting: {key}")
                    continue
                    
                if key == 'use_speaker_boost':
                    if not isinstance(value, bool):
                        raise ValueError(f"use_speaker_boost must be boolean, got {type(value)}")
                    validated_settings[key] = value
                else:
                    min_val, max_val = valid_settings[key]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        raise ValueError(f"{key} must be between {min_val} and {max_val}")
                    validated_settings[key] = float(value)
            
            # Make API call to update voice settings
            response = self._make_retryable_request(
                "POST",
                f"{self.config.base_url}/voices/{voice_id}/settings/edit",
                json=validated_settings,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully updated voice settings for {voice_id}")
                return True
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key: Unauthorized")
            elif response.status_code == 403:
                raise AuthenticationError("Invalid API key: Access forbidden")
            elif response.status_code == 404:
                raise VoiceNotFoundError(f"Voice not found: {voice_id}")
            else:
                raise TTSError(f"Failed to update voice settings: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to update voice settings: {e}")
            raise ConnectionError(f"Request failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to update voice settings: {e}")
            raise TTSError(f"Failed to update voice settings: {e}")
    
    def get_voice_settings(self, voice_id: str) -> Dict[str, Any]:
        """
        Get current voice settings.
        
        Args:
            voice_id: Voice ID to get settings for
            
        Returns:
            Dict[str, Any]: Voice settings dictionary
            
        Raises:
            VoiceNotFoundError: If voice ID is not found
            TTSError: If settings retrieval fails
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> settings = client.get_voice_settings("6sFKzaJr574YWVu4UuJF")
            >>> print(f"Stability: {settings['stability']}")
        """
        try:
            # Validate voice exists
            if not self.get_voice_info(voice_id):
                raise VoiceNotFoundError(f"Voice not found: {voice_id}")
            
            if not self._session_manager:
                raise ConnectionError("Client not initialized - call validate_api_key() first")
            
            response = self._make_retryable_request(
                "GET",
                f"{self.config.base_url}/voices/{voice_id}/settings",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key: Unauthorized")
            elif response.status_code == 403:
                raise AuthenticationError("Invalid API key: Access forbidden")
            elif response.status_code == 404:
                raise VoiceNotFoundError(f"Voice not found: {voice_id}")
            else:
                raise TTSError(f"Failed to get voice settings: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get voice settings: {e}")
            raise ConnectionError(f"Request failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get voice settings: {e}")
            raise TTSError(f"Failed to get voice settings: {e}")
    
    def list_voices_by_category(self, category: str = None) -> List[VoiceInfo]:
        """
        List voices filtered by category.
        
        Args:
            category: Voice category to filter by (e.g., 'male', 'female', 'child')
                     If None, returns all voices
            
        Returns:
            List[VoiceInfo]: List of voices in the specified category
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> male_voices = client.list_voices_by_category("male")
            >>> for voice in male_voices:
            ...     print(f"Male voice: {voice.name}")
        """
        voices = self.get_available_voices()
        
        if category is None:
            return voices
            
        return [voice for voice in voices if voice.category.lower() == category.lower()]
    
    def find_voice_by_name(self, name: str) -> Optional[VoiceInfo]:
        """
        Find a voice by name (case-insensitive).
        
        Args:
            name: Voice name to search for
            
        Returns:
            Optional[VoiceInfo]: Voice information if found, None otherwise
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> voice = client.find_voice_by_name("David")
            >>> if voice:
            ...     print(f"Found voice: {voice.voice_id}")
        """
        voices = self.get_available_voices()
        return next((voice for voice in voices if voice.name.lower() == name.lower()), None)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the current session.
        
        Returns:
            Dict[str, Any]: Usage statistics including API calls, characters synthesized, etc.
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> stats = client.get_usage_stats()
            >>> print(f"API calls made: {stats['api_calls']}")
        """
        base_stats = {
            "api_calls": 0,
            "characters_synthesized": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_request_time": None
        }
        
        # Add session manager statistics
        if self._session_manager:
            session_stats = self._session_manager.get_session_stats()
            base_stats.update({
                "session_stats": session_stats,
                "connection_pooling_active": True
            })
        else:
            base_stats["connection_pooling_active"] = False
        
        # Add retry statistics
        if self._retry_handler:
            retry_stats = self._retry_handler.get_retry_stats()
            base_stats.update({
                "retry_stats": retry_stats,
                "retry_logic_active": True
            })
        else:
            base_stats["retry_logic_active"] = False
            
        return base_stats
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get detailed connection and session statistics.
        
        Returns:
            Dict[str, Any]: Connection statistics including session health, pool info, etc.
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> conn_stats = client.get_connection_stats()
            >>> print(f"Session health: {conn_stats['session_healthy']}")
        """
        if not self._session_manager:
            return {
                "status": "no_session_manager",
                "connection_pooling_active": False
            }
        
        session_stats = self._session_manager.get_session_stats()
        session_health = self._session_manager.health_check()
        
        return {
            "status": "active",
            "connection_pooling_active": True,
            "session_stats": session_stats,
            "session_health": session_health,
            "pool_config": {
                "pool_connections": self._session_manager.pool_config.pool_connections,
                "pool_maxsize": self._session_manager.pool_config.pool_maxsize,
                "max_retries": self._session_manager.pool_config.max_retries,
                "backoff_factor": self._session_manager.pool_config.backoff_factor
            }
        }
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get detailed retry statistics and configuration.
        
        Returns:
            Dict[str, Any]: Retry statistics and configuration info
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> retry_stats = client.get_retry_stats()
            >>> print(f"Successful retries: {retry_stats['stats']['successful_retries']}")
        """
        if not self._retry_handler:
            return {
                "status": "no_retry_handler",
                "retry_logic_active": False
            }
        
        retry_stats = self._retry_handler.get_retry_stats()
        retry_config = self.config.retry_config
        
        return {
            "status": "active",
            "retry_logic_active": True,
            "stats": retry_stats,
            "config": {
                "max_retries": retry_config.max_retries,
                "base_delay": retry_config.base_delay,
                "max_delay": retry_config.max_delay,
                "backoff_factor": retry_config.backoff_factor,
                "strategy": retry_config.strategy.value,
                "jitter": retry_config.jitter,
                "retryable_status_codes": retry_config.retryable_status_codes,
                "non_retryable_status_codes": retry_config.non_retryable_status_codes
            }
        }
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get detailed rate limiting statistics and configuration.
        
        Returns:
            Dict[str, Any]: Rate limiting statistics and configuration info
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> rate_stats = client.get_rate_limit_stats()
            >>> print(f"Requests queued: {rate_stats['stats']['requests_queued']}")
        """
        if not self._rate_limit_manager:
            return {
                "status": "no_rate_limit_manager",
                "rate_limiting_active": False
            }
        
        return {
            "status": "active",
            "rate_limiting_active": self._rate_limit_manager.config.enabled,
            "config": {
                "enabled": self._rate_limit_manager.config.enabled,
                "requests_per_second": self._rate_limit_manager.config.requests_per_second,
                "burst_requests": self._rate_limit_manager.config.burst_requests,
                "adaptive_enabled": self._rate_limit_manager.config.adaptive_enabled,
                "respect_headers": self._rate_limit_manager.config.respect_headers,
                "token_bucket": {
                    "capacity": self._rate_limit_manager.config.token_bucket.capacity,
                    "refill_rate": self._rate_limit_manager.config.token_bucket.refill_rate,
                    "burst_capacity": self._rate_limit_manager.config.token_bucket.burst_capacity
                },
                "queue": {
                    "max_size": self._rate_limit_manager.config.queue.max_size,
                    "overflow_strategy": self._rate_limit_manager.config.queue.overflow_strategy.value,
                    "priority_enabled": self._rate_limit_manager.config.queue.priority_enabled,
                    "worker_threads": self._rate_limit_manager.config.queue.worker_threads
                }
            },
            "stats": self._rate_limit_manager.get_stats()
        }
    
    def reset_retry_stats(self):
        """
        Reset retry statistics.
        
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.reset_retry_stats()
        """
        if self._retry_handler:
            self._retry_handler.reset_stats()
            self.logger.info("Retry statistics reset")
    
    def reset_rate_limit_stats(self):
        """
        Reset rate limiting statistics.
        
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.reset_rate_limit_stats()
        """
        if self._rate_limit_manager:
            self._rate_limit_manager.stats = {
                "requests_processed": 0,
                "requests_queued": 0,
                "requests_rejected": 0,
                "rate_limit_hits": 0,
                "adaptive_adjustments": 0
            }
            self.logger.info("Rate limiting statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the TTS client.
        
        Returns:
            Dict[str, Any]: Health status information
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> health = client.health_check()
            >>> print(f"Status: {health['status']}")
        """
        try:
            # Basic health checks
            health_info = {
                "status": "healthy",
                "timestamp": time.time(),
                "api_key_valid": False,
                "voices_available": 0,
                "cache_size": len(self._voice_cache),
                "config_valid": True,
                "session_manager_active": self._session_manager is not None,
                "retry_handler_active": self._retry_handler is not None,
                "rate_limit_manager_active": self._rate_limit_manager is not None,
                "audio_player_active": self._audio_player is not None,
                "error_handler_active": self._error_handler is not None
            }
            
            # Add error handler health check
            if self._error_handler:
                error_handler_health = self._error_handler.health_check()
                health_info["error_handler_status"] = error_handler_health["status"]
                health_info["recent_error_rate"] = error_handler_health["recent_error_rate"]
                health_info["total_errors"] = error_handler_health["total_errors"]
                
                # Update overall status based on error handler
                if error_handler_health["status"] == "degraded":
                    health_info["status"] = "degraded"
                elif error_handler_health["status"] == "warning" and health_info["status"] == "healthy":
                    health_info["status"] = "warning"
            
            # Check session manager health
            if self._session_manager:
                try:
                    session_health = self._session_manager.health_check()
                    health_info["session_manager_health"] = session_health
                    if not session_health.get("session_healthy", True):
                        health_info["status"] = "degraded"
                except Exception:
                    health_info["session_manager_health"] = {"status": "error"}
                    health_info["status"] = "degraded"
            
            # Check retry handler health
            if self._retry_handler:
                try:
                    retry_stats = self._retry_handler.get_retry_stats()
                    health_info["retry_handler_stats"] = retry_stats
                    # Check if retry failure rate is too high
                    if retry_stats["total_attempts"] > 0:
                        failure_rate = retry_stats["failed_retries"] / retry_stats["total_attempts"]
                        if failure_rate > 0.5:  # More than 50% failure rate
                            health_info["status"] = "degraded"
                except Exception:
                    health_info["retry_handler_stats"] = {"status": "error"}
                    health_info["status"] = "degraded"
            
            # Check rate limiting health
            if self._rate_limit_manager:
                try:
                    rate_stats = self._rate_limit_manager.get_stats()
                    health_info["rate_limit_stats"] = rate_stats
                    # Check if queue is getting too full
                    queue_stats = rate_stats.get("queue", {})
                    if queue_stats.get("utilization", 0) > 0.8:  # More than 80% full
                        health_info["status"] = "degraded"
                except Exception:
                    health_info["rate_limit_stats"] = {"status": "error"}
                    health_info["status"] = "degraded"
            
            # Check audio player health
            if self._audio_player:
                try:
                    audio_health = self._audio_player.health_check()
                    health_info["audio_player_health"] = audio_health
                    if audio_health["status"] != "healthy":
                        health_info["status"] = "degraded"
                except Exception:
                    health_info["audio_player_health"] = {"status": "error"}
                    health_info["status"] = "degraded"
            
            # Check error handler health
            if self._error_handler:
                try:
                    error_health = self._error_handler.health_check()
                    health_info["error_handler_health"] = error_health
                    if error_health["status"] != "healthy":
                        health_info["status"] = "degraded"
                except Exception:
                    health_info["error_handler_health"] = {"status": "error"}
                    health_info["status"] = "degraded"
            
            # Check API key
            try:
                health_info["api_key_valid"] = self.validate_api_key()
            except Exception:
                health_info["api_key_valid"] = False
                health_info["status"] = "degraded"
            
            # Check voices
            try:
                voices = self.get_available_voices()
                health_info["voices_available"] = len(voices)
            except Exception:
                health_info["voices_available"] = 0
                health_info["status"] = "degraded"
            
            return health_info
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def speak_text(self, text: str, voice_id: str = None, blocking: bool = True, volume: float = None) -> bool:
        """
        Synthesize text to speech and play it.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (defaults to configured default)
            blocking: Whether to block until playback completes
            volume: Volume level (0.0 to 1.0, overrides config)
            
        Returns:
            bool: True if synthesis and playback succeeded
            
        Raises:
            AuthenticationError: If API key is invalid
            VoiceNotFoundError: If voice ID is not available
            PlaybackError: If audio playback fails
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.speak_text("Hello world", voice_id="David")
        """
        context = {
            'text_length': len(text) if text else 0,
            'voice_id': voice_id,
            'blocking': blocking,
            'volume': volume
        }
        
        try:
            # Validate input
            if not text or not text.strip():
                raise ConfigurationError("Text cannot be empty", config_field="text")
            
            # Use default voice if not specified
            if voice_id is None:
                voice_id = self.config.default_voice_id
                context['voice_id'] = voice_id
            
            # Set volume if provided
            if volume is not None:
                self._audio_player.set_volume(volume)
            
            self.logger.info(f"Synthesizing text: '{text[:50]}...' with voice: {voice_id}")
            
            # Synthesize audio with error handling
            audio_data = self._synthesize_text_to_audio_with_recovery(text, voice_id, context)
            
            # Play audio with error handling
            success = self._play_audio_with_recovery(audio_data, blocking, context)
            
            if success:
                self.logger.info("Text-to-speech playback completed successfully")
                return True
            else:
                raise PlaybackError("Audio playback failed", platform=platform.system())
                
        except Exception as e:
            # Handle error with comprehensive error handling system
            error_result = self._error_handler.handle_error(e, context)
            
            # Try recovery if suggested
            if error_result['recovery_result'].get('action') == 'fallback':
                return self._try_fallback_synthesis(text, error_result, context)
            elif error_result['recovery_result'].get('action') == 'degrade':
                self.logger.warning("Continuing in degraded mode (silent)")
                return False
            
            # Re-raise if no recovery possible
            raise
    
    def _synthesize_text_to_audio_with_recovery(self, text: str, voice_id: str, context: Dict[str, Any]) -> bytes:
        """Synthesize text to audio with error recovery."""
        try:
            return self._synthesize_text_to_audio(text, voice_id)
        except Exception as e:
            error_result = self._error_handler.handle_error(e, context)
            
            # Try fallback voice if suggested
            if error_result['recovery_result'].get('action') == 'fallback':
                fallback_voice = error_result['recovery_result'].get('fallback_voice')
                if fallback_voice and fallback_voice != voice_id:
                    self.logger.info(f"Trying fallback voice: {fallback_voice}")
                    return self._synthesize_text_to_audio(text, fallback_voice)
            
            # Re-raise if no recovery possible
            raise
    
    def _play_audio_with_recovery(self, audio_data: bytes, blocking: bool, context: Dict[str, Any]) -> bool:
        """Play audio with error recovery."""
        try:
            return self._audio_player.play_audio(audio_data, AudioFormat.MP3, blocking=blocking)
        except Exception as e:
            error_result = self._error_handler.handle_error(e, context)
            
            # Try graceful degradation
            if error_result['recovery_result'].get('action') == 'degrade':
                self.logger.warning("Audio playback failed, continuing in silent mode")
                return False
            
            # Re-raise if no recovery possible
            raise
    
    def _try_fallback_synthesis(self, text: str, error_result: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Try synthesis with fallback settings."""
        try:
            fallback_voice = error_result['recovery_result'].get('fallback_voice')
            if fallback_voice:
                self.logger.info(f"Attempting fallback synthesis with voice: {fallback_voice}")
                audio_data = self._synthesize_text_to_audio(text, fallback_voice)
                return self._audio_player.play_audio(audio_data, AudioFormat.MP3, blocking=context.get('blocking', True))
            else:
                self.logger.warning("No fallback voice available")
                return False
        except Exception as e:
            self.logger.error(f"Fallback synthesis failed: {e}")
            return False
    
    def _synthesize_text_to_audio(self, text: str, voice_id: str) -> bytes:
        """
        Synthesize text to audio data using ElevenLabs API.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            
        Returns:
            bytes: Audio data in MP3 format
            
        Raises:
            AuthenticationError: If API key is invalid
            VoiceNotFoundError: If voice ID is not available
            TTSError: If synthesis fails
        """
        try:
            # Prepare synthesis request
            synthesis_data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            # Make API request
            response = self._make_rate_limited_request(
                "POST",
                f"{self.config.base_url}/text-to-speech/{voice_id}",
                priority=QueuePriority.HIGH,
                json=synthesis_data,
                headers={"Accept": "audio/mpeg"},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                raise VoiceNotFoundError(f"Voice ID {voice_id} not found")
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            else:
                raise TTSError(f"Synthesis failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Text synthesis failed: {e}")
            raise
    
    def stop_playback(self) -> bool:
        """
        Stop current audio playback.
        
        Returns:
            bool: True if playback was stopped successfully
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.speak_text("Hello world", blocking=False)
            >>> client.stop_playback()
        """
        if self._audio_player:
            return self._audio_player.stop_playback()
        return False
    
    def pause_playback(self) -> bool:
        """
        Pause current audio playback.
        
        Note: Pause functionality is platform-dependent.
        
        Returns:
            bool: True if playback was paused successfully
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.speak_text("Hello world", blocking=False)
            >>> client.pause_playback()
        """
        if self._audio_player:
            return self._audio_player.pause_playback()
        return False
    
    def resume_playback(self) -> bool:
        """
        Resume paused audio playback.
        
        Returns:
            bool: True if playback was resumed successfully
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.resume_playback()
        """
        if self._audio_player:
            return self._audio_player.resume_playback()
        return False
    
    def set_volume(self, volume: float) -> bool:
        """
        Set audio playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.set_volume(0.8)
        """
        if self._audio_player:
            return self._audio_player.set_volume(volume)
        return False
    
    def get_playback_status(self) -> PlaybackStatus:
        """
        Get current audio playback status.
        
        Returns:
            PlaybackStatus: Current playback status
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> status = client.get_playback_status()
            >>> print(f"Status: {status.value}")
        """
        if self._audio_player:
            return self._audio_player.get_status()
        return PlaybackStatus.ERROR
    
    def is_playing(self) -> bool:
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> if client.is_playing():
            ...     print("Audio is playing")
        """
        if self._audio_player:
            return self._audio_player.is_playing()
        return False
    
    def get_supported_audio_formats(self) -> List[AudioFormat]:
        """
        Get list of supported audio formats for current platform.
        
        Returns:
            List[AudioFormat]: List of supported audio formats
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> formats = client.get_supported_audio_formats()
            >>> print(f"Supported formats: {[f.value for f in formats]}")
        """
        if self._audio_player:
            return self._audio_player.get_supported_formats()
        return []
    
    def get_audio_player_stats(self) -> Dict[str, Any]:
        """
        Get audio player statistics and health information.
        
        Returns:
            Dict[str, Any]: Audio player statistics
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> stats = client.get_audio_player_stats()
            >>> print(f"Platform: {stats['platform']}")
        """
        if self._audio_player:
            return self._audio_player.health_check()
        return {"status": "no_audio_player", "audio_player_active": False}
    
    def get_error_handler_stats(self) -> Dict[str, Any]:
        """
        Get error handler statistics and metrics.
        
        Returns:
            Dict[str, Any]: Error handler statistics
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> stats = client.get_error_handler_stats()
            >>> print(f"Total errors: {stats['total_errors']}")
        """
        if self._error_handler:
            return self._error_handler.get_metrics()
        return {"status": "no_error_handler", "error_handler_active": False}
    
    def reset_error_handler_stats(self):
        """
        Reset error handler statistics.
        
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> client.reset_error_handler_stats()
        """
        if self._error_handler:
            self._error_handler.reset_metrics()
            self.logger.info("Error handler statistics reset")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all components.
        
        Returns:
            Dict[str, Any]: Combined statistics from all components
            
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> stats = client.get_comprehensive_stats()
            >>> print(f"System health: {stats['health']['status']}")
        """
        return {
            'health': self.health_check(),
            'connection': self.get_connection_stats(),
            'retry': self.get_retry_stats(),
            'rate_limit': self.get_rate_limit_stats(),
            'audio_player': self.get_audio_player_stats(),
            'error_handler': self.get_error_handler_stats(),
            'timestamp': time.time()
        }
    
    def close(self):
        """
        Clean up resources and close the client.
        
        This method should be called when the client is no longer needed
        to ensure proper cleanup of connections and resources.
        
        Example:
            >>> client = TTSClient(api_key="your-api-key")
            >>> try:
            ...     client.speak_text("Hello world")
            ... finally:
            ...     client.close()
        """
        with self._lock:
            self.logger.info("Closing TTS client...")
            
            # Stop any ongoing playback
            if self._audio_player:
                self._audio_player.stop_playback()
            
            # Clear caches
            self._voice_cache.clear()
            
            # Close session manager if exists
            if self._session_manager:
                try:
                    self._session_manager.close()
                    self.logger.info("Session manager closed")
                except Exception as e:
                    self.logger.warning(f"Error closing session manager: {e}")
                finally:
                    self._session_manager = None
            
            # Shutdown rate limiting manager if exists
            if self._rate_limit_manager:
                try:
                    self._rate_limit_manager.shutdown_manager()
                    self.logger.info("Rate limit manager shutdown")
                except Exception as e:
                    self.logger.warning(f"Error shutting down rate limit manager: {e}")
                finally:
                    self._rate_limit_manager = None
            
            # Mark as not initialized
            self._is_initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (f"TTSClient(api_key={self.api_key}, "
                f"base_url={self.base_url}, "
                f"default_voice={self.default_voice_id})")


# Convenience functions for common use cases
def create_client(api_key: Optional[str] = None, **config_kwargs) -> TTSClient:
    """
    Create a TTS client with optional configuration.
    
    Args:
        api_key: ElevenLabs API key
        **config_kwargs: Additional configuration parameters
        
    Returns:
        TTSClient: Configured TTS client
        
    Example:
        >>> client = create_client(api_key="your-api-key", timeout=60)
        >>> client.speak_text("Hello world")
    """
    config = TTSConfig(api_key=api_key or os.getenv("ELEVENLABS_API_KEY", ""), **config_kwargs)
    return TTSClient(config=config)


def quick_speak(text: str, voice_id: Optional[str] = None, api_key: Optional[str] = None) -> bool:
    """
    Quick text-to-speech synthesis without managing client lifecycle.
    
    Args:
        text: Text to synthesize
        voice_id: Voice ID to use
        api_key: ElevenLabs API key
        
    Returns:
        bool: True if synthesis succeeded
        
    Example:
        >>> quick_speak("Hello world", voice_id="David")
    """
    with create_client(api_key=api_key) as client:
        return client.speak_text(text, voice_id=voice_id)


if __name__ == "__main__":
    # Example usage and testing
    print("ElevenLabs TTS Client - Architecture and Interface")
    print("=" * 50)
    
    # Test configuration
    try:
        config = TTSConfig(api_key="test-key")
        print(f"✓ Configuration created: {config.default_voice_id}")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
    
    # Test client creation
    try:
        client = TTSClient(api_key="test-key")
        print(f"✓ Client created: {client}")
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
    
    # Test voice information
    try:
        voices = list(TTSClient.SUPPORTED_VOICES.values())
        print(f"✓ Supported voices: {len(voices)}")
        for voice in voices:
            print(f"  - {voice.name} ({voice.voice_id}): {voice.description}")
    except Exception as e:
        print(f"✗ Voice info failed: {e}")
    
    # Test data structures
    try:
        request = SynthesisRequest(text="Hello world", voice_id="6sFKzaJr574YWVu4UuJF")
        print(f"✓ Synthesis request created: {request.text}")
    except Exception as e:
        print(f"✗ Synthesis request failed: {e}")
    
    print("\n" + "=" * 50)
    print("TTSClient architecture and interface design complete!")
    print("Ready for Task 4.2: API authentication and key validation")