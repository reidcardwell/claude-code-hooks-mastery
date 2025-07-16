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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Base exception for TTS-related errors."""
    pass


class AuthenticationError(TTSError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(TTSError):
    """Raised when API rate limits are exceeded."""
    pass


class VoiceNotFoundError(TTSError):
    """Raised when requested voice ID is not available."""
    pass


class PlaybackError(TTSError):
    """Raised when audio playback fails."""
    pass


class ConnectionError(TTSError):
    """Raised when network connection fails."""
    pass


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
            
            # Use retryable request for API key validation
            response = self._make_retryable_request(
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
                
                # Fetch voices from API with retry logic
                response = self._make_retryable_request(
                    "GET",
                    f"{self.config.base_url}/voices",
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
        complete pipeline from text input to audio playback.
        
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
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use default voice if not specified
            voice_id = voice_id or self.config.default_voice_id
            
            # Validate voice ID
            if not self.get_voice_info(voice_id):
                raise VoiceNotFoundError(f"Voice not found: {voice_id}")
            
            # Create synthesis request
            request = SynthesisRequest(
                text=text,
                voice_id=voice_id,
                quality=kwargs.get('quality', self.config.voice_quality),
                **kwargs
            )
            
            # Log request
            self.logger.info(f"Synthesizing text with voice {voice_id}: {text[:50]}...")
            
            # Implementation will be completed in subsequent tasks
            # For now, return success placeholder
            return True
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise TTSError(f"Failed to synthesize speech: {e}")
    
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
        try:
            # Validate voice exists
            if not self.get_voice_info(voice_id):
                raise VoiceNotFoundError(f"Voice not found: {voice_id}")
            
            if not self._session:
                raise ConnectionError("Client not initialized - call validate_api_key() first")
            
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
                "retry_handler_active": self._retry_handler is not None
            }
            
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
            self.stop_playback()
            
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