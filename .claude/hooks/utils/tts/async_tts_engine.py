#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["elevenlabs>=1.0.0", "requests>=2.31.0", "python-dotenv>=1.0.0"]
# ///

"""
Asynchronous TTS Execution Engine

This module provides a non-blocking TTS execution system designed to process
speech synthesis requests in the background without impacting Claude Code's
command execution performance.

Key Features:
- Non-blocking TTS operations using asyncio
- Concurrent request processing with configurable limits
- Priority-based queue management for recent commands
- Resource cleanup and memory management
- Graceful shutdown handling
- Thread-safe operation guarantees

Architecture Overview:
- AsyncTTSEngine: Main coordinator class
- TTSQueue: Queue management with concurrency control
- TTSWorker: Background worker for processing requests
- TTSRequest: Request data structure with priority
- AsyncTTSContext: Context manager for resource management

Usage:
    engine = AsyncTTSEngine()
    await engine.start()
    await engine.speak_async("Hello world", tts_config)
    await engine.shutdown()
"""

import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set
from pathlib import Path
import signal
import threading
import heapq
import uuid
from contextlib import asynccontextmanager, contextmanager
import gc
import psutil
import os
import traceback
import resource
import sys
import json
from collections import defaultdict

# Configure thread-safe logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('.tts_engine.log')
    ]
)
logger = logging.getLogger(__name__)


class ThreadSafeLogger:
    """Thread-safe logger wrapper to prevent race conditions in logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._lock = threading.Lock()
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        with self._lock:
            self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        with self._lock:
            self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        with self._lock:
            self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        with self._lock:
            self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        with self._lock:
            self.logger.critical(msg, *args, **kwargs)


class ThreadSafeCounters:
    """Thread-safe counter management for performance metrics"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Union[int, float]] = {}
    
    def increment(self, key: str, value: Union[int, float] = 1) -> Union[int, float]:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
            return self._counters[key]
    
    def decrement(self, key: str, value: Union[int, float] = 1) -> Union[int, float]:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) - value
            return self._counters[key]
    
    def set(self, key: str, value: Union[int, float]) -> None:
        with self._lock:
            self._counters[key] = value
    
    def get(self, key: str) -> Union[int, float]:
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_all(self) -> Dict[str, Union[int, float]]:
        with self._lock:
            return self._counters.copy()
    
    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
    
    def atomic_max_update(self, key: str, value: Union[int, float]) -> Union[int, float]:
        """Atomically update key to max of current value and new value"""
        with self._lock:
            current = self._counters.get(key, 0)
            new_value = max(current, value)
            self._counters[key] = new_value
            return new_value


class LockManager:
    """Manages lock acquisition order to prevent deadlocks"""
    
    def __init__(self):
        self._lock_order = {
            'queue': 1,
            'resource': 2,
            'memory': 3,
            'worker': 4,
            'performance': 5,
            'shutdown': 6
        }
        self._locks: Dict[str, asyncio.Lock] = {}
        self._creation_lock = threading.Lock()
    
    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock with the given name"""
        with self._creation_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]
    
    @asynccontextmanager
    async def acquire_locks(self, *lock_names: str):
        """Acquire multiple locks in consistent order to prevent deadlocks"""
        if not lock_names:
            yield
            return
        
        # Sort by predefined order to prevent deadlocks
        sorted_names = sorted(lock_names, key=lambda x: self._lock_order.get(x, 999))
        locks = [self.get_lock(name) for name in sorted_names]
        
        acquired = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            # Release in reverse order
            for lock in reversed(acquired):
                lock.release()


class ThreadSafeSet:
    """Thread-safe set operations"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._set: Set[Any] = set()
    
    def add(self, item: Any) -> None:
        with self._lock:
            self._set.add(item)
    
    def remove(self, item: Any) -> None:
        with self._lock:
            self._set.discard(item)
    
    def contains(self, item: Any) -> bool:
        with self._lock:
            return item in self._set
    
    def size(self) -> int:
        with self._lock:
            return len(self._set)
    
    def clear(self) -> None:
        with self._lock:
            self._set.clear()
    
    def copy(self) -> Set[Any]:
        with self._lock:
            return self._set.copy()


# Create global instances
safe_logger = ThreadSafeLogger(__name__)
global_lock_manager = LockManager()

# Constants
DEFAULT_MAX_CONCURRENT_REQUESTS = 3
DEFAULT_QUEUE_SIZE = 50
DEFAULT_WORKER_THREADS = 2
DEFAULT_PRIORITY_BOOST_THRESHOLD = 2.0  # seconds
DEFAULT_CLEANUP_INTERVAL = 30.0  # seconds
DEFAULT_SHUTDOWN_TIMEOUT = 10.0  # seconds
DEFAULT_MEMORY_THRESHOLD = 512 * 1024 * 1024  # 512MB
DEFAULT_RESOURCE_LEAK_CHECK_INTERVAL = 60.0  # seconds
DEFAULT_STALE_REQUEST_AGE = 300.0  # 5 minutes


class TTSPriority(Enum):
    """Priority levels for TTS requests"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TTSShutdownState(Enum):
    """Shutdown states for graceful termination"""
    RUNNING = "running"
    SHUTDOWN_INITIATED = "shutdown_initiated"
    DRAINING_REQUESTS = "draining_requests"
    STOPPING_WORKERS = "stopping_workers"
    CLEANING_RESOURCES = "cleaning_resources"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    FORCE_SHUTDOWN = "force_shutdown"


class TTSRequestStatus(Enum):
    """Status of TTS requests"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TTSRequest:
    """
    Thread-safe TTS request with priority and metadata
    """
    text: str
    config: Dict[str, Any]
    priority: TTSPriority = TTSPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TTSRequestStatus = TTSRequestStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    context: Optional[Dict[str, Any]] = None
    _priority_score: float = field(default=0.0, init=False)
    _last_priority_boost: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def __post_init__(self):
        # Calculate initial priority score
        self._calculate_priority_score()
        
        # Auto-boost priority for recent requests
        if time.time() - self.created_at < DEFAULT_PRIORITY_BOOST_THRESHOLD:
            if self.priority == TTSPriority.NORMAL:
                self.priority = TTSPriority.HIGH
                self._last_priority_boost = time.time()
    
    def _calculate_priority_score(self) -> None:
        """Calculate priority score based on priority level, age, and starvation prevention"""
        current_time = time.time()
        age = current_time - self.created_at
        
        # Base priority score
        base_score = self.priority.value * 1000
        
        # Age factor (older requests get slightly higher priority to prevent starvation)
        age_factor = min(age * 10, 500)  # Cap at 500 points for age
        
        # Starvation prevention boost for requests waiting too long
        starvation_boost = 0
        if age > 30:  # 30 seconds threshold
            starvation_boost = min((age - 30) * 50, 1000)  # Progressive boost
        
        # Recent command boost
        recent_boost = 0
        if current_time - self.created_at < DEFAULT_PRIORITY_BOOST_THRESHOLD:
            recent_boost = 200
        
        # Retry penalty (lower priority for failed requests)
        retry_penalty = self.retry_count * 50
        
        self._priority_score = base_score + age_factor + starvation_boost + recent_boost - retry_penalty
    
    def update_priority_score(self) -> None:
        """Thread-safe update of priority score"""
        with self._lock:
            old_score = self._priority_score
            self._calculate_priority_score()
            
            # Log significant priority changes
            if abs(self._priority_score - old_score) > 100:
                safe_logger.debug(f"Priority score updated for {self.request_id}: {old_score:.1f} -> {self._priority_score:.1f}")
    
    def get_priority_metrics(self) -> Dict[str, Any]:
        """Get priority metrics for monitoring - thread-safe"""
        with self._lock:
            current_time = time.time()
            return {
                "request_id": self.request_id,
                "priority_level": self.priority.name,
                "priority_score": self._priority_score,
                "age_seconds": current_time - self.created_at,
                "retry_count": self.retry_count,
                "has_recent_boost": (current_time - self.created_at) < DEFAULT_PRIORITY_BOOST_THRESHOLD,
                "last_boost_time": self._last_priority_boost
            }
    
    def get_status_safe(self) -> TTSRequestStatus:
        """Thread-safe status getter"""
        with self._lock:
            return self.status
    
    def set_status_safe(self, status: TTSRequestStatus) -> None:
        """Thread-safe status setter"""
        with self._lock:
            self.status = status
    
    def get_priority_score_safe(self) -> float:
        """Thread-safe priority score getter"""
        with self._lock:
            return self._priority_score
    
    def __lt__(self, other):
        """Priority queue comparison - higher priority score first"""
        # Update scores before comparison to ensure freshness
        self.update_priority_score()
        other.update_priority_score()
        return self.get_priority_score_safe() > other.get_priority_score_safe()
    
    def is_expired(self, max_age: float = 60.0) -> bool:
        """Check if request has expired"""
        return time.time() - self.created_at > max_age
    
    def can_retry(self) -> bool:
        """Thread-safe check if request can be retried"""
        with self._lock:
            return self.retry_count < self.max_retries
    
    def mark_retry(self):
        """Thread-safe mark request for retry"""
        with self._lock:
            self.retry_count += 1
            self.status = TTSRequestStatus.PENDING


class TTSResourceType(Enum):
    """Types of TTS resources for cleanup tracking"""
    AUDIO_STREAM = "audio_stream"
    API_CONNECTION = "api_connection"
    THREAD_POOL = "thread_pool"
    MEMORY_BUFFER = "memory_buffer"
    TEMPORARY_FILE = "temporary_file"
    CACHE_ENTRY = "cache_entry"


@dataclass
class TTSResource:
    """Represents a TTS resource that needs cleanup"""
    resource_id: str
    resource_type: TTSResourceType
    resource_ref: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    cleanup_handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_stale(self, max_age: float = DEFAULT_STALE_REQUEST_AGE) -> bool:
        """Check if resource is stale and should be cleaned up"""
        return time.time() - self.last_accessed > max_age
    
    def update_access_time(self):
        """Update last accessed time"""
        self.last_accessed = time.time()
    
    def cleanup(self):
        """Clean up the resource"""
        try:
            if self.cleanup_handler:
                self.cleanup_handler(self.resource_ref)
            elif hasattr(self.resource_ref, 'close'):
                self.resource_ref.close()
            elif hasattr(self.resource_ref, 'shutdown'):
                self.resource_ref.shutdown()
        except Exception as e:
            logger.error(f"Error cleaning up resource {self.resource_id}: {e}")


class TTSResourceManager:
    """Thread-safe TTS resource manager with automatic cleanup"""
    
    def __init__(self):
        self._resources: Dict[str, TTSResource] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._leak_detection_task: Optional[asyncio.Task] = None
        self._memory_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._counters = ThreadSafeCounters()
        self._main_lock = threading.Lock()  # Main lock for resource dictionary operations
        
    async def start(self):
        """Start the resource manager"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._leak_detection_task = asyncio.create_task(self._leak_detection_loop())
        self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
        safe_logger.info("TTSResourceManager started")
    
    async def stop(self):
        """Stop the resource manager and cleanup all resources"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._leak_detection_task, self._memory_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up all resources
        await self._cleanup_all_resources()
        cleanup_count = self._counters.get('cleanup_count')
        safe_logger.info(f"TTSResourceManager stopped (cleaned up {cleanup_count} resources)")
    
    def register_resource(self, resource: TTSResource) -> str:
        """Thread-safe register a resource for cleanup tracking"""
        resource_id = resource.resource_id
        with self._main_lock:
            self._resources[resource_id] = resource
            
        # Update counters atomically
        self._counters.increment('total_memory_usage', resource.size_bytes)
        self._counters.atomic_max_update('peak_memory_usage', self._counters.get('total_memory_usage'))
        
        safe_logger.debug(f"Registered resource {resource_id} ({resource.resource_type.value})")
        return resource_id
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Thread-safe unregister and cleanup a resource"""
        with self._main_lock:
            if resource_id not in self._resources:
                return False
            
            resource = self._resources[resource_id]
            resource.cleanup()
            del self._resources[resource_id]
            
        # Update counters atomically
        self._counters.decrement('total_memory_usage', resource.size_bytes)
        self._counters.increment('cleanup_count')
        
        safe_logger.debug(f"Unregistered resource {resource_id}")
        return True
    
    def access_resource(self, resource_id: str) -> Optional[TTSResource]:
        """Thread-safe access a resource and update its access time"""
        with self._main_lock:
            if resource_id not in self._resources:
                return None
            resource = self._resources[resource_id]
            resource.update_access_time()
            return resource
    
    @contextmanager
    def resource_context(self, resource_type: TTSResourceType, resource_ref: Any, 
                        cleanup_handler: Optional[Callable] = None, size_bytes: int = 0):
        """Context manager for automatic resource cleanup"""
        resource = TTSResource(
            resource_id=str(uuid.uuid4()),
            resource_type=resource_type,
            resource_ref=resource_ref,
            cleanup_handler=cleanup_handler,
            size_bytes=size_bytes
        )
        
        try:
            resource_id = self.register_resource(resource)
            yield resource_ref
        finally:
            self.unregister_resource(resource_id)
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        while self._running:
            try:
                await asyncio.sleep(DEFAULT_CLEANUP_INTERVAL)
                await self._cleanup_stale_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_stale_resources(self):
        """Clean up stale resources - thread-safe"""
        stale_resources = []
        current_time = time.time()
        
        # Get snapshot of resources to check
        with self._main_lock:
            resources_snapshot = list(self._resources.items())
        
        # Check which resources are stale
        for resource_id, resource in resources_snapshot:
            if resource.is_stale():
                stale_resources.append(resource_id)
        
        # Clean up stale resources
        for resource_id in stale_resources:
            self.unregister_resource(resource_id)
            
        if stale_resources:
            safe_logger.info(f"Cleaned up {len(stale_resources)} stale resources")
    
    async def _leak_detection_loop(self):
        """Background task for resource leak detection"""
        while self._running:
            try:
                await asyncio.sleep(DEFAULT_RESOURCE_LEAK_CHECK_INTERVAL)
                await self._detect_leaks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                safe_logger.error(f"Error in leak detection loop: {e}")
    
    async def _detect_leaks(self):
        """Detect potential resource leaks - thread-safe"""
        current_time = time.time()
        long_lived_resources = []
        
        # Get snapshot of resources to check
        with self._main_lock:
            resources_snapshot = list(self._resources.items())
        
        # Check for long-lived resources
        for resource_id, resource in resources_snapshot:
            age = current_time - resource.created_at
            if age > DEFAULT_STALE_REQUEST_AGE * 2:  # 10 minutes
                long_lived_resources.append((resource_id, age, resource.resource_type))
        
        if long_lived_resources:
            safe_logger.warning(f"Potential resource leaks detected: {len(long_lived_resources)} long-lived resources")
            for resource_id, age, resource_type in long_lived_resources:
                safe_logger.warning(f"  - {resource_id} ({resource_type.value}): {age:.1f}s old")
            self._counters.increment('leak_detection_count', len(long_lived_resources))
    
    async def _memory_monitor_loop(self):
        """Background task for memory monitoring"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._monitor_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")
    
    async def _monitor_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            
            if current_memory > DEFAULT_MEMORY_THRESHOLD:
                logger.warning(f"High memory usage detected: {current_memory / 1024 / 1024:.1f}MB")
                await self._emergency_cleanup()
                
                # Force garbage collection
                gc.collect()
                
                # Check memory again after cleanup
                new_memory = psutil.Process().memory_info().rss
                logger.info(f"Memory after cleanup: {new_memory / 1024 / 1024:.1f}MB")
        except Exception as e:
            logger.error(f"Error monitoring memory: {e}")
    
    async def _emergency_cleanup(self):
        """Perform emergency cleanup when memory usage is high"""
        # Clean up all stale resources immediately
        await self._cleanup_stale_resources()
        
        # Clean up oldest resources if still over threshold
        if len(self._resources) > 10:
            oldest_resources = sorted(
                self._resources.items(),
                key=lambda x: x[1].created_at
            )[:len(self._resources) // 2]
            
            for resource_id, _ in oldest_resources:
                self.unregister_resource(resource_id)
            
            logger.info(f"Emergency cleanup removed {len(oldest_resources)} oldest resources")
    
    async def _cleanup_all_resources(self):
        """Clean up all registered resources"""
        resource_ids = list(self._resources.keys())
        for resource_id in resource_ids:
            self.unregister_resource(resource_id)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics - thread-safe"""
        resource_types = {}
        
        # Get snapshot of resources
        with self._main_lock:
            resources_snapshot = list(self._resources.values())
        
        # Count resource types
        for resource in resources_snapshot:
            resource_type = resource.resource_type.value
            if resource_type not in resource_types:
                resource_types[resource_type] = 0
            resource_types[resource_type] += 1
        
        # Get counter values atomically
        counters = self._counters.get_all()
        
        return {
            "total_resources": len(resources_snapshot),
            "total_memory_usage": counters.get('total_memory_usage', 0),
            "peak_memory_usage": counters.get('peak_memory_usage', 0),
            "cleanup_count": counters.get('cleanup_count', 0),
            "leak_detection_count": counters.get('leak_detection_count', 0),
            "resource_types": resource_types
        }


class TTSAudioBuffer:
    """Manages audio buffers with size limits and efficient memory usage"""
    
    def __init__(self, max_buffer_size: int = 10 * 1024 * 1024):  # 10MB default
        self._buffers: Dict[str, bytes] = {}
        self._buffer_metadata: Dict[str, Dict[str, Any]] = {}
        self._max_buffer_size = max_buffer_size
        self._current_size = 0
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def store_audio(self, request_id: str, audio_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Store audio data with metadata, returns False if size limit exceeded"""
        with self._lock:
            data_size = len(audio_data)
            
            # Check if adding this buffer would exceed limit
            if self._current_size + data_size > self._max_buffer_size:
                # Try to free space by removing oldest buffers
                if not self._free_space_for_size(data_size):
                    logger.warning(f"Cannot store audio buffer {request_id}: size limit exceeded")
                    return False
            
            # Store the audio and metadata
            self._buffers[request_id] = audio_data
            self._buffer_metadata[request_id] = metadata
            self._access_times[request_id] = time.time()
            self._current_size += data_size
            
            logger.debug(f"Stored audio buffer {request_id} ({data_size} bytes)")
            return True
    
    def get_audio(self, request_id: str) -> Optional[bytes]:
        """Retrieve audio data and update access time"""
        with self._lock:
            if request_id in self._buffers:
                self._access_times[request_id] = time.time()
                return self._buffers[request_id]
            return None
    
    def remove_audio(self, request_id: str) -> bool:
        """Remove audio buffer"""
        with self._lock:
            if request_id in self._buffers:
                data_size = len(self._buffers[request_id])
                del self._buffers[request_id]
                del self._buffer_metadata[request_id]
                del self._access_times[request_id]
                self._current_size -= data_size
                logger.debug(f"Removed audio buffer {request_id} ({data_size} bytes)")
                return True
            return False
    
    def _free_space_for_size(self, needed_size: int) -> bool:
        """Free space by removing oldest buffers"""
        if needed_size > self._max_buffer_size:
            return False
        
        # Sort by access time (oldest first)
        sorted_buffers = sorted(self._access_times.items(), key=lambda x: x[1])
        
        freed_space = 0
        for request_id, _ in sorted_buffers:
            if self._current_size - freed_space + needed_size <= self._max_buffer_size:
                break
            
            buffer_size = len(self._buffers[request_id])
            del self._buffers[request_id]
            del self._buffer_metadata[request_id]
            del self._access_times[request_id]
            freed_space += buffer_size
            logger.debug(f"Freed audio buffer {request_id} ({buffer_size} bytes)")
        
        self._current_size -= freed_space
        return self._current_size + needed_size <= self._max_buffer_size
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        with self._lock:
            return {
                "buffer_count": len(self._buffers),
                "current_size": self._current_size,
                "max_size": self._max_buffer_size,
                "usage_percentage": (self._current_size / self._max_buffer_size) * 100,
                "oldest_buffer_age": time.time() - min(self._access_times.values()) if self._access_times else 0
            }


class TTSResultCache:
    """LRU cache for frequently used TTS results"""
    
    def __init__(self, max_entries: int = 100, max_memory: int = 50 * 1024 * 1024):  # 50MB default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._memory_usage: Dict[str, int] = {}
        self._max_entries = max_entries
        self._max_memory = max_memory
        self._current_memory = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
    def _generate_cache_key(self, text: str, voice_id: str, settings: Dict[str, Any]) -> str:
        """Generate cache key from TTS parameters"""
        import hashlib
        key_data = f"{text}:{voice_id}:{sorted(settings.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, text: str, voice_id: str, settings: Dict[str, Any]) -> Optional[bytes]:
        """Get cached TTS result"""
        cache_key = self._generate_cache_key(text, voice_id, settings)
        
        with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                self._hits += 1
                logger.debug(f"Cache hit for TTS result {cache_key[:8]}...")
                return self._cache[cache_key]['data']
            
            self._misses += 1
            return None
    
    def put(self, text: str, voice_id: str, settings: Dict[str, Any], audio_data: bytes) -> bool:
        """Cache TTS result"""
        cache_key = self._generate_cache_key(text, voice_id, settings)
        data_size = len(audio_data)
        
        with self._lock:
            # Check if we need to evict entries
            while (len(self._cache) >= self._max_entries or 
                   self._current_memory + data_size > self._max_memory):
                if not self._access_order:
                    logger.warning("Cannot cache TTS result: limits exceeded")
                    return False
                
                # Remove least recently used
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    old_size = self._memory_usage[lru_key]
                    del self._cache[lru_key]
                    del self._memory_usage[lru_key]
                    self._current_memory -= old_size
                    logger.debug(f"Evicted cache entry {lru_key[:8]}... ({old_size} bytes)")
            
            # Add new entry
            self._cache[cache_key] = {
                'data': audio_data,
                'timestamp': time.time(),
                'access_count': 1
            }
            self._memory_usage[cache_key] = data_size
            self._access_order.append(cache_key)
            self._current_memory += data_size
            
            logger.debug(f"Cached TTS result {cache_key[:8]}... ({data_size} bytes)")
            return True
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_usage.clear()
            self._current_memory = 0
            logger.debug("Cleared TTS result cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "memory_usage": self._current_memory,
                "max_memory": self._max_memory,
                "memory_usage_percentage": (self._current_memory / self._max_memory) * 100,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }


class TTSMemoryManager:
    """Comprehensive memory management for TTS operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._max_total_memory = config.get('max_total_memory', 100 * 1024 * 1024)  # 100MB default
        self._memory_pressure_threshold = config.get('memory_pressure_threshold', 0.8)  # 80%
        self._gc_threshold = config.get('gc_threshold', 0.9)  # 90%
        self._oom_threshold = config.get('oom_threshold', 0.95)  # 95%
        
        # Initialize components
        self._audio_buffer = TTSAudioBuffer(config.get('max_buffer_size', 10 * 1024 * 1024))
        self._result_cache = TTSResultCache(
            config.get('max_cache_entries', 100),
            config.get('max_cache_memory', 50 * 1024 * 1024)
        )
        
        # Memory monitoring
        self._memory_stats = {
            'total_allocations': 0,
            'peak_usage': 0,
            'pressure_events': 0,
            'gc_events': 0,
            'oom_events': 0
        }
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start memory management and monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._memory_monitoring_loop())
        logger.info("TTSMemoryManager started")
    
    async def stop(self):
        """Stop memory management"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Clear all cached data
        self._result_cache.clear()
        logger.info("TTSMemoryManager stopped")
    
    def get_audio_buffer(self) -> TTSAudioBuffer:
        """Get the audio buffer manager"""
        return self._audio_buffer
    
    def get_result_cache(self) -> TTSResultCache:
        """Get the result cache manager"""
        return self._result_cache
    
    def get_current_memory_usage(self) -> int:
        """Get current total memory usage"""
        buffer_stats = self._audio_buffer.get_memory_stats()
        cache_stats = self._result_cache.get_cache_stats()
        return buffer_stats['current_size'] + cache_stats['memory_usage']
    
    def get_memory_pressure_level(self) -> float:
        """Get memory pressure level (0.0 to 1.0)"""
        current_usage = self.get_current_memory_usage()
        return current_usage / self._max_total_memory
    
    async def handle_memory_pressure(self):
        """Handle memory pressure situations"""
        pressure_level = self.get_memory_pressure_level()
        
        if pressure_level >= self._oom_threshold:
            # Critical: Clear everything
            self._memory_stats['oom_events'] += 1
            logger.critical(f"OOM threshold reached ({pressure_level:.1%}), clearing all memory")
            self._result_cache.clear()
            # Audio buffers are cleared by freeing space as needed
            
        elif pressure_level >= self._gc_threshold:
            # High: Aggressive cleanup
            self._memory_stats['gc_events'] += 1
            logger.warning(f"GC threshold reached ({pressure_level:.1%}), performing aggressive cleanup")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear half of cache entries
            cache_stats = self._result_cache.get_cache_stats()
            entries_to_clear = cache_stats['entries'] // 2
            for _ in range(entries_to_clear):
                if self._result_cache._access_order:
                    lru_key = self._result_cache._access_order.pop(0)
                    if lru_key in self._result_cache._cache:
                        size = self._result_cache._memory_usage[lru_key]
                        del self._result_cache._cache[lru_key]
                        del self._result_cache._memory_usage[lru_key]
                        self._result_cache._current_memory -= size
            
        elif pressure_level >= self._memory_pressure_threshold:
            # Medium: Normal pressure response
            self._memory_stats['pressure_events'] += 1
            logger.info(f"Memory pressure detected ({pressure_level:.1%}), performing cleanup")
            
            # Clear oldest 25% of cache
            cache_stats = self._result_cache.get_cache_stats()
            entries_to_clear = cache_stats['entries'] // 4
            for _ in range(entries_to_clear):
                if self._result_cache._access_order:
                    lru_key = self._result_cache._access_order.pop(0)
                    if lru_key in self._result_cache._cache:
                        size = self._result_cache._memory_usage[lru_key]
                        del self._result_cache._cache[lru_key]
                        del self._result_cache._memory_usage[lru_key]
                        self._result_cache._current_memory -= size
    
    async def _memory_monitoring_loop(self):
        """Background task to monitor memory usage"""
        while self._running:
            try:
                current_usage = self.get_current_memory_usage()
                self._memory_stats['peak_usage'] = max(self._memory_stats['peak_usage'], current_usage)
                
                # Check for memory pressure
                pressure_level = self.get_memory_pressure_level()
                if pressure_level >= self._memory_pressure_threshold:
                    await self.handle_memory_pressure()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(10)  # Longer delay on error
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        buffer_stats = self._audio_buffer.get_memory_stats()
        cache_stats = self._result_cache.get_cache_stats()
        current_usage = self.get_current_memory_usage()
        
        return {
            "total_memory_usage": current_usage,
            "max_total_memory": self._max_total_memory,
            "memory_usage_percentage": (current_usage / self._max_total_memory) * 100,
            "pressure_level": self.get_memory_pressure_level(),
            "audio_buffer": buffer_stats,
            "result_cache": cache_stats,
            "memory_events": self._memory_stats,
            "thresholds": {
                "pressure": self._memory_pressure_threshold,
                "gc": self._gc_threshold,
                "oom": self._oom_threshold
            }
        }


class TTSShutdownManager:
    """Manages graceful shutdown process for TTS operations"""
    
    def __init__(self, engine_name: str = "AsyncTTSEngine"):
        self.engine_name = engine_name
        self.state = TTSShutdownState.RUNNING
        self.shutdown_start_time: Optional[float] = None
        self.shutdown_reason: Optional[str] = None
        self.in_flight_requests: Dict[str, TTSRequest] = {}
        self.shutdown_callbacks: List[Callable] = []
        self.force_shutdown_event = asyncio.Event()
        self.shutdown_complete_event = asyncio.Event()
        self.shutdown_lock = asyncio.Lock()
        
        # Shutdown statistics
        self.shutdown_stats = {
            "total_shutdowns": 0,
            "graceful_shutdowns": 0,
            "forced_shutdowns": 0,
            "requests_drained": 0,
            "requests_lost": 0,
            "average_shutdown_time": 0.0
        }
    
    async def initiate_shutdown(self, reason: str = "manual", timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> bool:
        """
        Initiate graceful shutdown process
        
        Args:
            reason: Reason for shutdown (signal, manual, error, etc.)
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            bool: True if shutdown was successful, False if forced
        """
        async with self.shutdown_lock:
            if self.state != TTSShutdownState.RUNNING:
                logger.warning(f"Shutdown already in progress (state: {self.state.value})")
                return await self._wait_for_shutdown_complete(timeout)
            
            self.shutdown_start_time = time.time()
            self.shutdown_reason = reason
            self.state = TTSShutdownState.SHUTDOWN_INITIATED
            self.shutdown_stats["total_shutdowns"] += 1
            
            logger.info(f"Initiating graceful shutdown of {self.engine_name} (reason: {reason})")
            
            # Create shutdown timeout task
            timeout_task = asyncio.create_task(self._shutdown_timeout_handler(timeout))
            
            try:
                # Execute shutdown phases
                success = await self._execute_shutdown_phases(timeout)
                
                if success:
                    self.shutdown_stats["graceful_shutdowns"] += 1
                    logger.info(f"Graceful shutdown completed in {time.time() - self.shutdown_start_time:.2f}s")
                else:
                    self.shutdown_stats["forced_shutdowns"] += 1
                    logger.warning(f"Forced shutdown after {time.time() - self.shutdown_start_time:.2f}s")
                
                return success
                
            except Exception as e:
                logger.error(f"Shutdown failed: {e}")
                self.state = TTSShutdownState.FORCE_SHUTDOWN
                return False
            finally:
                timeout_task.cancel()
                self.shutdown_complete_event.set()
                self._update_shutdown_stats()
    
    async def _execute_shutdown_phases(self, timeout: float) -> bool:
        """Execute shutdown phases with timeout protection"""
        phase_timeout = timeout / 4  # Divide timeout among phases
        
        try:
            # Phase 1: Drain requests
            self.state = TTSShutdownState.DRAINING_REQUESTS
            logger.info("Phase 1: Draining in-flight requests...")
            if not await self._drain_requests(phase_timeout):
                return False
            
            # Phase 2: Stop workers
            self.state = TTSShutdownState.STOPPING_WORKERS
            logger.info("Phase 2: Stopping workers...")
            if not await self._stop_workers(phase_timeout):
                return False
            
            # Phase 3: Clean resources
            self.state = TTSShutdownState.CLEANING_RESOURCES
            logger.info("Phase 3: Cleaning resources...")
            if not await self._clean_resources(phase_timeout):
                return False
            
            # Phase 4: Complete shutdown
            self.state = TTSShutdownState.SHUTDOWN_COMPLETE
            logger.info("Phase 4: Shutdown complete")
            return True
            
        except asyncio.TimeoutError:
            logger.error("Shutdown phase timed out")
            return False
        except Exception as e:
            logger.error(f"Shutdown phase failed: {e}")
            return False
    
    async def _drain_requests(self, timeout: float) -> bool:
        """Drain in-flight requests with timeout"""
        try:
            start_time = time.time()
            drained_count = 0
            
            # Wait for in-flight requests to complete
            while self.in_flight_requests and (time.time() - start_time) < timeout:
                # Check request status and remove completed ones
                completed_requests = []
                for request_id, request in self.in_flight_requests.items():
                    if request.status in [TTSRequestStatus.COMPLETED, TTSRequestStatus.FAILED, TTSRequestStatus.CANCELLED]:
                        completed_requests.append(request_id)
                
                for request_id in completed_requests:
                    del self.in_flight_requests[request_id]
                    drained_count += 1
                
                if self.in_flight_requests:
                    await asyncio.sleep(0.1)  # Short delay before next check
            
            remaining_requests = len(self.in_flight_requests)
            if remaining_requests > 0:
                logger.warning(f"Forcing shutdown with {remaining_requests} requests still in-flight")
                
                # Save state for persistence
                await self.save_state_to_file()
                
                self.shutdown_stats["requests_lost"] += remaining_requests
                return False
            
            self.shutdown_stats["requests_drained"] += drained_count
            logger.info(f"Successfully drained {drained_count} requests")
            return True
            
        except Exception as e:
            logger.error(f"Request draining failed: {e}")
            return False
    
    async def _stop_workers(self, timeout: float) -> bool:
        """Stop worker processes with timeout"""
        # Call engine-specific worker shutdown if callback is registered
        for callback in self.shutdown_callbacks:
            if hasattr(callback, '__name__') and 'stop_workers' in callback.__name__:
                try:
                    return await callback(timeout)
                except Exception as e:
                    logger.error(f"Worker shutdown callback failed: {e}")
                    return False
        return True
    
    async def _clean_resources(self, timeout: float) -> bool:
        """Clean up resources with timeout"""
        # Call engine-specific resource cleanup if callback is registered
        for callback in self.shutdown_callbacks:
            if hasattr(callback, '__name__') and 'clean_resources' in callback.__name__:
                try:
                    return await callback(timeout)
                except Exception as e:
                    logger.error(f"Resource cleanup callback failed: {e}")
                    return False
        return True
    
    async def _shutdown_timeout_handler(self, timeout: float):
        """Handle shutdown timeout by forcing shutdown"""
        try:
            await asyncio.sleep(timeout)
            if self.state not in [TTSShutdownState.SHUTDOWN_COMPLETE, TTSShutdownState.FORCE_SHUTDOWN]:
                logger.critical(f"Shutdown timeout reached ({timeout}s), forcing shutdown")
                self.state = TTSShutdownState.FORCE_SHUTDOWN
                self.force_shutdown_event.set()
        except asyncio.CancelledError:
            pass  # Timeout was cancelled, shutdown completed normally
    
    async def _wait_for_shutdown_complete(self, timeout: float) -> bool:
        """Wait for shutdown to complete"""
        try:
            await asyncio.wait_for(self.shutdown_complete_event.wait(), timeout=timeout)
            return self.state == TTSShutdownState.SHUTDOWN_COMPLETE
        except asyncio.TimeoutError:
            return False
    
    def _update_shutdown_stats(self):
        """Update shutdown statistics"""
        if self.shutdown_start_time:
            shutdown_time = time.time() - self.shutdown_start_time
            total_shutdowns = self.shutdown_stats["total_shutdowns"]
            current_avg = self.shutdown_stats["average_shutdown_time"]
            
            # Update rolling average
            self.shutdown_stats["average_shutdown_time"] = (
                (current_avg * (total_shutdowns - 1) + shutdown_time) / total_shutdowns
            )
    
    def add_shutdown_callback(self, callback: Callable):
        """Add callback to be called during shutdown"""
        self.shutdown_callbacks.append(callback)
    
    def register_in_flight_request(self, request: TTSRequest):
        """Register a request as in-flight"""
        self.in_flight_requests[request.request_id] = request
    
    def unregister_in_flight_request(self, request_id: str):
        """Unregister a request from in-flight tracking"""
        self.in_flight_requests.pop(request_id, None)
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status"""
        return {
            "state": self.state.value,
            "shutdown_time": time.time() - self.shutdown_start_time if self.shutdown_start_time else 0,
            "shutdown_reason": self.shutdown_reason,
            "in_flight_requests": len(self.in_flight_requests),
            "statistics": self.shutdown_stats.copy()
        }
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        return self.state != TTSShutdownState.RUNNING
    
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete"""
        return self.state in [TTSShutdownState.SHUTDOWN_COMPLETE, TTSShutdownState.FORCE_SHUTDOWN]
    
    async def save_state_to_file(self, state_file: str = ".tts_shutdown_state.json"):
        """Save current in-flight requests to file for state persistence"""
        try:
            import json
            
            # Prepare state data
            state_data = {
                "timestamp": time.time(),
                "engine_name": self.engine_name,
                "shutdown_state": self.state.value,
                "shutdown_reason": self.shutdown_reason,
                "in_flight_requests": []
            }
            
            # Serialize in-flight requests
            for request_id, request in self.in_flight_requests.items():
                request_data = {
                    "request_id": request_id,
                    "text": request.text,
                    "config": request.config,
                    "priority": request.priority.value,
                    "status": request.status.value,
                    "created_at": request.created_at,
                    "context": request.context
                }
                state_data["in_flight_requests"].append(request_data)
            
            # Write to file
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Saved shutdown state to {state_file} ({len(self.in_flight_requests)} requests)")
            
        except Exception as e:
            logger.error(f"Failed to save shutdown state: {e}")
    
    async def load_state_from_file(self, state_file: str = ".tts_shutdown_state.json") -> List[TTSRequest]:
        """Load previously saved in-flight requests from file"""
        try:
            import json
            from pathlib import Path
            
            if not Path(state_file).exists():
                return []
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Reconstruct requests
            recovered_requests = []
            for request_data in state_data.get("in_flight_requests", []):
                request = TTSRequest(
                    text=request_data["text"],
                    config=request_data["config"],
                    priority=TTSPriority(request_data["priority"]),
                    context=request_data.get("context", {})
                )
                request.request_id = request_data["request_id"]
                request.status = TTSRequestStatus(request_data["status"])
                request.created_at = request_data["created_at"]
                recovered_requests.append(request)
            
            logger.info(f"Loaded {len(recovered_requests)} requests from {state_file}")
            
            # Clean up state file
            Path(state_file).unlink()
            
            return recovered_requests
            
        except Exception as e:
            logger.error(f"Failed to load shutdown state: {e}")
            return []


class AsyncTTSInterface(ABC):
    """
    Abstract interface for async TTS components
    """
    
    @abstractmethod
    async def start(self) -> None:
        """Start the async component"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the async component"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        pass


class AsyncTTSQueue(AsyncTTSInterface):
    """
    Thread-safe async queue manager for TTS requests with priority and concurrency control
    """
    
    def __init__(self, max_size: int = DEFAULT_QUEUE_SIZE):
        self.max_size = max_size
        self._queue: List[TTSRequest] = []
        self._active_requests: Dict[str, TTSRequest] = {}
        self._completed_requests: Dict[str, TTSRequest] = {}
        self._semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._priority_reorder_task: Optional[asyncio.Task] = None
        self._counters = ThreadSafeCounters()
        self._main_lock = threading.Lock()  # Main lock for queue operations
        
    async def start(self) -> None:
        """Start the queue and background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._priority_reorder_task = asyncio.create_task(self._priority_reorder_loop())
        safe_logger.info("AsyncTTSQueue started")
    
    async def stop(self) -> None:
        """Stop the queue and cleanup"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._priority_reorder_task:
            self._priority_reorder_task.cancel()
            try:
                await self._priority_reorder_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending requests - thread-safe
        with self._main_lock:
            for request in self._queue:
                request.set_status_safe(TTSRequestStatus.CANCELLED)
            self._queue.clear()
        
        safe_logger.info("AsyncTTSQueue stopped")
    
    async def enqueue(self, request: TTSRequest) -> bool:
        """
        Add request to queue with priority handling - thread-safe
        
        Returns:
            bool: True if enqueued successfully, False if queue full
        """
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                if len(self._queue) >= self.max_size:
                    # Remove oldest low-priority request if queue is full
                    if not self._make_room():
                        return False
                
                # Update metrics atomically
                self._counters.increment('total_requests')
                if request.priority == TTSPriority.HIGH and time.time() - request.created_at < DEFAULT_PRIORITY_BOOST_THRESHOLD:
                    self._counters.increment('priority_boosts')
                
                # Insert in priority order
                heapq.heappush(self._queue, request)
                
                safe_logger.debug(f"Enqueued request {request.request_id} with priority {request.priority}")
                return True
    
    async def dequeue(self) -> Optional[TTSRequest]:
        """
        Remove and return highest priority request - thread-safe
        
        Returns:
            TTSRequest or None if queue is empty
        """
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                if not self._queue:
                    return None
                
                request = heapq.heappop(self._queue)
                self._active_requests[request.request_id] = request
                
                safe_logger.debug(f"Dequeued request {request.request_id}")
                return request
    
    async def complete_request(self, request_id: str, success: bool = True) -> None:
        """Mark request as completed - thread-safe"""
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                if request_id in self._active_requests:
                    request = self._active_requests.pop(request_id)
                    final_status = TTSRequestStatus.COMPLETED if success else TTSRequestStatus.FAILED
                    request.set_status_safe(final_status)
                    
                    # Update metrics atomically
                    wait_time = time.time() - request.created_at
                    current_avg = self._counters.get('average_wait_time')
                    total_requests = self._counters.get('total_requests')
                    if total_requests > 0:
                        new_avg = ((current_avg * (total_requests - 1) + wait_time) / total_requests)
                        self._counters.set('average_wait_time', new_avg)
                    else:
                        self._counters.set('average_wait_time', wait_time)
                    
                    self._completed_requests[request_id] = request
                    self._semaphore.release()
    
    def _make_room(self) -> bool:
        """Remove oldest low-priority request to make room - thread-safe"""
        if not self._queue:
            return False
            
        # Find oldest low-priority request
        oldest_idx = -1
        oldest_time = float('inf')
        
        for i, request in enumerate(self._queue):
            if request.priority == TTSPriority.LOW and request.created_at < oldest_time:
                oldest_idx = i
                oldest_time = request.created_at
        
        if oldest_idx >= 0:
            removed = self._queue.pop(oldest_idx)
            removed.set_status_safe(TTSRequestStatus.CANCELLED)
            heapq.heapify(self._queue)  # Restore heap property
            safe_logger.debug(f"Removed old request {removed.request_id} to make room")
            return True
        
        return False
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired requests"""
        while True:
            try:
                await asyncio.sleep(DEFAULT_CLEANUP_INTERVAL)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired requests - thread-safe"""
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                # Clean completed requests
                expired_ids = [
                    req_id for req_id, request in self._completed_requests.items()
                    if request.is_expired()
                ]
                for req_id in expired_ids:
                    del self._completed_requests[req_id]
                
                # Clean expired pending requests
            self._queue = [req for req in self._queue if not req.is_expired()]
            heapq.heapify(self._queue)
            
            if expired_ids:
                logger.debug(f"Cleaned up {len(expired_ids)} expired requests")
    
    async def _priority_reorder_loop(self) -> None:
        """Background task to reorder queue based on updated priorities"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._reorder_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in priority reorder loop: {e}")
    
    async def _reorder_queue(self) -> None:
        """Reorder queue based on updated priority scores - thread-safe"""
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                if len(self._queue) <= 1:
                    return
                
                # Update priority scores for all requests
                starvation_prevented = 0
                for request in self._queue:
                    old_score = request._priority_score
                    request.update_priority_score()
                    
                    # Check if starvation prevention was triggered
                    if request._priority_score - old_score > 500:  # Significant boost
                        starvation_prevented += 1
                
                # Reorder queue if priorities have changed significantly
                if starvation_prevented > 0:
                    heapq.heapify(self._queue)
                    self._priority_metrics["starvation_prevented"] += starvation_prevented
                    self._priority_metrics["reorders_performed"] += 1
                    safe_logger.debug(f"Reordered queue: prevented {starvation_prevented} starvation cases")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return queue health status - thread-safe"""
        async with global_lock_manager.acquire_locks('queue'):
            with self._main_lock:
                # Calculate priority distribution
                priority_distribution = {p.name: 0 for p in TTSPriority}
                for request in self._queue:
                    priority_distribution[request.priority.name] += 1
                
                # Get priority metrics for active requests
                active_priorities = [
                    req.get_priority_metrics() for req in self._active_requests.values()
                ]
                
                return {
                    "queue_size": len(self._queue),
                    "max_size": self.max_size,
                    "active_requests": len(self._active_requests),
                    "completed_requests": len(self._completed_requests),
                    "semaphore_value": self._semaphore._value,
                    "is_healthy": len(self._queue) < self.max_size * 0.8,
                    "priority_distribution": priority_distribution,
                    "priority_metrics": self._priority_metrics.copy(),
                    "active_request_priorities": active_priorities
                }


class AsyncTTSWorker(AsyncTTSInterface):
    """
    Background worker for processing TTS requests with ThreadPoolExecutor infrastructure
    """
    
    def __init__(self, worker_id: str, queue: AsyncTTSQueue, tts_client, max_thread_workers: int = 2, resource_manager: Optional[TTSResourceManager] = None, memory_manager: Optional[TTSMemoryManager] = None, engine_ref: Optional['AsyncTTSEngine'] = None):
        self.worker_id = worker_id
        self.queue = queue
        self.tts_client = tts_client
        self.max_thread_workers = max_thread_workers
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.engine_ref = engine_ref
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._last_error_time = 0.0
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._thread_pool_resource_id: Optional[str] = None
        self._thread_health_monitor: Optional[asyncio.Task] = None
        self._restart_count = 0
        self._last_restart_time = 0.0
        self._thread_status: Dict[str, Dict[str, Any]] = {}
        self._work_distribution_stats = {
            "total_distributed": 0,
            "pending_work": 0,
            "active_threads": 0,
            "thread_utilization": 0.0
        }
        
    async def start(self) -> None:
        """Start the worker with thread pool infrastructure"""
        if self._running:
            return
            
        self._running = True
        
        # Initialize thread pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.max_thread_workers,
            thread_name_prefix=f"tts-worker-{self.worker_id}"
        )
        
        # Register thread pool with resource manager
        if self.resource_manager:
            resource = TTSResource(
                resource_id=f"thread-pool-{self.worker_id}",
                resource_type=TTSResourceType.THREAD_POOL,
                resource_ref=self._thread_pool,
                cleanup_handler=lambda tp: tp.shutdown(wait=False, cancel_futures=True)
            )
            self._thread_pool_resource_id = self.resource_manager.register_resource(resource)
        
        # Start main worker loop
        self._task = asyncio.create_task(self._worker_loop())
        
        # Start thread health monitoring
        self._thread_health_monitor = asyncio.create_task(self._thread_health_monitor_loop())
        
        logger.info(f"Worker {self.worker_id} started with {self.max_thread_workers} thread workers")
    
    async def stop(self) -> None:
        """Stop the worker with graceful thread shutdown"""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel health monitoring
        if self._thread_health_monitor:
            self._thread_health_monitor.cancel()
            try:
                await self._thread_health_monitor
            except asyncio.CancelledError:
                pass
        
        # Cancel main task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Unregister and shutdown thread pool
        if self._thread_pool_resource_id and self.resource_manager:
            self.resource_manager.unregister_resource(self._thread_pool_resource_id)
            self._thread_pool_resource_id = None
        elif self._thread_pool:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            self._thread_pool = None
        
        logger.info(f"Worker {self.worker_id} stopped (restarts: {self._restart_count})")
    
    async def _worker_loop(self) -> None:
        """Main worker processing loop with enhanced error handling"""
        while self._running:
            try:
                # Check if we need to restart due to excessive errors
                if self._consecutive_errors >= 5:
                    await self._restart_worker()
                    continue
                
                request = await self.queue.dequeue()
                if request:
                    await self._process_request(request)
                    self._consecutive_errors = 0  # Reset on success
                else:
                    await asyncio.sleep(0.1)  # Brief pause if no requests
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self._error_count += 1
                self._consecutive_errors += 1
                self._last_error_time = time.time()
                await asyncio.sleep(min(2 ** self._consecutive_errors, 10))  # Exponential backoff
    
    async def _process_request(self, request: TTSRequest) -> None:
        """Process a single TTS request with enhanced thread management"""
        if not self._thread_pool:
            raise RuntimeError("Thread pool not initialized")
        
        thread_id = None
        
        try:
            request.status = TTSRequestStatus.PROCESSING
            
            # Update work distribution stats
            self._work_distribution_stats["total_distributed"] += 1
            self._work_distribution_stats["pending_work"] += 1
            
            # Submit to thread pool with timeout
            loop = asyncio.get_event_loop()
            future = self._thread_pool.submit(self._synthesize_tts, request)
            
            # Track thread assignment
            thread_id = id(future)
            self._thread_status[str(thread_id)] = {
                "request_id": request.request_id,
                "start_time": time.time(),
                "status": "running"
            }
            
            # Wait for completion with timeout
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                future.cancel()
                raise TimeoutError(f"TTS request {request.request_id} timed out")
            finally:
                # Update thread status
                if thread_id and str(thread_id) in self._thread_status:
                    self._thread_status[str(thread_id)]["status"] = "completed"
                    self._thread_status[str(thread_id)]["end_time"] = time.time()
                
                self._work_distribution_stats["pending_work"] -= 1
            
            # Use engine's completion handler if available
            if self.engine_ref:
                await self.engine_ref._handle_request_completion(request.request_id, success=True)
            else:
                await self.queue.complete_request(request.request_id, success=True)
            self._processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process request {request.request_id}: {e}")
            
            # Update thread status on error
            if thread_id and str(thread_id) in self._thread_status:
                self._thread_status[str(thread_id)]["status"] = "failed"
                self._thread_status[str(thread_id)]["error"] = str(e)
            
            if request.can_retry():
                request.mark_retry()
                await self.queue.enqueue(request)
            else:
                # Use engine's completion handler if available
                if self.engine_ref:
                    await self.engine_ref._handle_request_completion(request.request_id, success=False)
                else:
                    await self.queue.complete_request(request.request_id, success=False)
            
            self._error_count += 1
            self._consecutive_errors += 1
    
    def _synthesize_tts(self, request: TTSRequest) -> None:
        """Synchronous TTS synthesis with thread-safe execution and memory management"""
        try:
            # Set thread name for monitoring
            current_thread = threading.current_thread()
            original_name = current_thread.name
            current_thread.name = f"tts-{self.worker_id}-{request.request_id[:8]}"
            
            try:
                # Check for cached audio in request context
                if "cached_audio" in request.context:
                    logger.debug(f"Using cached audio for request {request.request_id}")
                    # Use cached audio - we would play it here
                    # For now, just log and continue to avoid breaking existing functionality
                    pass
                
                # Extract TTS parameters
                voice_id = request.config.get('voice_id', '6HWqrqOzDfj3UnywjJoZ')
                tts_settings = {k: v for k, v in request.config.items() if k != 'voice_id'}
                
                # Check cache if we have memory manager
                audio_data = None
                if self.memory_manager:
                    audio_data = self.memory_manager.get_result_cache().get(request.text, voice_id, tts_settings)
                
                if audio_data:
                    # Cache hit - use cached audio
                    logger.debug(f"Cache hit for request {request.request_id}")
                    # Store in audio buffer for playback
                    if self.memory_manager:
                        self.memory_manager.get_audio_buffer().store_audio(
                            request.request_id, 
                            audio_data, 
                            {"voice_id": voice_id, "timestamp": time.time()}
                        )
                else:
                    # Cache miss - synthesize and cache
                    logger.debug(f"Cache miss for request {request.request_id}, synthesizing...")
                    
                    # Import here to avoid circular imports
                    from .elevenlabs_tts import speak_text
                    
                    # Synthesize TTS
                    speak_text(request.text, voice_id)
                    
                    # TODO: Get the actual audio data from speak_text and cache it
                    # For now, we'll create a placeholder to maintain the caching structure
                    if self.memory_manager:
                        # This is a placeholder - in a real implementation, we'd get the audio data
                        placeholder_audio = b"placeholder_audio_data"
                        self.memory_manager.get_result_cache().put(request.text, voice_id, tts_settings, placeholder_audio)
                        
                        # Also store in audio buffer
                        self.memory_manager.get_audio_buffer().store_audio(
                            request.request_id, 
                            placeholder_audio, 
                            {"voice_id": voice_id, "timestamp": time.time()}
                        )
                
            finally:
                # Restore original thread name
                current_thread.name = original_name
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    async def _thread_health_monitor_loop(self) -> None:
        """Monitor thread health and performance"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._check_thread_health()
                await self._update_work_distribution_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Thread health monitor error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _check_thread_health(self) -> None:
        """Check thread pool health and identify issues"""
        if not self._thread_pool:
            return
        
        current_time = time.time()
        
        # Check for stuck threads
        stuck_threads = []
        for thread_id, status in self._thread_status.items():
            if status["status"] == "running":
                run_time = current_time - status["start_time"]
                if run_time > 60:  # 1 minute threshold
                    stuck_threads.append((thread_id, run_time))
        
        if stuck_threads:
            logger.warning(f"Worker {self.worker_id} has {len(stuck_threads)} stuck threads")
            
            # If too many stuck threads, restart worker
            if len(stuck_threads) >= self.max_thread_workers:
                logger.error(f"Worker {self.worker_id} has all threads stuck, restarting...")
                await self._restart_worker()
        
        # Cleanup old thread status entries
        cutoff_time = current_time - 300  # 5 minutes
        old_entries = [
            tid for tid, status in self._thread_status.items()
            if status.get("end_time", current_time) < cutoff_time
        ]
        
        for tid in old_entries:
            del self._thread_status[tid]
    
    async def _update_work_distribution_stats(self) -> None:
        """Update work distribution statistics"""
        if not self._thread_pool:
            return
        
        # Calculate thread utilization
        active_threads = sum(1 for status in self._thread_status.values() if status["status"] == "running")
        self._work_distribution_stats["active_threads"] = active_threads
        self._work_distribution_stats["thread_utilization"] = active_threads / self.max_thread_workers
    
    async def _restart_worker(self) -> None:
        """Restart worker after failures"""
        current_time = time.time()
        
        # Rate limit restarts
        if current_time - self._last_restart_time < 60:  # 1 minute cooldown
            logger.warning(f"Worker {self.worker_id} restart rate limited")
            return
        
        logger.info(f"Restarting worker {self.worker_id} after {self._consecutive_errors} consecutive errors")
        
        # Shutdown current thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
        
        # Create new thread pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.max_thread_workers,
            thread_name_prefix=f"tts-worker-{self.worker_id}-restart-{self._restart_count}"
        )
        
        # Reset counters
        self._consecutive_errors = 0
        self._restart_count += 1
        self._last_restart_time = current_time
        self._thread_status.clear()
        
        logger.info(f"Worker {self.worker_id} restarted successfully")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return comprehensive worker health status"""
        current_time = time.time()
        
        # Thread pool status
        thread_pool_status = {
            "initialized": self._thread_pool is not None,
            "max_workers": self.max_thread_workers,
            "active_threads": self._work_distribution_stats["active_threads"],
            "utilization": self._work_distribution_stats["thread_utilization"],
            "pending_work": self._work_distribution_stats["pending_work"]
        }
        
        # Error analysis
        error_analysis = {
            "total_errors": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "error_rate": self._error_count / max(self._processed_count, 1),
            "last_error_age": current_time - self._last_error_time if self._last_error_time > 0 else None,
            "restart_count": self._restart_count,
            "last_restart_age": current_time - self._last_restart_time if self._last_restart_time > 0 else None
        }
        
        # Thread health
        thread_health = {
            "total_threads": len(self._thread_status),
            "running_threads": sum(1 for s in self._thread_status.values() if s["status"] == "running"),
            "completed_threads": sum(1 for s in self._thread_status.values() if s["status"] == "completed"),
            "failed_threads": sum(1 for s in self._thread_status.values() if s["status"] == "failed"),
            "stuck_threads": sum(1 for s in self._thread_status.values() 
                               if s["status"] == "running" and current_time - s["start_time"] > 60)
        }
        
        return {
            "worker_id": self.worker_id,
            "is_running": self._running,
            "processed_count": self._processed_count,
            "thread_pool_status": thread_pool_status,
            "error_analysis": error_analysis,
            "thread_health": thread_health,
            "work_distribution_stats": self._work_distribution_stats.copy(),
            "is_healthy": (
                self._running and 
                self._consecutive_errors < 3 and
                thread_health["stuck_threads"] == 0 and
                (self._thread_pool is not None)
            )
        }


@asynccontextmanager
async def async_tts_context(max_workers: int = DEFAULT_WORKER_THREADS):
    """
    Async context manager for TTS engine lifecycle
    
    Usage:
        async with async_tts_context() as engine:
            await engine.speak_async("Hello", config)
    """
    engine = AsyncTTSEngine(max_workers=max_workers)
    try:
        await engine.start()
        yield engine
    finally:
        await engine.shutdown()


class TTSPerformanceMetrics:
    """Thread-safe performance metrics collection for TTS operations"""
    
    def __init__(self):
        self._counters = ThreadSafeCounters()
        self._samples_lock = threading.Lock()
        self._samples = {
            'queue_depth_samples': [],
            'queue_latency_samples': [],
            'throughput_samples': [],
            'bottleneck_events': [],
            'performance_alerts': []
        }
        self._request_start_times: Dict[str, float] = {}
        self._request_times_lock = threading.Lock()
        self._last_throughput_check = time.time()
        self._throughput_window = 60.0  # 1 minute window
        
        # Initialize default counter values
        self._counters.set('min_execution_time', float('inf'))
        
    def record_request_start(self, request_id: str):
        """Record when a request started processing - thread-safe"""
        current_time = time.time()
        
        with self._request_times_lock:
            self._request_start_times[request_id] = current_time
        
        # Update counters atomically
        self._counters.increment('total_requests')
        concurrent = self._counters.increment('concurrent_requests')
        self._counters.atomic_max_update('peak_concurrent_requests', concurrent)
    
    def record_request_completion(self, request_id: str, success: bool = True):
        """Record when a request completed - thread-safe"""
        current_time = time.time()
        
        with self._request_times_lock:
            if request_id not in self._request_start_times:
                return
            execution_time = current_time - self._request_start_times[request_id]
            del self._request_start_times[request_id]
        
        # Update execution time metrics atomically
        self._counters.increment('total_execution_time', execution_time)
        current_min = self._counters.get('min_execution_time')
        if execution_time < current_min:
            self._counters.set('min_execution_time', execution_time)
        self._counters.atomic_max_update('max_execution_time', execution_time)
        
        # Update success/failure counts
        if success:
            self._counters.increment('successful_requests')
        else:
            self._counters.increment('failed_requests')
        
        # Update average execution time
        completed_requests = self._counters.get('successful_requests') + self._counters.get('failed_requests')
        if completed_requests > 0:
            total_time = self._counters.get('total_execution_time')
            self._counters.set('average_execution_time', total_time / completed_requests)
        
        self._counters.decrement('concurrent_requests')
    
    def record_queue_depth(self, depth: int):
        """Record current queue depth"""
        with self._lock:
            self._metrics['queue_depth_samples'].append({
                'timestamp': time.time(),
                'depth': depth
            })
            # Keep only last 100 samples
            if len(self._metrics['queue_depth_samples']) > 100:
                self._metrics['queue_depth_samples'].pop(0)
    
    def record_queue_latency(self, latency: float):
        """Record queue latency (time from enqueue to dequeue)"""
        with self._lock:
            self._metrics['queue_latency_samples'].append({
                'timestamp': time.time(),
                'latency': latency
            })
            # Keep only last 100 samples
            if len(self._metrics['queue_latency_samples']) > 100:
                self._metrics['queue_latency_samples'].pop(0)
    
    def record_throughput(self):
        """Record current throughput (requests per second)"""
        current_time = time.time()
        with self._lock:
            time_diff = current_time - self._last_throughput_check
            if time_diff >= self._throughput_window:
                requests_in_window = self._metrics['successful_requests'] + self._metrics['failed_requests']
                throughput = requests_in_window / time_diff if time_diff > 0 else 0
                
                self._metrics['throughput_samples'].append({
                    'timestamp': current_time,
                    'throughput': throughput
                })
                
                # Keep only last 60 samples (1 hour at 1-minute intervals)
                if len(self._metrics['throughput_samples']) > 60:
                    self._metrics['throughput_samples'].pop(0)
                
                self._last_throughput_check = current_time
    
    def record_bottleneck_event(self, event_type: str, details: Dict[str, Any]):
        """Record a performance bottleneck event"""
        with self._lock:
            self._metrics['bottleneck_events'].append({
                'timestamp': time.time(),
                'type': event_type,
                'details': details
            })
            # Keep only last 50 events
            if len(self._metrics['bottleneck_events']) > 50:
                self._metrics['bottleneck_events'].pop(0)
    
    def record_performance_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Record a performance alert"""
        with self._lock:
            self._metrics['performance_alerts'].append({
                'timestamp': time.time(),
                'type': alert_type,
                'message': message,
                'severity': severity
            })
            # Keep only last 25 alerts
            if len(self._metrics['performance_alerts']) > 25:
                self._metrics['performance_alerts'].pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            if metrics['total_requests'] > 0:
                metrics['success_rate'] = (
                    metrics['successful_requests'] / metrics['total_requests'] * 100
                )
                metrics['failure_rate'] = (
                    metrics['failed_requests'] / metrics['total_requests'] * 100
                )
            else:
                metrics['success_rate'] = 0.0
                metrics['failure_rate'] = 0.0
            
            # Calculate queue statistics
            if metrics['queue_depth_samples']:
                depths = [s['depth'] for s in metrics['queue_depth_samples']]
                metrics['average_queue_depth'] = sum(depths) / len(depths)
                metrics['max_queue_depth'] = max(depths)
            else:
                metrics['average_queue_depth'] = 0.0
                metrics['max_queue_depth'] = 0
            
            # Calculate latency statistics
            if metrics['queue_latency_samples']:
                latencies = [s['latency'] for s in metrics['queue_latency_samples']]
                metrics['average_queue_latency'] = sum(latencies) / len(latencies)
                metrics['max_queue_latency'] = max(latencies)
            else:
                metrics['average_queue_latency'] = 0.0
                metrics['max_queue_latency'] = 0.0
            
            # Calculate current throughput
            if metrics['throughput_samples']:
                recent_throughput = metrics['throughput_samples'][-1]['throughput']
                metrics['current_throughput'] = recent_throughput
            else:
                metrics['current_throughput'] = 0.0
            
            return metrics
    
    def reset_metrics(self):
        """Reset all metrics to initial state"""
        with self._lock:
            self._metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': 0.0,
                'total_execution_time': 0.0,
                'queue_depth_samples': [],
                'queue_latency_samples': [],
                'concurrent_requests': 0,
                'peak_concurrent_requests': 0,
                'throughput_samples': [],
                'bottleneck_events': [],
                'performance_alerts': []
            }
            self._request_start_times.clear()
            self._last_throughput_check = time.time()


class TTSPerformanceMonitor:
    """Comprehensive performance monitoring and alerting system"""
    
    def __init__(self, metrics: TTSPerformanceMetrics, config: Dict[str, Any]):
        self.metrics = metrics
        self.config = config
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance thresholds
        self.thresholds = {
            'max_execution_time': config.get('max_execution_time', 30.0),
            'max_queue_depth': config.get('max_queue_depth', 20),
            'max_queue_latency': config.get('max_queue_latency', 10.0),
            'min_throughput': config.get('min_throughput', 1.0),
            'max_concurrent_requests': config.get('max_concurrent_requests', 10),
            'max_failure_rate': config.get('max_failure_rate', 10.0),
            'bottleneck_threshold': config.get('bottleneck_threshold', 5.0)
        }
        
        # Dashboard configuration
        self.dashboard_config = {
            'log_interval': config.get('log_interval', 60),  # seconds
            'alert_cooldown': config.get('alert_cooldown', 300),  # 5 minutes
            'performance_log_file': config.get('performance_log_file', '.tts_performance.log')
        }
        
        self._last_alert_times: Dict[str, float] = {}
        self._performance_history: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start performance monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("TTSPerformanceMonitor started")
        
    async def stop(self):
        """Stop performance monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("TTSPerformanceMonitor stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_performance_metrics()
                await self._update_dashboard()
                await asyncio.sleep(self.dashboard_config['log_interval'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_performance_metrics(self):
        """Check performance metrics against thresholds"""
        current_metrics = self.metrics.get_metrics()
        
        # Check execution time
        if current_metrics['max_execution_time'] > self.thresholds['max_execution_time']:
            await self._trigger_alert(
                'execution_time',
                f"High execution time detected: {current_metrics['max_execution_time']:.2f}s",
                'warning'
            )
        
        # Check queue depth
        if current_metrics.get('max_queue_depth', 0) > self.thresholds['max_queue_depth']:
            await self._trigger_alert(
                'queue_depth',
                f"High queue depth: {current_metrics['max_queue_depth']}",
                'warning'
            )
            
            # Record bottleneck event
            self.metrics.record_bottleneck_event('high_queue_depth', {
                'queue_depth': current_metrics['max_queue_depth'],
                'threshold': self.thresholds['max_queue_depth']
            })
        
        # Check queue latency
        if current_metrics.get('max_queue_latency', 0) > self.thresholds['max_queue_latency']:
            await self._trigger_alert(
                'queue_latency',
                f"High queue latency: {current_metrics['max_queue_latency']:.2f}s",
                'warning'
            )
        
        # Check throughput
        if current_metrics.get('current_throughput', 0) < self.thresholds['min_throughput']:
            await self._trigger_alert(
                'low_throughput',
                f"Low throughput: {current_metrics['current_throughput']:.2f} req/s",
                'warning'
            )
        
        # Check failure rate
        if current_metrics.get('failure_rate', 0) > self.thresholds['max_failure_rate']:
            await self._trigger_alert(
                'high_failure_rate',
                f"High failure rate: {current_metrics['failure_rate']:.1f}%",
                'error'
            )
        
        # Check concurrent requests
        if current_metrics['concurrent_requests'] > self.thresholds['max_concurrent_requests']:
            await self._trigger_alert(
                'high_concurrency',
                f"High concurrent requests: {current_metrics['concurrent_requests']}",
                'warning'
            )
        
        # Detect bottlenecks
        await self._detect_bottlenecks(current_metrics)
    
    async def _detect_bottlenecks(self, metrics: Dict[str, Any]):
        """Detect performance bottlenecks"""
        # Check if average execution time is increasing
        if len(self._performance_history) >= 3:
            recent_avg_times = [
                h.get('average_execution_time', 0) 
                for h in self._performance_history[-3:]
            ]
            if all(recent_avg_times[i] < recent_avg_times[i+1] for i in range(len(recent_avg_times)-1)):
                self.metrics.record_bottleneck_event('increasing_execution_time', {
                    'trend': recent_avg_times,
                    'current_avg': metrics['average_execution_time']
                })
        
        # Check for queue buildup
        if metrics.get('average_queue_depth', 0) > self.thresholds['bottleneck_threshold']:
            if metrics['concurrent_requests'] < self.thresholds['max_concurrent_requests']:
                self.metrics.record_bottleneck_event('queue_buildup', {
                    'queue_depth': metrics['average_queue_depth'],
                    'concurrent_requests': metrics['concurrent_requests']
                })
    
    async def _trigger_alert(self, alert_type: str, message: str, severity: str):
        """Trigger performance alert with cooldown"""
        current_time = time.time()
        last_alert_time = self._last_alert_times.get(alert_type, 0)
        
        if current_time - last_alert_time >= self.dashboard_config['alert_cooldown']:
            self.metrics.record_performance_alert(alert_type, message, severity)
            self._last_alert_times[alert_type] = current_time
            
            # Log alert
            log_level = logging.ERROR if severity == 'error' else logging.WARNING
            logger.log(log_level, f"TTS Performance Alert ({alert_type}): {message}")
    
    async def _update_dashboard(self):
        """Update performance dashboard and logs"""
        current_metrics = self.metrics.get_metrics()
        
        # Add to history
        self._performance_history.append({
            'timestamp': time.time(),
            **current_metrics
        })
        
        # Keep only last 60 entries (1 hour at 1-minute intervals)
        if len(self._performance_history) > 60:
            self._performance_history.pop(0)
        
        # Log performance summary
        logger.info(
            f"TTS Performance: {current_metrics['total_requests']} requests, "
            f"{current_metrics['success_rate']:.1f}% success, "
            f"{current_metrics['average_execution_time']:.2f}s avg time, "
            f"{current_metrics.get('current_throughput', 0):.2f} req/s throughput"
        )
        
        # Write to performance log file
        await self._write_performance_log(current_metrics)
    
    async def _write_performance_log(self, metrics: Dict[str, Any]):
        """Write performance metrics to log file"""
        try:
            log_entry = {
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics
            }
            
            log_file = self.dashboard_config['performance_log_file']
            with open(log_file, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")
                
        except Exception as e:
            logger.error(f"Failed to write performance log: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        current_metrics = self.metrics.get_metrics()
        
        return {
            'current_metrics': current_metrics,
            'performance_history': self._performance_history[-20:],  # Last 20 entries
            'thresholds': self.thresholds,
            'recent_alerts': current_metrics.get('performance_alerts', [])[-10:],  # Last 10 alerts
            'bottleneck_events': current_metrics.get('bottleneck_events', [])[-10:],  # Last 10 events
            'health_status': self._calculate_health_status(current_metrics)
        }
    
    def _calculate_health_status(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall health status"""
        issues = []
        
        if metrics.get('failure_rate', 0) > self.thresholds['max_failure_rate']:
            issues.append('high_failure_rate')
        
        if metrics.get('max_queue_depth', 0) > self.thresholds['max_queue_depth']:
            issues.append('high_queue_depth')
        
        if metrics.get('current_throughput', 0) < self.thresholds['min_throughput']:
            issues.append('low_throughput')
        
        if metrics.get('max_execution_time', 0) > self.thresholds['max_execution_time']:
            issues.append('high_execution_time')
        
        if len(issues) == 0:
            return 'healthy'
        elif len(issues) <= 2:
            return 'warning'
        else:
            return 'critical'


class AsyncTTSEngine:
    """
    Main asynchronous TTS execution engine
    
    Coordinates queue management, worker threads, and resource cleanup
    for non-blocking TTS operations.
    """
    
    def __init__(self, max_workers: int = DEFAULT_WORKER_THREADS, memory_config: Optional[Dict[str, Any]] = None, performance_config: Optional[Dict[str, Any]] = None):
        self.max_workers = max_workers
        self.queue = AsyncTTSQueue()
        self.workers: List[AsyncTTSWorker] = []
        self.tts_client = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._memory_monitor_task: Optional[asyncio.Task] = None
        self.resource_manager = TTSResourceManager()
        
        # Initialize memory management
        self._memory_config = memory_config or {
            'max_total_memory': 100 * 1024 * 1024,  # 100MB
            'max_buffer_size': 10 * 1024 * 1024,    # 10MB
            'max_cache_entries': 100,
            'max_cache_memory': 50 * 1024 * 1024,   # 50MB
            'memory_pressure_threshold': 0.8,
            'gc_threshold': 0.9,
            'oom_threshold': 0.95
        }
        self.memory_manager = TTSMemoryManager(self._memory_config)
        
        # Initialize performance monitoring
        self._performance_config = performance_config or {
            'max_execution_time': 30.0,
            'max_queue_depth': 20,
            'max_queue_latency': 10.0,
            'min_throughput': 1.0,
            'max_concurrent_requests': 10,
            'max_failure_rate': 10.0,
            'bottleneck_threshold': 5.0,
            'log_interval': 60,
            'alert_cooldown': 300,
            'performance_log_file': '.tts_performance.log'
        }
        self.performance_metrics = TTSPerformanceMetrics()
        self.performance_monitor = TTSPerformanceMonitor(self.performance_metrics, self._performance_config)
        
        # Initialize shutdown manager
        self.shutdown_manager = TTSShutdownManager("AsyncTTSEngine")
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            signal_names = {
                signal.SIGTERM: "SIGTERM",
                signal.SIGINT: "SIGINT"
            }
            signal_name = signal_names.get(signum, str(signum))
            logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
            
            # Use shutdown manager for graceful shutdown
            asyncio.create_task(self.shutdown_manager.initiate_shutdown(
                reason=f"signal_{signal_name.lower()}",
                timeout=DEFAULT_SHUTDOWN_TIMEOUT
            ))
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.debug("Signal handlers registered for SIGTERM and SIGINT")
        except ValueError as e:
            # Signal handlers can only be set in the main thread
            logger.warning(f"Could not set signal handlers: {e}")
        except Exception as e:
            logger.error(f"Error setting up signal handlers: {e}")
    
    async def start(self) -> None:
        """Start the TTS engine"""
        if self._running:
            return
        
        logger.info("Starting AsyncTTSEngine...")
        
        # Start resource manager
        await self.resource_manager.start()
        
        # Start memory management
        await self.memory_manager.start()
        
        # Start performance monitoring
        await self.performance_monitor.start()
        
        # Register shutdown callbacks
        self.shutdown_manager.add_shutdown_callback(self._shutdown_phase_stop_workers)
        self.shutdown_manager.add_shutdown_callback(self._shutdown_phase_clean_resources)
        
        # Initialize TTS client
        await self._initialize_tts_client()
        
        # Start queue
        await self.queue.start()
        
        # Start workers with enhanced thread pool support
        for i in range(self.max_workers):
            worker = AsyncTTSWorker(
                f"worker-{i}", 
                self.queue, 
                self.tts_client,
                max_thread_workers=2,  # Each worker gets 2 threads
                resource_manager=self.resource_manager,
                memory_manager=self.memory_manager,
                engine_ref=self
            )
            await worker.start()
            self.workers.append(worker)
        
        # Start memory monitoring
        self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
        
        # Recover from previous shutdown if applicable
        await self.recover_from_previous_shutdown()
        
        self._running = True
        logger.info(f"AsyncTTSEngine started with {len(self.workers)} workers")
    
    async def speak_async(self, text: str, config: Dict[str, Any], priority: TTSPriority = TTSPriority.NORMAL) -> str:
        """
        Asynchronously queue text for TTS synthesis
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            priority: Request priority
            
        Returns:
            str: Request ID for tracking
        """
        if not self._running:
            raise RuntimeError("TTS engine not running")
        
        # Check cache first
        voice_id = config.get('voice_id', 'default')
        tts_settings = {k: v for k, v in config.items() if k != 'voice_id'}
        
        cached_audio = self.memory_manager.get_result_cache().get(text, voice_id, tts_settings)
        if cached_audio:
            # Cache hit - play immediately without queuing
            logger.debug(f"Cache hit for text: {text[:50]}...")
            # TODO: Implement direct audio playback for cached results
            # For now, we'll still queue but with a cache flag
            request = TTSRequest(
                text=text,
                config=config,
                priority=priority,
                context={"source": "async_engine", "cached_audio": cached_audio}
            )
        else:
            # Cache miss - normal request
            request = TTSRequest(
                text=text,
                config=config,
                priority=priority,
                context={"source": "async_engine"}
            )
        
        # Record performance metrics
        self.performance_metrics.record_request_start(request.request_id)
        
        # Record queue metrics
        queue_depth = len(self.queue._queue)
        self.performance_metrics.record_queue_depth(queue_depth)
        
        # Register request for shutdown tracking
        self.shutdown_manager.register_in_flight_request(request)
        
        success = await self.queue.enqueue(request)
        if not success:
            # Remove from tracking if enqueue failed
            self.shutdown_manager.unregister_in_flight_request(request.request_id)
            self.performance_metrics.record_request_completion(request.request_id, success=False)
            raise RuntimeError("TTS queue is full")
        
        return request.request_id
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return self.memory_manager.get_memory_stats()
    
    def get_audio_buffer(self) -> TTSAudioBuffer:
        """Get access to audio buffer manager"""
        return self.memory_manager.get_audio_buffer()
    
    def get_result_cache(self) -> TTSResultCache:
        """Get access to result cache"""
        return self.memory_manager.get_result_cache()
    
    async def force_memory_cleanup(self):
        """Force memory cleanup and garbage collection"""
        await self.memory_manager.handle_memory_pressure()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.get_metrics()
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return self.performance_monitor.get_dashboard_data()
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status"""
        return self.shutdown_manager.get_shutdown_status()
    
    def is_shutting_down(self) -> bool:
        """Check if engine is in shutdown process"""
        return self.shutdown_manager.is_shutting_down()
    
    async def _handle_request_completion(self, request_id: str, success: bool = True):
        """Handle request completion and cleanup"""
        # Record performance metrics
        self.performance_metrics.record_request_completion(request_id, success)
        
        # Remove from shutdown tracking
        self.shutdown_manager.unregister_in_flight_request(request_id)
        
        # Mark as completed in queue
        await self.queue.complete_request(request_id, success)
    
    async def recover_from_previous_shutdown(self):
        """Recover and requeue requests from previous shutdown"""
        try:
            recovered_requests = await self.shutdown_manager.load_state_from_file()
            
            if recovered_requests:
                logger.info(f"Recovering {len(recovered_requests)} requests from previous shutdown")
                
                for request in recovered_requests:
                    # Reset status to pending for retry
                    request.status = TTSRequestStatus.PENDING
                    
                    # Re-register for tracking
                    self.shutdown_manager.register_in_flight_request(request)
                    
                    # Re-enqueue the request
                    success = await self.queue.enqueue(request)
                    if not success:
                        logger.warning(f"Failed to requeue recovered request {request.request_id}")
                        self.shutdown_manager.unregister_in_flight_request(request.request_id)
                
                logger.info(f"Successfully recovered {len(recovered_requests)} requests")
            
        except Exception as e:
            logger.error(f"Failed to recover from previous shutdown: {e}")
    
    async def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> bool:
        """
        Gracefully shutdown the TTS engine using shutdown manager
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if shutdown was successful, False if forced
        """
        if not self._running:
            return True
        
        # Use shutdown manager for graceful shutdown
        success = await self.shutdown_manager.initiate_shutdown(
            reason="manual",
            timeout=timeout
        )
        
        return success
    
    async def _shutdown_phase_stop_workers(self, timeout: float) -> bool:
        """Stop workers during shutdown phase"""
        try:
            logger.info("Stopping TTS workers...")
            
            # Stop workers
            worker_tasks = [worker.stop() for worker in self.workers]
            if worker_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*worker_tasks, return_exceptions=True),
                    timeout=timeout
                )
            
            self.workers.clear()
            return True
            
        except asyncio.TimeoutError:
            logger.error("Worker shutdown timed out")
            return False
        except Exception as e:
            logger.error(f"Worker shutdown failed: {e}")
            return False
    
    async def _shutdown_phase_clean_resources(self, timeout: float) -> bool:
        """Clean resources during shutdown phase"""
        try:
            logger.info("Cleaning up TTS resources...")
            
            # Stop memory monitoring
            if self._memory_monitor_task:
                self._memory_monitor_task.cancel()
                try:
                    await asyncio.wait_for(self._memory_monitor_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            # Stop performance monitoring
            await asyncio.wait_for(self.performance_monitor.stop(), timeout=timeout/5)
            
            # Stop queue
            await asyncio.wait_for(self.queue.stop(), timeout=timeout/5)
            
            # Stop resource manager
            await asyncio.wait_for(self.resource_manager.stop(), timeout=timeout/5)
            
            # Stop memory management
            await asyncio.wait_for(self.memory_manager.stop(), timeout=timeout/5)
            
            # Final cleanup
            await asyncio.wait_for(self._cleanup_resources(), timeout=timeout/5)
            
            # Set engine state
            self._running = False
            self._shutdown_event.set()
            
            return True
            
        except asyncio.TimeoutError:
            logger.error("Resource cleanup timed out")
            return False
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            return False
    
    async def _initialize_tts_client(self) -> None:
        """Initialize TTS client"""
        try:
            # Import here to avoid circular imports
            from .elevenlabs_client import TTSClient
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv('ELEVENLABS_API_KEY')
            
            if api_key and api_key != 'your-api-key-here':
                self.tts_client = TTSClient(api_key=api_key)
                if not self.tts_client.validate_api_key():
                    logger.warning("TTS API key validation failed")
            else:
                logger.warning("No valid TTS API key found")
                
        except Exception as e:
            logger.error(f"Failed to initialize TTS client: {e}")
    
    async def _memory_monitor_loop(self) -> None:
        """Monitor memory usage and trigger cleanup"""
        while self._running:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > 500:  # 500MB threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    await self._cleanup_resources()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_resources(self) -> None:
        """Cleanup resources and trigger garbage collection"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear weak references
            weakref.finalize(self, lambda: None)
            
            logger.debug("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health status with thread pool metrics"""
        queue_health = await self.queue.health_check()
        
        worker_health = []
        total_threads = 0
        active_threads = 0
        total_restarts = 0
        
        for worker in self.workers:
            health = await worker.health_check()
            worker_health.append(health)
            
            # Aggregate thread pool metrics
            if "thread_pool_status" in health:
                thread_status = health["thread_pool_status"]
                total_threads += thread_status.get("max_workers", 0)
                active_threads += thread_status.get("active_threads", 0)
            
            # Aggregate restart counts
            if "error_analysis" in health:
                total_restarts += health["error_analysis"].get("restart_count", 0)
        
        process = psutil.Process()
        
        # Thread pool summary
        thread_pool_summary = {
            "total_thread_capacity": total_threads,
            "active_threads": active_threads,
            "thread_utilization": active_threads / max(total_threads, 1),
            "total_worker_restarts": total_restarts,
            "thread_pool_efficiency": sum(
                w.get("thread_pool_status", {}).get("utilization", 0) 
                for w in worker_health
            ) / max(len(worker_health), 1)
        }
        
        # Resource manager stats
        resource_stats = self.resource_manager.get_resource_stats()
        
        # Performance metrics
        performance_metrics = self.performance_metrics.get_metrics()
        performance_dashboard = self.performance_monitor.get_dashboard_data()
        
        return {
            "engine_running": self._running,
            "queue_health": queue_health,
            "worker_health": worker_health,
            "healthy_workers": sum(1 for w in worker_health if w["is_healthy"]),
            "total_workers": len(self.workers),
            "thread_pool_summary": thread_pool_summary,
            "resource_stats": resource_stats,
            "performance_metrics": performance_metrics,
            "performance_dashboard": performance_dashboard,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "is_healthy": (
                self._running and 
                queue_health["is_healthy"] and 
                sum(1 for w in worker_health if w["is_healthy"]) >= len(self.workers) // 2 and
                thread_pool_summary["thread_utilization"] < 0.9 and  # Not overloaded
                resource_stats["total_resources"] < 100 and  # Resource limit
                performance_dashboard["health_status"] in ["healthy", "warning"]  # Performance health
            )
        }


# Event loop management utilities
def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get existing event loop or create new one"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async_tts_engine(coro):
    """
    Run async TTS engine in current or new event loop
    
    Usage:
        run_async_tts_engine(engine.speak_async("Hello", config))
    """
    loop = get_or_create_event_loop()
    
    if loop.is_running():
        # Running in existing loop, create task
        return asyncio.create_task(coro)
    else:
        # Run in new loop
        return loop.run_until_complete(coro)


# Coroutine patterns for TTS operations
async def batch_speak_async(texts: List[str], config: Dict[str, Any], 
                           engine: AsyncTTSEngine) -> List[str]:
    """
    Batch process multiple TTS requests
    
    Args:
        texts: List of texts to synthesize
        config: TTS configuration
        engine: TTS engine instance
        
    Returns:
        List of request IDs
    """
    tasks = []
    for text in texts:
        task = engine.speak_async(text, config)
        tasks.append(task)
    
    return await asyncio.gather(*tasks)


async def speak_with_timeout(text: str, config: Dict[str, Any], 
                            engine: AsyncTTSEngine, timeout: float = 30.0) -> str:
    """
    Speak with timeout protection
    
    Args:
        text: Text to synthesize
        config: TTS configuration
        engine: TTS engine instance
        timeout: Timeout in seconds
        
    Returns:
        Request ID
        
    Raises:
        asyncio.TimeoutError: If operation times out
    """
    return await asyncio.wait_for(
        engine.speak_async(text, config),
        timeout=timeout
    )


async def speak_with_retry(text: str, config: Dict[str, Any], 
                          engine: AsyncTTSEngine, max_retries: int = 3) -> str:
    """
    Speak with automatic retry on failure
    
    Args:
        text: Text to synthesize
        config: TTS configuration
        engine: TTS engine instance
        max_retries: Maximum retry attempts
        
    Returns:
        Request ID
        
    Raises:
        Exception: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return await engine.speak_async(text, config)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise last_error


# Testing utilities
async def test_async_engine():
    """Test the async TTS engine with thread pool infrastructure"""
    print("Testing AsyncTTSEngine with Thread Pool Infrastructure...")
    
    async with async_tts_context() as engine:
        # Test basic functionality
        request_id = await engine.speak_async("Hello, world!", {"voice_id": "6HWqrqOzDfj3UnywjJoZ"})
        print(f"Queued request: {request_id}")
        
        # Test initial health check
        health = await engine.health_check()
        print(f"Initial health status:")
        print(f"  - Engine running: {health['engine_running']}")
        print(f"  - Workers: {health['healthy_workers']}/{health['total_workers']}")
        print(f"  - Thread pool: {health['thread_pool_summary']['active_threads']}/{health['thread_pool_summary']['total_thread_capacity']}")
        print(f"  - Thread utilization: {health['thread_pool_summary']['thread_utilization']:.2%}")
        
        # Test batch processing to stress thread pool
        texts = [
            "First message",
            "Second message", 
            "Third message",
            "Fourth message",
            "Fifth message"
        ]
        request_ids = await batch_speak_async(texts, {"voice_id": "6HWqrqOzDfj3UnywjJoZ"}, engine)
        print(f"Batch requests queued: {len(request_ids)}")
        
        # Monitor thread pool during processing
        for i in range(3):
            await asyncio.sleep(1)
            health = await engine.health_check()
            print(f"Processing... Active threads: {health['thread_pool_summary']['active_threads']}")
        
        # Final comprehensive health check
        final_health = await engine.health_check()
        print(f"\nFinal health status:")
        print(f"  - Engine healthy: {final_health['is_healthy']}")
        print(f"  - Queue size: {final_health['queue_health']['queue_size']}")
        print(f"  - Total restarts: {final_health['thread_pool_summary']['total_worker_restarts']}")
        print(f"  - Thread pool efficiency: {final_health['thread_pool_summary']['thread_pool_efficiency']:.2%}")
        
        # Test resource management
        resource_stats = final_health['resource_stats']
        print(f"\nResource Management:")
        print(f"  - Total resources: {resource_stats['total_resources']}")
        print(f"  - Memory usage: {resource_stats['total_memory_usage'] / 1024 / 1024:.1f}MB")
        print(f"  - Peak memory: {resource_stats['peak_memory_usage'] / 1024 / 1024:.1f}MB")
        print(f"  - Cleanup count: {resource_stats['cleanup_count']}")
        print(f"  - Leak detection count: {resource_stats['leak_detection_count']}")
        print(f"  - Resource types: {resource_stats['resource_types']}")
        
        # Test performance monitoring
        performance_metrics = final_health['performance_metrics']
        performance_dashboard = final_health['performance_dashboard']
        print(f"\nPerformance Monitoring:")
        print(f"  - Total requests: {performance_metrics['total_requests']}")
        print(f"  - Success rate: {performance_metrics['success_rate']:.1f}%")
        print(f"  - Average execution time: {performance_metrics['average_execution_time']:.2f}s")
        print(f"  - Current throughput: {performance_metrics['current_throughput']:.2f} req/s")
        print(f"  - Peak concurrent requests: {performance_metrics['peak_concurrent_requests']}")
        print(f"  - Health status: {performance_dashboard['health_status']}")
        
        # Test worker health details
        for i, worker in enumerate(final_health['worker_health']):
            print(f"  - Worker {i}:")
            print(f"    - Processed: {worker['processed_count']}")
            print(f"    - Errors: {worker['error_analysis']['total_errors']}")
            print(f"    - Thread utilization: {worker['thread_pool_status']['utilization']:.2%}")
            print(f"    - Healthy: {worker['is_healthy']}")
    
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_async_engine())