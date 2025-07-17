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
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from pathlib import Path
import signal
import threading
import heapq
import uuid
from contextlib import asynccontextmanager
import gc
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_CONCURRENT_REQUESTS = 3
DEFAULT_QUEUE_SIZE = 50
DEFAULT_WORKER_THREADS = 2
DEFAULT_PRIORITY_BOOST_THRESHOLD = 2.0  # seconds
DEFAULT_CLEANUP_INTERVAL = 30.0  # seconds
DEFAULT_SHUTDOWN_TIMEOUT = 10.0  # seconds


class TTSPriority(Enum):
    """Priority levels for TTS requests"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


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
    Represents a TTS request with priority and metadata
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
        """Update priority score (called periodically to prevent starvation)"""
        old_score = self._priority_score
        self._calculate_priority_score()
        
        # Log significant priority changes
        if abs(self._priority_score - old_score) > 100:
            logger.debug(f"Priority score updated for {self.request_id}: {old_score:.1f} -> {self._priority_score:.1f}")
    
    def get_priority_metrics(self) -> Dict[str, Any]:
        """Get priority metrics for monitoring"""
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
    
    def __lt__(self, other):
        """Priority queue comparison - higher priority score first"""
        # Update scores before comparison to ensure freshness
        self.update_priority_score()
        other.update_priority_score()
        return self._priority_score > other._priority_score
    
    def is_expired(self, max_age: float = 60.0) -> bool:
        """Check if request has expired"""
        return time.time() - self.created_at > max_age
    
    def can_retry(self) -> bool:
        """Check if request can be retried"""
        return self.retry_count < self.max_retries
    
    def mark_retry(self):
        """Mark request for retry"""
        self.retry_count += 1
        self.status = TTSRequestStatus.PENDING


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
    Async queue manager for TTS requests with priority and concurrency control
    """
    
    def __init__(self, max_size: int = DEFAULT_QUEUE_SIZE):
        self.max_size = max_size
        self._queue: List[TTSRequest] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self._active_requests: Dict[str, TTSRequest] = {}
        self._completed_requests: Dict[str, TTSRequest] = {}
        self._semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._priority_reorder_task: Optional[asyncio.Task] = None
        self._priority_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "priority_boosts": 0,
            "starvation_prevented": 0,
            "reorders_performed": 0,
            "average_wait_time": 0.0
        }
        
    async def start(self) -> None:
        """Start the queue and background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._priority_reorder_task = asyncio.create_task(self._priority_reorder_loop())
        logger.info("AsyncTTSQueue started")
    
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
        
        # Cancel all pending requests
        async with self._lock:
            for request in self._queue:
                request.status = TTSRequestStatus.CANCELLED
            self._queue.clear()
        
        logger.info("AsyncTTSQueue stopped")
    
    async def enqueue(self, request: TTSRequest) -> bool:
        """
        Add request to queue with priority handling
        
        Returns:
            bool: True if enqueued successfully, False if queue full
        """
        async with self._not_full:
            if len(self._queue) >= self.max_size:
                # Remove oldest low-priority request if queue is full
                await self._make_room()
                if len(self._queue) >= self.max_size:
                    return False
            
            # Update metrics
            self._priority_metrics["total_requests"] += 1
            if request.priority == TTSPriority.HIGH and time.time() - request.created_at < DEFAULT_PRIORITY_BOOST_THRESHOLD:
                self._priority_metrics["priority_boosts"] += 1
            
            # Insert in priority order
            heapq.heappush(self._queue, request)
            self._not_empty.notify()
            
            logger.debug(f"Enqueued request {request.request_id} with priority {request.priority}")
            return True
    
    async def dequeue(self) -> Optional[TTSRequest]:
        """
        Remove and return highest priority request
        
        Returns:
            TTSRequest or None if queue is empty
        """
        async with self._not_empty:
            while not self._queue:
                await self._not_empty.wait()
            
            request = heapq.heappop(self._queue)
            self._active_requests[request.request_id] = request
            self._not_full.notify()
            
            logger.debug(f"Dequeued request {request.request_id}")
            return request
    
    async def complete_request(self, request_id: str, success: bool = True) -> None:
        """Mark request as completed"""
        async with self._lock:
            if request_id in self._active_requests:
                request = self._active_requests.pop(request_id)
                request.status = TTSRequestStatus.COMPLETED if success else TTSRequestStatus.FAILED
                
                # Update metrics
                wait_time = time.time() - request.created_at
                current_avg = self._priority_metrics["average_wait_time"]
                total_requests = self._priority_metrics["total_requests"]
                self._priority_metrics["average_wait_time"] = (
                    (current_avg * (total_requests - 1) + wait_time) / total_requests
                    if total_requests > 0 else wait_time
                )
                
                self._completed_requests[request_id] = request
                self._semaphore.release()
    
    async def _make_room(self) -> None:
        """Remove oldest low-priority request to make room"""
        if not self._queue:
            return
            
        # Find oldest low-priority request
        oldest_idx = -1
        oldest_time = float('inf')
        
        for i, request in enumerate(self._queue):
            if request.priority == TTSPriority.LOW and request.created_at < oldest_time:
                oldest_idx = i
                oldest_time = request.created_at
        
        if oldest_idx >= 0:
            removed = self._queue.pop(oldest_idx)
            removed.status = TTSRequestStatus.CANCELLED
            heapq.heapify(self._queue)  # Restore heap property
            logger.debug(f"Removed old request {removed.request_id} to make room")
    
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
        """Remove expired requests"""
        async with self._lock:
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
        """Reorder queue based on updated priority scores"""
        async with self._lock:
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
                logger.debug(f"Reordered queue: prevented {starvation_prevented} starvation cases")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return queue health status"""
        async with self._lock:
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
    
    def __init__(self, worker_id: str, queue: AsyncTTSQueue, tts_client, max_thread_workers: int = 2):
        self.worker_id = worker_id
        self.queue = queue
        self.tts_client = tts_client
        self.max_thread_workers = max_thread_workers
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._last_error_time = 0.0
        self._thread_pool: Optional[ThreadPoolExecutor] = None
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
        
        # Shutdown thread pool
        if self._thread_pool:
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
                await self.queue.complete_request(request.request_id, success=False)
            
            self._error_count += 1
            self._consecutive_errors += 1
    
    def _synthesize_tts(self, request: TTSRequest) -> None:
        """Synchronous TTS synthesis with thread-safe execution"""
        try:
            # Set thread name for monitoring
            current_thread = threading.current_thread()
            original_name = current_thread.name
            current_thread.name = f"tts-{self.worker_id}-{request.request_id[:8]}"
            
            try:
                # Import here to avoid circular imports
                from .elevenlabs_tts import speak_text
                
                voice_id = request.config.get('voice_id', '6HWqrqOzDfj3UnywjJoZ')
                speak_text(request.text, voice_id)
                
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


class AsyncTTSEngine:
    """
    Main asynchronous TTS execution engine
    
    Coordinates queue management, worker threads, and resource cleanup
    for non-blocking TTS operations.
    """
    
    def __init__(self, max_workers: int = DEFAULT_WORKER_THREADS):
        self.max_workers = max_workers
        self.queue = AsyncTTSQueue()
        self.workers: List[AsyncTTSWorker] = []
        self.tts_client = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._memory_monitor_task: Optional[asyncio.Task] = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self) -> None:
        """Start the TTS engine"""
        if self._running:
            return
        
        logger.info("Starting AsyncTTSEngine...")
        
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
                max_thread_workers=2  # Each worker gets 2 threads
            )
            await worker.start()
            self.workers.append(worker)
        
        # Start memory monitoring
        self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
        
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
        
        request = TTSRequest(
            text=text,
            config=config,
            priority=priority,
            context={"source": "async_engine"}
        )
        
        success = await self.queue.enqueue(request)
        if not success:
            raise RuntimeError("TTS queue is full")
        
        return request.request_id
    
    async def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> None:
        """
        Gracefully shutdown the TTS engine
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return
        
        logger.info("Shutting down AsyncTTSEngine...")
        self._running = False
        
        # Stop memory monitoring
        if self._memory_monitor_task:
            self._memory_monitor_task.cancel()
        
        # Stop workers
        worker_tasks = [worker.stop() for worker in self.workers]
        if worker_tasks:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=timeout
            )
        
        # Stop queue
        await self.queue.stop()
        
        # Cleanup
        await self._cleanup_resources()
        
        self._shutdown_event.set()
        logger.info("AsyncTTSEngine shutdown complete")
    
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
        
        return {
            "engine_running": self._running,
            "queue_health": queue_health,
            "worker_health": worker_health,
            "healthy_workers": sum(1 for w in worker_health if w["is_healthy"]),
            "total_workers": len(self.workers),
            "thread_pool_summary": thread_pool_summary,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "is_healthy": (
                self._running and 
                queue_health["is_healthy"] and 
                sum(1 for w in worker_health if w["is_healthy"]) >= len(self.workers) // 2 and
                thread_pool_summary["thread_utilization"] < 0.9  # Not overloaded
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