#!/usr/bin/env python3
"""
Comprehensive Thread-Safety Test Suite for TTS Engine

This test suite validates thread-safety across all TTS engine components
and detects race conditions, deadlocks, and other concurrency issues.
"""

import asyncio
import threading
import time
import random
import concurrent.futures
import queue
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
import logging
from contextlib import contextmanager

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result with timing and success information"""
    test_name: str
    success: bool
    duration: float
    errors: List[str]
    metrics: Dict[str, Any]


class ThreadSafetyTester:
    """Comprehensive thread-safety testing framework"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.failed_tests: List[str] = []
        self.race_conditions_detected: List[str] = []
        self.deadlocks_detected: List[str] = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all thread-safety tests"""
        logger.info("Starting comprehensive thread-safety test suite...")
        
        test_methods = [
            self.test_concurrent_counter_operations,
            self.test_lock_manager_deadlock_prevention,
            self.test_thread_safe_logger,
            self.test_tts_request_thread_safety,
            self.test_resource_manager_concurrency,
            self.test_queue_operations_thread_safety,
            self.test_performance_metrics_thread_safety,
            self.test_memory_management_thread_safety,
            self.test_high_concurrency_stress,
            self.test_race_condition_detection
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}...")
                result = test_method()
                self.results.append(result)
                if not result.success:
                    self.failed_tests.append(test_method.__name__)
                    logger.error(f"FAILED: {test_method.__name__}")
                else:
                    logger.info(f"PASSED: {test_method.__name__}")
            except Exception as e:
                logger.error(f"ERROR in {test_method.__name__}: {e}")
                self.failed_tests.append(test_method.__name__)
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    success=False,
                    duration=0.0,
                    errors=[str(e)],
                    metrics={}
                ))
        
        return self.generate_report()
    
    def test_concurrent_counter_operations(self) -> TestResult:
        """Test ThreadSafeCounters under high concurrency"""
        from async_tts_engine import ThreadSafeCounters
        
        start_time = time.time()
        errors = []
        
        try:
            counters = ThreadSafeCounters()
            num_threads = 50
            operations_per_thread = 1000
            
            def worker(worker_id: int):
                for i in range(operations_per_thread):
                    # Mix of operations
                    counters.increment('test_counter', 1)
                    counters.decrement('test_counter', 0.5)
                    counters.atomic_max_update('max_counter', random.randint(1, 100))
                    counters.get('test_counter')
                    
                    # Test different keys
                    counters.increment(f'worker_{worker_id}', 1)
                    counters.get_all()
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)
            
            # Verify final state
            final_count = counters.get('test_counter')
            expected_count = num_threads * operations_per_thread * 0.5  # 1 - 0.5 = 0.5 per operation
            
            if abs(final_count - expected_count) > 0.1:
                errors.append(f"Counter mismatch: expected {expected_count}, got {final_count}")
            
            # Verify worker counters
            for i in range(num_threads):
                worker_count = counters.get(f'worker_{i}')
                if worker_count != operations_per_thread:
                    errors.append(f"Worker {i} counter mismatch: expected {operations_per_thread}, got {worker_count}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_concurrent_counter_operations",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'threads': num_threads,
                'operations_per_thread': operations_per_thread,
                'final_count': final_count if 'final_count' in locals() else 0
            }
        )
    
    def test_lock_manager_deadlock_prevention(self) -> TestResult:
        """Test LockManager deadlock prevention"""
        from async_tts_engine import LockManager
        
        start_time = time.time()
        errors = []
        
        try:
            lock_manager = LockManager()
            deadlock_detected = False
            
            async def worker_1():
                try:
                    async with lock_manager.acquire_locks('resource', 'queue', 'memory'):
                        await asyncio.sleep(0.1)
                        async with lock_manager.acquire_locks('worker'):
                            await asyncio.sleep(0.1)
                except Exception as e:
                    nonlocal deadlock_detected
                    deadlock_detected = True
                    errors.append(f"Worker 1 failed: {e}")
            
            async def worker_2():
                try:
                    async with lock_manager.acquire_locks('memory', 'worker', 'queue'):
                        await asyncio.sleep(0.1)
                        async with lock_manager.acquire_locks('resource'):
                            await asyncio.sleep(0.1)
                except Exception as e:
                    nonlocal deadlock_detected
                    deadlock_detected = True
                    errors.append(f"Worker 2 failed: {e}")
            
            # Run concurrent lock acquisition
            async def run_test():
                await asyncio.gather(
                    worker_1(),
                    worker_2(),
                    return_exceptions=True
                )
            
            asyncio.run(run_test())
            
            if deadlock_detected:
                errors.append("Deadlock detected in lock manager")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_lock_manager_deadlock_prevention",
            success=success,
            duration=duration,
            errors=errors,
            metrics={'deadlock_detected': deadlock_detected if 'deadlock_detected' in locals() else True}
        )
    
    def test_thread_safe_logger(self) -> TestResult:
        """Test ThreadSafeLogger under concurrent logging"""
        from async_tts_engine import ThreadSafeLogger
        
        start_time = time.time()
        errors = []
        
        try:
            logger = ThreadSafeLogger(__name__)
            num_threads = 20
            messages_per_thread = 100
            
            def worker(worker_id: int):
                for i in range(messages_per_thread):
                    logger.info(f"Worker {worker_id} message {i}")
                    logger.debug(f"Worker {worker_id} debug {i}")
                    logger.warning(f"Worker {worker_id} warning {i}")
                    
                    # Test different message types
                    if i % 10 == 0:
                        logger.error(f"Worker {worker_id} error {i}")
            
            # Run concurrent logging
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)
            
            # Check for any exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Logging thread failed: {e}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_thread_safe_logger",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'threads': num_threads,
                'messages_per_thread': messages_per_thread
            }
        )
    
    def test_tts_request_thread_safety(self) -> TestResult:
        """Test TTSRequest thread-safety"""
        try:
            from async_tts_engine import TTSRequest, TTSPriority, TTSRequestStatus
        except ImportError:
            return TestResult(
                test_name="test_tts_request_thread_safety",
                success=False,
                duration=0.0,
                errors=["Cannot import TTSRequest classes"],
                metrics={}
            )
        
        start_time = time.time()
        errors = []
        
        try:
            request = TTSRequest(
                text="Test message",
                config={"voice_id": "test"},
                priority=TTSPriority.NORMAL
            )
            
            num_threads = 30
            operations_per_thread = 100
            
            def worker(worker_id: int):
                for i in range(operations_per_thread):
                    # Test thread-safe operations
                    request.update_priority_score()
                    request.get_priority_metrics()
                    request.get_status_safe()
                    request.get_priority_score_safe()
                    
                    # Test status changes
                    if i % 10 == 0:
                        request.set_status_safe(TTSRequestStatus.PROCESSING)
                    if i % 20 == 0:
                        request.set_status_safe(TTSRequestStatus.PENDING)
                    
                    # Test retry operations
                    if i % 50 == 0:
                        if request.can_retry():
                            request.mark_retry()
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)
            
            # Verify final state
            final_status = request.get_status_safe()
            final_score = request.get_priority_score_safe()
            
            # Check for any exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Request operation failed: {e}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_tts_request_thread_safety",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'threads': num_threads,
                'operations_per_thread': operations_per_thread,
                'final_status': final_status.value if 'final_status' in locals() else 'unknown'
            }
        )
    
    def test_resource_manager_concurrency(self) -> TestResult:
        """Test TTSResourceManager thread-safety"""
        try:
            from async_tts_engine import TTSResourceManager, TTSResource, TTSResourceType
        except ImportError:
            return TestResult(
                test_name="test_resource_manager_concurrency",
                success=False,
                duration=0.0,
                errors=["Cannot import TTSResourceManager classes"],
                metrics={}
            )
        
        start_time = time.time()
        errors = []
        
        try:
            async def run_test():
                manager = TTSResourceManager()
                await manager.start()
                
                num_threads = 20
                resources_per_thread = 50
                
                def worker(worker_id: int):
                    resource_ids = []
                    for i in range(resources_per_thread):
                        # Create resource
                        resource = TTSResource(
                            resource_id=f"worker_{worker_id}_resource_{i}",
                            resource_type=TTSResourceType.MEMORY_BUFFER,
                            resource_ref=f"test_resource_{worker_id}_{i}",
                            size_bytes=random.randint(1000, 10000)
                        )
                        
                        # Register resource
                        resource_id = manager.register_resource(resource)
                        resource_ids.append(resource_id)
                        
                        # Access resource
                        if i % 5 == 0:
                            manager.access_resource(resource_id)
                    
                    # Clean up half the resources
                    for i, resource_id in enumerate(resource_ids[:len(resource_ids)//2]):
                        manager.unregister_resource(resource_id)
                
                # Run concurrent operations
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(worker, i) for i in range(num_threads)]
                    concurrent.futures.wait(futures)
                
                # Get final stats
                stats = manager.get_resource_stats()
                
                # Check for any exceptions
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"Resource operation failed: {e}")
                
                await manager.stop()
                return stats
            
            stats = asyncio.run(run_test())
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
            stats = {}
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_resource_manager_concurrency",
            success=success,
            duration=duration,
            errors=errors,
            metrics=stats
        )
    
    def test_queue_operations_thread_safety(self) -> TestResult:
        """Test AsyncTTSQueue thread-safety"""
        try:
            from async_tts_engine import AsyncTTSQueue, TTSRequest, TTSPriority
        except ImportError:
            return TestResult(
                test_name="test_queue_operations_thread_safety",
                success=False,
                duration=0.0,
                errors=["Cannot import AsyncTTSQueue classes"],
                metrics={}
            )
        
        start_time = time.time()
        errors = []
        
        try:
            async def run_test():
                queue = AsyncTTSQueue(max_size=100)
                await queue.start()
                
                num_producers = 10
                num_consumers = 5
                requests_per_producer = 20
                
                async def producer(producer_id: int):
                    for i in range(requests_per_producer):
                        request = TTSRequest(
                            text=f"Producer {producer_id} message {i}",
                            config={"voice_id": "test"},
                            priority=random.choice(list(TTSPriority))
                        )
                        
                        success = await queue.enqueue(request)
                        if not success:
                            errors.append(f"Producer {producer_id} failed to enqueue request {i}")
                
                async def consumer(consumer_id: int):
                    processed = 0
                    while processed < (num_producers * requests_per_producer) // num_consumers:
                        request = await queue.dequeue()
                        if request:
                            await queue.complete_request(request.request_id, success=True)
                            processed += 1
                        else:
                            await asyncio.sleep(0.01)  # Short delay if queue empty
                
                # Run producers and consumers concurrently
                producers = [producer(i) for i in range(num_producers)]
                consumers = [consumer(i) for i in range(num_consumers)]
                
                await asyncio.gather(*producers, *consumers, return_exceptions=True)
                
                # Get final health check
                health = await queue.health_check()
                
                await queue.stop()
                return health
            
            health = asyncio.run(run_test())
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
            health = {}
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_queue_operations_thread_safety",
            success=success,
            duration=duration,
            errors=errors,
            metrics=health
        )
    
    def test_performance_metrics_thread_safety(self) -> TestResult:
        """Test TTSPerformanceMetrics thread-safety"""
        try:
            from async_tts_engine import TTSPerformanceMetrics
        except ImportError:
            return TestResult(
                test_name="test_performance_metrics_thread_safety",
                success=False,
                duration=0.0,
                errors=["Cannot import TTSPerformanceMetrics"],
                metrics={}
            )
        
        start_time = time.time()
        errors = []
        
        try:
            metrics = TTSPerformanceMetrics()
            num_threads = 25
            operations_per_thread = 200
            
            def worker(worker_id: int):
                for i in range(operations_per_thread):
                    request_id = f"worker_{worker_id}_request_{i}"
                    
                    # Start request
                    metrics.record_request_start(request_id)
                    
                    # Simulate processing time
                    time.sleep(random.uniform(0.001, 0.01))
                    
                    # Complete request
                    success = random.random() > 0.1  # 90% success rate
                    metrics.record_request_completion(request_id, success)
                    
                    # Record queue metrics
                    metrics.record_queue_depth(random.randint(1, 20))
                    metrics.record_queue_latency(random.uniform(0.1, 2.0))
                    
                    # Record throughput periodically
                    if i % 10 == 0:
                        metrics.record_throughput()
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)
            
            # Get final metrics
            final_metrics = metrics.get_metrics()
            
            # Verify metrics consistency
            expected_total = num_threads * operations_per_thread
            actual_total = final_metrics.get('total_requests', 0)
            
            if abs(actual_total - expected_total) > expected_total * 0.05:  # 5% tolerance
                errors.append(f"Total requests mismatch: expected {expected_total}, got {actual_total}")
            
            # Check for any exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Metrics operation failed: {e}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
            final_metrics = {}
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_performance_metrics_thread_safety",
            success=success,
            duration=duration,
            errors=errors,
            metrics=final_metrics
        )
    
    def test_memory_management_thread_safety(self) -> TestResult:
        """Test memory management components thread-safety"""
        try:
            from async_tts_engine import TTSAudioBuffer, TTSResultCache
        except ImportError:
            return TestResult(
                test_name="test_memory_management_thread_safety",
                success=False,
                duration=0.0,
                errors=["Cannot import memory management classes"],
                metrics={}
            )
        
        start_time = time.time()
        errors = []
        
        try:
            audio_buffer = TTSAudioBuffer(max_buffer_size=1024*1024)  # 1MB
            result_cache = TTSResultCache(max_entries=100, max_memory=1024*1024)
            
            num_threads = 15
            operations_per_thread = 100
            
            def worker(worker_id: int):
                for i in range(operations_per_thread):
                    # Audio buffer operations
                    audio_data = f"audio_data_{worker_id}_{i}".encode() * 100
                    buffer_id = f"buffer_{worker_id}_{i}"
                    
                    audio_buffer.store_audio(buffer_id, audio_data, {"worker": worker_id})
                    audio_buffer.get_audio(buffer_id)
                    
                    if i % 5 == 0:
                        audio_buffer.remove_audio(buffer_id)
                    
                    # Result cache operations
                    text = f"text_{worker_id}_{i}"
                    voice_id = f"voice_{worker_id % 3}"
                    settings = {"speed": 1.0, "worker": worker_id}
                    result_data = f"result_{worker_id}_{i}".encode() * 50
                    
                    result_cache.put(text, voice_id, settings, result_data)
                    cached_result = result_cache.get(text, voice_id, settings)
                    
                    if i % 10 == 0:
                        result_cache.get_cache_stats()
                    
                    if i % 20 == 0:
                        audio_buffer.get_memory_stats()
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)
            
            # Get final stats
            buffer_stats = audio_buffer.get_memory_stats()
            cache_stats = result_cache.get_cache_stats()
            
            # Check for any exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Memory operation failed: {e}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during test: {e}")
            success = False
            buffer_stats = {}
            cache_stats = {}
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_memory_management_thread_safety",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'buffer_stats': buffer_stats,
                'cache_stats': cache_stats
            }
        )
    
    def test_high_concurrency_stress(self) -> TestResult:
        """Stress test with high concurrency"""
        start_time = time.time()
        errors = []
        
        try:
            from async_tts_engine import ThreadSafeCounters, ThreadSafeLogger
            
            # High concurrency stress test
            num_threads = 100
            operations_per_thread = 1000
            
            counters = ThreadSafeCounters()
            logger = ThreadSafeLogger(__name__)
            
            def stress_worker(worker_id: int):
                for i in range(operations_per_thread):
                    # Mix of operations
                    counters.increment('stress_counter')
                    counters.atomic_max_update('max_stress', i)
                    counters.get_all()
                    
                    # Logging under stress
                    if i % 100 == 0:
                        logger.info(f"Stress worker {worker_id} at {i}")
                    
                    # Random operations
                    if random.random() < 0.1:
                        counters.decrement('stress_counter', 0.5)
                    
                    if random.random() < 0.05:
                        counters.reset()
            
            # Run high concurrency stress test
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures, timeout=30)  # 30 second timeout
            
            # Check results
            final_counters = counters.get_all()
            
            # Check for any exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Stress test failed: {e}")
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during stress test: {e}")
            success = False
            final_counters = {}
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_high_concurrency_stress",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'threads': num_threads,
                'operations_per_thread': operations_per_thread,
                'final_counters': final_counters
            }
        )
    
    def test_race_condition_detection(self) -> TestResult:
        """Test for race condition detection"""
        start_time = time.time()
        errors = []
        race_conditions = []
        
        try:
            # Simulate potential race condition scenario
            shared_resource = {'value': 0, 'operations': []}
            lock = threading.Lock()
            
            def racy_worker(worker_id: int):
                for i in range(100):
                    # Intentionally racy operation
                    old_value = shared_resource['value']
                    time.sleep(0.001)  # Simulate processing time
                    new_value = old_value + 1
                    
                    # Check for race condition
                    with lock:
                        if shared_resource['value'] != old_value:
                            race_conditions.append(f"Race condition detected by worker {worker_id} at iteration {i}")
                        shared_resource['value'] = new_value
                        shared_resource['operations'].append(f"worker_{worker_id}_op_{i}")
            
            # Run workers that might have race conditions
            num_workers = 20
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(racy_worker, i) for i in range(num_workers)]
                concurrent.futures.wait(futures)
            
            # Analyze results
            expected_value = num_workers * 100
            actual_value = shared_resource['value']
            
            if actual_value != expected_value:
                errors.append(f"Race condition caused incorrect final value: expected {expected_value}, got {actual_value}")
            
            if race_conditions:
                errors.extend(race_conditions)
            
            success = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception during race condition test: {e}")
            success = False
        
        duration = time.time() - start_time
        return TestResult(
            test_name="test_race_condition_detection",
            success=success,
            duration=duration,
            errors=errors,
            metrics={
                'race_conditions_detected': len(race_conditions),
                'expected_value': expected_value if 'expected_value' in locals() else 0,
                'actual_value': actual_value if 'actual_value' in locals() else 0
            }
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.results)
        avg_duration = total_duration / max(total_tests, 1)
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / max(total_tests, 1)) * 100,
                'total_duration': total_duration,
                'average_duration': avg_duration
            },
            'failed_tests': self.failed_tests,
            'race_conditions_detected': self.race_conditions_detected,
            'deadlocks_detected': self.deadlocks_detected,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'duration': r.duration,
                    'errors': r.errors,
                    'metrics': r.metrics
                }
                for r in self.results
            ]
        }


def main():
    """Main test runner"""
    print("=" * 80)
    print("TTS Engine Thread-Safety Test Suite")
    print("=" * 80)
    
    tester = ThreadSafetyTester()
    report = tester.run_all_tests()
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    summary = report['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Average Duration: {summary['average_duration']:.2f}s")
    
    if report['failed_tests']:
        print(f"\nFAILED TESTS:")
        for test in report['failed_tests']:
            print(f"  - {test}")
    
    if report['race_conditions_detected']:
        print(f"\nRACE CONDITIONS DETECTED:")
        for race in report['race_conditions_detected']:
            print(f"  - {race}")
    
    if report['deadlocks_detected']:
        print(f"\nDEADLOCKS DETECTED:")
        for deadlock in report['deadlocks_detected']:
            print(f"  - {deadlock}")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for result in report['detailed_results']:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status}: {result['test_name']} ({result['duration']:.2f}s)")
        
        if result['errors']:
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
        
        if result['metrics']:
            print("  Metrics:")
            for key, value in result['metrics'].items():
                print(f"    {key}: {value}")
        print()
    
    # Return success/failure for CI/CD
    return summary['failed_tests'] == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)