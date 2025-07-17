# Async TTS Engine Architecture

## Overview

The Async TTS Engine provides a non-blocking, high-performance text-to-speech execution system designed to process speech synthesis requests without impacting Claude Code's command execution performance.

## Architecture Components

### Core Components

#### 1. AsyncTTSEngine
- **Role**: Main coordinator and entry point
- **Responsibilities**:
  - Lifecycle management (start/stop)
  - Worker thread coordination
  - Resource cleanup
  - Signal handling for graceful shutdown
  - Memory monitoring

#### 2. AsyncTTSQueue
- **Role**: Priority-based queue management
- **Responsibilities**:
  - Request prioritization using heapq
  - Concurrency control with semaphores
  - Queue overflow handling
  - Expired request cleanup
  - Thread-safe operations

#### 3. AsyncTTSWorker
- **Role**: Background TTS processing
- **Responsibilities**:
  - Request processing in dedicated threads
  - Error handling and retry logic
  - Performance metrics tracking
  - Thread pool management

#### 4. TTSRequest
- **Role**: Request data structure
- **Responsibilities**:
  - Priority calculation
  - Status tracking
  - Retry management
  - Expiration handling

## Design Patterns

### 1. Async Architecture Pattern

```python
# Non-blocking operation pattern
async def speak_async(text: str, config: Dict) -> str:
    request = TTSRequest(text, config)
    await queue.enqueue(request)  # Returns immediately
    return request.request_id
```

**Benefits**:
- Zero blocking of main thread
- Immediate response to user
- Background processing

### 2. Priority Queue Pattern

```python
# Priority-based processing
class TTSRequest:
    def __lt__(self, other):
        return self.priority.value > other.priority.value
```

**Benefits**:
- Recent commands get priority
- Prevents stale request buildup
- Intelligent queue management

### 3. Worker Pool Pattern

```python
# Concurrent processing
workers = [AsyncTTSWorker(f"worker-{i}", queue, client) 
          for i in range(max_workers)]
```

**Benefits**:
- Configurable concurrency
- Fault tolerance
- Resource isolation

### 4. Context Manager Pattern

```python
# Resource management
async with async_tts_context() as engine:
    await engine.speak_async("Hello", config)
```

**Benefits**:
- Automatic cleanup
- Exception safety
- Resource guarantee

## Interfaces

### AsyncTTSInterface
Base interface for all async components:

```python
class AsyncTTSInterface(ABC):
    @abstractmethod
    async def start(self) -> None: ...
    
    @abstractmethod
    async def stop(self) -> None: ...
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]: ...
```

### Event Loop Management
Utilities for managing asyncio event loops:

```python
def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    # Gets existing loop or creates new one
    
def run_async_tts_engine(coro):
    # Runs coroutine in appropriate loop
```

## Context Managers

### AsyncTTSContext
Provides automatic resource management:

```python
@asynccontextmanager
async def async_tts_context(max_workers: int = 2):
    engine = AsyncTTSEngine(max_workers=max_workers)
    try:
        await engine.start()
        yield engine
    finally:
        await engine.shutdown()
```

## Coroutine Patterns

### 1. Batch Processing
```python
async def batch_speak_async(texts: List[str], config: Dict, 
                           engine: AsyncTTSEngine) -> List[str]:
    tasks = [engine.speak_async(text, config) for text in texts]
    return await asyncio.gather(*tasks)
```

### 2. Timeout Protection
```python
async def speak_with_timeout(text: str, config: Dict, 
                            engine: AsyncTTSEngine, timeout: float = 30.0) -> str:
    return await asyncio.wait_for(
        engine.speak_async(text, config),
        timeout=timeout
    )
```

### 3. Retry Logic
```python
async def speak_with_retry(text: str, config: Dict, 
                          engine: AsyncTTSEngine, max_retries: int = 3) -> str:
    for attempt in range(max_retries + 1):
        try:
            return await engine.speak_async(text, config)
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    raise last_error
```

## Memory Management

### Resource Cleanup
- Automatic garbage collection triggers
- Weak reference cleanup
- Periodic memory monitoring
- Resource usage thresholds

### Memory Monitoring
```python
async def _memory_monitor_loop(self):
    while self._running:
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 500:  # 500MB threshold
            await self._cleanup_resources()
```

## Error Handling

### Graceful Degradation
- Request retry with exponential backoff
- Circuit breaker pattern support
- Fallback mechanisms
- Error categorization

### Exception Hierarchy
- Base exceptions for different failure types
- Specific error handling strategies
- Recovery mechanisms

## Performance Considerations

### Concurrency Control
- Semaphore-based request limiting (max 3 concurrent)
- Worker thread pooling
- Non-blocking queue operations

### Optimization Strategies
- Request deduplication
- Priority boosting for recent commands
- Expired request cleanup
- Memory pressure handling

## Signal Handling

### Graceful Shutdown
```python
def _setup_signal_handlers(self):
    def signal_handler(signum, frame):
        asyncio.create_task(self.shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
```

## Integration Points

### Claude Code Hook Integration
- Non-blocking hook execution
- Immediate command response
- Background TTS processing

### Configuration Integration
- Dynamic configuration loading
- Hot-reload support
- Environment variable integration

## Testing Strategy

### Unit Testing
- Component isolation
- Mock dependencies
- Async test patterns

### Integration Testing
- End-to-end workflow
- Performance benchmarks
- Resource usage validation

## Deployment Considerations

### Resource Requirements
- Memory: ~50MB base + queue overhead
- CPU: Minimal (background processing)
- Network: Dependent on API usage

### Monitoring
- Health check endpoints
- Performance metrics
- Error rate tracking
- Resource usage monitoring

## Future Enhancements

### Planned Features
- Advanced caching strategies
- Request batching optimization
- Multi-provider support
- Enhanced metrics collection

### Scalability
- Horizontal scaling support
- Load balancing
- Distributed queue management