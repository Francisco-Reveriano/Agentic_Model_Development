"""
SSE Heartbeat for Long-Running Operations

Provides heartbeat/keep-alive mechanism for Server-Sent Events to maintain
connections during long-running agent executions. Prevents timeout disconnections
and provides real-time status updates to connected clients.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
from datetime import datetime


@dataclass
class HeartbeatConfig:
    """
    Configuration for SSE heartbeat mechanism.

    Heartbeats are lightweight keep-alive messages sent to prevent connection
    timeouts during long operations. Configurable interval and message format.
    """

    enabled: bool = True
    """Enable heartbeat (default: True)"""

    interval: int = 15
    """Heartbeat interval in seconds (default: 15)"""

    message: str = "heartbeat"
    """Message to send on heartbeat (default: 'heartbeat')"""

    include_timestamp: bool = True
    """Include ISO timestamp in heartbeat (default: True)"""

    include_uptime: bool = True
    """Include elapsed seconds in heartbeat (default: True)"""

    verbose: bool = False
    """Log heartbeat events (default: False)"""


async def emit_heartbeat(
    queue: asyncio.Queue,
    interval: int = 15,
    start_time: Optional[float] = None,
    config: Optional[HeartbeatConfig] = None,
) -> None:
    """
    Emit periodic heartbeat messages to SSE queue.

    Runs indefinitely, emitting heartbeat messages at specified intervals.
    Typically run as a background task during pipeline execution.

    Args:
        queue: asyncio.Queue to emit heartbeats to
        interval: Interval in seconds (default: 15)
        start_time: Start timestamp for uptime calculation (default: now)
        config: HeartbeatConfig instance (optional)

    Example:
        >>> queue = asyncio.Queue()
        >>> task = asyncio.create_task(emit_heartbeat(queue, interval=30))
        >>> # Later, cancel when done:
        >>> task.cancel()
    """
    if config is None:
        config = HeartbeatConfig(interval=interval)

    if not config.enabled:
        return

    if start_time is None:
        start_time = time.time()

    try:
        while True:
            await asyncio.sleep(config.interval)

            # Build heartbeat message
            message = config.message

            if config.include_timestamp:
                timestamp = datetime.utcnow().isoformat() + "Z"
                message = f"{message}|timestamp={timestamp}"

            if config.include_uptime:
                uptime = int(time.time() - start_time)
                message = f"{message}|uptime={uptime}"

            # Emit to queue
            await queue.put(f"data: {message}\n\n")

            if config.verbose:
                print(f"[Heartbeat] {message}")

    except asyncio.CancelledError:
        if config.verbose:
            print("[Heartbeat] Task cancelled")
        pass


def create_heartbeat_task(
    queue: asyncio.Queue,
    interval: int = 15,
) -> asyncio.Task:
    """
    Create a heartbeat task for an async context.

    Args:
        queue: asyncio.Queue for heartbeats
        interval: Interval in seconds

    Returns:
        asyncio.Task that emits heartbeats

    Example:
        >>> queue = asyncio.Queue()
        >>> heartbeat_task = create_heartbeat_task(queue, interval=30)
        >>> try:
        ...     # Do long-running work
        ...     result = await some_long_operation()
        ... finally:
        ...     heartbeat_task.cancel()
    """
    return asyncio.create_task(emit_heartbeat(queue, interval=interval))


def create_reconnectable_sse_handler() -> dict:
    """
    Create configuration for reconnectable SSE handler.

    Returns handler metadata supporting automatic reconnection on client side.

    Returns:
        Dictionary with:
        - max_reconnect_delay: Maximum backoff delay (seconds)
        - initial_reconnect_delay: Initial reconnect delay (seconds)
        - reconnect_strategy: 'exponential' or 'fixed'
        - heartbeat_interval: Expected heartbeat interval (seconds)

    Example:
        >>> handler_config = create_reconnectable_sse_handler()
        >>> # Send to client, client uses for reconnection logic
    """
    return {
        "max_reconnect_delay": 30,
        "initial_reconnect_delay": 1,
        "reconnect_strategy": "exponential",
        "backoff_multiplier": 2.0,
        "max_retries": 5,
        "heartbeat_interval": 15,
    }


class SSEHeartbeatManager:
    """
    Manages heartbeat tasks for multiple concurrent SSE streams.

    Coordinates heartbeats across multiple client connections during
    a pipeline run.
    """

    def __init__(self, interval: int = 15, config: Optional[HeartbeatConfig] = None):
        """
        Initialize heartbeat manager.

        Args:
            interval: Default heartbeat interval (seconds)
            config: HeartbeatConfig instance
        """
        self.interval = interval
        self.config = config or HeartbeatConfig(interval=interval)
        self.queues: dict = {}
        self.tasks: dict = {}
        self.start_time = time.time()

    async def add_stream(self, stream_id: str, queue: asyncio.Queue) -> None:
        """
        Add a new SSE stream to receive heartbeats.

        Args:
            stream_id: Unique identifier for stream
            queue: asyncio.Queue for this stream
        """
        self.queues[stream_id] = queue

        # Start heartbeat task for this stream
        task = asyncio.create_task(
            emit_heartbeat(
                queue,
                interval=self.interval,
                start_time=self.start_time,
                config=self.config,
            )
        )
        self.tasks[stream_id] = task

        if self.config.verbose:
            print(f"[SSE] Added stream {stream_id}")

    def remove_stream(self, stream_id: str) -> None:
        """
        Remove an SSE stream and cancel its heartbeat.

        Args:
            stream_id: Identifier of stream to remove
        """
        if stream_id in self.tasks:
            self.tasks[stream_id].cancel()
            del self.tasks[stream_id]

        if stream_id in self.queues:
            del self.queues[stream_id]

        if self.config.verbose:
            print(f"[SSE] Removed stream {stream_id}")

    async def broadcast_message(self, message: str) -> None:
        """
        Broadcast a message to all connected streams.

        Args:
            message: Message to broadcast (SSE format)
        """
        for queue in self.queues.values():
            try:
                await queue.put(message)
            except asyncio.QueueFull:
                if self.config.verbose:
                    print("[SSE] Queue full, skipping message")

    def get_active_streams_count(self) -> int:
        """
        Get number of active SSE streams.

        Returns:
            Count of connected streams
        """
        return len(self.queues)

    def get_uptime(self) -> int:
        """
        Get elapsed time since manager started.

        Returns:
            Elapsed seconds
        """
        return int(time.time() - self.start_time)


async def emit_status_update(
    queue: asyncio.Queue,
    stage: str,
    status: str,
    details: Optional[dict] = None,
) -> None:
    """
    Emit a status update message to SSE queue.

    Convenience function for agent status reporting.

    Args:
        queue: asyncio.Queue
        stage: Current stage name
        status: Status message
        details: Optional dictionary of additional info

    Example:
        >>> await emit_status_update(queue, "pd_modeling", "training_candidate", {"model": "xgboost"})
    """
    message = f"stage={stage}|status={status}"

    if details:
        for key, value in details.items():
            message = f"{message}|{key}={value}"

    message = f"data: {message}\n\n"
    await queue.put(message)


async def monitor_with_heartbeat(
    operation: Callable[..., Any],
    queue: asyncio.Queue,
    heartbeat_interval: int = 15,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run an async operation with concurrent heartbeat monitoring.

    Starts heartbeat in background while operation executes. Ensures
    client stays connected even if operation takes time.

    Args:
        operation: Async callable to execute
        queue: SSE queue for heartbeats
        heartbeat_interval: Heartbeat interval (seconds)
        *args: Arguments to operation
        **kwargs: Keyword arguments to operation

    Returns:
        Result of operation

    Example:
        >>> async def slow_operation():
        ...     await asyncio.sleep(120)  # 2 minutes
        ...     return "done"
        >>> queue = asyncio.Queue()
        >>> result = await monitor_with_heartbeat(slow_operation, queue)
    """
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(
        emit_heartbeat(queue, interval=heartbeat_interval)
    )

    try:
        # Run the main operation
        result = await operation(*args, **kwargs)
        return result
    finally:
        # Cancel heartbeat when done
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
