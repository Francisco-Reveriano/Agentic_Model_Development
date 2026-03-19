"""
Agent Timeout Management

Provides per-agent timeout configuration and wrapper functions to prevent long-running
agents from blocking the pipeline. Timeouts are configurable per agent type and can be
overridden at runtime.
"""

import functools
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class AgentTimeoutConfig:
    """
    Configuration for per-agent timeout limits.

    Each timeout represents the maximum wall-clock time an agent should spend
    executing. If exceeded, the agent task is interrupted and can be retried
    with reduced scope or parameters.

    All timeouts in seconds.
    """

    DATA_AGENT: int = 300  # 5 minutes
    FEATURE_AGENT: int = 600  # 10 minutes
    PD_AGENT: int = 1800  # 30 minutes
    LGD_AGENT: int = 1200  # 20 minutes
    EAD_AGENT: int = 900  # 15 minutes
    EL_AGENT: int = 300  # 5 minutes
    REPORT_AGENT: int = 300  # 5 minutes

    def get_timeout(self, agent_name: str) -> int:
        """
        Get timeout for a specific agent.

        Args:
            agent_name: Name of agent (e.g., "data_agent", "pd_agent")

        Returns:
            Timeout in seconds. Defaults to 600 if agent not found.
        """
        # Normalize agent name to config attribute
        attr_name = agent_name.upper().replace("-", "_")
        if not attr_name.endswith("_AGENT"):
            attr_name = attr_name.upper() + "_AGENT"

        return getattr(self, attr_name, 600)

    def set_timeout(self, agent_name: str, timeout_seconds: int) -> None:
        """
        Set timeout for a specific agent.

        Args:
            agent_name: Name of agent (e.g., "data_agent")
            timeout_seconds: New timeout value in seconds
        """
        attr_name = agent_name.upper().replace("-", "_")
        if not attr_name.endswith("_AGENT"):
            attr_name = attr_name.upper() + "_AGENT"

        if hasattr(self, attr_name):
            setattr(self, attr_name, timeout_seconds)


class TimeoutException(Exception):
    """Raised when an agent execution exceeds its timeout limit."""

    pass


def wrap_with_timeout(
    agent_call: Callable[..., Any],
    timeout_seconds: int,
    agent_name: str = "unknown_agent",
) -> Callable[..., Any]:
    """
    Wrap an agent call with a timeout decorator.

    If the agent takes longer than timeout_seconds, raises TimeoutException.
    Works on Unix-like systems using signal.SIGALRM. On Windows, timeout
    is checked via elapsed time (not guaranteed to interrupt immediately).

    Args:
        agent_call: Callable agent function or method
        timeout_seconds: Maximum execution time in seconds
        agent_name: Name of agent for logging/debugging

    Returns:
        Wrapped function that enforces timeout

    Raises:
        TimeoutException: If execution exceeds timeout

    Example:
        >>> from backend.agents.data_agent import create_data_agent
        >>> agent = create_data_agent()
        >>> wrapped_call = wrap_with_timeout(agent.run, 300, "data_agent")
        >>> result = wrapped_call(data)
    """

    @functools.wraps(agent_call)
    def timeout_handler(*args: Any, **kwargs: Any) -> Any:
        # Try Unix signal-based timeout first
        try:
            def signal_timeout_handler(signum, frame):
                raise TimeoutException(
                    f"Agent '{agent_name}' exceeded timeout of {timeout_seconds}s"
                )

            # Set alarm
            signal.signal(signal.SIGALRM, signal_timeout_handler)
            signal.alarm(timeout_seconds)

            try:
                result = agent_call(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)

            return result

        except AttributeError:
            # signal module not available (Windows), use elapsed time check
            start_time = time.time()

            result = agent_call(*args, **kwargs)

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutException(
                    f"Agent '{agent_name}' exceeded timeout of {timeout_seconds}s "
                    f"(took {elapsed:.1f}s)"
                )

            return result

    return timeout_handler


def create_timeout_config(overrides: Optional[dict] = None) -> AgentTimeoutConfig:
    """
    Create timeout configuration with optional overrides.

    Args:
        overrides: Dictionary of {agent_name: timeout_seconds} to override defaults

    Returns:
        AgentTimeoutConfig with applied overrides

    Example:
        >>> config = create_timeout_config({
        ...     "data_agent": 600,
        ...     "pd_agent": 3600
        ... })
        >>> timeout = config.get_timeout("data_agent")
        600
    """
    config = AgentTimeoutConfig()

    if overrides:
        for agent_name, timeout_seconds in overrides.items():
            config.set_timeout(agent_name, timeout_seconds)

    return config
