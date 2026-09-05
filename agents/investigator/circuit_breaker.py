from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Awaitable, Callable, TypeVar


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Async circuit breaker with CLOSED, OPEN, and HALF_OPEN states."""

    def __init__(self, *, failure_threshold: int = 3, recovery_timeout: float = 5.0):
        if failure_threshold < 1 or recovery_timeout <= 0:
            raise ValueError("failure_threshold must be positive and recovery_timeout must be greater than zero")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if self.state is CircuitState.OPEN:
                if self.opened_at is None or asyncio.get_running_loop().time() - self.opened_at < self.recovery_timeout:
                    raise CircuitOpenError("circuit is open")
                self.state = CircuitState.HALF_OPEN
        try:
            result = await operation()
        except Exception:
            async with self._lock:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.opened_at = asyncio.get_running_loop().time()
            raise
        async with self._lock:
            self.consecutive_failures = 0
            self.state = CircuitState.CLOSED
            self.opened_at = None
        return result


class CircuitBreakerRegistry:
    def __init__(self, *, failure_threshold: int = 3, recovery_timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._breakers: dict[str, CircuitBreaker] = {}

    def for_node(self, node_id: str) -> CircuitBreaker:
        if not node_id.strip():
            raise ValueError("node_id must not be empty")
        if node_id not in self._breakers:
            self._breakers[node_id] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
            )
        return self._breakers[node_id]


__all__ = ["CircuitBreaker", "CircuitBreakerRegistry", "CircuitOpenError", "CircuitState"]
