"""Per-node circuit breaker (Proposal §8.3; ADR 0012 §3).

``closed`` → ``open`` after ``failure_threshold`` consecutive failures. While open, a
call is refused until ``open_seconds`` have passed. Then one probe is allowed
(``half_open``): success closes the breaker, failure opens it again. Every state
change is passed to ``on_transition`` (the runner writes it as a metric event).
"""

from __future__ import annotations

import time
from collections.abc import Callable

Clock = Callable[[], float]


class CircuitBreaker:
    def __init__(
        self,
        node: str,
        *,
        failure_threshold: int = 2,
        open_seconds: float = 30.0,
        clock: Clock = time.monotonic,
        on_transition: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.node, self.threshold, self.open_seconds = node, failure_threshold, open_seconds
        self.clock = clock
        self.on_transition = on_transition
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0

    def _move(self, to: str) -> None:
        if to != self.state:
            frm, self.state = self.state, to
            if self.on_transition is not None:
                self.on_transition(self.node, frm, to)

    def allow(self) -> bool:
        if self.state == "open":
            if self.clock() - self.opened_at >= self.open_seconds:
                self._move("half_open")
                return True
            return False
        return True

    def success(self) -> None:
        self.failures = 0
        self._move("closed")

    def failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.opened_at = self.clock()
            self._move("open")
