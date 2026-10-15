"""The bus interface shared by the SQLite and in-memory implementations.

**Semantics: at-least-once delivery, with idempotent consumers.**

* ``publish`` stores a message once per ``msg_id``. Publishing the same alert again
  is a no-op.
* ``claim`` leases up to ``limit`` available messages of a topic to one consumer,
  for ``lease_s`` seconds, and counts the attempt.
* ``ack`` succeeds only for the consumer holding the message. Call it after the
  handler's own writes have committed.
* An unacknowledged message becomes available again when its lease expires. After
  ``max_attempts`` claims it is dead-lettered instead.
* ``release`` gives a message back early, optionally with a delay, for example as
  backoff after a transient error.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

Clock = Callable[[], float]
TOPICS = ("alerts.site_a", "alerts.site_b", "actions.site_a", "actions.site_b", "approvals")
DEFAULT_LEASE_S = 30.0
MAX_ATTEMPTS = 5


def stamp(t: float) -> str:
    """Fixed-width UTC timestamp; lexicographic order equals time order."""
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def check_topic(topic: str) -> None:
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; expected one of {TOPICS}")


@dataclass(frozen=True)
class Message:
    msg_id: str
    topic: str
    payload: dict[str, Any]
    attempts: int


class Bus(Protocol):
    def publish(self, topic: str, payload: dict[str, Any], msg_id: str) -> bool: ...

    def claim(
        self, topic: str, consumer: str, *, lease_s: float = DEFAULT_LEASE_S, limit: int = 1
    ) -> list[Message]: ...

    def ack(self, msg_id: str, consumer: str) -> bool: ...

    def release(self, msg_id: str, consumer: str, delay_s: float = 0.0) -> bool: ...

    def pending(self, topic: str) -> int: ...

    def dead(self, topic: str) -> list[str]: ...


def system_clock() -> float:
    return time.time()
