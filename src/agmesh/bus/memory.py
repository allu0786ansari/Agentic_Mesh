"""In-memory bus with the same semantics as :class:`SqliteBus` (for unit tests)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from agmesh.bus.base import DEFAULT_LEASE_S, MAX_ATTEMPTS, Clock, Message, check_topic, system_clock


@dataclass
class _Row:
    msg_id: str
    topic: str
    payload: dict[str, Any]
    seq: int
    available_at: float
    attempts: int = 0
    lease_owner: str | None = None
    lease_until: float | None = None
    acked: bool = False
    failed: bool = False


class MemoryBus:
    def __init__(self, *, clock: Clock = system_clock, max_attempts: int = MAX_ATTEMPTS) -> None:
        self.clock, self.max_attempts = clock, max_attempts
        self.rows: dict[str, _Row] = {}

    def publish(self, topic: str, payload: dict[str, Any], msg_id: str) -> bool:
        check_topic(topic)
        if msg_id in self.rows:
            return False
        self.rows[msg_id] = _Row(
            msg_id, topic, copy.deepcopy(payload), len(self.rows), self.clock()
        )
        return True

    def _free(self, row: _Row, now: float) -> bool:
        return (
            not row.acked and not row.failed and (row.lease_until is None or row.lease_until <= now)
        )

    def claim(
        self, topic: str, consumer: str, *, lease_s: float = DEFAULT_LEASE_S, limit: int = 1
    ) -> list[Message]:
        check_topic(topic)
        now = self.clock()
        out: list[Message] = []
        for row in sorted(self.rows.values(), key=lambda r: r.seq):
            if row.topic != topic or not self._free(row, now):
                continue
            if row.attempts >= self.max_attempts:
                row.failed, row.lease_owner, row.lease_until = True, None, None
                continue
            if row.available_at > now or len(out) >= limit:
                continue
            row.lease_owner, row.lease_until = consumer, now + lease_s
            row.attempts += 1
            out.append(Message(row.msg_id, topic, copy.deepcopy(row.payload), row.attempts))
        return out

    def ack(self, msg_id: str, consumer: str) -> bool:
        row = self.rows.get(msg_id)
        if row is None or row.acked or row.lease_owner != consumer:
            return False
        row.acked, row.lease_owner, row.lease_until = True, None, None
        return True

    def release(self, msg_id: str, consumer: str, delay_s: float = 0.0) -> bool:
        row = self.rows.get(msg_id)
        if row is None or row.acked or row.lease_owner != consumer:
            return False
        row.lease_owner, row.lease_until = None, None
        row.available_at = self.clock() + delay_s
        return True

    def pending(self, topic: str) -> int:
        return sum(r.topic == topic and not r.acked and not r.failed for r in self.rows.values())

    def dead(self, topic: str) -> list[str]:
        return [
            r.msg_id
            for r in sorted(self.rows.values(), key=lambda r: r.seq)
            if r.topic == topic and r.failed
        ]
