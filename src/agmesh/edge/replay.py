"""Replay pacing: a window is emitted at its end time, scaled by a speed factor.

* ``speed = 1``: real time.
* ``speed = 60``: one data minute per wall second.
* ``speed = 0``: as fast as possible.

Gaps in the data longer than ``max_gap_s`` (between partitions, or across days) are
shortened to ``max_gap_s``, so a replay does not sleep for hours. ``emit`` returns
the wall time at which the window was handed to the scorer. Detection latency is
measured from that moment (ADR 0009 §8).
"""

from __future__ import annotations

import time
from collections.abc import Callable


class Pacer:
    def __init__(
        self,
        speed: float = 0.0,
        max_gap_s: float = 60.0,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.speed, self.max_gap = speed, max_gap_s
        self.clock, self.sleep = clock, sleep
        self.t_data: float | None = None
        self.due: float | None = None
        self.slept_s = 0.0

    def emit(self, t_end: float) -> float:
        """Wait until the window ending at data time ``t_end`` is due; return wall time."""
        now = self.clock()
        if self.speed == 0:
            self.t_data = t_end
            return now
        if self.t_data is None or self.due is None:
            self.t_data, self.due = t_end, now
            return now
        gap = max(0.0, min(t_end - self.t_data, self.max_gap))
        self.t_data = t_end
        self.due += gap / self.speed
        wait = self.due - now
        if wait > 0:
            self.sleep(wait)
            self.slept_s += wait
        return self.clock()
