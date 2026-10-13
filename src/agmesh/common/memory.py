"""Background memory sampling for the resource benchmarks (Week 1 spikes, X7).

Resident set size (RSS) double-counts shared library pages when several processes
load the same libraries (for example, every Ray worker loads torch). On Linux the
sampler therefore also records proportional set size (PSS), which splits shared
pages between the processes that use them and is the fairer total.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import psutil

ProcessSelector = Callable[[], Iterable[psutil.Process]]

_MB = 2**20


def process_tree(root_pid: int | None = None) -> ProcessSelector:
    """Select a process and all of its descendants (default: the current process)."""

    def select() -> list[psutil.Process]:
        try:
            root = psutil.Process(root_pid)
            return [root, *root.children(recursive=True)]
        except psutil.Error:
            return []

    return select


def processes_matching(substring: str) -> ProcessSelector:
    """Select every process whose name or command line contains ``substring``."""
    needle = substring.lower()

    def select() -> list[psutil.Process]:
        found: list[psutil.Process] = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            except psutil.Error:
                continue
            if needle in name or needle in cmdline:
                found.append(proc)
        return found

    return select


@dataclass
class MemorySample:
    t: float
    rss_mb: float
    pss_mb: float | None
    n_processes: int
    system_used_mb: float


@dataclass
class MemorySampler:
    """Sample memory of the selected processes on a background thread."""

    selector: ProcessSelector
    interval_s: float = 0.5
    samples: list[MemorySample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _t0: float = field(default=0.0, repr=False)

    def sample_once(self) -> MemorySample:
        rss = 0
        pss = 0
        pss_available = True
        count = 0
        for proc in self.selector():
            try:
                rss += proc.memory_info().rss
                count += 1
                if pss_available:
                    full = proc.memory_full_info()
                    value = getattr(full, "pss", None)
                    if value is None:
                        pss_available = False
                    else:
                        pss += value
            except psutil.AccessDenied:
                pss_available = False
            except psutil.Error:
                continue
        sample = MemorySample(
            t=round(time.monotonic() - self._t0, 3),
            rss_mb=round(rss / _MB, 1),
            pss_mb=round(pss / _MB, 1) if pss_available and count else None,
            n_processes=count,
            system_used_mb=round(psutil.virtual_memory().used / _MB, 1),
        )
        self.samples.append(sample)
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> MemorySampler:
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.interval_s + 1)
        self.sample_once()

    def __enter__(self) -> MemorySampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def summary(
        self, baseline_system_used_mb: float | None = None
    ) -> dict[str, float | int | None]:
        if not self.samples:
            return {"n_samples": 0}
        pss_values = [s.pss_mb for s in self.samples if s.pss_mb is not None]
        peak_system = max(s.system_used_mb for s in self.samples)
        result: dict[str, float | int | None] = {
            "n_samples": len(self.samples),
            "peak_rss_mb": max(s.rss_mb for s in self.samples),
            "peak_pss_mb": max(pss_values) if pss_values else None,
            "peak_processes": max(s.n_processes for s in self.samples),
            "peak_system_used_mb": peak_system,
        }
        if baseline_system_used_mb is not None:
            result["peak_system_delta_mb"] = round(peak_system - baseline_system_used_mb, 1)
        return result
