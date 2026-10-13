"""Tests for the background memory sampler."""

from __future__ import annotations

import time

from agmesh.common.memory import MemorySampler, process_tree, processes_matching


def test_sampler_measures_current_process() -> None:
    with MemorySampler(process_tree(), interval_s=0.05) as sampler:
        time.sleep(0.2)
    summary = sampler.summary(baseline_system_used_mb=0.0)
    n_samples = summary["n_samples"]
    assert isinstance(n_samples, int)
    assert n_samples >= 2
    peak_rss = summary["peak_rss_mb"]
    assert isinstance(peak_rss, float)
    assert peak_rss > 0
    assert summary["peak_processes"] == 1
    assert "peak_system_delta_mb" in summary


def test_selector_with_no_match_is_empty() -> None:
    sampler = MemorySampler(processes_matching("no-such-process-agmesh-xyz"))
    sample = sampler.sample_once()
    assert sample.n_processes == 0
    assert sample.rss_mb == 0
