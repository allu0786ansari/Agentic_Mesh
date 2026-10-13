"""Tests for the dependency-free pcap/pcapng timestamp reader and interval maths."""

from __future__ import annotations

from pathlib import Path

import pytest

from agmesh.feasibility.captures import (
    CaptureFormatError,
    capture_span,
    intersect,
    merge_intervals,
    scan_hosts,
    total_length,
)
from tests.fixtures import write_pcap, write_pcapng

T0 = 1_686_000_000.0  # an arbitrary 2023 epoch time


@pytest.mark.parametrize("nanoseconds", [False, True])
def test_pcap_span(tmp_path: Path, nanoseconds: bool) -> None:
    path = write_pcap(tmp_path / "a.pcap", [T0 + 5.25, T0, T0 + 60.5], nanoseconds=nanoseconds)
    span = capture_span(path)
    assert span.fmt == "pcap"
    assert span.n_packets == 3
    assert span.first_ts == pytest.approx(T0)
    assert span.last_ts == pytest.approx(T0 + 60.5)
    assert span.duration_s == pytest.approx(60.5)
    assert not span.truncated


@pytest.mark.parametrize("tsresol", [None, 9])
def test_pcapng_span(tmp_path: Path, tsresol: int | None) -> None:
    path = write_pcapng(tmp_path / "a.pcapng", [T0, T0 + 1.5, T0 + 30.0], tsresol=tsresol)
    span = capture_span(path)
    assert span.fmt == "pcapng"
    assert span.n_packets == 3
    assert span.first_ts == pytest.approx(T0, abs=1e-3)
    assert span.last_ts == pytest.approx(T0 + 30.0, abs=1e-3)


def test_truncated_pcap_is_flagged(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "a.pcap", [T0, T0 + 1])
    path.write_bytes(path.read_bytes()[:-2])
    assert capture_span(path).truncated


def test_non_capture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.pcap"
    path.write_text("hello world", encoding="utf-8")
    with pytest.raises(CaptureFormatError):
        capture_span(path)


def test_interval_maths() -> None:
    merged = merge_intervals([(5, 10), (0, 3), (2, 4), (9, 12)])
    assert merged == [(0, 4), (5, 12)]
    assert intersect(merged, [(3, 6), (11, 20)]) == [(3, 4), (5, 6), (11, 12)]
    assert total_length(merged) == 11
    assert intersect([(0, 1)], [(1, 2)]) == []


def test_scan_hosts_counts_hosts_and_watched_windows(tmp_path: Path) -> None:
    attacker, hmi, ied = "185.175.0.7", "185.175.0.3", "185.175.0.4"
    times = [T0, T0 + 1, T0 + 12, T0 + 15, T0 + 45]
    pairs = [(hmi, ied), (attacker, ied), (ied, attacker), (hmi, ied), (attacker, ied)]
    path = write_pcap(tmp_path / "a.pcap", times, hosts=pairs)
    scan = scan_hosts(path, attacker)
    assert scan.watched_packets == 3
    assert scan.watched_windows(10.0) == 3  # windows 0, 1 and 4 from the first packet
    assert scan.hosts[ied] == 5
    assert scan.hosts[attacker] == 3


def test_scan_hosts_ignores_non_ip_frames(tmp_path: Path) -> None:
    path = write_pcap(tmp_path / "a.pcap", [T0, T0 + 1])  # opaque 4-byte frames
    scan = scan_hosts(path, "185.175.0.7")
    assert scan.watched_packets == 0
    assert not scan.hosts
