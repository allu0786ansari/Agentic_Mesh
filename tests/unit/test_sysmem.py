"""Memory readings from /proc (ADR 0013 §6), on a fake /proc tree."""

from __future__ import annotations

from pathlib import Path

from agmesh.common.sysmem import available_mb, llm_server_rss_mb


def _proc(tmp: Path) -> Path:
    (tmp / "meminfo").write_text("MemTotal: 6000000 kB\nMemAvailable:  163840 kB\n")
    for pid, comm, rss in (("10", "llama-server", 5237996), ("11", "bash", 2856), ("12", "x", 0)):
        d = tmp / pid
        d.mkdir()
        (d / "comm").write_text(comm + "\n")
        (d / "status").write_text(f"Name: {comm}\nVmRSS:\t{rss} kB\n")
    (tmp / "self").mkdir()
    return tmp


def test_readings(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    assert available_mb(proc) == 160
    assert llm_server_rss_mb(proc) == 5115


def test_missing_proc_gives_none(tmp_path: Path) -> None:
    assert available_mb(tmp_path / "nope") is None
    assert llm_server_rss_mb(tmp_path / "nope") is None
    (tmp_path / "1").mkdir()
    assert llm_server_rss_mb(tmp_path) is None
