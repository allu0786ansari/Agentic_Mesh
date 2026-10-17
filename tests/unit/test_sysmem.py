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


def test_server_info_reads_cache_ram_from_cmdline_or_environment(tmp_path: Path) -> None:
    from agmesh.common.sysmem import llm_server_info

    d = tmp_path / "10"
    d.mkdir()
    (d / "comm").write_text("llama-server\n")
    (d / "cmdline").write_bytes(b"/usr/local/lib/ollama/llama-server\0-c\x004096\0-np\x001\0")
    (d / "environ").write_bytes(b"HOME=/x\0LLAMA_ARG_CACHE_RAM=0\0")
    info = llm_server_info(tmp_path)
    assert info is not None
    assert info["cache_ram"] == "0" and info["env_readable"] and info["pid"] == 10
    (d / "cmdline").write_bytes(b"llama-server\0--cache-ram\x008192\0")
    info = llm_server_info(tmp_path)
    assert info is not None and info["cache_ram"] == "8192"
    (d / "environ").unlink()
    (d / "cmdline").write_bytes(b"llama-server\0")
    info = llm_server_info(tmp_path)
    assert info is not None and info["cache_ram"] is None and not info["env_readable"]
    assert llm_server_info(tmp_path / "none") is None
