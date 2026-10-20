"""X7 resources (ADR 0019 §1): roles, peaks, the budget and out-of-memory exits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agmesh.evaluation import resources as rs


def test_roles_from_command_lines() -> None:
    assert rs.role_of(["python", "-m", "agmesh.nodeapi.serve", "--node", "b1"]) == "node_api"
    assert rs.role_of(["python", "-m", "agmesh.mesh.runner", "--site", "site_a"]) == "mesh"
    assert rs.role_of(["/usr/local/bin/ollama", "serve"]) == "llm"
    assert rs.role_of(["/usr/lib/ollama/llama-server", "--model", "x"]) == "llm"
    assert rs.role_of(["python", "-m", "agmesh.edge.runtime"]) == "edge"
    assert rs.role_of(["bash"]) is None


def _line(used: float, swap: float, procs: list[tuple[str, float]]) -> dict[str, Any]:
    return {"t": 0.0, "system": {"total_mb": 5927.0, "used_mb": used, "swap_used_mb": swap},
            "processes": [{"role": r, "pss_mb": v} for r, v in procs]}  # fmt: skip


def test_summarise_takes_peaks_per_role_and_flags_the_budget() -> None:
    lines: list[dict[str, Any]] = [
        {"meta": {"mode": "watch"}},
        _line(3000, 0, [("node_api", 100), ("node_api", 110), ("llm", 2200)]),
        _line(5100, 300, [("node_api", 105), ("llm", 2400), ("mesh", 600)]),
        {"exit": {"returncode": 0}},
    ]
    s = rs.summarise(lines)
    assert s["peak_pss_mb_by_role"] == {"llm": 2400.0, "mesh": 600.0, "node_api": 210.0}
    assert s["peak_stage_pss_mb"] == 3105.0
    assert s["peak_system_used_mb"] == 5100.0 and s["peak_swap_used_mb"] == 300.0
    assert s["within_budget"]
    over = rs.summarise([_line(6100, 0, []), {"exit": {"returncode": 0}}])
    assert not over["within_budget"]
    killed = rs.summarise([_line(1000, 0, []), {"exit": {"returncode": -9}}])
    assert killed["likely_oom_kill"] and not killed["within_budget"]


def test_run_samples_a_command_and_report_needs_both_stages(tmp_path: Path) -> None:
    cmd = [sys.executable, "-c", "import time; x = bytearray(30_000_000); time.sleep(1.2)"]
    path, code = rs.run_command("training", cmd, out=tmp_path)
    assert code == 0
    import json

    lines = [json.loads(x) for x in path.read_text().splitlines()]
    s = rs.summarise(lines)
    assert s["samples"] >= 2 and s["peak_pss_mb_by_role"]["training"] > 25
    rep = json.loads(rs.report(tmp_path).read_text())
    assert rep["missing_stages"] == ["operations"] and not rep["passed"]
