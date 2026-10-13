"""Tests for hashing, git state and environment description."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agmesh.common.provenance import build_provenance, environment, sha256_file, utc_now_iso


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"agentic-mesh" * 300_000  # spans several read chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_utc_timestamp_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now_iso())


def test_environment_reports_machine_facts() -> None:
    env = environment(lockfile="does-not-exist.lock")
    assert env["memory_total_mb"] > 0
    assert env["cpu_count_logical"] >= 1
    assert env["lockfile_sha256"] is None
    assert len(env["hostname_hash"]) == 12


def test_build_provenance_includes_params_hash(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text("a: 1\n", encoding="utf-8")
    record = build_provenance(params_path=params, extra={"note": "x"})
    assert record["params_sha256"] == sha256_file(params)
    assert record["note"] == "x"
    assert {"git_sha", "git_branch", "git_dirty", "created_at", "environment"} <= record.keys()
