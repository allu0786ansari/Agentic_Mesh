"""Tests for artifact writing and metric events."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from agmesh.common.artifacts import read_artifact, write_artifact, write_event


def test_artifact_round_trip(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "out" / "a.json",
        kind="test.kind",
        data_source="fixture",
        payload={"value": np.float32(0.5), "array": np.arange(3), "where": tmp_path},
        params_path=None,
    )
    document = read_artifact(path)
    assert document["kind"] == "test.kind"
    assert document["data_source"] == "fixture"
    assert document["value"] == pytest.approx(0.5)
    assert document["array"] == [0, 1, 2]
    assert "provenance" in document
    assert not list(path.parent.glob("*.tmp")), "temporary file left behind"


def test_non_finite_floats_become_strings(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "a.json",
        kind="k",
        data_source="runtime",
        payload={"eps": math.inf, "nested": [{"x": -math.inf}], "nan": math.nan},
        params_path=None,
    )
    document = read_artifact(path)
    assert document["eps"] == "inf"
    assert document["nested"][0]["x"] == "-inf"
    assert document["nan"] == "nan"


def test_invalid_data_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_source"):
        write_artifact(
            tmp_path / "a.json",
            kind="k",
            data_source="made_up",  # type: ignore[arg-type]
            payload={},
            params_path=None,
        )


def test_events_are_appended_as_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "metrics.jsonl"
    write_event(path, "round", {"loss": 1.0}, run_id="r1", data_source="runtime")
    write_event(path, "round", {"loss": 0.5}, run_id="r1", data_source="runtime", session_id="s")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["payload"]["loss"] for line in lines] == [1.0, 0.5]
    assert "session_id" not in lines[0]
    assert lines[1]["session_id"] == "s"
    assert all(line["data_source"] == "runtime" for line in lines)
