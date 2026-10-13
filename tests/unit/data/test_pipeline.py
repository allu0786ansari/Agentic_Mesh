"""The command-line stages run end to end on the fixture and record a check artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agmesh.data import pipeline
from tests.unit.data.conftest import REPO_PARAMS


def test_cli_runs_all_stages(fixture_paths: dict[str, Path], tmp_path: Path) -> None:
    interim, parts, out = tmp_path / "interim", tmp_path / "parts", tmp_path / "check.json"
    base = ["--params", str(REPO_PARAMS)]
    assert (
        pipeline.main(
            [
                *base,
                "prepare",
                "--modbus-dir",
                str(fixture_paths["modbus_dir"]),
                "--interim-dir",
                str(interim),
            ]
        )
        == 0
    )
    assert (
        pipeline.main(
            [
                *base,
                "partition",
                "--hai-dir",
                str(fixture_paths["hai_dir"]),
                "--interim-dir",
                str(interim),
                "--out",
                str(parts),
            ]
        )
        == 0
    )
    code = pipeline.main(
        [
            *base,
            "check",
            "--out",
            str(parts),
            "--no-expectations",
            "--fixture",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["data_source"] == "fixture"
    assert artifact["passed"] == artifact["total"] > 0


def test_expectations_fail_on_fixture(
    fixture_paths: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real-data expectations must not pass on synthetic data."""
    interim, parts = tmp_path / "interim", tmp_path / "parts"
    base = ["--params", str(REPO_PARAMS)]
    pipeline.main(
        [
            *base,
            "prepare",
            "--modbus-dir",
            str(fixture_paths["modbus_dir"]),
            "--interim-dir",
            str(interim),
        ]
    )
    pipeline.main(
        [
            *base,
            "partition",
            "--hai-dir",
            str(fixture_paths["hai_dir"]),
            "--interim-dir",
            str(interim),
            "--out",
            str(parts),
        ]
    )
    expectations = REPO_PARAMS.parent / "partition_expectations.yaml"
    assert yaml.safe_load(expectations.read_text(encoding="utf-8"))["site_a"]["events_total"] == 50
    code = pipeline.main(
        [
            *base,
            "check",
            "--out",
            str(parts),
            "--expectations",
            str(expectations),
            "--fixture",
            "--output",
            str(tmp_path / "c.json"),
        ]
    )
    assert code == 1
    assert "FAILED site_a.events_total" in capsys.readouterr().out
