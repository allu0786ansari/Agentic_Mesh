"""The sealed test set, result artifacts and the checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agmesh.common.config import Params
from agmesh.common.provenance import utc_now_iso
from agmesh.data.access import load_partition
from agmesh.evaluation import checker
from agmesh.evaluation.results import DETECTION_METRICS, write_result
from agmesh.evaluation.sealed import FinalAccess, SealedTestError, read_log
from tests.conftest import REPO_PARAMS

METRICS = {name: 0.5 for name in DETECTION_METRICS}


def _root(site_a_run: tuple[dict[str, Any], Path]) -> Path:
    return site_a_run[1].parent


def test_test_split_is_sealed(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = _root(site_a_run)
    val = load_partition(params, "site_a", "a1", "val_attack", root=root)
    assert len(val.features) == len(val.labels.target) > 0
    with pytest.raises(SealedTestError, match="sealed"):
        load_partition(params, "site_a", "a1", "test", root=root)
    log = tmp_path / "test_access.log"
    unfrozen = FinalAccess("cfg", 11, log_path=log, params_path=REPO_PARAMS)
    with pytest.raises(SealedTestError, match="g4-frozen"):
        load_partition(params, "site_a", "a1", "test", root=root, final=unfrozen)
    allowed = FinalAccess("cfg", 11, log_path=log, params_path=REPO_PARAMS, require_frozen=False)
    test = load_partition(params, "site_a", "a1", "test", root=root, final=allowed)
    assert test.labels.target.sum() > 0
    entries = read_log(log)
    assert len(entries) == 1
    assert entries[0]["config_id"] == "cfg" and entries[0]["seed"] == 11 and entries[0]["git_sha"]
    with pytest.raises(ValueError, match="unknown partition"):
        load_partition(params, "site_a", "a1", "holdout", root=root)
    with pytest.raises(KeyError):
        load_partition(params, "site_a", "a9", "train", root=root)


def _write(root: Path, **overrides: Any) -> Path:
    args: dict[str, Any] = {
        "experiment": "X1",
        "config_id": "site_a__fedavg__eps10",
        "seed": 11,
        "split": "val",
        "data_source": "runtime",
        "metrics": METRICS,
        "manifest_sha256": "abc",
        "started_at": utc_now_iso(),
        "params_path": REPO_PARAMS,
        "scores": {"nodes": np.array(["a1"]), "a1__scores": np.zeros(3, np.float32)},
    }
    args.update(overrides)
    return write_result(root, **args)


def test_write_result_layout_and_schema(tmp_path: Path) -> None:
    path = _write(tmp_path)
    assert path == tmp_path / "X1" / "site_a__fedavg__eps10" / "seed_11.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["kind"] == "result" and doc["provenance"]["manifest_sha256"] == "abc"
    with pytest.raises(ValueError):
        _write(tmp_path, config_id="bad/id")
    with pytest.raises(ValueError):
        _write(tmp_path, experiment="X9")


def _problems(findings: list[checker.Finding], name: str) -> list[str]:
    return next(f.problems for f in findings if name in f.path)


def test_checker_accepts_runtime_and_rejects_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    log = tmp_path / "test_access.log"
    good = _write(results, config_id="good")
    _write(results, config_id="fixture", data_source="fixture")
    _write(results, config_id="partial", metrics={"auroc": 0.9})
    _write(results, config_id="sealed", split="test")
    findings = checker.check(results, log)
    dirty = json.loads(good.read_text(encoding="utf-8"))["provenance"]["git_dirty"]
    assert any("fixture" in p for p in _problems(findings, "fixture"))
    assert any("metrics missing" in p for p in _problems(findings, "partial"))
    assert any("access log" in p for p in _problems(findings, "sealed"))
    good_problems = _problems(findings, "good")
    if dirty:
        assert len(good_problems) == 1
        assert good_problems[0].startswith("produced from a working tree with uncommitted")
    else:
        assert good_problems == []


def test_checker_verifies_score_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    log = tmp_path / "log"
    _write(results, config_id="none", scores=None)
    tampered = _write(results, config_id="tampered")
    gone = _write(results, config_id="gone")
    tampered.with_name("seed_11.scores.npz").write_bytes(b"changed")
    gone.with_name("seed_11.scores.npz").unlink()
    findings = checker.check(results, log)
    assert any("no per-window score file" in p for p in _problems(findings, "none"))
    assert any("does not match its recorded hash" in p for p in _problems(findings, "tampered"))
    assert any("is missing" in p for p in _problems(findings, "gone"))
    doc = json.loads((results / "X1" / "none" / "seed_11.json").read_text(encoding="utf-8"))
    assert "scores_file" not in doc


def test_checker_matches_the_access_log(tmp_path: Path) -> None:
    results = tmp_path / "results"
    path = _write(results, config_id="sealed", split="test")
    doc = json.loads(path.read_text(encoding="utf-8"))
    log = tmp_path / "test_access.log"
    log.write_text(
        json.dumps({"config_id": "sealed", "seed": 11, "git_sha": doc["provenance"]["git_sha"]})
        + "\n",
        encoding="utf-8",
    )
    problems = checker.check(results, log)[0].problems
    assert not any("access log" in p for p in problems)


def test_checker_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results"
    _write(results, config_id="fixture", data_source="fixture")
    (results / "X2" / "broken").mkdir(parents=True)
    (results / "X2" / "broken" / "seed_1.json").write_text("{not json", encoding="utf-8")
    code = checker.main(["--results", str(results), "--access-log", str(tmp_path / "log")])
    out = capsys.readouterr().out
    assert code == 1 and "REJECTED" in out and "0/2 result artifacts accepted" in out
    assert checker.main(["--results", str(tmp_path / "empty")]) == 0
