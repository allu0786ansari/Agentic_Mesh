"""The completion tier and the CC2 access rule (ADR 0019 §6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agmesh.evaluation import completion, sealed
from agmesh.evaluation.checker import _access_problems


def _entry(client: str, run: str, sha: str = "abc", cid: str = "X1/c") -> dict[str, object]:
    return {"config_id": cid, "seed": 11, "client": client, "run_id": run, "git_sha": sha}


def test_access_rule_wants_every_client_of_the_run_at_its_commit() -> None:
    raw = {"provenance": {"git_sha": "abc"}, "access_run_id": "r2",
           "per_client": {"a1": {}, "a2": {}}, "validation": {"comparison": {}}}  # fmt: skip
    ok = [_entry("a1", "r1"), _entry("a1", "r2"), _entry("a2", "r2")]  # r1 crashed earlier
    assert _access_problems("X1", "c", 11, raw, ok) == []
    assert "clients without" in _access_problems("X1", "c", 11, raw, ok[:2])[0]
    wrong_commit = [_entry("a1", "r2", "zzz"), _entry("a2", "r2", "zzz")]
    assert "no access entry for this run" in _access_problems("X1", "c", 11, raw, wrong_commit)[0]
    assert "no matching entry" in _access_problems("X1", "c", 11, raw, [])[0]
    no_cmp = {**raw, "validation": {}}
    assert any("validation comparison" in p for p in _access_problems("X1", "c", 11, no_cmp, ok))


def test_completion_reports_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(completion, "frozen_sha", lambda *a, **k: "t" * 40)
    res = tmp_path / "results"
    (res / "reproduce").mkdir(parents=True)
    (res / "reproduce" / "reproduction.json").write_text(json.dumps({"passed": True}))
    (res / "X7").mkdir()
    (res / "X7" / "resources.json").write_text(json.dumps({"passed": False}))
    log = tmp_path / "log"
    log.write_text(
        json.dumps({"config_id": "edge__site_a", "seed": 0, "client": "a1", "frozen_sha": "x"})
        + "\n"
        + json.dumps(
            {"config_id": "protocol_inspection_week14", "out_of_band": True, "client": "b1"}
        )
        + "\n"
    )
    rep = completion.completion(res, log, run_contract_tests=False)
    assert rep["status"]["CC7"] and not rep["status"]["CC8"]
    assert not rep["status"]["CC6"]
    assert any("fusion_test.json" in p for p in rep["problems"]["CC6"])
    assert any("site_b" in p for p in rep["problems"]["CC6"])  # no test edge run for B
    assert any("frozen_sha x" in p for p in rep["problems"]["CC2"])
    assert any("out-of-band" in n for n in rep["notes"]["CC2"])
    assert not rep["passed"]
    assert sealed.FROZEN_TAG == "g4-frozen"


def test_a_waived_cc7_is_reported_as_waived_not_as_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(completion, "frozen_sha", lambda *a, **k: None)
    res = tmp_path / "results"
    (res / "reproduce").mkdir(parents=True)
    (res / "reproduce" / "waiver.json").write_text(
        json.dumps({"criterion": "CC7", "decided_by": "author", "reason": "time"})
    )
    rep = completion.completion(res, tmp_path / "log", run_contract_tests=False)
    assert "CC7" in rep["waived"] and "CC7" not in rep["status"]
    assert not rep["status"]["CC8"]  # still required
