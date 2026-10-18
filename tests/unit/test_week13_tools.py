"""Week 13 tools (ADR 0017): session export for X4, and the five-seed model check."""

from __future__ import annotations

import json
from pathlib import Path

from agmesh.experiments import x3_models, x4_export
from agmesh.mesh.sessions import SessionStore


def test_export_writes_one_record_per_session_without_evidence_text(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "storage.db")
    sid = store.open("site_a", ["x1", "x2"], "run")[0].session_id
    incident = {"alert_ids": ["x1", "x2"], "start_s": 10.0, "end_s": 70.0}
    store.record_step(sid, "triage", "ok", {**incident, "nodes": ["a1"], "severity": "high"})
    store.record_step(sid, "attribute", "full", {
        "attribution_mode": "full", "technique_id": "T0836", "tactics": ["Impair Process Control"],
        "ranked": ["T0836", "T0806", "T0814", "T0801"], "confidence": 0.8,
        "candidates": ["T0836"] * 10, "evidence_text": "SECRET-EVIDENCE",
        "llm": {"seconds": 14.2, "validation": {"schema_valid": True, "grounded": True,
                                                 "valid": True, "technique_id": "T0836"}},
    })  # fmt: skip
    store.conn.close()
    out = x4_export.export(tmp_path)
    (rec,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert rec["incident"] == {"start_s": 10.0, "end_s": 70.0, "nodes": ["a1"],
                               "severity": "high", "alerts": 2}  # fmt: skip
    assert rec["ranked_top3"] == ["T0836", "T0806", "T0814"] and rec["candidates_n"] == 10
    assert rec["llm"]["grounded"] is True and rec["llm"]["seconds"] == 14.2
    assert "SECRET-EVIDENCE" not in out.read_text()


def test_verify_flags_mismatches_and_missing_models(tmp_path: Path) -> None:
    rerun, ref, models = tmp_path / "rerun", tmp_path / "ref", tmp_path / "models"
    for root, auroc in ((rerun, 0.9), (ref, 0.9)):
        for site in x3_models.SITES:
            for seed in x3_models.SEEDS:
                p = root / "X1" / x3_models.CONFIG.format(site=site) / f"seed_{seed}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"metrics": {"auroc": auroc, "ttd_s": None}}))
    for site in x3_models.SITES:
        for seed in x3_models.SEEDS:
            d = x3_models.model_dir(site, seed, models)
            d.mkdir(parents=True)
            (d / "model.json").write_text("{}")
    assert x3_models.verify(rerun, ref, models) == []
    bad = rerun / "X1" / "site_b__fedprox__eps3" / "seed_23.json"
    bad.write_text(json.dumps({"metrics": {"auroc": 0.9001}}))
    (x3_models.model_dir("site_a", 11, models) / "model.json").unlink()
    problems = x3_models.verify(rerun, ref, models)
    assert any("seed 23: auroc" in p for p in problems) and any("no model" in p for p in problems)
