"""Replay → alert on fixture partitions for both sites (Plan Week 8 ``make edge``).

A randomly initialised network stands in for the frozen global model. The runtime
is model-agnostic, so this exercises everything after training:

* calibration;
* streaming scores against batch scores;
* evidence records in the node stores and alerts on the bus;
* deterministic IDs;
* the sealed test split.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from agmesh.common.config import Params
from agmesh.common.schemas import Alert, EvidenceRecord
from agmesh.data.access import Site, load_partition
from agmesh.edge import runtime
from agmesh.edge.bundle import GlobalModel
from agmesh.experiments import x1_baselines as x1
from agmesh.federated.protocol import ModelSpec, Normalisation
from agmesh.federated.server import initial_weights


def _model(params: Params, site: Site, root: Path, out: Path) -> Path:
    train = load_partition(params, site, x1._clients(params, site)[0], "train", root=root)
    raw = x1._raw(site, train, params)
    norm = Normalisation(raw.mean(axis=0), np.where(raw.std(axis=0) < 1e-6, 1.0, raw.std(axis=0)))
    if site == "site_a":
        cfg = params.models.vae
        spec = ModelSpec(
            "vae", raw.shape[1] * cfg.segments_per_window, tuple(cfg.hidden), cfg.latent
        )
    else:
        spec = ModelSpec("ae", raw.shape[1], tuple(params.models.ae.hidden), 0)
    model = GlobalModel(site, spec, initial_weights(spec, 7), norm, {"note": "random init (test)"})
    model.save(out / site)
    assert GlobalModel.load(out / site).version == model.version
    return out / site


@pytest.mark.parametrize("site", ["site_a", "site_b"])
def test_replay_produces_valid_alerts_and_evidence(
    params: Params,
    site: Site,
    site_a_run: tuple[dict[str, Any], Path],
    site_b_run: tuple[dict[str, Any], Path],
    tmp_path: Path,
) -> None:
    root = site_a_run[1].parent
    model_dir = _model(params, site, root, tmp_path / "models")
    paths = [
        runtime.run(params, site, model_dir, root, tmp_path / f"edge{i}", data_source="fixture")
        for i in (1, 2)
    ]
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    out = paths[0].parent
    assert summary["data_source"] == "fixture" and summary["split"] == "val"
    for name, check in summary["stream_vs_batch"].items():
        assert check["scored_mask_mismatches"] == 0, name
        assert check["max_rel_diff"] < 1e-3, name
        assert check["alert_decision_mismatches"] <= max(1, check["windows"] // 1000), name
    alerts_total = sum(s.get("alerts", 0) for s in summary["streams"].values())
    assert alerts_total > 0 and summary["bus"]["messages"] == alerts_total
    assert sum(summary["evidence"]["records"].values()) == alerts_total
    for node, c in summary["nodes"].items():
        assert c["threshold_val_fpr"] <= params.evaluation.target_window_fpr + 1e-9, node
        assert (out / f"calibration_{node}.json").is_file()

    con = sqlite3.connect(out / "storage.db")
    payloads = [json.loads(p) for (p,) in con.execute("SELECT payload FROM bus_messages")]
    alerts = [Alert.model_validate(p) for p in payloads]
    assert {a.site_id for a in alerts} == {site}
    node = alerts[0].node_id
    ev_con = sqlite3.connect(out / f"evidence_{node}.db")
    rec = EvidenceRecord.model_validate_json(
        ev_con.execute(
            "SELECT record FROM evidence WHERE evidence_id = ?", (alerts[0].evidence_id,)
        ).fetchone()[0]
    )
    rec.check_site()
    assert (rec.protocol_summary is not None) == (site == "site_b")
    lines = (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == alerts_total and json.loads(lines[0])["event"] == "detection"
    assert summary["detection_latency_ms"]["p95"] < 1000.0

    # deterministic IDs: a second replay of the same model and data yields the same alerts
    con2 = sqlite3.connect(paths[1].parent / "storage.db")
    ids1 = sorted(r[0] for r in con.execute("SELECT msg_id FROM bus_messages"))
    ids2 = sorted(r[0] for r in con2.execute("SELECT msg_id FROM bus_messages"))
    assert ids1 == ids2


def test_test_split_is_sealed(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    model_dir = _model(params, "site_a", root, tmp_path / "models")
    with pytest.raises(SystemExit, match="--final"):
        runtime.run(params, "site_a", model_dir, root, tmp_path / "edge", split="test")
