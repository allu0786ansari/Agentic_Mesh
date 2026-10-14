"""Model inputs, networks, baselines and the X1 baseline runner (fixture data only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agmesh.common.config import Params
from agmesh.data.access import load_partition
from agmesh.evaluation.detection import evaluate_node, macro
from agmesh.models import inputs as mi
from agmesh.models.baselines import IForestDetector, PcaDetector, RuleDetector
from tests.conftest import REPO_PARAMS

torch = pytest.importorskip("torch")
from agmesh.models import nets  # noqa: E402  (needs torch)


def _root(run: tuple[dict[str, Any], Path]) -> Path:
    return run[1].parent


def test_scaler_clips_and_survives_constant_columns() -> None:
    x = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    scaler = mi.Scaler.fit(x, clip=1.0)
    z = scaler.transform([[100.0, 5.0], [2.0, 5.0]])
    assert z.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        mi.Scaler.fit(np.empty((0, 2)), clip=1.0)


def test_site_a_inputs_segment_means_and_flat(
    params: Params, site_a_run: tuple[dict[str, Any], Path]
) -> None:
    part = load_partition(params, "site_a", "a1", "train", root=_root(site_a_run))
    scaler = mi.Scaler.fit(mi.site_a_raw(part, params), 6.0)
    inp = mi.site_a_input(part, scaler, params)
    assert inp.segments == 6 and inp.n_inputs == 6 * len(inp.base_features)
    assert len(inp) == site_a_run[0]["nodes"]["a1"]["partitions"]["train"]["windows"]
    assert not inp.label.any()
    raw = params.model_dump()
    raw["models"]["vae"]["input_mode"] = "flat"
    raw["models"]["vae"]["exclude_features"] = ["P1_FT01"]
    flat = mi.site_a_input(
        part,
        scaler.__class__.fit(mi.site_a_raw(part, Params.model_validate(raw)), 6.0),
        Params.model_validate(raw),
    )
    assert flat.segments == 60 and "P1_FT01" not in flat.base_features
    errors = np.ones((len(flat), flat.n_inputs), dtype=np.float32)
    assert mi.per_feature(errors, flat).shape == (len(flat), len(flat.base_features))
    raw["models"]["vae"]["exclude_features"] = ["nope"]
    with pytest.raises(ValueError, match="unknown"):
        mi.site_a_raw(part, Params.model_validate(raw))


def test_site_b_inputs_drop_identity_features(
    params: Params, site_b_run: tuple[dict[str, Any], Path]
) -> None:
    part = load_partition(params, "site_b", "b1", "val_attack", root=_root(site_b_run))
    raw = mi.site_b_raw(part, params)
    inp = mi.site_b_input(part, mi.Scaler.fit(raw, 6.0), params)
    assert "new_peers" not in inp.base_features and "peers" not in inp.base_features
    assert inp.n_inputs == 16 and inp.label.sum() == part.labels.target.sum()


def test_networks_are_opacus_safe_and_learn() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 12)).astype(np.float32)
    for kind in ("vae", "ae"):
        model = nets.build(kind, 12, [16, 8], latent=4, seed=3)
        assert all(isinstance(m, nets.OPACUS_SAFE_LAYERS) for m in model.modules())
        result = nets.fit(
            model, x[:400], x[400:], lr=1e-2, batch=64, max_epochs=8, patience=3, seed=1
        )
        assert result.history[result.best_epoch - 1]["val_loss"] <= result.history[0]["val_loss"]
        errors = nets.reconstruction_errors(model, x[:5])
        assert errors.shape == (5, 12) and (errors >= 0).all()
        again = nets.build(kind, 12, [16, 8], latent=4, seed=3)
        nets.fit(again, x[:400], x[400:], lr=1e-2, batch=64, max_epochs=8, patience=3, seed=1)
        assert nets.state_sha256(again) == nets.state_sha256(model)  # seeded and deterministic
        assert nets.parameter_count(model) > 0


def test_reconstruction_scores_flag_outliers() -> None:
    rng = np.random.default_rng(1)
    normal = rng.normal(size=(600, 6)).astype(np.float32)
    weird = normal[:20] + 8.0
    pca = PcaDetector(0.95).fit(normal)
    forest = IForestDetector(50, 128, seed=0).fit(normal)
    for detector in (pca, forest):
        assert detector.score(weird).mean() > detector.score(normal).mean()
    assert pca.n_components >= 1 and pca.errors(weird).shape == (20, 6)
    assert RuleDetector(2).score(np.eye(3)).tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(RuntimeError):
        PcaDetector(0.9).score(normal)
    with pytest.raises(RuntimeError):
        IForestDetector(5, 5, 0).score(normal)


def test_evaluate_node_and_macro() -> None:
    starts = np.arange(20) * 10.0
    labels = np.zeros(20, dtype=bool)
    labels[10:12] = True
    scores = labels.astype(float) + 0.01 * np.arange(20) / 20
    out = evaluate_node(
        np.zeros(100),
        scores,
        labels,
        starts,
        window_length_s=10,
        stride_s=10,
        events=[(100.0, 120.0)],
        target_fpr=0.01,
        tolerance_s=60,
    )
    assert out["auroc"] == 1.0 and out["event_recall"] == 1.0
    assert out["window_fpr"] == 17 / 18  # every normal window but the first is above 0
    assert macro({"a": {"x": 1.0}, "b": {"x": float("nan")}, "c": {}}, ("x", "y")) == {
        "x": 1.0,
        "y": None,
    }


def test_x1_runner_writes_checkable_artifacts(
    params: Params,
    site_a_run: tuple[dict[str, Any], Path],
    site_b_run: tuple[dict[str, Any], Path],
    tmp_path: Path,
) -> None:
    from agmesh.evaluation.checker import check
    from agmesh.experiments import x1_baselines

    raw = params.model_dump()
    raw["models"]["vae"]["max_epochs"] = 2
    raw["models"]["ae"]["max_epochs"] = 2
    quick = Params.model_validate(raw)
    parts = _root(site_a_run)
    assert _root(site_b_run) == parts
    summary = x1_baselines.run(
        quick,
        ["site_a", "site_b"],
        ["vae", "ae", "pca", "rule_new_peer"],
        parts,
        tmp_path / "results",
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    ids = {row["config_id"] for row in summary}
    assert ids == {
        "site_a__local__vae",
        "site_a__central__vae",
        "site_a__local__pca",
        "site_a__central__pca",
        "site_b__local__ae",
        "site_b__central__ae",
        "site_b__local__pca",
        "site_b__central__pca",
        "site_b__local__rule_new_peer",
    }
    doc = json.loads(Path(summary[0]["path"]).read_text(encoding="utf-8"))
    assert (
        doc["split"] == "val"
        and doc["data_source"] == "fixture"
        and set(doc["per_client"]) == {"a1", "a2", "a3", "a4"}
    )
    rule = json.loads(
        (tmp_path / "results/X1/site_b__local__rule_new_peer/seed_11.json").read_text(
            encoding="utf-8"
        )
    )
    assert rule["per_client"]["b1"]["event_recall"] == 1.0
    assert rule["per_client"]["b2"]["events"] == 0
    findings = check(tmp_path / "results", tmp_path / "log")
    assert findings and all(any("fixture" in p for p in f.problems) for f in findings)
