"""Figures are drawn from artifacts only (ADR 0016 §6): a smoke test on a tiny result tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from agmesh.experiments import figures


def _result(root: Path, exp: str, cid: str, seed: int, target: str) -> None:
    p = root / exp / cid / f"seed_{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    plan = {"noise_multiplier": 3.2, "steps": 600, "n_windows": 3060}
    clients = {"a1": {"epsilon": 2.99, "target_epsilon": 3.0, "delta": 1e-5, "plan": plan}}
    p.write_text(json.dumps({
        "split": "val", "config_id": cid, "target_epsilon": target,
        "metrics": {"auroc": 0.9, "auprc": 0.7},
        "privacy": {"clients": clients},
        "federation": {"round_log": [{"val_loss": 1.0 - 0.1 * r} for r in range(5)]},
    }))  # fmt: skip


def test_figures_and_privacy_table_come_from_artifacts(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    ci = {"observed": 0.9, "low": 0.85, "high": 0.95}
    configs = {f"X1/site_a__{s}__eps{e}": {"auroc": ci, "auprc": ci}
               for s in ("fedavg", "fedprox") for e in figures.EPS}  # fmt: skip
    configs |= {f"X2/site_a__{s}__eps3__het__drop0.2": {"auroc": ci} for s in ("fedavg", "fedprox")}
    configs |= {f"X1/site_a__{s}__eps3": {"auroc": ci} for s in ("fedavg", "fedprox")}
    for exp in ("X1", "X2"):
        (tmp_path / exp).mkdir(parents=True, exist_ok=True)
        (tmp_path / exp / "analysis_val.json").write_text(json.dumps({"configs": configs}))
    _result(tmp_path, "X1", "site_a__fedprox__eps3", 11, "3")
    _result(tmp_path, "X1", "site_a__central__vae", 11, "inf")
    assert figures.main(["--results", str(tmp_path), "--out", str(tmp_path / "fig")]) == 0
    out = tmp_path / "fig"
    assert (out / "privacy_utility_site_a_auroc.svg").is_file()
    assert (out / "convergence_site_a.png").is_file()
    table = (out / "privacy_table.md").read_text()
    assert "site_a__fedprox__eps3" in table and "central__vae" not in table  # non-private skipped
