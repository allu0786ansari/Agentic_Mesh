"""Contract (CC1): labels can never reach model or scoring code as features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agmesh.common.config import Params
from agmesh.common.frames import LABEL_COLUMNS, LabelFreeFrame, LabelLeakError, is_label_column
from agmesh.data.access import load_partition
from agmesh.data.manifest import PARTITIONS
from agmesh.evaluation.diagnostics import label_proxies
from agmesh.evaluation.sealed import FinalAccess
from tests.conftest import REPO_PARAMS


def test_label_columns_are_refused() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "attack": [0, 1], "label": [0, 1]})
    for bad in (["x", "attack"], ["label"], ["attack_P2"]):
        with pytest.raises(LabelLeakError):
            LabelFreeFrame((*bad,), np.zeros((2, len(bad)), dtype=np.float32))
    with pytest.raises(LabelLeakError):
        LabelFreeFrame.from_frame(frame, ["x"], forbidden=["x"])
    with pytest.raises(KeyError):
        LabelFreeFrame.from_frame(frame, ["missing"])
    assert all(is_label_column(c) for c in LABEL_COLUMNS)


def test_features_are_read_only() -> None:
    frame = LabelFreeFrame.from_frame(pd.DataFrame({"x": [1.0, 2.0]}), ["x"])
    with pytest.raises(ValueError):
        frame.values[0, 0] = 5.0
    assert len(frame.take([1])) == 1 and frame.n_features == 1


@pytest.mark.parametrize("site", ["site_a", "site_b"])
def test_every_partition_is_label_free(
    site: str,
    params: Params,
    site_a_run: tuple[dict[str, Any], Path],
    site_b_run: tuple[dict[str, Any], Path],
    tmp_path: Path,
) -> None:
    root = site_a_run[1].parent if site == "site_a" else site_b_run[1].parent
    clients = ("a1", "a2", "a3", "a4") if site == "site_a" else ("b1", "b2", "b3")
    final = FinalAccess(
        "contract", 0, log_path=tmp_path / "log", params_path=REPO_PARAMS, require_frozen=False
    )
    for client in clients:
        for part in PARTITIONS:
            p = load_partition(params, site, client, part, root=root, final=final)  # type: ignore[arg-type]
            assert isinstance(p.features, LabelFreeFrame)
            assert not set(p.features.columns) & LABEL_COLUMNS
            assert not any(is_label_column(c) for c in p.features.columns)
            assert len(p.features) == len(p.labels.target)
            # a renamed label column would reproduce the label on Site A, where every
            # feature is a sensor reading; Site B's identity features are proxies by
            # construction and are reported by evaluation.diagnostics instead
            if site == "site_a" and p.labels.target.any() and not p.labels.target.all():
                copies = [
                    x for x in label_proxies(p.features, p.labels.target) if x.identical_to_label
                ]
                assert not copies


def test_site_b_identity_features_are_reported_as_proxies(
    params: Params, site_b_run: tuple[dict[str, Any], Path]
) -> None:
    root = site_b_run[1].parent
    p = load_partition(params, "site_b", "b1", "val_attack", root=root)
    names = {proxy.feature for proxy in label_proxies(p.features, p.labels.target)}
    assert "new_peers" in names
    exact = [x for x in label_proxies(p.features, p.labels.target) if x.identical_to_label]
    assert "new_peers" in [x.feature for x in exact]
