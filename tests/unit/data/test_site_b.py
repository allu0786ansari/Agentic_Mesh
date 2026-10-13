"""Site B: window grid, idle cut, per-node vocabulary, labels, episodes and the partition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.config import Params
from agmesh.data import site_b
from agmesh.data.checks import structural_site_b


def test_grid_keeps_only_whole_windows_inside_captures() -> None:
    windows = site_b.grid_windows([(103.0, 151.0), (200.0, 230.0)], 10)
    assert windows.tolist() == [11, 12, 13, 14, 20, 21, 22]


def test_idle_intervals_and_drop() -> None:
    activity = np.array([10.0, 20.0, 200.0, 210.0])
    intervals = site_b.idle_intervals(activity, [(0.0, 300.0)], max_gap=60.0)
    assert intervals == [(20.0, 200.0), (210.0, 300.0)]
    windows = np.arange(0, 30)
    keep = site_b.drop_idle(windows, intervals, 10)
    kept = windows[keep].tolist()
    assert kept == [0, 1, 20]


def test_partition_labels_only_attacker_windows(site_b_run: tuple[dict[str, Any], Path]) -> None:
    manifest, _out = site_b_run
    assert all(c["ok"] for c in structural_site_b(manifest))
    assert len(manifest["episodes"]) == 6
    by_device = {n["device"]: n for n in manifest["nodes"].values()}
    assert by_device["ied1a"]["attack_day_attack_windows"] > 0
    assert by_device["ied1b"]["attack_day_attack_windows"] == 0
    assert by_device["scada-hmi"]["attack_day_attack_windows"] == 0
    assert by_device["ied1a"]["idle_windows_dropped"]["benign"] >= 30  # the 5-minute idle stretch
    b1 = manifest["nodes"]["b1"]["partitions"]
    assert b1["val_attack"]["episodes"] == [0, 1, 2, 3]
    assert b1["test"]["episodes"] == [4, 5]


def test_features_flag_what_the_attacker_changed(site_b_run: tuple[dict[str, Any], Path]) -> None:
    _manifest, out = site_b_run
    train = pd.read_parquet(out / "b1" / "train.parquet")
    attack = pd.concat(pd.read_parquet(out / "b1" / f"{p}.parquet") for p in ("val_attack", "test"))
    hit = attack[attack["label"] == 1]
    normal = attack[attack["label"] == 0]
    for column in ("new_peers", "unseen_addr", "oversized_qty", "unanswered", "stacked_frames"):
        assert train[column].sum() == 0, column
    assert (hit["new_peers"] >= 1).all()
    assert hit["unseen_addr"].sum() > 0 and hit["oversized_qty"].sum() > 0
    assert hit["stacked_frames"].sum() == 1
    assert normal[["new_peers", "unseen_addr", "oversized_qty"]].to_numpy().sum() == 0
    assert list(train.columns[2 : 2 + len(site_b.FEATURES)]) == list(site_b.FEATURES)


def test_every_node_shares_one_schema(site_b_run: tuple[dict[str, Any], Path]) -> None:
    _manifest, out = site_b_run
    columns = {
        c: list(pd.read_parquet(out / c / "train.parquet").columns) for c in ("b1", "b2", "b3")
    }
    assert columns["b1"] == columns["b2"] == columns["b3"]
    hmi = pd.read_parquet(out / "b3" / "train.parquet")
    assert hmi["req_sent"].sum() > 0 and hmi["req_recv"].sum() == 0
    assert (hmi["peers"] == 3).mean() > 0.9


def test_partition_is_deterministic(
    site_b_run: tuple[dict[str, Any], Path], params: Params, interim_dir: Path, tmp_path: Path
) -> None:
    manifest, _out = site_b_run
    assert site_b.partition(params, interim_dir, tmp_path / "site_b") == manifest
