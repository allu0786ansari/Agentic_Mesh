"""Site A: feature selection, event segments, stratified assignment and the partition."""

from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agmesh.common.config import Params
from agmesh.data import hai, site_a
from agmesh.data.checks import structural_site_a
from agmesh.data.manifest import read_manifest
from agmesh.data.windows import frame_window_starts


def _write(path: Path, frame: pd.DataFrame) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return path


def test_read_file_fills_short_gaps_and_drops_long_ones(tmp_path: Path, params: Params) -> None:
    values = np.arange(20, dtype=float)
    values[3] = np.nan  # one missing second: forward-filled
    values[10:17] = np.nan  # seven missing seconds: the last two rows are dropped
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2020-07-11", periods=20, freq="s").astype(str),
            "P1_X": values,
            "attack": 0,
            "attack_P1": 0,
            "attack_P2": 0,
            "attack_P3": 0,
        }
    )
    report = hai.LoadReport()
    out = hai.read_file(_write(tmp_path / "train1.csv.gz", frame), params.data.hai, report)
    assert len(out) == 18
    assert report.rows_dropped_missing == 2
    assert out.loc[3, "P1_X"] == 2.0
    assert out["attack"].dtype == np.int8


def test_feature_spec_drops_constant_and_duplicate(
    fixture_paths: dict[str, Path], params: Params
) -> None:
    frames, _paths, _report = hai.load(fixture_paths["hai_dir"], params.data.hai)
    spec = hai.build_feature_spec(
        [f for n, f in frames.items() if n.startswith("train")], params.data.hai
    )
    assert "P1_PCV02D" in spec.constant and "P1_PCV02D" not in spec.features
    assert spec.dropped == ["P4_HT_PS"] and "P4_ST_PS" in spec.features
    assert not {"attack", "attack_P1", "time"} & set(spec.features)


def _segments(counts: dict[str, int]) -> list[site_a.Segment]:
    out = []
    for combo, n in counts.items():
        for i in range(n):
            out.append(site_a.Segment(f"{combo}:{i:02d}", "test1", 0, 1, 0, 1, combo))
    return out


def test_assignment_matches_the_hai_event_mix() -> None:
    # HAI 21.03: 50 events by targeted process (data card)
    mix = {"P1": 23, "P2": 11, "P1+P2": 8, "P1+P3": 3, "P2+P3": 3, "P3": 2}
    result = site_a.assign_segments(_segments(mix), ["a1", "a2", "a3", "a4"], 0.30, seed=2026)
    val = Counter(a.segment.processes for a in result if a.partition == "val_attack")
    assert sum(val.values()) == 15
    assert val == {"P1": 7, "P2": 3, "P1+P2": 2, "P1+P3": 1, "P2+P3": 1, "P3": 1}
    for part in ("val_attack", "test"):
        per_client = Counter(a.client for a in result if a.partition == part)
        assert set(per_client) == {"a1", "a2", "a3", "a4"}
        assert max(per_client.values()) - min(per_client.values()) <= 1
    again = site_a.assign_segments(_segments(mix), ["a1", "a2", "a3", "a4"], 0.30, seed=2026)
    assert again == result


def test_event_segments_have_gaps_and_contain_their_event(params: Params) -> None:
    label = np.zeros(1000, dtype=np.int8)
    label[100:150] = 1
    label[600:700] = 1
    frame = pd.DataFrame({"attack": label, "attack_P1": label, "attack_P2": 0, "attack_P3": 0})
    segs = site_a.event_segments("test9", frame, params, gap_rows=60)
    assert [s.segment_id for s in segs] == ["test9:e00", "test9:e01"]
    first, second = segs
    assert first.start_row == 0 and second.stop_row == 1000
    assert second.start_row - first.stop_row == 60
    for s in segs:
        assert s.start_row <= s.event_start_row < s.event_stop_row <= s.stop_row
        assert s.processes == "P1"


def test_partition_on_fixture(site_a_run: tuple[dict[str, Any], Path], params: Params) -> None:
    manifest, out = site_a_run
    assert manifest == read_manifest(out)
    assert manifest["features"]["n_features"] == 6
    assert manifest["events"]["total"] == 14
    assert manifest["events"]["val_attack"] == 4
    assert all(c["ok"] for c in structural_site_a(manifest, params))
    length = params.data.windows.site_a.length_s
    for client, node in manifest["nodes"].items():
        for part, summary in node["partitions"].items():
            frame = pd.read_parquet(out / client / f"{part}.parquet")
            assert len(frame) == summary["rows"]
            starts = frame_window_starts(frame, length, length)
            runs_at_start = frame["run_id"].to_numpy()[starts]
            runs_at_end = frame["run_id"].to_numpy()[starts + length - 1]
            assert (runs_at_start == runs_at_end).all()
    a1 = pd.read_parquet(out / "a1" / "train.parquet")
    a1_val = pd.read_parquet(out / "a1" / "val_normal.parquet")
    gap = (a1_val["time"].min() - a1["time"].max()).total_seconds()
    assert gap > params.data.splits.gap_windows * length


def test_partition_is_deterministic(
    site_a_run: tuple[dict[str, Any], Path],
    params: Params,
    fixture_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    manifest, _out = site_a_run
    again = site_a.partition(params, fixture_paths["hai_dir"], tmp_path / "site_a")
    assert again == manifest


def test_regime_scheme_is_refused(
    params: Params, fixture_paths: dict[str, Path], tmp_path: Path
) -> None:
    raw = params.model_dump()
    raw["data"]["partition"]["site_a"]["scheme"] = "regime"
    with pytest.raises(ValueError, match="regime"):
        site_a.partition(Params.model_validate(raw), fixture_paths["hai_dir"], tmp_path)


def test_iid_scheme_balances_blocks(
    params: Params, fixture_paths: dict[str, Path], tmp_path: Path
) -> None:
    raw = params.model_dump()
    raw["data"]["partition"]["site_a"]["scheme"] = "iid"
    raw["data"]["partition"]["site_a"]["block_hours"] = 1
    manifest = site_a.partition(Params.model_validate(raw), fixture_paths["hai_dir"], tmp_path)
    runs = [
        n["partitions"]["train"]["runs"] + n["partitions"]["val_normal"]["runs"]
        for n in manifest["nodes"].values()
    ]
    assert len(set(runs)) == 1
    assert all(c["ok"] for c in structural_site_a(manifest, Params.model_validate(raw)))
