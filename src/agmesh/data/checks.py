"""Integrity and label checks over the partition manifests (Implementation Plan, Week 2).

Structural checks hold for any data (fixture or real): no event in two partitions,
no attack labels in training or normal validation, partitions in time order on
Site B, every Site A client holding attack segments. Expectation checks compare
the counts with ``configs/partition_expectations.yaml`` and only make sense on the
analysed datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agmesh.common.config import Params

Check = dict[str, Any]


def _check(name: str, ok: bool, detail: Any = None) -> Check:
    return {"check": name, "ok": bool(ok), "detail": detail}


def structural_site_a(manifest: dict[str, Any], params: Params) -> list[Check]:
    checks: list[Check] = []
    features = manifest["features"]["features"]
    labels = {params.data.hai.label_column, *params.data.hai.process_label_columns}
    checks.append(_check("site_a.no_label_column_is_a_feature", not labels & set(features)))
    segments = manifest["events"]["segments"]
    ids = [s["segment_id"] for s in segments]
    checks.append(_check("site_a.each_event_in_one_partition", len(ids) == len(set(ids))))
    for client, node in manifest["nodes"].items():
        parts = node["partitions"]
        clean = parts["train"]["attack_rows"] == 0 and parts["val_normal"]["attack_rows"] == 0
        checks.append(_check(f"site_a.{client}.train_and_val_normal_unlabelled", clean))
        has_events = bool(parts["val_attack"]["events"]) and bool(parts["test"]["events"])
        checks.append(_check(f"site_a.{client}.has_attack_segments", has_events))
        if manifest["scheme"] == "time":  # iid blocks are shuffled, so no time order
            ordered = (parts["train"]["end"] or "") < (parts["val_normal"]["start"] or "~")
            checks.append(_check(f"site_a.{client}.val_normal_after_train", ordered))
    return checks


def structural_site_b(manifest: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for client, node in manifest["nodes"].items():
        parts = node["partitions"]
        clean = parts["train"]["attack_windows"] == 0 and parts["val_normal"]["attack_windows"] == 0
        checks.append(_check(f"site_b.{client}.train_and_val_normal_unlabelled", clean))
        order = [parts[p]["start"] for p in ("train", "val_normal", "val_attack", "test")]
        ends = [parts[p]["end"] for p in ("train", "val_normal", "val_attack", "test")]
        in_order = all(
            ends[i] is None or order[i + 1] is None or ends[i] < order[i + 1] for i in range(3)
        )
        checks.append(_check(f"site_b.{client}.partitions_in_time_order", in_order))
        overlap = set(parts["val_attack"]["episodes"]) & set(parts["test"]["episodes"])
        checks.append(_check(f"site_b.{client}.each_episode_in_one_partition", not overlap))
        labelled = parts["val_attack"]["attack_windows"] + parts["test"]["attack_windows"]
        lost = node["attack_day_attack_windows"] - labelled
        checks.append(_check(f"site_b.{client}.no_attack_window_lost_at_boundary", lost == 0, lost))
    return checks


def expected_site_a(manifest: dict[str, Any], expected: dict[str, Any]) -> list[Check]:
    actual = {
        "n_features": manifest["features"]["n_features"],
        "training_constant_features": len(manifest["features"]["training_constant"]),
        "events_total": manifest["events"]["total"],
        "events_val_attack": manifest["events"]["val_attack"],
        "events_test": manifest["events"]["test"],
    }
    return [
        _check(
            f"site_a.{key}",
            actual.get(key) == value,
            {"expected": value, "actual": actual.get(key)},
        )
        for key, value in expected.items()
    ]


def expected_site_b(manifest: dict[str, Any], expected: dict[str, Any]) -> list[Check]:
    episodes = manifest["episodes"]
    by_device = {node["device"]: node for node in manifest["nodes"].values()}
    checks = [
        _check(
            "site_b.episodes",
            len(episodes) == expected["episodes"],
            {"expected": expected["episodes"], "actual": len(episodes)},
        ),
        _check(
            "site_b.val_attack_episodes",
            sum(e["partition"] == "val_attack" for e in episodes)
            == expected["val_attack_episodes"],
        ),
    ]
    for device, count in expected["attack_windows"].items():
        actual = by_device[device]["attack_day_attack_windows"] if device in by_device else None
        checks.append(
            _check(
                f"site_b.{device}.attack_windows",
                actual == count,
                {"expected": count, "actual": actual},
            )
        )
    dropped = sum(
        n["idle_windows_dropped"]["attack_day_with_attacker_traffic"] for n in by_device.values()
    )
    checks.append(
        _check(
            "site_b.attack_windows_dropped_as_idle",
            dropped == expected["attack_windows_dropped_as_idle"],
            {"expected": expected["attack_windows_dropped_as_idle"], "actual": dropped},
        )
    )
    return checks


def run_checks(
    manifests: dict[str, dict[str, Any]], params: Params, expected: dict[str, Any] | None
) -> list[Check]:
    checks: list[Check] = []
    if "site_a" in manifests:
        checks += structural_site_a(manifests["site_a"], params)
        if expected and "site_a" in expected:
            checks += expected_site_a(manifests["site_a"], expected["site_a"])
    if "site_b" in manifests:
        checks += structural_site_b(manifests["site_b"])
        if expected and "site_b" in expected:
            checks += expected_site_b(manifests["site_b"], expected["site_b"])
    return checks


def partition_dirs(root: Path) -> dict[str, Path]:
    return {
        site: root / site
        for site in ("site_a", "site_b")
        if (root / site / "manifest.json").is_file()
    }
