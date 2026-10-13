"""Tests for the Gate G0 data inspection on synthetic HAI and Modbus layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agmesh.feasibility.data_inspect import (
    classify_capture,
    gate_hints,
    inspect_hai,
    inspect_modbus,
    label_events,
    split_columns,
)
from tests.fixtures import write_hai_csv, write_lfs_pointer, write_pcap

DEVICES = ["ied1a", "ied1b", "scada-hmi"]
T0 = 1_686_000_000.0


def test_split_columns_finds_time_and_labels() -> None:
    time_col, labels, features = split_columns(["time", "P1_A", "attack", "attack_P2", "P2_B"])
    assert time_col == "time"
    assert labels == ["attack", "attack_P2"]
    assert features == ["P1_A", "P2_B"]


def test_label_events_counts_contiguous_runs() -> None:
    label = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    seconds = np.arange(len(label), dtype=float)
    events = label_events(label, seconds)
    assert [e["rows"] for e in events] == [2, 1, 3]
    assert events[0]["start_s"] == 1.0
    assert label_events(np.zeros(4), np.arange(4.0)) == []


def test_inspect_hai(tmp_path: Path) -> None:
    root = tmp_path / "hai-21.03"
    write_hai_csv(root / "train1.csv.gz", hours=13, level_shift_every_s=6 * 3600, seed=1)
    write_hai_csv(root / "train2.csv.gz", hours=13, level_shift_every_s=6 * 3600, seed=2)
    write_hai_csv(root / "test1.csv.gz", hours=2, attack_windows=[(100, 400), (3000, 3100)])
    write_lfs_pointer(root / "test2.csv.gz")

    report = inspect_hai(root, block_hours=6, n_clients=2)
    assert report["status"] == "ok"
    assert report["release_hint"] == "hai-21.03"
    assert report["n_features"] == 5
    assert report["feature_lists_consistent"]
    assert report["test_attack_events"] == 2
    assert report["lfs_pointer_files"] == ["test2.csv.gz"]
    test1 = report["files"]["test"][0]
    assert test1["median_interval_s"] == pytest.approx(1.0)
    assert test1["events"]["duration_s_max"] == pytest.approx(300.0)
    regime = report["regime_analysis"]
    assert regime["complete_train_blocks"] == 4
    assert regime["candidates"][0]["feature"] == "P1_F00"  # the feature with level shifts


def test_inspect_hai_missing_dir(tmp_path: Path) -> None:
    assert inspect_hai(tmp_path / "nope", 6, 4)["status"] == "missing"


def test_classify_capture() -> None:
    assert classify_capture(Path("attack/ied1a/x.pcap"), DEVICES) == ("attack", "ied1a")
    assert classify_capture(Path("benign/scada_hmi/y.pcap"), DEVICES) == ("benign", "scada-hmi")
    assert classify_capture(Path("benign/ied4c/z.pcap"), DEVICES) == ("benign", "ied4c")
    assert classify_capture(Path("z.pcap"), DEVICES) == ("unknown", "unknown")


ATTACKER = "185.175.0.7"
HMI = "185.175.0.3"


def _modbus_layout(
    root: Path, attack_offsets: dict[str, float], attacked: tuple[str, ...] = ("ied1a",)
) -> None:
    for device in DEVICES:
        write_pcap(
            root / "benign" / device / "b1.pcap", [T0, T0 + 3600], hosts=[(HMI, "10.0.0.1")] * 2
        )
    for device, offset in attack_offsets.items():
        start = T0 + 10_000 + offset
        src = ATTACKER if device in attacked else HMI
        write_pcap(
            root / "attack" / device / "a1.pcap",
            [start, start + 25, start + 1800],
            hosts=[(HMI, "10.0.0.1"), (src, "10.0.0.1"), (HMI, "10.0.0.1")],
        )


def test_inspect_modbus_reports_overlap_and_logs(tmp_path: Path) -> None:
    root = tmp_path / "modbus2023"
    _modbus_layout(root, {"ied1a": 0.0, "ied1b": 600.0, "scada-hmi": 5000.0})
    (root / "docs").mkdir()
    (root / "docs" / "attack_log.csv").write_text("time,attack\n1,T0855\n", encoding="utf-8")

    report = inspect_modbus(root, DEVICES, attacker_ip=ATTACKER)
    assert report["n_captures"] == 6
    assert report["captures_by_split_and_device"]["attack"] == {
        "ied1a": 1,
        "ied1b": 1,
        "scada-hmi": 1,
    }
    attack = report["overlap"]["attack"]
    assert attack["pairwise_overlap_s"]["ied1a|ied1b"] == pytest.approx(1200.0)
    assert attack["pairwise_overlap_s"]["ied1a|scada-hmi"] == 0.0
    assert attack["all_devices_overlap_s"] == 0.0
    assert report["overlap"]["benign"]["all_devices_overlap_s"] == pytest.approx(3600.0)
    assert report["possible_label_or_log_files"] == ["docs/attack_log.csv"]
    assert report["mitre_ids_in_text_files"] == {"attack_log.csv": ["T0855"]}

    assert report["attacker_by_split_and_device"]["attack"]["ied1a"] == {"packets": 1, "windows": 1}
    assert report["attacker_by_split_and_device"]["attack"]["ied1b"] == {"packets": 0, "windows": 0}
    hints = gate_hints({"status": "ok", "release_hint": "hai-21.03"}, report, "time")
    assert hints["devices_overlap_in_time"] is True
    assert hints["attacker_seen_on_devices"] == ["ied1a"]
    assert hints["rq2_fusion_data"] == "single_device_attack"
    assert hints["site_a_partition_scheme"] == "time"
    assert hints["site_b_attack_labels"] == "attack_logs_available"
    assert hints["site_b_attribution_ground_truth"] == "dataset_mitre_mapping_found"


def test_gate_hints_multi_device_attack(tmp_path: Path) -> None:
    root = tmp_path / "modbus2023"
    _modbus_layout(root, {"ied1a": 0.0, "ied1b": 0.0}, attacked=("ied1a", "ied1b"))
    hints = gate_hints({"status": "ok"}, inspect_modbus(root, DEVICES, attacker_ip=ATTACKER))
    assert hints["rq2_fusion_data"] == "real_multi_device_attack"
    assert hints["site_b_attack_labels"] == "attacker_address"


def test_gate_hints_without_overlap_or_logs(tmp_path: Path) -> None:
    root = tmp_path / "modbus2023"
    _modbus_layout(root, {"ied1a": 0.0, "ied1b": 3000.0}, attacked=())
    hints = gate_hints({"status": "ok"}, inspect_modbus(root, DEVICES, attacker_ip=ATTACKER))
    assert hints["rq2_fusion_data"] == "no_attacker_traffic_found"
    assert hints["site_b_attack_labels"] == "capture_level_only"
    assert hints["site_b_attribution_ground_truth"] == "needs_mapping_protocol_or_manual_check"


def test_main_writes_artifact_with_hints(tmp_path: Path) -> None:
    from agmesh.common.artifacts import read_artifact
    from agmesh.feasibility.data_inspect import main

    params = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"
    hai = tmp_path / "hai-21.03"
    write_hai_csv(hai / "train1.csv.gz", hours=1)
    write_hai_csv(hai / "test1.csv.gz", hours=1, attack_windows=[(10, 20)])
    modbus = tmp_path / "modbus2023"
    _modbus_layout(modbus, {"ied1a": 0.0, "ied1b": 0.0})
    out = tmp_path / "inspection.json"
    code = main(
        [
            "--params",
            str(params),
            "--hai-dir",
            str(hai),
            "--modbus-dir",
            str(modbus),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    document = read_artifact(out)
    assert document["kind"] == "feasibility.data_inspection"
    assert document["g0_hints"]["hai_usable"] is True
    assert document["g0_hints"]["rq2_fusion_data"] == "single_device_attack"
    checks = {c["check"]: c for c in document["expectation_checks"]}
    assert checks["hai.test_attack_events"]["actual"] == 1
    assert (
        checks["hai.test_attack_events"]["ok"] is False
    )  # synthetic data differs from the real files


def test_main_reports_missing_data(tmp_path: Path) -> None:
    from agmesh.feasibility.data_inspect import main

    params = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"
    code = main(
        [
            "--params",
            str(params),
            "--hai-dir",
            str(tmp_path / "a"),
            "--modbus-dir",
            str(tmp_path / "b"),
            "--output",
            str(tmp_path / "o.json"),
        ]
    )
    assert code == 1


def test_check_expectations_flags_differences() -> None:
    from agmesh.feasibility.data_inspect import check_expectations

    actual = {"hai": {"rows": 10, "events": 2}, "modbus": {"captures": {"attack": {"ied1a": 1}}}}
    expected = {"hai": {"rows": 10, "events": 3}, "modbus": {"captures": {"attack": {"ied1a": 1}}}}
    result = {c["check"]: c["ok"] for c in check_expectations(actual, expected)}
    assert result == {"hai.rows": True, "hai.events": False, "modbus.captures.attack.ied1a": True}


def test_repository_expectations_file_is_valid() -> None:
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "data_expectations.yaml"
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert expected["hai"]["test_attack_events"] == 50
    assert expected["modbus"]["attacker_windows"]["ied1a"] == 51
