from pathlib import Path

import pandas as pd

from scripts.prepare_week2_data import (
    NODE_SPECS,
    _aggregate_packets,
    _split_capture_paths,
    _write_node,
)


def test_modbus_packet_windows_have_stable_features_and_labels() -> None:
    packets = pd.DataFrame(
        {
            "timestamp_epoch": ["1000.1", "1000.9", "1011.0"],
            "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "dst_ip": ["10.0.0.2", "10.0.0.2", "10.0.0.1"],
            "src_port": ["1000", "1000", "1001"],
            "dst_port": ["502", "502", "502"],
            "frame_len": ["100", "120", "80"],
            "tcp_flags": ["0x0002", "0x0010", "0x0004"],
            "modbus_func": ["3", "16", "3"],
            "modbus_unit": ["1", "1", "1"],
            "tcp_stream": ["1", "1", "2"],
        }
    )

    windows = _aggregate_packets(packets, label=1, window_seconds=10)

    assert len(windows) == 2
    assert windows["label"].tolist() == [1, 1]
    assert windows.loc[0, "packet_count"] == 2
    assert windows.loc[0, "byte_count"] == 220
    assert windows.loc[0, "syn_count"] == 1
    assert windows.loc[0, "modbus_read_count"] == 1
    assert windows.loc[0, "modbus_write_count"] == 1


def test_modbus_capture_split_is_by_file(tmp_path: Path) -> None:
    (tmp_path / "benign").mkdir()
    (tmp_path / "attack").mkdir()
    for index in range(5):
        (tmp_path / "benign" / f"normal-{index}.pcap").write_bytes(b"pcap")
    (tmp_path / "attack" / "attack-0.pcap").write_bytes(b"pcap")

    train, held_out, attack = _split_capture_paths(tmp_path, "", seed=42)

    assert len(train) == 4
    assert len(held_out) == 1
    assert len(attack) == 1
    assert set(train).isdisjoint(held_out)


def test_six_node_contract_writes_registry_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "feature": [1.0], "label": [0]})
    metadata_paths = []
    for node in NODE_SPECS:
        metadata_paths.append(_write_node(node, frame, frame, tmp_path, seed=42))

    assert len(metadata_paths) == 6
    assert {path.parent.name for path in metadata_paths} == {f"node_{index:02d}" for index in range(1, 7)}
