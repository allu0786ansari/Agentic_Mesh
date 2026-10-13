"""Tests for raw-data checksum recording and verification."""

from __future__ import annotations

from pathlib import Path

from agmesh.common.checksums import is_lfs_pointer, record, verify
from tests.fixtures import write_lfs_pointer


def _make_raw(root: Path) -> None:
    (root / "hai").mkdir(parents=True)
    (root / "hai" / "train1.csv.gz").write_bytes(b"a" * 2000)
    (root / "modbus").mkdir()
    (root / "modbus" / "x.pcap").write_bytes(b"b" * 100)
    write_lfs_pointer(root / "hai" / "test9.csv.gz")


def test_record_lists_files_and_lfs_pointers(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _make_raw(root)
    payload = record(root, tmp_path / "sums.json")
    assert payload["n_files"] == 3
    assert payload["lfs_pointers"] == ["hai/test9.csv.gz"]


def test_verify_detects_changes(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _make_raw(root)
    manifest = tmp_path / "sums.json"
    record(root, manifest)
    assert verify(root, manifest).ok

    (root / "modbus" / "x.pcap").write_bytes(b"c" * 100)
    (root / "hai" / "train1.csv.gz").unlink()
    (root / "new.bin").write_bytes(b"new")
    report = verify(root, manifest)
    assert not report.ok
    assert report.changed == ["modbus/x.pcap"]
    assert report.missing == ["hai/train1.csv.gz"]
    assert report.unexpected == ["new.bin"]


def test_lfs_pointer_detection(tmp_path: Path) -> None:
    pointer = write_lfs_pointer(tmp_path / "p.csv.gz")
    real = tmp_path / "r.csv.gz"
    real.write_bytes(b"version" + b"x" * 5000)
    assert is_lfs_pointer(pointer)
    assert not is_lfs_pointer(real)
