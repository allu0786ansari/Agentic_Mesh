"""Phase 5 data (ADR 0021): checksum verification and the sealed final paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from agmesh.data import phase5
from agmesh.evaluation import sealed


def test_repository_config_seals_the_final_data_and_lists_every_hai_file() -> None:
    cfg = phase5.load_config()
    rel = cfg["hai"]["releases"]
    assert rel["hai-22.04"]["role"] == "development" and len(rel["hai-22.04"]["files"]) == 10
    assert rel["hai-23.05"]["role"] == "final" and "hai-test1.csv" in rel["hai-23.05"]["files"]
    assert cfg["modbus2023"]["scenarios"]["compromised-scada"]["role"] == "final"
    assert {p.as_posix() for p in sealed.sealed_paths()} == {
        "data/raw/hai/hai-23.05", "data/raw/modbus2023/compromised-scada"}  # fmt: skip


def test_verify_hai_checks_size_and_sha256(tmp_path: Path) -> None:
    good, bad = b"timestamp,x\n1,2\n", b"other"
    cfg = {"hai": {"releases": {"hai-x": {"role": "development", "dir": "d", "files": {
        "a.csv": {"sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good)},
        "b.csv": {"sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good)},
        "c.csv": {"sha256": "0" * 64, "bytes": 1}}}}}}  # fmt: skip
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.csv").write_bytes(good)
    (tmp_path / "d" / "b.csv").write_bytes(bad)
    rep = phase5.verify_hai(cfg, tmp_path)
    assert rep["hai-x/a.csv"]["ok"] and not rep["hai-x/b.csv"]["ok"]
    assert rep["hai-x/c.csv"]["problem"] == "missing"


def test_sealed_paths_refuse_reads_before_the_final_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "phase5.yaml"
    sealed_dir = tmp_path / "data" / "final"
    sealed_dir.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump({"sealed": [sealed_dir.as_posix()]}))
    sealed.assert_not_sealed(tmp_path / "data" / "dev.csv", config=cfg)  # not sealed: fine
    with pytest.raises(sealed.SealedTestError, match="sealed final data"):
        sealed.assert_not_sealed(sealed_dir / "x.csv", config=cfg)
    monkeypatch.setattr(sealed, "frozen_problem", lambda *a, **k: "no g5-frozen tag")
    acc = sealed.FinalAccess("G5/x", 0, log_path=tmp_path / "log")
    with pytest.raises(sealed.SealedTestError, match="g5-frozen"):
        sealed.assert_not_sealed(sealed_dir / "x.csv", acc, config=cfg)
    monkeypatch.setattr(sealed, "frozen_problem", lambda *a, **k: None)
    sealed.assert_not_sealed(sealed_dir / "x.csv", acc, config=cfg)
    assert (tmp_path / "log").read_text().count("G5/x") == 1
