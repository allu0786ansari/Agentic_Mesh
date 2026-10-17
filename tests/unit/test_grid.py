"""The X1/X2 matrix (ADR 0016): expansion, artifact paths, completeness, privacy rule."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from agmesh.experiments.grid import (
    MATRIX,
    completeness,
    describe,
    expand,
    privacy_violations,
    remaining,
)

PARTS = Path("data/partitions")


def test_matrix_matches_adr_0016() -> None:
    runs = expand(MATRIX)
    assert len(runs) == 180
    assert Counter(r.group for r in runs) == {
        "X1-fed": 80, "X1-h1c": 10, "X1-local": 20, "X1-central": 20,
        "X2-het": 20, "X2-iid": 10, "X2-mu-het": 10, "X2-mu-homog": 10,
    }  # fmt: skip
    assert {r.seed for r in runs} == {11, 23, 37, 41, 53}
    keys = [(r.experiment, r.config_id(PARTS), r.seed) for r in runs]
    assert len(set(keys)) == len(keys)  # no two runs share an artifact
    assert round(sum(r.minutes for r in runs) / 60) == 30


def test_config_ids_and_commands() -> None:
    runs = {(r.group, r.config_id(PARTS)) for r in expand(MATRIX)}
    for expected in [
        ("X1-fed", "site_a__fedprox__eps3"),
        ("X1-fed", "site_b__fedavg__epsinf"),
        ("X1-h1c", "site_a__fedprox__eps10__h1c"),
        ("X1-local", "site_a__local__vae__eps3"),
        ("X1-central", "site_b__central__ae__epsinf"),
        ("X2-het", "site_b__fedprox__eps3__het__drop0.2"),
        ("X2-iid", "site_a__fedavg__eps3__iid"),
        ("X2-mu-het", "site_a__fedprox__eps3__het__drop0.2__mu0.001"),
        ("X2-mu-homog", "site_a__fedprox__eps3__mu0.1"),
    ]:
        assert expected in runs
    r = next(r for r in expand(MATRIX) if r.group == "X2-mu-het")
    argv = r.argv(Path("results"), Path("configs/params.yaml"))
    assert argv[argv.index("--experiment") + 1] == "X2"
    assert "--heterogeneous" in argv and argv[argv.index("--mu") + 1] == "0.001"
    assert "het" in describe(r)
    inf = next(r for r in expand(MATRIX) if math.isinf(r.epsilon))
    assert (
        inf.argv(Path("r"), Path("p"))[inf.argv(Path("r"), Path("p")).index("--epsilon") + 1]
        == "inf"
    )


def _artifact(
    root: Path, exp: str, cid: str, seed: int, target: str, eps: dict[str, float]
) -> None:
    path = root / exp / cid / f"seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clients = {c: {"epsilon": e} for c, e in eps.items()}
    path.write_text(json.dumps({"target_epsilon": target, "privacy": {"clients": clients}}))


def test_completeness_and_privacy_rule(tmp_path: Path) -> None:
    runs = expand(MATRIX)
    first = runs[0]
    _artifact(tmp_path, first.experiment, first.config_id(PARTS), first.seed, "1", {"a1": 0.99})
    assert len(remaining(runs, tmp_path, PARTS)) == 179
    missing = completeness(tmp_path, PARTS, ["X1"])
    assert len(missing) == 129 and all(m.startswith("X1/") for m in missing)
    assert privacy_violations(tmp_path, ["X1"]) == []
    _artifact(tmp_path, "X1", "site_a__fedavg__eps3", 23, "3", {"a1": 2.9, "a2": 3.2})
    _artifact(tmp_path, "X1", "site_a__fedavg__epsinf", 23, "inf", {"a1": math.inf})
    (bad,) = privacy_violations(tmp_path, ["X1"])
    assert "a2" in bad and "a1" not in bad
