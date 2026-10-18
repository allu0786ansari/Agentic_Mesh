"""ADR 0016 §4: one-sided claims, test families, and the lock-only provenance rule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from agmesh.evaluation import checker
from agmesh.experiments import x1x2_analysis as an


def test_claims_count_the_right_draws_against_them() -> None:
    d = np.array([-0.06, -0.04, 0.0, 0.02, 0.03])
    assert an._p(d, "noninferior", 0.05) == pytest.approx(2 / 6)  # only -0.06 <= -0.05
    assert an._p(d, "at_or_above", 0.0) == pytest.approx(3 / 6)  # -0.06, -0.04
    assert an._p(d, "greater", 0.0) == pytest.approx(4 / 6)  # <= 0: three, plus one
    assert an._p(d, "less", 0.0) == pytest.approx(4 / 6)  # >= 0: three, plus one
    zeros = np.zeros(999)  # Site B: AUROC 1.000 everywhere
    assert an._p(zeros, "at_or_above", 0.0) == pytest.approx(1 / 1000)
    assert an._p(zeros, "greater", 0.0) == 1.0
    with pytest.raises(ValueError):
        an._p(d, "other", 0.0)


def test_families_match_adr_0016() -> None:
    x1, x2 = an.x1_tests(), an.x2_tests()

    def fam(ts: list[an.Test], f: str) -> list[an.Test]:
        return [t for t in ts if t.family == f]

    rq1 = fam(x1, "RQ1")
    assert [t.label for t in rq1] == ["PT1 site_a", "PT2 site_a", "PT1 site_b", "PT2 site_b"]
    assert all(t.a[1].endswith("__fedprox__eps3") for t in rq1)
    assert {t.b[1] for t in rq1} == {
        "site_a__central__vae", "site_a__local__vae", "site_b__central__ae", "site_b__local__ae"
    }  # fmt: skip
    assert rq1[0].claim == "noninferior" and rq1[0].margin == 0.05
    assert rq1[1].claim == "at_or_above"
    assert len(fam(x2, "H1b")) == 4 and len(fam(x1, "H1c")) == 2
    assert all(t.unpaired for t in x2 if "IID" in t.label)
    assert len(an.curve("X1")) == 16 and an.curve("X2") == set()


def test_a_lock_only_dirty_tree_passes_only_with_a_committed_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checker, "committed_lock_hashes", lambda: frozenset({"abc"}))
    prov: dict[str, Any] = {
        "git_dirty_paths": ["uv.lock"],
        "environment": {"lockfile_sha256": "abc"},
    }
    assert checker._lock_only_and_committed(prov)
    assert not checker._lock_only_and_committed({**prov, "environment": {"lockfile_sha256": "x"}})
    assert not checker._lock_only_and_committed({**prov, "git_dirty_paths": ["uv.lock", "a.py"]})
    assert not checker._lock_only_and_committed({**prov, "git_dirty_paths": []})
