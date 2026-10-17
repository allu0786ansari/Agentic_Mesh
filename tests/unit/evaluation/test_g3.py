"""Gate G3's pre-declared rule (ADR 0014 §9)."""

from __future__ import annotations

from typing import Any

import pytest

from agmesh.evaluation.g3 import decide


def r(
    site: str, validity: float = 1.0, i2d_ms: float = 17_000.0, inv_ms: float = 150.0, **kw: Any
) -> dict[str, Any]:
    return {
        "site": site,
        "mode": "full",
        "nodes": "https",
        "validity": validity,
        "incident_to_dispatch_p95_ms": i2d_ms,
        "investigation_p95_ms": inv_ms,
        **kw,
    }


def test_full_stays_default_when_both_sites_pass() -> None:
    out = decide([r("site_a"), r("site_b")], 24.0)
    assert out["default_mode"] == "full" and out["l_llm_s"] == 24.0


@pytest.mark.parametrize(
    "bad",
    [r("site_b", validity=0.94), r("site_b", i2d_ms=24_001.0), r("site_b", inv_ms=2_001.0)],
)
def test_any_failure_switches_to_retrieval_only(bad: dict[str, Any]) -> None:
    assert decide([r("site_a"), bad], 24.0)["default_mode"] == "retrieval_only"


def test_only_full_runs_over_https_decide() -> None:
    local = r("site_b", validity=0.0, nodes="local")
    assert decide([r("site_a"), r("site_b"), local], 24.0)["default_mode"] == "full"
    with pytest.raises(SystemExit, match="site_b"):
        decide([r("site_a"), local], 24.0)
