"""Grounding: an out-of-candidate ID always falls back (Plan Week 10 validation; ADR 0012 §4)."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from agmesh.mesh.attribution import AttributionAgent
from tests.mesh_doubles import UNAVAILABLE, ScriptedLLM, fixed_rank, record

ORDER = [
    "T0836",
    "T1692.001",
    "T0806",
    "T1692.002",
    "T0814",
    "T0801",
    "T0861",
    "T0888",
    "T0846",
    "T0866",
    "T0802",
]
RECS = [record("a1", 1_594_136_100, [("P1_B3004", "up", "high")])]


def agent(reply: Any, **kw: Any) -> AttributionAgent:
    return AttributionAgent(
        fixed_rank(ORDER),
        ScriptedLLM(reply),
        k=10,
        mitigations={"T0836": [["M0947", "Audit"]]},
        **kw,
    )


def ok(tid: str, conf: float = 0.8) -> dict[str, Any]:
    return {"technique_id": tid, "tactic": "x", "confidence": conf, "rationale": "r"}


def test_valid_choice_is_kept_and_ranked_first() -> None:
    a = agent(ok("T0806")).attribute("site_a", RECS, "full")
    assert (a.attribution_mode, a.technique_id, a.confidence) == ("full", "T0806", 0.8)
    assert a.ranked[:3] == ["T0806", "T0836", "T1692.001"] and len(a.candidates) == 10
    assert a.tactics == ["Impair Process Control"]  # from the pin, not the LLM's 'x'
    assert a.llm is not None and a.llm["validation"]["grounded"]


def test_every_invalid_output_falls_back_to_retrieval() -> None:
    cases = [
        ok("T0866"),  # rank 10: a candidate, valid
        ok("T0802"),  # rank 11: NOT a candidate
        ok("T9999"),
        ok("T0836", 0.3),  # below the confidence floor
        "not json",
        {"technique_id": "T0836"},  # schema incomplete
        UNAVAILABLE,  # timeout or no server
    ]
    modes = [agent(c).attribute("site_a", RECS, "full") for c in cases]
    assert modes[0].attribution_mode == "full" and modes[0].technique_id == "T0866"
    for a in modes[1:]:
        assert a.attribution_mode == "retrieval_fallback" and a.technique_id == "T0836", a.reason
    assert (
        modes[1].llm is not None and modes[1].llm["validation"]["grounded"] is False
    )  # raw grounding recorded


@settings(max_examples=200, deadline=None)
@given(
    tid=st.text(min_size=0, max_size=12) | st.sampled_from(ORDER), conf=st.floats(-1, 2) | st.none()
)
def test_the_attributed_technique_is_always_a_candidate(tid: str, conf: float | None) -> None:
    a = agent({"technique_id": tid, "tactic": "t", "confidence": conf, "rationale": "r"}).attribute(
        "site_a", RECS, "full"
    )
    assert a.technique_id in a.candidates
    if a.attribution_mode == "full":
        assert a.technique_id == tid and conf is not None and 0.5 <= conf <= 1.0
    else:
        assert a.attribution_mode == "retrieval_fallback" and a.technique_id == ORDER[0]


def test_other_modes() -> None:
    r = agent(ok("T0806")).attribute("site_a", RECS, "retrieval_only")
    assert (r.technique_id, r.llm, r.mitigations) == ("T0836", None, [["M0947", "Audit"]])
    rule = agent(ok("T0806")).attribute("site_a", RECS, "rule_only")
    assert rule.technique_id == "T0836" and rule.attribution_mode == "rule_only"  # setpoint root
    nr = AttributionAgent(None, ScriptedLLM(ok("T0827")), k=10).attribute(
        "site_a", RECS, "llm_no_retrieval"
    )
    assert nr.attribution_mode == "abstained"  # T0827 (Impact-only) is not among the 92
    assert len(nr.candidates) == 92 and "T0827" not in nr.candidates
    good = AttributionAgent(None, ScriptedLLM(ok("T1692.001")), k=10).attribute(
        "site_a", RECS, "llm_no_retrieval"
    )
    assert good.technique_id == "T1692.001" and good.ranked == ["T1692.001"]
    none = agent(ok("T0806")).attribute("site_a", [], "full")
    assert none.attribution_mode == "abstained" and none.technique_id is None
