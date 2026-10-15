"""STIX parsing, passages, stores, retriever, renderer and the rule baseline (ADR 0010)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from agmesh.common.schemas import EvidenceRecord
from agmesh.evaluation import attack_ics
from agmesh.kb import chunking, stix_parser
from agmesh.kb.embed import HashingEncoder
from agmesh.kb.render import render
from agmesh.kb.retriever import Retriever
from agmesh.kb.store import MemoryStore
from agmesh.mesh import rules

VERSION = "sha256:" + "ab" * 32


def _ref(tid: str) -> list[dict[str, str]]:
    return [{"source_name": "mitre-attack", "external_id": tid}]


def tiny_bundle() -> dict[str, Any]:
    tactic: dict[str, Any] = {
        "type": "x-mitre-tactic",
        "id": "x-mitre-tactic--1",
        "name": "Impair Process Control",
        "x_mitre_shortname": "impair-process-control",
    }
    t1: dict[str, Any] = {
        "type": "attack-pattern",
        "id": "attack-pattern--1",
        "name": "Modify Parameter",
        "external_references": _ref("T0836"),
        "description": "Adversaries may modify parameters (Citation: X). See [Y](https://e.org).",
        "kill_chain_phases": [{"phase_name": "impair-process-control"}],
    }
    t2: dict[str, Any] = {
        **t1,
        "id": "attack-pattern--2",
        "name": "Command Message",
        "external_references": _ref("T1692.001"),
        "x_mitre_is_subtechnique": True,
    }
    parent: dict[str, Any] = {
        **t1,
        "id": "attack-pattern--3",
        "name": "Unauthorized Message",
        "external_references": _ref("T1692"),
    }
    old: dict[str, Any] = {
        **t1,
        "id": "attack-pattern--4",
        "name": "Old",
        "external_references": _ref("T0855"),
        "revoked": True,
    }
    analytic: dict[str, Any] = {
        "type": "x-mitre-analytic",
        "id": "x-mitre-analytic--1",
        "name": "Analytic 1",
        "description": "Watch <code>setpoint</code> writes.",
    }
    strategy: dict[str, Any] = {
        "type": "x-mitre-detection-strategy",
        "id": "x-mitre-detection-strategy--1",
        "name": "D",
        "x_mitre_analytic_refs": ["x-mitre-analytic--1"],
    }
    mit: dict[str, Any] = {
        "type": "course-of-action",
        "id": "course-of-action--1",
        "name": "Audit",
        "external_references": _ref("M0947"),
    }

    def rel(kind: str, src: str, dst: str, n: int) -> dict[str, str]:
        return {
            "type": "relationship",
            "id": f"relationship--{n}",
            "relationship_type": kind,
            "source_ref": src,
            "target_ref": dst,
        }

    return {
        "objects": [
            tactic, t1, t2, parent, old, analytic, strategy, mit,
            rel("detects", strategy["id"], t1["id"], 1),
            rel("mitigates", mit["id"], t1["id"], 2),
            rel("subtechnique-of", t2["id"], parent["id"], 3),
        ]
    }  # fmt: skip


def test_parser_keeps_active_techniques_with_detection_and_mitigations() -> None:
    techs = stix_parser.parse(tiny_bundle())
    assert [t.technique_id for t in techs] == ["T0836", "T1692", "T1692.001"]
    t = techs[0]
    assert t.description == "Adversaries may modify parameters. See Y."
    assert t.analytics[0].description == "Watch setpoint writes."
    assert t.mitigations == (("M0947", "Audit"),) and t.tactics == ("Impair Process Control",)
    assert techs[2].parent == "T1692" and techs[2].is_subtechnique
    passages = chunking.technique_passages(techs)
    assert [p.passage_id for p in passages][:2] == ["T0836/summary", "T0836/detection/0"]
    assert passages[0].text.startswith("Modify Parameter. Tactics: Impair Process Control.")


def test_store_retriever_and_dimensions() -> None:
    techs = stix_parser.parse(tiny_bundle())
    enc = HashingEncoder()
    store = MemoryStore()
    from agmesh.kb.build import build

    manifest = build(techs, enc, store)
    assert (
        manifest["techniques_indexed"] == 3 and manifest["collections"]["mitre_ics"]["dim"] == 384
    )
    assert manifest["collections"]["incident_memory"]["points"] == 0
    ranked = Retriever(store, enc).rank("setpoint writes watched")
    assert {h.technique_id for h in ranked} == {"T0836", "T1692", "T1692.001"}
    assert ranked[0].technique_id == "T0836" and ranked[0].passage_id == "T0836/detection/0"
    small = MemoryStore()
    small.create("mitre_ics", 16)
    with pytest.raises(ValueError, match="dimension"):
        Retriever(small, enc)


def test_nist_sections() -> None:
    text = (
        "Preface words\n1 Introduction\nalpha beta\n1.1 Purpose and Scope\n"
        + "gamma " * 400
        + "\nA Acronyms\nx y"
    )
    sections = chunking.split_sections(text)
    assert [s for s, _ in sections] == [
        "front",
        "1 Introduction",
        "1.1 Purpose and Scope",
        "A Acronyms",
    ]
    passages = chunking.nist_passages(text, max_words=180)
    assert {p.kind for p in passages} == {"nist"} and all(p.technique_id is None for p in passages)
    assert sum(p.section == "1.1 Purpose and Scope" for p in passages) == 3


def _rec(node: str, t0: float, feats: list[tuple[str, str, str]], **over: Any) -> EvidenceRecord:
    from agmesh.edge.evidence import iso

    site = "site_a" if node.startswith("a") else "site_b"
    base: dict[str, Any] = {
        "evidence_id": f"{int(t0):032x}"[-32:], "site_id": site, "node_id": node,
        "window_start": iso(t0), "window_end": iso(t0 + (60 if site == "site_a" else 10)),
        "model_version": VERSION, "score": 2.0, "threshold": 1.0, "calibrated_prob": 0.9,
        "top_features": [{"name": n, "direction": d, "magnitude": m} for n, d, m in feats],
        "provenance": {"run_id": "r", "data_source": "fixture"},
    }  # fmt: skip
    base.update(over)
    return EvidenceRecord.model_validate(base)


SUMMARY = {
    "read_ops": "11-100", "write_ops": ">100", "diagnostic_ops": "0", "exception_responses": "1-10",
    "request_rate": "high", "new_peer": True, "unseen_address": False, "oversized_quantity": False,
    "malformed_frames": True, "stacked_frames": False, "slow_responses": False,
}  # fmt: skip


def test_renderer_is_deterministic_and_names_no_technique() -> None:
    a = [
        _rec(
            "a2",
            1_594_136_100,
            [("P1_LIT01", "up", "high"), ("P1_B3004", "up", "medium")],
            constants_changed=["P1_PCV02D"],
        ),
        _rec("a2", 1_594_136_160, [("P1_LIT01", "down", "high"), ("P2_SIT02", "down", "low")]),
    ]
    text = render("site_a", list(reversed(a)))
    assert text == render("site_a", a)
    assert "2 anomalous 60-second windows over 2 minutes" in text
    assert (
        "sensor reading water level of the return water tank (P1_LIT01) rose, high deviation"
        in text
    )
    assert "setpoint water level setpoint (return water tank) (P1_B3004)" in text
    assert "sensor reading P2_SIT02 fell" in text  # undocumented point: type only
    assert "position command for the PCV2 valve (P1_PCV02D)" in text
    b = [_rec("b1", 1_675_223_500, [("req_write", "up", "high")], protocol_summary=SUMMARY)]
    tb = render("site_b", b)
    assert "write requests >100" in tb and "request rate high" in tb
    assert "requests from a peer never seen in training" in tb and "malformed or truncated" in tb
    names = [v["name"].lower() for v in attack_ics.table().values()]
    ids = list(attack_ics.table())
    for t in (text, tb):
        low = t.lower()
        assert not [n for n in names if re.search(rf"\b{re.escape(n)}\b", low)], t
        assert not [i for i in ids if i.lower() in low]


def test_rule_baseline() -> None:
    a_sp = [
        _rec("a1", 100, [("P1_FT01", "up", "high")], constants_changed=[]),
        _rec("a1", 40, [("P1_LCV01D", "up", "low"), ("P1_B3004", "up", "low")]),
    ]
    assert rules.decide("site_a", a_sp).technique == "T0836"  # earliest record, setpoint first
    a_co = [_rec("a1", 40, [("P1_FCV03D", "up", "high"), ("P1_FT01", "up", "high")])]
    assert rules.decide("site_a", a_co).technique == "T1692.001"
    a_pv = [_rec("a1", 40, [("P1_FT01", "up", "high")])]
    assert rules.decide("site_a", a_pv).technique == "T1692.002"
    a_const = [_rec("a1", 40, [("P1_FT01", "up", "high")], constants_changed=["P2_RTR"])]
    assert rules.decide("site_a", a_const).technique == "T0836"  # a changed setpoint constant
    assert rules.decide("site_a", []).technique is None
    flood = [_rec("b1", 0, [], protocol_summary=SUMMARY)]
    assert rules.decide("site_b", flood).rule_id == "b1"
    reads = {**SUMMARY, "write_ops": "0", "malformed_frames": False}
    slow = [_rec("b1", 0, [], protocol_summary=reads), _rec("b1", 700, [], protocol_summary=reads)]
    assert rules.decide("site_b", slow).technique == "T0801"
    assert rules.decide("site_b", slow[:1]).technique == "T0861"
    bad = [_rec("b1", 0, [], protocol_summary={**reads, "oversized_quantity": True})]
    assert rules.decide("site_b", bad).technique == "T0814"
    table = json.loads(json.dumps(rules.load()))
    assert all(
        attack_ics.check_ids({r["technique"]}) == []
        for s in ("site_a", "site_b")
        for r in table[s]["rules"]
    )
