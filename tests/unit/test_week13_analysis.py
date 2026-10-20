"""Week 13 analysis (ADR 0017): X4 scoring, X3 detectors and τ, robustness injections."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agmesh.common.config import load_params
from agmesh.common.frames import LabelFreeFrame
from agmesh.data.access import Labels, Partition
from agmesh.experiments import robustness as rb
from agmesh.experiments import x3_fusion as xf
from agmesh.experiments import x4_score as x4
from agmesh.mesh import fusion

# ---------------------------------------------------------------------------- X4


def _session(
    sid: str, start: float, end: float, tid: str | None, top3: list[str], **llm: Any
) -> dict[str, Any]:
    return {
        "session_id": sid,
        "opened_at": "2026-10-17T00:00:00Z",
        "status": "COMPLETED",
        "attribution_mode": "full" if tid else "abstained",
        "technique_id": tid,
        "ranked_top3": top3,
        "incident": {"start_s": start, "end_s": end, "nodes": ["a1"]},
        "llm": {
            "called": bool(llm),
            "error_kind": None,
            "schema_valid": llm.get("valid"),
            "grounded": llm.get("grounded"),
            "valid": llm.get("valid"),
            "raw_technique": llm.get("raw"),
        },
    }


def _event(key: str, start: float, end: float, acceptable: tuple[str, ...]) -> x4.Event:
    return x4.Event(
        key, ("a1",), start, end, acceptable[0], acceptable, ("Impair Process Control",)
    )


def test_x4_matches_with_the_tolerance_after_the_event_only() -> None:
    e = _event("e", 100.0, 200.0, ("T0836",))
    assert x4.matches({"start_s": 250.0, "end_s": 300.0}, e, 60.0)  # within end + 60
    assert not x4.matches({"start_s": 260.0, "end_s": 300.0}, e, 60.0)
    assert x4.matches({"start_s": 0.0, "end_s": 100.0}, e, 60.0)  # touches the start
    assert not x4.matches({"start_s": 0.0, "end_s": 99.0}, e, 60.0)


def test_x4_scores_the_earliest_matching_incident_and_counts_misses_as_wrong() -> None:
    events = [
        _event("e1", 100, 200, ("T0836",)),
        _event("e2", 1000, 1100, ("T1692.001",)),
        _event("e3", 5000, 5100, ("T0806",)),
    ]
    sessions = [
        _session("late", 150, 160, "T0836", ["T0836"]),
        _session("early", 90, 120, "T0855", ["T0855", "T0836"]),  # revoked -> T1692.001
        _session("e2", 1000, 1050, "T0855", ["T0855"]),
        _session("fp", 3000, 3010, "T0836", ["T0836"]),
    ]
    m = x4.site_a_mode(sessions, events, 60.0)
    rows = {r["event"]: r for r in m["per_event"]}
    assert rows["e1"]["session_id"] == "early" and rows["e1"]["technique"] == "T1692.001"
    assert not rows["e1"]["top1"] and rows["e1"]["top3_hit"]
    assert rows["e2"]["top1"]  # T0855 is scored as T1692.001
    assert not rows["e3"]["matched"]
    assert m["top1"]["estimate"] == pytest.approx(1 / 3)
    assert m["top3"]["estimate"] == pytest.approx(2 / 3)
    assert m["top1_answered"] == pytest.approx(1 / 2)
    assert m["false_positive_incidents"] == 1 and m["missed"] == 1


def test_x4_a_parent_does_not_match_a_sub_technique() -> None:
    e = [_event("e", 0, 10, ("T1692.001",))]
    (row,) = x4.per_event([_session("s", 0, 5, "T1692", ["T1692"])], e, 60.0)
    assert not row["top1"] and not row["top3_hit"]


def test_x4_checks_flag_missing_modes_and_ungrounded_dispatch(tmp_path: Path) -> None:
    good = _session("ok", 0, 5, "T0836", ["T0836"], valid=True, grounded=True, raw="T0836")
    bad = _session("bad", 0, 5, "T9999", ["T9999"])
    wrong = _session("wrong", 0, 5, "T0836", ["T0836"], valid=True, grounded=True, raw="T0806")
    run = x4.Run("site_a", "full", tmp_path / "r", {"sessions": 4}, [good, bad, wrong])
    problems = x4.checks([run])
    assert any("no rule_only run" in p for p in problems)
    assert any("T9999 is not a candidate" in p for p in problems)
    assert any("valid answer is not dispatched" in p for p in problems)
    assert any("3 exported, 4 in run" in p for p in problems)


def test_x4_majority_baseline_counts_acceptable_sets() -> None:
    events = [
        _event("a", 0, 1, ("T0836",)),
        _event("b", 0, 1, ("T0836", "T1692.001")),
        _event("c", 0, 1, ("T1692.001",)),
        _event("d", 0, 1, ("T0836",)),
    ]
    b = x4.baselines(events, {"constant_answer_baseline": {"T0836": 0.64}})
    assert b["majority_class"]["technique"] == "T0836"
    assert b["majority_class"]["top1"]["estimate"] == pytest.approx(0.75)
    assert b["constant_answer"]["T0836"]["these_events"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------- X3


def _fa(node: str, t: float, p: float, lat: bool = False) -> fusion.FusionAlert:
    return fusion.FusionAlert(f"{node}{t}", node, t, p, lat)


def test_x3_alarm_rules_fire_at_the_alert_that_meets_them() -> None:
    inc = fusion.fuse(
        [_fa("b1", 10, 0.2), _fa("b1", 20, 0.3), _fa("b2", 30, 0.5), _fa("b3", 40, 0.9, lat=True)]
    )
    assert xf.alarm(inc, "single", 1.0, 10.0) == xf.Alarm(10, 0, 40)  # ends at its context
    assert xf.alarm(inc, "2of3", 1.0, 10.0) == xf.Alarm(30, 20, 40)  # latency-only b3 ignored
    assert xf.alarm(inc, "noisy_or", 0.6, 10.0) == xf.Alarm(30, 20, 40)  # 1 - .7 * .5 = .65
    assert xf.alarm(inc, "noisy_or", 0.7, 10.0) is None


def _wa(node: str, t: float, p: float) -> xf.WindowAlert:
    return xf.WindowAlert(node, t - 10.0, t, p, False)


def test_x3_matched_tau_is_the_smallest_value_within_the_best_nodes_rate() -> None:
    params = load_params()
    alerts = [_wa("b1", 1000.0 * i, 0.1 * i) for i in range(1, 6)]  # 5 b1 incidents
    alerts += [_wa("b2", 1000.0 * i + 5, 0.0) for i in range(1, 6)]
    alerts += [_wa("b3", 7000.0, 0.6), _wa("b3", 9000.0, 0.7)]  # b3: 2 incidents, the best
    out = xf.matched_tau(alerts, params, hours=1.0)
    assert out["best_node"] == "b3" and out["best_fa_per_h"] == 2.0
    # joint incident scores .1 .2 .3 .4 .5 .6 .7: at .6, two alarms per hour
    assert out["tau"] == pytest.approx(0.6)


def test_x3_matched_tau_is_infinite_when_every_benign_score_is_zero() -> None:
    params = load_params()
    alerts = [_wa("b1", 1000.0 * i, 0.0) for i in range(1, 4)] + [_wa("b2", 9000.0, 0.0)]
    assert xf.matched_tau(alerts, params, hours=1.0)["tau"] == math.inf


def test_x3_detect_uses_the_tolerance_and_never_a_negative_delay() -> None:
    events = {"e": (100.0, 160.0)}
    early = xf.Alarm(t_s=95.0, start_s=85.0, end_s=120.0)
    late = xf.Alarm(t_s=225.0, start_s=215.0, end_s=230.0)
    assert xf.detect(events, [early], 60.0) == {"e": 0.0}
    assert xf.detect(events, [late], 60.0) == {"e": 125.0}
    assert xf.detect(events, [xf.Alarm(235.0, 225.0, 240.0)], 60.0) == {"e": None}


def _partition(cols: tuple[str, ...], values: np.ndarray, t: np.ndarray) -> Partition:
    labels = Labels(np.zeros(t.size, np.int8), t.astype(float), np.zeros(t.size).astype(str))
    return Partition(
        "site_b", "b1", "val_normal", LabelFreeFrame(cols, values.astype(np.float32)), labels, "x"
    )


def test_x3_injection_sets_req_write_and_raises_req_recv_only() -> None:
    cols = ("req_recv", "req_write", "peers")
    part = _partition(cols, np.ones((5, 3)), np.arange(5) * 10.0)
    out = xf.injected(part, [1, 2], 5.0)
    v = out.features.values
    assert v[1].tolist() == [6.0, 5.0, 1.0] and v[2].tolist() == [6.0, 5.0, 1.0]
    assert v[0].tolist() == [1.0, 1.0, 1.0] and part.features.values[1, 1] == 1.0


# ---------------------------------------------------------------------- robustness


def test_perturbations_change_only_the_planned_rows() -> None:
    cols = ("a", "b")
    t = np.arange(200) * 10.0
    values = np.stack([np.arange(200.0), np.full(200, 7.0)], axis=1)
    part = _partition(cols, values, t)
    sd = {"a": 2.0, "b": 1.0}
    inj = [
        {"kind": "spike", "start_s": 100.0, "end_s": 110.0, "feature": "a"},
        {"kind": "drift", "start_s": 500.0, "end_s": 540.0, "feature": "b"},
        {"kind": "stuck", "start_s": 800.0, "end_s": 830.0, "feature": "a"},
        {"kind": "replay", "start_s": 1500.0, "end_s": 1520.0, "feature": None},
    ]
    out, touched = rb.perturb(part, inj, sd)
    v = out.features.values
    assert v[10, 0] == 10 + 12  # +6 sd
    assert v[50:54, 1].tolist() == pytest.approx([7.75, 8.5, 9.25, 10.0])  # ramp to +3 sd
    assert v[80:83, 0].tolist() == [79.0, 79.0, 79.0]  # frozen at the last value
    assert v[150:152, 0].tolist() == [90.0, 91.0]  # the 10 min (60 rows) before
    assert touched.sum() == 1 + 4 + 3 + 2
    changed = np.flatnonzero((v != part.features.values).any(axis=1))
    assert set(changed) <= set(np.flatnonzero(touched))


def test_robustness_uses_only_the_seeds_that_reproduce_per_site() -> None:
    problems = [
        "site_a__fedprox__eps3 seed 23: auroc 0.1 != X1 0.2",
        "site_a__fedprox__eps3 seed 41: no model",
    ]
    by = rb.verified_seeds([11, 23, 41, 53], problems)
    assert by["site_a"]["used"] == [11, 53]
    assert sorted(by["site_a"]["excluded"]) == ["23", "41"]
    assert by["site_b"] == {"used": [11, 23, 41, 53], "excluded": {}}


def test_x4_diversity_flags_a_constant_answerer() -> None:
    const = x4.diversity(["T0836", "T0836", None, "T0836"])
    assert const["constant"] and const["distinct"] == 1 and const["modal_share"] == 1.0
    assert const["answered"] == 3
    mixed = x4.diversity(["T0836", "T1692.001", "T0836", "T0801"])
    assert not mixed["constant"] and mixed["distinct"] == 3
    assert mixed["modal"] == "T0836" and mixed["modal_share"] == 0.5
    assert x4.diversity([None, None])["answered"] == 0
