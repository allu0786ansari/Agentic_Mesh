"""Site B fusion as pure functions (Plan Week 8 validation; ADR 0009 §7)."""

from __future__ import annotations

import math

import pytest

from agmesh.mesh import fusion as fu

A = fu.FusionAlert


def test_noisy_or() -> None:
    assert fu.noisy_or([]) == 0.0
    assert fu.noisy_or([0.5]) == 0.5
    assert math.isclose(fu.noisy_or([0.5, 0.5]), 0.75)
    assert math.isclose(fu.noisy_or([0.2, 0.3, 0.1]), 1 - 0.8 * 0.7 * 0.9)
    with pytest.raises(ValueError):
        fu.noisy_or([1.2])


def test_incidents_chain_within_delta_t() -> None:
    alerts = [
        A("x1", "b1", 0.0, 0.6),
        A("x2", "b2", 50.0, 0.5),  # within 60 s of x1
        A("x3", "b1", 105.0, 0.9),  # within 60 s of x2 (chaining), 105 s after x1
        A("x4", "b3", 200.0, 0.4),  # gap of 95 s: a new incident
    ]
    found = fu.incidents(reversed(alerts), 60.0)
    assert [i.alert_ids for i in found] == [["x1", "x2", "x3"], ["x4"]]
    first = found[0]
    assert first.nodes == ("b1", "b2") and first.n_nodes == 2
    assert math.isclose(first.noisy_or, 1 - (1 - 0.9) * (1 - 0.5))  # per node: its largest p
    assert first.confirmed(2) and not first.confirmed(3)
    assert (first.start_s, first.end_s) == (0.0, 105.0)


def test_latency_only_alerts_are_context_not_votes() -> None:
    alerts = [A("x1", "b1", 0.0, 0.7), A("x2", "b3", 5.0, 0.9, latency_only=True)]
    (inc,) = fu.incidents(alerts, 60.0)
    assert inc.nodes == ("b1",) and [a.alert_id for a in inc.context] == ["x2"]
    assert inc.noisy_or == 0.7 and not inc.confirmed(2)
    (raw,) = fu.incidents(alerts, 60.0, exclude_latency=False)
    assert raw.confirmed(2) and raw.context == ()
    (only,) = fu.incidents([alerts[1]], 60.0)
    assert only.noisy_or == 0.0 and only.n_nodes == 0


def test_from_alert_and_sensitivity() -> None:
    payload = {
        "alert_id": "a" * 32, "node_id": "b2", "window_end": "2023-02-01T03:52:10.000Z",
        "calibrated_prob": 0.25, "latency_only": False,
    }  # fmt: skip
    fa = fu.from_alert(payload)
    assert fa.node_id == "b2" and fa.prob == 0.25 and fa.t_end_s == 1675223530.0
    grid = fu.sensitivity(
        [A("x1", "b1", 0.0, 0.5), A("x2", "b2", 45.0, 0.5)], deltas=(30, 60), ks=(1, 2)
    )
    rows = {(r["delta_t_s"], r["k"]): (r["incidents"], r["confirmed"]) for r in grid.rows}
    assert rows == {(30, 1): (2, 2), (30, 2): (2, 0), (60, 1): (1, 1), (60, 2): (1, 1)}
    with pytest.raises(ValueError):
        fu.incidents([], 0)
