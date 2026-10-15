"""Evidence minimisation (Plan §6.3, CC5; ADR 0009 §3): only allowlisted fields leave a node.

The property test feeds the evidence builder random private inputs: errors,
residuals, raw Modbus counters and raw constant values. It then checks the
serialised record and alert:

* every number sits under ``score``, ``threshold`` or ``calibrated_prob``;
* every string is an identifier, a time, an enumeration value or an allowed feature
  name;
* no raw input value appears anywhere in the text.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agmesh.common.schemas import (
    FEATURE_PATTERN,
    ID_PATTERN,
    ISO_PATTERN,
    NUMERIC_FIELDS,
    RUN_PATTERN,
    VERSION_PATTERN,
    Alert,
    EvidenceRecord,
    check_features,
)
from agmesh.edge import evidence as ev

SITE_B = [
    "req_sent", "req_recv", "req_read", "req_write", "req_other", "resp_exception",
    "unseen_addr", "unseen_addr_write", "oversized_qty", "tcp_syn", "tcp_rst",
    "stacked_frames", "malformed_frames", "latency_median_ms", "slow_responses", "unanswered",
]  # fmt: skip
COUNTERS = [*SITE_B, "peers", "new_peers"]
SITE_A = [f"P1_X{i:02d}" for i in range(12)]
CONSTANTS = ["P1_PP02D", "P2_ASD", "P3_LH"]
ENUM = {
    "up", "down", "low", "medium", "high", "normal", "0", "1-10", "11-100", ">100",
    "site_a", "site_b", "runtime", "fixture", "synthetic",
}  # fmt: skip
VERSION = "sha256:" + "ab" * 32
FLOATS = st.floats(min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False)


def _walk(value: Any, key: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [item for k, v in value.items() for item in _walk(v, k)]
    if isinstance(value, list):
        return [item for v in value for item in _walk(v, key)]
    return [(key, value)]


def _check_minimised(doc: dict[str, Any], allowed_names: set[str]) -> None:
    for key, value in _walk(doc):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            assert key in NUMERIC_FIELDS, f"numeric field {key!r} = {value}"
            continue
        assert isinstance(value, str), (key, value)
        ok = (
            value in ENUM
            or value in allowed_names
            or re.fullmatch(ID_PATTERN, value)
            or re.fullmatch(ISO_PATTERN, value)
            or re.fullmatch(VERSION_PATTERN, value)
            or (key in ("run_id",) and re.fullmatch(RUN_PATTERN, value))
            or (key == "node_id" and re.fullmatch(r"[ab][1-9]", value))
        )
        assert ok, f"string field {key!r} = {value!r} is not allowlisted"


def _free_text(doc: dict[str, Any]) -> str:
    """The record's text without its allowed numbers, IDs and times."""
    kept = [
        str(v)
        for k, v in _walk(doc)
        if k not in NUMERIC_FIELDS
        and not (
            isinstance(v, str) and (re.fullmatch(ID_PATTERN, v) or re.fullmatch(ISO_PATTERN, v))
        )
        and not (isinstance(v, str) and re.fullmatch(VERSION_PATTERN, v))
    ]
    return " ".join(kept)


def _raw_marks(values: list[float]) -> list[str]:
    """Distinctive renderings of raw values; none may appear in the output text."""
    return [repr(float(v)) for v in values if abs(v) > 1.5 and float(v) != int(v)]


@settings(max_examples=60, deadline=None)
@given(
    errors=st.lists(FLOATS, min_size=len(SITE_B), max_size=len(SITE_B)),
    residuals=st.lists(
        st.floats(-1e3, 1e3, allow_nan=False), min_size=len(SITE_B), max_size=len(SITE_B)
    ),
    counters=st.lists(
        st.floats(0, 1e5, allow_nan=False), min_size=len(COUNTERS), max_size=len(COUNTERS)
    ),
    score=FLOATS,
    prob=st.floats(0, 1),
    t0=st.floats(1.6e9, 1.8e9),
)
def test_site_b_records_hold_no_raw_values(
    errors: list[float],
    residuals: list[float],
    counters: list[float],
    score: float,
    prob: float,
    t0: float,
) -> None:
    buckets = {n: [0.5, 2.0, 9.0] for n in SITE_B}
    feats = ev.top_features(errors, residuals, SITE_B, buckets)
    summary = ev.protocol_summary(dict(zip(COUNTERS, counters, strict=True)), [3.0, 40.0])
    record, alert = ev.build(
        site="site_b", node="b1", window_start_s=t0, window_end_s=t0 + 10,
        model_version=VERSION, score=score, threshold=1.0, calibrated_prob=prob,
        features=feats, run_id="edge__site_b__val__x", data_source="runtime", summary=summary,
    )  # fmt: skip
    check_features(record, SITE_B)
    for doc in (json.loads(record.model_dump_json()), json.loads(alert.model_dump_json())):
        _check_minimised(doc, set(SITE_B))
    text = _free_text(json.loads(record.model_dump_json())) + _free_text(
        json.loads(alert.model_dump_json())
    )
    for mark in _raw_marks(errors + residuals + counters):
        assert mark not in text
    assert len(record.top_features) <= 5


@settings(max_examples=40, deadline=None)
@given(
    errors=st.lists(FLOATS, min_size=len(SITE_A), max_size=len(SITE_A)),
    constants=st.lists(st.floats(-1e4, 1e4, allow_nan=False), min_size=3 * 5, max_size=3 * 5),
)
def test_site_a_records_name_changed_constants_without_values(
    errors: list[float], constants: list[float]
) -> None:
    window = np.asarray(constants, dtype=np.float64).reshape(5, 3)
    reference = dict(zip(CONSTANTS, window[0].tolist(), strict=True))
    changed = ev.constants_changed(window, CONSTANTS, reference)
    feats = ev.top_features(
        errors, np.ones(len(SITE_A)), SITE_A, {n: [1.0, 5.0, 50.0] for n in SITE_A}
    )
    record, _alert = ev.build(
        site="site_a", node="a3", window_start_s=1.7e9, window_end_s=1.7e9 + 60,
        model_version=VERSION, score=4.2, threshold=3.1, calibrated_prob=0.7,
        features=feats, run_id="r", data_source="runtime", changed=changed,
    )  # fmt: skip
    check_features(record, [*SITE_A, *CONSTANTS])
    _check_minimised(json.loads(record.model_dump_json()), set(SITE_A) | set(CONSTANTS))
    text = _free_text(json.loads(record.model_dump_json()))
    for mark in _raw_marks(errors + constants):
        assert mark not in text


def _record(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evidence_id": "0" * 32, "site_id": "site_a", "node_id": "a1",
        "window_start": "2020-07-07T15:35:00.000Z", "window_end": "2020-07-07T15:36:00.000Z",
        "model_version": VERSION, "score": 1.0, "threshold": 0.5, "calibrated_prob": 0.9,
        "top_features": [], "provenance": {"run_id": "r", "data_source": "runtime"},
    }  # fmt: skip
    base.update(over)
    return base


@pytest.mark.parametrize(
    "over",
    [
        {"raw_value": 12.5},  # extra field
        {
            "top_features": [
                {"name": "P1_FT01", "direction": "up", "magnitude": "low", "value": 3.0}
            ]
        },
        {
            "top_features": [
                {"name": f"P1_F{i}", "direction": "up", "magnitude": "low"} for i in range(6)
            ]
        },
        {"top_features": [{"name": "12.5", "direction": "up", "magnitude": "low"}]},
        {"calibrated_prob": 1.2},
        {"score": float("nan")},
        {"node_id": "185.175.0.4"},
        {"window_end": "1596123360.0"},
        {"provenance": {"run_id": "r", "data_source": "runtime", "host": "10.0.0.1"}},
    ],
)
def test_outbound_record_rejects_anything_not_allowlisted(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate(_record(**over))


def test_site_rules_and_feature_allowlist() -> None:
    rec = EvidenceRecord.model_validate(_record())
    rec.check_site()
    with pytest.raises(ValueError, match="protocol_summary"):
        EvidenceRecord.model_validate(_record(site_id="site_b", node_id="b1")).check_site()
    named = EvidenceRecord.model_validate(
        _record(top_features=[{"name": "P9_SECRET", "direction": "up", "magnitude": "high"}])
    )
    with pytest.raises(ValueError, match="not in the node's list"):
        check_features(named, ["P1_FT01"])
    assert re.fullmatch(FEATURE_PATTERN, "P1_FT01")
    alert = {
        "alert_id": "1" * 32, "evidence_id": "0" * 32, "site_id": "site_b", "node_id": "b2",
        "window_end": "2023-02-01T03:52:00.000Z", "calibrated_prob": 0.4, "latency_only": False,
        "run_id": "r", "data_source": "runtime",
    }  # fmt: skip
    Alert.model_validate(alert)
    for bad in ({"src_ip": "185.175.0.7"}, {"score": 3.0}, {"calibrated_prob": -0.1}):
        with pytest.raises(ValidationError):
            Alert.model_validate({**alert, **bad})
