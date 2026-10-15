"""First-pass ground-truth tool: variable typing, deviation evidence, rule and logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.config import Params
from agmesh.evaluation import ground_truth as gt
from agmesh.evaluation.sealed import read_log


def test_variable_types_follow_hai_naming() -> None:
    cases = {
        "P1_B2004": "setpoint",
        "P3_LH": "setpoint",
        "P4_ST_LD": "setpoint",
        "P4_ST_PS": "setpoint",
        "P1_FCV01D": "command",
        "P3_LCV01D": "command",
        "P2_Emerg": "command",
        "P4_ST_GOV": "command",
        "P1_FCV01Z": "sensor",
        "P1_FT01": "sensor",
        "P1_FT01Z": "sensor",
        "P2_SIT01": "sensor",
        "P4_ST_PT01": "sensor",
        "P4_HT_PO": "sensor",
        "P4_HT_FD": "unknown",
        "P2_ASD": "setpoint",  # auto speed demand (P2_AutoSD)
        "P2_RTR": "setpoint",  # RPM trip rate
        "P2_VTR01": "setpoint",  # vibration trip limit
        "P2_CO_rpm": "command",  # speed-controller output (P2_SCO)
    }
    assert {name: gt.variable_type(name) for name in cases} == cases


def test_command_before_sensor_maps_to_unauthorized_command() -> None:
    n = 100
    context = pd.DataFrame({"P1_FCV03D": np.full(n, 50.0), "P1_FT03": 10 + np.sin(np.arange(n))})
    event = pd.DataFrame(
        {"P1_FCV03D": np.full(40, 80.0), "P1_FT03": np.r_[np.full(5, 10.0), np.full(35, 30.0)]}
    )
    devs = gt.deviations(context, event, ["P1_FCV03D", "P1_FT03"])
    assert [d["variable"] for d in devs] == ["P1_FCV03D", "P1_FT03"]
    decision = gt.rule(devs)
    assert decision["primary"] == "T1692.001" and decision["acceptable"] == ["T1692.001"]


def test_sensor_only_and_setpoint_ties() -> None:
    sensor_only = [
        {"variable": "P2_SIT01", "type": "sensor", "first_row": 0, "share": 1.0, "excess": 3.0}
    ]
    assert gt.rule(sensor_only)["acceptable"] == ["T1692.002"]
    tie = [
        *sensor_only,
        {"variable": "P1_B4005", "type": "setpoint", "first_row": 2, "share": 0.5, "excess": 1.0},
    ]
    assert gt.rule(tie)["primary"] == "T0836"  # simultaneous: the setpoint is the cause
    assert gt.rule([])["primary"] is None
    noise = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.0]})
    assert gt.deviations(noise, pd.DataFrame({"x": [1.05, 0.95]}), ["x"]) == []


def test_site_a_draft_is_logged_and_site_b_complete(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    log = tmp_path / "access.log"
    doc = gt.site_a(params, root, tmp_path / "gt", log)
    assert doc["pass"] == 1 and doc["reviewed"] is False
    assert len(doc["events"]) == site_a_run[0]["events"]["total"]
    entries = read_log(log)
    assert {e["config_id"] for e in entries} == {"ground_truth_site_a"} and len(entries) == 4
    for event in doc["events"].values():
        assert set(event["tactics"]) <= {t for ts in gt.TACTICS.values() for t in ts}
        assert event["documented"] is None and event["basis"].startswith("inferred only")
        assert event["primary"] == event["inferred"]["primary"] and event["start"]
    b = gt.site_b(tmp_path / "gt")
    assert sorted(b["events"]) == [str(i) for i in range(7)]
    saved = json.loads((tmp_path / "gt" / "mapping_site_b.json").read_text(encoding="utf-8"))
    assert saved["events"]["5"]["primary"] == "T0806"
    assert "events" in gt.summary(b)


def test_documented_timetable_and_rule() -> None:
    from collections import Counter
    from datetime import datetime

    from agmesh.evaluation import hai_attacks as ha

    assert len(ha.TIMETABLE) == 50
    assert Counter(len(a.primitives) for a in ha.TIMETABLE) == {1: 25, 2: 25}
    assert len({a.attack_id for a in ha.TIMETABLE}) == 50
    by_id = {a.attack_id: a for a in ha.TIMETABLE}
    assert ha.techniques(by_id["A101"]) == ("T0836", ["T0836"])  # setpoint
    assert ha.techniques(by_id["A103"]) == ("T1692.001", ["T1692.001"])  # control output
    assert ha.techniques(by_id["A202"]) == ("T0836", ["T0836", "T1692.002"])  # sensor hidden
    assert ha.techniques(by_id["A203"]) == ("T1692.001", ["T1692.001", "T0806"])  # short-term
    primary, acceptable = ha.techniques(by_id["A301"])  # CO-ST + SP: the setpoint leads
    assert primary == "T0836" and set(acceptable) == {"T1692.001", "T0806", "T0836"}
    assert by_id["A220"].primitives[0].columns == ("P3_LCV01D",)  # erratum corrected
    assert by_id["A104"].primitives[0].columns == ("P2_CO_rpm",)
    assert by_id["A402"].processes == ["P1", "P3"]
    assert max(ha.constant_baseline().values()) <= 0.65  # no constant answer wins
    hit = ha.match(datetime(2020, 7, 10, 3, 3, 11), ["P1"])
    assert hit is not None and hit.attack_id == "A209"
    assert ha.match(datetime(2020, 7, 10, 3, 3, 11), ["P2"]) is None  # processes must agree
    assert ha.match(datetime(2020, 7, 10, 3, 10, 0), ["P1"]) is None  # too far


def test_committed_mappings_use_the_pinned_attack_release() -> None:
    from agmesh.evaluation import attack_ics

    ids = set(gt.TACTICS)
    for site in ("a", "b"):
        doc = json.loads((gt.OUT_DIR / f"mapping_site_{site}.json").read_text(encoding="utf-8"))
        assert doc["protocol_version"] == gt.PROTOCOL_VERSION
        for event in doc["events"].values():
            ids |= {event["primary"], *event["acceptable"]}
            expected = {t for x in event["acceptable"] for t in gt.TACTICS[x]}
            assert event["tactics"] == sorted(expected)
    assert attack_ics.check_ids(ids) == []
    table = attack_ics.table()
    assert all(sorted(gt.TACTICS[t]) == table[t]["tactics"] for t in gt.TACTICS)


def test_revoked_ids_follow_mitre_revoked_by() -> None:
    from agmesh.evaluation import attack_ics

    assert attack_ics.canonical("T0855") == "T1692.001"
    assert attack_ics.canonical("T0856") == "T1692.002"
    assert attack_ics.canonical("T0836") == "T0836"
    assert attack_ics.canonical("T9999") == "T9999"  # unknown: unchanged, matches nothing
    assert attack_ics.tactics("T0855") == ["Evasion", "Impair Process Control"]
    assert attack_ics.check_ids({"T0855"}) == ["T0855: revoked in ATT&CK for ICS 20.0"]


def test_relabel_reapplies_the_rule_without_data(tmp_path: Path) -> None:
    from agmesh.evaluation import hai_attacks as ha

    v10 = {
        "site": "site_a",
        "protocol_version": "1.0",
        "reviews": [{"pass": 1}],
        "events": {
            "s1": {
                "attack_id": "A203",
                "primary": "T0855",
                "acceptable": ["T0855", "T0831", "T0806"],
                "tactics": ["Impact", "Impair Process Control"],
                "inferred": {"primary": "T0856", "acceptable": ["T0856", "T0832", "T0855"]},
            }
        },
    }
    path = tmp_path / "mapping_site_a.json"
    path.write_text(json.dumps(v10), encoding="utf-8")
    doc = gt.relabel(path)
    event = doc["events"]["s1"]
    by_id = {a.attack_id: a for a in ha.TIMETABLE}
    assert (event["primary"], event["acceptable"]) == ha.techniques(by_id["A203"])
    assert event["tactics"] == ["Evasion", "Impair Process Control"]
    assert event["inferred"] == {"primary": "T1692.002", "acceptable": ["T1692.002", "T1692.001"]}
    assert doc["reviews"] == [{"pass": 1}] and doc["protocol_version"] == "1.1"
