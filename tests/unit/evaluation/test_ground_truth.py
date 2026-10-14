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
    assert decision["primary"] == "T0855" and "T0831" in decision["acceptable"]


def test_sensor_only_and_setpoint_ties() -> None:
    sensor_only = [
        {"variable": "P2_SIT01", "type": "sensor", "first_row": 0, "share": 1.0, "excess": 3.0}
    ]
    assert gt.rule(sensor_only)["acceptable"] == ["T0856", "T0832"]
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
    b = gt.site_b(tmp_path / "gt")
    assert sorted(b["events"]) == [str(i) for i in range(7)]
    saved = json.loads((tmp_path / "gt" / "mapping_site_b.json").read_text(encoding="utf-8"))
    assert saved["events"]["5"]["primary"] == "T0806"
    assert "events" in gt.summary(b)
