"""Attribution ground truth (docs/mapping_protocol.md, v1.1).

    python -m agmesh.evaluation.ground_truth site_a   # reads HAI test events (logged)
    python -m agmesh.evaluation.ground_truth site_b   # writes the 7-episode mapping
    python -m agmesh.evaluation.ground_truth relabel  # re-applies the rule; no data read

Site A: every HAI attack event (validation and test) is matched by its start time and
labelled processes to the attack timetable published with the dataset
(:mod:`agmesh.evaluation.hai_attacks`). Its techniques follow from the documented
target points by the rule in that module. That is the ground truth.

Independently, the tool also infers a technique from the data alone: which variables
deviate during the event relative to the normal rows before it, typed by the HAI
point list, with the earliest deviation as root. That inference is kept under
``inferred`` as a cross-check and as a measure of how much of the truth is visible in
the data. Reading the test events is a labelling step: it goes through the
sealed-test gate with the configuration ``ground_truth_site_a`` and is logged. No
model or threshold is involved.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.provenance import git_state, utc_now_iso
from agmesh.data.manifest import read_manifest
from agmesh.evaluation import hai_attacks as ha
from agmesh.evaluation.sealed import FinalAccess, authorise

PROTOCOL_VERSION = "1.1"
OUT_DIR = Path("evaluation/ground_truth")

#: Technique -> tactics in the pinned ATT&CK for ICS v20.0 (:mod:`attack_ics`; a test
#: checks this table against the pin).
TACTICS = {
    "T0836": ["Impair Process Control"],  # Modify Parameter
    "T1692.001": ["Evasion", "Impair Process Control"],  # Command Message (was T0855)
    "T1692.002": ["Evasion", "Impair Process Control"],  # Reporting Message (was T0856)
    "T0888": ["Discovery"],  # Remote System Information Discovery
    "T0846": ["Discovery"],  # Remote System Discovery
    "T0861": ["Collection"],  # Point & Tag Identification
    "T0801": ["Collection"],  # Monitor Process State
    "T0806": ["Impair Process Control"],  # Brute Force I/O
    "T0814": ["Inhibit Response Function"],  # Denial of Service
}

# Typed from the HAI point list (technical details, pp. 12-14). P2_ASD/P2_MSD are the
# auto/manual speed demands, P2_RTR and P2_VTR* the trip limits (setpoints);
# P2_CO_rpm is the speed controller's output (P2_SCO), i.e. a command.
_SETPOINT = re.compile(
    r"^P1_B\d|^P1_STSP$|^P3_L[HL]|_LD$|^P4_(ST|HT)_PS$|^P2_(ASD|MSD|RTR|VTR\d+)$"
)
_COMMAND = re.compile(
    r"^P\d_(FCV|LCV|PCV|PP|LCP)\w*D$|^P2_(AutoGO|ManualGO|OnOff|Emerg|TripEx|CO_rpm)$"
    r"|^P4_ST_GOV$"
)
_SENSOR = re.compile(
    r"Z$|_(FT|LIT|PIT|TIT|SIT|VT|VXT|VYT|FIT|PT|TT)\d|_PO$|_24Vdc$|_HILout$|_FT\d+Z?$"
)


def variable_type(name: str) -> str:
    """setpoint, command, sensor or unknown, from the HAI naming convention."""
    if _SETPOINT.search(name):
        return "setpoint"
    if _COMMAND.search(name):
        return "command"
    if _SENSOR.search(name):
        return "sensor"
    return "unknown"


SUSTAINED = 0.2  # share of event rows outside the pre-event range
MARGIN = 0.1  # widening of the pre-event range, as a share of its width
TIE_ROWS = 5  # deviations starting within this many rows of each other are simultaneous
PRECEDENCE = {"setpoint": 0, "command": 1, "sensor": 2, "unknown": 3}


def deviations(
    context: pd.DataFrame, event: pd.DataFrame, columns: list[str]
) -> list[dict[str, Any]]:
    """Variables that leave their pre-event range during the event, persistently.

    A value deviates if it lies outside the range seen in the normal rows before the
    event, widened by ``MARGIN`` of its width (or by a small absolute amount for a
    constant variable). A variable counts if at least ``SUSTAINED`` of the event rows
    deviate; ``first_row`` is the first such row.
    """
    found = []
    for col in columns:
        ref = context[col].to_numpy(dtype=np.float64)
        vals = event[col].to_numpy(dtype=np.float64)
        if ref.size == 0 or vals.size == 0:
            continue
        lo, hi = float(ref.min()), float(ref.max())
        pad = MARGIN * (hi - lo) if hi > lo else max(1e-6, 1e-3 * abs(lo))
        outside = (vals < lo - pad) | (vals > hi + pad)
        share = float(outside.mean())
        if share >= SUSTAINED:
            width = (hi - lo) or 1.0
            excess = float(np.max(np.maximum(lo - vals, vals - hi)) / width)
            found.append(
                {
                    "variable": col,
                    "type": variable_type(col),
                    "first_row": int(np.flatnonzero(outside)[0]),
                    "share": round(share, 3),
                    "excess": round(excess, 3),
                }
            )
    return sorted(found, key=lambda d: (d["first_row"], PRECEDENCE[d["type"]], -d["excess"]))


def root_cause(devs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Earliest typed deviation; among near-simultaneous ones, setpoint > command > sensor."""
    typed = [d for d in devs if d["type"] != "unknown"]
    if not typed:
        return None
    start = typed[0]["first_row"]
    simultaneous = [d for d in typed if d["first_row"] - start <= TIE_ROWS]
    return min(simultaneous, key=lambda d: (PRECEDENCE[d["type"]], d["first_row"], -d["excess"]))


def rule(devs: list[dict[str, Any]]) -> dict[str, Any]:
    """The protocol's first-pass rule (mapping_protocol.md §2)."""
    types = {d["type"] for d in devs}
    first = root_cause(devs)
    if first is None:
        return {"primary": None, "acceptable": [], "basis": "no typed variable deviates"}
    if first["type"] == "setpoint":
        primary, acceptable = "T0836", ["T0836", "T1692.001"]
    elif first["type"] == "command":
        primary, acceptable = "T1692.001", ["T1692.001"]
    else:
        primary, acceptable = "T1692.002", ["T1692.002"]
        if types & {"command", "setpoint"}:
            acceptable.append("T1692.001")
    return {
        "primary": primary,
        "acceptable": acceptable,
        "basis": f"root deviation: {first['variable']} ({first['type']})",
    }


def site_a(
    params: Params,
    root: Path,
    out: Path,
    log_path: Path,
    *,
    require_documented: bool = False,
) -> dict[str, Any]:
    manifest = read_manifest(root / "site_a")
    features = list(manifest["features"]["features"]) + list(
        manifest["features"]["training_constant"]
    )
    cfg = params.data.hai
    access = FinalAccess("ground_truth_site_a", 0, log_path=log_path, require_frozen=False)
    events: dict[str, Any] = {}
    for client in manifest["nodes"]:
        for part in ("val_attack", "test"):
            if part == "test":
                authorise(access, site="site_a", client=client)
            frame = pd.read_parquet(root / "site_a" / client / f"{part}.parquet")
            for segment_id, seg in frame.groupby("run_id", sort=True):
                label = seg[cfg.label_column].to_numpy() > 0
                if not label.any():
                    continue
                first = int(np.flatnonzero(label)[0])
                last = int(np.flatnonzero(label)[-1])
                context = seg.iloc[:first]
                event = seg.iloc[first : last + 1]
                procs = [
                    c.removeprefix("attack_") for c in cfg.process_label_columns if event[c].any()
                ]
                # only variables of the targeted processes (named by the labels) count
                candidates = [
                    c
                    for c in features
                    if c in seg.columns and (not procs or c.split("_")[0] in procs)
                ]
                devs = deviations(context, event, candidates)
                inferred = rule(devs)
                start = pd.Timestamp(event["time"].iloc[0]).to_pydatetime()
                attack = ha.match(start, procs)
                record: dict[str, Any] = {
                    "partition": part,
                    "client": client,
                    "processes": procs,
                    "start": str(event["time"].iloc[0]),
                    "labelled_s": int(label.sum()),
                    "inferred": inferred,
                    "evidence": devs[:8],
                }
                if attack is not None:
                    primary, acceptable = ha.techniques(attack)
                    record.update(
                        attack_id=attack.attack_id,
                        documented={
                            "source": ha.SOURCE,
                            "start": attack.start,
                            "duration_s": attack.duration_s,
                            "primitives": [
                                {
                                    "code": p.code,
                                    "points": list(p.points),
                                    "columns": list(p.columns),
                                }
                                for p in attack.primitives
                            ],
                        },
                        primary=primary,
                        acceptable=acceptable,
                        basis="documented attack "
                        + attack.attack_id
                        + ": "
                        + " + ".join(
                            f"{p.code} ({', '.join(p.columns)})" for p in attack.primitives
                        ),
                    )
                elif require_documented:
                    raise ValueError(f"{segment_id}: no documented HAI 21.03 attack matches")
                else:
                    record.update(
                        attack_id=None,
                        documented=None,
                        primary=inferred["primary"],
                        acceptable=inferred["acceptable"],
                        basis="inferred only: " + inferred["basis"],
                    )
                record["tactics"] = _tactics(record["acceptable"])
                events[str(segment_id)] = record
    doc = _document("site_a", events)
    _write(doc, out / "mapping_site_a.json")
    return doc


SITE_B = {
    "0": (
        "Two reads, one asking for 65,535 values (a malformed length)",
        "T0888",
        ["T0888", "T0861", "T0846", "T0814"],
    ),
    "1": (
        "About 20 input-register reads; a stacked 23-frame packet",
        "T0861",
        ["T0861", "T0888", "T0814"],
    ),
    "2": ("Five reads across all four read function codes", "T0861", ["T0861", "T0888"]),
    "3": ("Two coil reads", "T0861", ["T0861", "T0801"]),
    "4": (
        "Slow scan: about one discrete-input read a minute for 20 minutes",
        "T0801",
        ["T0801", "T0861"],
    ),
    "5": (
        "65,535 coil writes to one address in 4 minutes",
        "T0806",
        ["T0806", "T1692.001", "T0814"],
    ),
    "6": ("21 coil writes to two addresses", "T1692.001", ["T1692.001"]),
}


def _tactics(acceptable: list[str]) -> list[str]:
    return sorted({t for tech in acceptable for t in TACTICS[tech]})


def _site_b_events() -> dict[str, Any]:
    return {
        ep: {
            "observed": text,
            "primary": primary,
            "acceptable": acceptable,
            "tactics": _tactics(acceptable),
        }
        for ep, (text, primary, acceptable) in SITE_B.items()
    }


def site_b(out: Path) -> dict[str, Any]:
    doc = _document("site_b", _site_b_events())
    _write(doc, out / "mapping_site_b.json")
    return doc


#: Rule v1.0 -> v1.1 for the data-only inference: the new IDs of ATT&CK v20.0, and no
#: Impact umbrella techniques. Applying it to a v1.0 inference gives exactly what
#: :func:`rule` now returns for the same deviations.
_V11 = {"T0855": "T1692.001", "T0856": "T1692.002", "T0831": None, "T0832": None}


def _translate(ids: list[str]) -> list[str]:
    out: list[str] = []
    for tid in ids:
        new = _V11.get(tid, tid)
        if new is not None and new not in out:
            out.append(new)
    return out


def relabel(path: Path) -> dict[str, Any]:
    """Re-apply the current rule to a mapping file, without reading any data.

    Site A: the documented truth of each event is recomputed from its ``attack_id``
    and the timetable. The stored data-only inference is translated by ``_V11``.
    Site B: the episodes are rewritten from :data:`SITE_B`. The review history is
    kept. Used for protocol v1.1 (pass 2), so the sealed test events are not read
    again for a change that does not depend on them.
    """
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if doc["site"] == "site_a":
        by_id = {a.attack_id: a for a in ha.TIMETABLE}
        for event in doc["events"].values():
            inferred = event["inferred"]
            if inferred.get("primary") is not None:
                inferred["primary"] = _translate([inferred["primary"]])[0]
            inferred["acceptable"] = _translate(inferred["acceptable"])
            if event.get("attack_id"):
                primary, acceptable = ha.techniques(by_id[event["attack_id"]])
            else:
                primary, acceptable = inferred["primary"], inferred["acceptable"]
            event["primary"], event["acceptable"] = primary, acceptable
            event["tactics"] = _tactics(acceptable)
        baseline = ha.constant_baseline()
        doc["constant_answer_baseline"] = {k: round(v, 2) for k, v in baseline.items()}
        doc["agreement_with_data_only_inference"] = agreement(doc)
    else:
        doc["events"] = _site_b_events()
    doc["protocol_version"] = PROTOCOL_VERSION
    doc["relabelled_at"] = utc_now_iso()
    doc["relabelled_git_sha"] = git_state()["git_sha"]
    _write(doc, path)
    return doc


def _document(site: str, events: dict[str, Any]) -> dict[str, Any]:
    return {
        "site": site,
        "protocol": "docs/mapping_protocol.md",
        "protocol_version": PROTOCOL_VERSION,
        "pass": 1,
        "reviewed": False,
        "generated_at": utc_now_iso(),
        "git_sha": git_state()["git_sha"],
        "reviews": [],
        "events": events,
    }


def _write(doc: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def agreement(doc: dict[str, Any]) -> dict[str, Any]:
    """How often the data-only inference agrees with the documented ground truth."""
    rows = [e for e in doc["events"].values() if e.get("documented") and "inferred" in e]
    if not rows:
        return {}
    same = sum(e["inferred"]["primary"] == e["primary"] for e in rows)
    inside = sum(e["inferred"]["primary"] in e["acceptable"] for e in rows)
    return {
        "events": len(rows),
        "inferred_primary_equals_documented": same,
        "inferred_primary_in_acceptable": inside,
    }


def summary(doc: dict[str, Any]) -> str:
    primary = Counter(str(e.get("primary")) for e in doc["events"].values())
    tactics = Counter(t for e in doc["events"].values() for t in e["tactics"])
    text = f"{len(doc['events'])} events; primary {dict(primary)}; tactics {dict(tactics)}"
    agree = agreement(doc)
    if agree:
        text += f"; data-only inference vs documented: {agree}"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="First-pass ground-truth mapping (draft)")
    parser.add_argument("site", choices=("site_a", "site_b", "relabel"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--access-log", type=Path, default=Path("results/test_access.log"))
    parser.add_argument(
        "--allow-undocumented",
        action="store_true",
        help="keep events with no documented attack (fixture data); real runs must match all",
    )
    args = parser.parse_args(argv)
    if args.site == "relabel":
        for site in ("a", "b"):
            print(summary(relabel(args.out / f"mapping_site_{site}.json")))
        return 0
    params = load_params(args.params)
    if args.site == "site_a":
        doc = site_a(
            params,
            args.partitions or params.paths.partitions_dir,
            args.out,
            args.access_log,
            require_documented=not args.allow_undocumented,
        )
    else:
        doc = site_b(args.out)
    print(summary(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
