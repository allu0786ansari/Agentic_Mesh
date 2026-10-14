"""First-pass attribution ground truth (docs/mapping_protocol.md).

    python -m agmesh.evaluation.ground_truth site_a   # reads HAI test events (logged)
    python -m agmesh.evaluation.ground_truth site_b   # writes the 7-episode candidate mapping

Site A: for every HAI attack event (validation and test), the tool finds which
variables deviate during the event relative to the normal rows before it in the same
segment, classifies them by type from the HAI naming convention, and applies the
protocol's rule to propose a primary technique and an acceptable set. Reading the
test events is a labelling step: it goes through the sealed-test gate with the
configuration ``ground_truth_site_a`` and is logged. No model or threshold is involved.

The output is a **draft** (``pass: 1, reviewed: false``). The author reviews it twice
and edits it before it is frozen at G1.
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
from agmesh.evaluation.sealed import FinalAccess, authorise

PROTOCOL_VERSION = "draft-1"
OUT_DIR = Path("evaluation/ground_truth")

#: Technique -> tactics (ATT&CK for ICS; checked against the STIX bundle in Week 9).
TACTICS = {
    "T0836": ["Impair Process Control"],  # Modify Parameter
    "T0855": ["Impair Process Control"],  # Unauthorized Command Message
    "T0831": ["Impact"],  # Manipulation of Control
    "T0856": ["Evasion", "Impair Process Control"],  # Spoof Reporting Message
    "T0832": ["Impact"],  # Manipulation of View
    "T0888": ["Discovery"],  # Remote System Information Discovery
    "T0846": ["Discovery"],  # Remote System Discovery
    "T0861": ["Collection"],  # Point & Tag Identification
    "T0801": ["Collection"],  # Monitor Process State
    "T0806": ["Impair Process Control"],  # Brute Force I/O
    "T0814": ["Inhibit Response Function"],  # Denial of Service
}

_SETPOINT = re.compile(r"^P1_B\d|^P1_STSP$|^P3_L[HL]$|_LD$|^P4_ST_PS$")
_COMMAND = re.compile(
    r"^P\d_(FCV|LCV|PCV|PP|LCP)\w*D$|^P2_(AutoGO|ManualGO|OnOff|Emerg|ASD|MSD|TripEx|RTR)$|^P4_ST_GOV$"
)
_SENSOR = re.compile(
    r"Z$|_(FT|LIT|PIT|TIT|SIT|VT|VXT|VYT|FIT|PT|TT)\d|_PO$|_CO_rpm$|_24Vdc$|_HILout$|_FT\d+Z?$"
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
        primary, acceptable = "T0836", ["T0836", "T0855"]
    elif first["type"] == "command":
        primary, acceptable = "T0855", ["T0855", "T0831"]
    else:
        primary, acceptable = "T0856", ["T0856", "T0832"]
        if types & {"command", "setpoint"}:
            acceptable.append("T0855")
    return {
        "primary": primary,
        "acceptable": acceptable,
        "basis": f"root deviation: {first['variable']} ({first['type']})",
    }


def site_a(params: Params, root: Path, out: Path, log_path: Path) -> dict[str, Any]:
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
                decision = rule(devs)
                events[str(segment_id)] = {
                    "partition": part,
                    "client": client,
                    "processes": procs,
                    **decision,
                    "tactics": sorted(
                        {t for tech in decision["acceptable"] for t in TACTICS[tech]}
                    ),
                    "evidence": devs[:8],
                }
    doc = _document("site_a", events)
    _write(doc, out / "mapping_site_a.json")
    return doc


def site_b(out: Path) -> dict[str, Any]:
    table = {
        "0": ("Two reads, one asking for 65,535 values", "T0888", ["T0888", "T0861", "T0846"]),
        "1": (
            "About 20 input-register reads; a stacked 23-frame packet",
            "T0861",
            ["T0861", "T0888"],
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
            ["T0806", "T0855", "T0814"],
        ),
        "6": ("21 coil writes to two addresses", "T0855", ["T0855", "T0831"]),
    }
    events = {
        ep: {
            "observed": text,
            "primary": primary,
            "acceptable": acceptable,
            "tactics": sorted({t for tech in acceptable for t in TACTICS[tech]}),
        }
        for ep, (text, primary, acceptable) in table.items()
    }
    doc = _document("site_b", events)
    _write(doc, out / "mapping_site_b.json")
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


def summary(doc: dict[str, Any]) -> str:
    primary = Counter(str(e.get("primary")) for e in doc["events"].values())
    tactics = Counter(t for e in doc["events"].values() for t in e["tactics"])
    return f"{len(doc['events'])} events; primary {dict(primary)}; tactics {dict(tactics)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="First-pass ground-truth mapping (draft)")
    parser.add_argument("site", choices=("site_a", "site_b"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--access-log", type=Path, default=Path("results/test_access.log"))
    args = parser.parse_args(argv)
    params = load_params(args.params)
    if args.site == "site_a":
        doc = site_a(
            params, args.partitions or params.paths.partitions_dir, args.out, args.access_log
        )
    else:
        doc = site_b(args.out)
    print(summary(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
