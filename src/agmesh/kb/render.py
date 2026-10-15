"""Evidence records → incident text (Plan Week 9 task 4; ADR 0010 §3).

The renderer is a deterministic template. It reads only the minimised outbound
records:

* feature names;
* directions and magnitude buckets;
* protocol buckets and flags;
* record times.

It never reads labels, scores or raw values. Its vocabulary comes from the
datasets' own documentation (``configs/kb/hai_points.yaml``,
``configs/kb/modbus_features.yaml``), and point types come from
``ground_truth.variable_type``. The template contains no technique names, IDs or
tactics.
"""

from __future__ import annotations

import collections
import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.schemas import EvidenceRecord
from agmesh.evaluation.ground_truth import variable_type
from agmesh.mesh.fusion import parse_iso

HAI_POINTS = Path("configs/kb/hai_points.yaml")
MODBUS_FEATURES = Path("configs/kb/modbus_features.yaml")
MAGNITUDE_RANK = {"low": 0, "medium": 1, "high": 2}
COUNT_RANK = {"0": 0, "1-10": 1, "11-100": 2, ">100": 3}
TYPE_PHRASE = {
    "setpoint": "setpoint",
    "command": "control command",
    "sensor": "sensor reading",
    "unknown": "process variable",
}
MAX_FEATURES = 5


@functools.lru_cache(maxsize=4)
def _yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


@dataclass(frozen=True)
class FeatureSummary:
    name: str
    records: int
    direction: str  # majority direction ("up" on a tie)
    magnitude: str  # highest magnitude seen


def summarise_features(records: Sequence[EvidenceRecord]) -> list[FeatureSummary]:
    count: collections.Counter[str] = collections.Counter()
    ups: collections.Counter[str] = collections.Counter()
    mag: dict[str, int] = {}
    for r in records:
        for f in r.top_features:
            count[f.name] += 1
            ups[f.name] += f.direction == "up"
            mag[f.name] = max(mag.get(f.name, 0), MAGNITUDE_RANK[f.magnitude])
    inverse = {v: k for k, v in MAGNITUDE_RANK.items()}
    out = [
        FeatureSummary(n, c, "up" if 2 * ups[n] >= c else "down", inverse[mag[n]])
        for n, c in count.items()
    ]
    return sorted(out, key=lambda s: (-s.records, -MAGNITUDE_RANK[s.magnitude], s.name))


def _span(records: Sequence[EvidenceRecord]) -> tuple[int, float]:
    starts = [parse_iso(r.window_start) for r in records]
    ends = [parse_iso(r.window_end) for r in records]
    return len(records), (max(ends) - min(starts)) / 60.0


def _minutes(m: float) -> str:
    return f"{m:.0f} minute" + ("" if round(m) == 1 else "s")


def _hai_feature(s: FeatureSummary) -> str:
    points = _yaml(HAI_POINTS)["points"]
    desc = (points.get(s.name) or {}).get("description")
    kind = TYPE_PHRASE[variable_type(s.name)]
    what = f"{kind} {desc[0].lower() + desc[1:]} ({s.name})" if desc else f"{kind} {s.name}"
    verb = "rose" if s.direction == "up" else "fell"
    return f"{what} {verb}, {s.magnitude} deviation"


def _modbus_feature(s: FeatureSummary) -> str:
    desc = _yaml(MODBUS_FEATURES)["features"].get(s.name, s.name)
    verb = "rose" if s.direction == "up" else "fell"
    return f"{desc} ({s.name}) {verb}, {s.magnitude} deviation"


def render_site_a(records: Sequence[EvidenceRecord]) -> str:
    if not records:
        raise ValueError("an incident needs at least one evidence record")
    records = sorted(records, key=lambda r: r.window_start)
    nodes = sorted({r.node_id for r in records})
    feats = summarise_features(records)[:MAX_FEATURES]
    processes = sorted(
        {f.name.split("_")[0] for f in feats}
        | {c.split("_")[0] for r in records for c in r.constants_changed}
    )
    names = _yaml(HAI_POINTS)["processes"]
    proc = ", ".join(f"{names.get(p, p)} {p}" for p in processes) or "unknown process"
    n, minutes = _span(records)
    parts = [
        f"Industrial control system incident at a boiler, turbine and water-treatment testbed "
        f"({proc}), node {', '.join(nodes)}: {n} anomalous 60-second windows "
        f"over {_minutes(minutes)}."
    ]
    if feats:
        parts.append("Deviating variables: " + "; ".join(_hai_feature(s) for s in feats) + ".")
    changed = sorted({c for r in records for c in r.constants_changed})
    if changed:
        points = _yaml(HAI_POINTS)["points"]
        items = []
        for c in changed:
            desc = (points.get(c) or {}).get("description")
            items.append(f"{desc[0].lower() + desc[1:]} ({c})" if desc else c)
        parts.append(
            "Variables that are constant in normal operation changed: " + "; ".join(items) + "."
        )
    return " ".join(parts)


def render_site_b(records: Sequence[EvidenceRecord]) -> str:
    if not records:
        raise ValueError("an incident needs at least one evidence record")
    records = sorted(records, key=lambda r: r.window_start)
    devices = _yaml(MODBUS_FEATURES)["devices"]
    nodes = sorted({r.node_id for r in records})
    where = "; ".join(f"{devices.get(n, n)} ({n})" for n in nodes)
    n, minutes = _span(records)
    parts = [
        f"Modbus/TCP incident at {where}: {n} anomalous 10-second windows over {_minutes(minutes)}."
    ]
    summaries = [r.protocol_summary for r in records if r.protocol_summary is not None]
    if summaries:

        def most(field: str) -> str:
            values: list[str] = [str(getattr(s, field)) for s in summaries]
            return max(values, key=lambda v: COUNT_RANK[v])

        rates = {s.request_rate for s in summaries}
        rate = "high" if "high" in rates else "low" if "low" in rates else "normal"
        parts.append(
            f"Per 10-second window, at most: read requests {most('read_ops')}, write requests "
            f"{most('write_ops')}, requests with other function codes {most('diagnostic_ops')}, "
            f"exception responses {most('exception_responses')}; request rate {rate}."
        )
        flags = [
            ("new_peer", "requests from a peer never seen in training"),
            ("unseen_address", "requests to register addresses never seen in training"),
            ("oversized_quantity", "requests for more values than ever seen in training"),
            ("malformed_frames", "malformed or truncated Modbus frames"),
            ("stacked_frames", "several Modbus frames stacked in one packet"),
            ("slow_responses", "unusually slow responses"),
        ]
        seen = [text for key, text in flags if any(getattr(s, key) for s in summaries)]
        if seen:
            parts.append("Observed: " + "; ".join(seen) + ".")
    feats = summarise_features(records)[:MAX_FEATURES]
    if feats:
        parts.append("Deviating counters: " + "; ".join(_modbus_feature(s) for s in feats) + ".")
    return " ".join(parts)


def render(site: str, records: Sequence[EvidenceRecord]) -> str:
    return render_site_a(records) if site == "site_a" else render_site_b(records)
