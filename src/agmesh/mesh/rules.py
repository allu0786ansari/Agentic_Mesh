"""The ``rule_only`` attribution baseline (ADR 0010 §6; ``configs/policy/rule_baseline.yaml``).

The rules are applied in order to one incident's minimised evidence records. The
first match gives the technique; when nothing matches, the baseline abstains. The
rule table was frozen before any X4 run.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.schemas import EvidenceRecord
from agmesh.evaluation.ground_truth import variable_type
from agmesh.mesh.fusion import parse_iso

RULES = Path("configs/policy/rule_baseline.yaml")
COUNT_RANK = {"0": 0, "1-10": 1, "11-100": 2, ">100": 3}


@dataclass(frozen=True)
class RuleDecision:
    technique: str | None  # None = abstain
    rule_id: str | None
    basis: str


@functools.lru_cache(maxsize=2)
def load(path: Path = RULES) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def root_type(
    records: Sequence[EvidenceRecord], precedence: Sequence[str]
) -> tuple[str | None, str | None]:
    """Type and name of the root deviation: the earliest record with a typed feature."""
    for r in sorted(records, key=lambda r: r.window_start):
        candidates = [(variable_type(f.name), f.name) for f in r.top_features]
        candidates += [(t, c) for c in r.constants_changed if (t := variable_type(c)) == "setpoint"]
        typed = [(t, n) for t, n in candidates if t in precedence]
        if typed:
            return min(typed, key=lambda tn: precedence.index(tn[0]))
    return None, None


def _site_b_match(cond: dict[str, Any], records: Sequence[EvidenceRecord]) -> bool:
    summaries = [r.protocol_summary for r in records if r.protocol_summary is not None]
    if not summaries:
        return False
    if "write_ops" in cond and not any(
        COUNT_RANK[s.write_ops] >= COUNT_RANK[cond["write_ops"]] for s in summaries
    ):
        return False
    if cond.get("any_writes") and not any(s.write_ops != "0" for s in summaries):
        return False
    if cond.get("any_reads") and not any(s.read_ops != "0" for s in summaries):
        return False
    if "any_flag" in cond and not any(getattr(s, f) for s in summaries for f in cond["any_flag"]):
        return False
    if "min_span_minutes" in cond:
        span = (
            max(parse_iso(r.window_end) for r in records)
            - min(parse_iso(r.window_start) for r in records)
        ) / 60
        if span < float(cond["min_span_minutes"]):
            return False
    return True


def decide(
    site: str, records: Sequence[EvidenceRecord], table: dict[str, Any] | None = None
) -> RuleDecision:
    rules = table or load()
    if not records:
        return RuleDecision(None, None, "no evidence")
    cfg = rules[site]
    if site == "site_a":
        kind, name = root_type(records, cfg["precedence"])
        for rule in cfg["rules"]:
            if rule["when"]["root_type"] == kind:
                return RuleDecision(
                    rule["technique"], rule["id"], f"root deviation {name} ({kind})"
                )
        return RuleDecision(None, None, "no typed deviation")
    for rule in cfg["rules"]:
        if _site_b_match(rule["when"], records):
            return RuleDecision(
                rule["technique"], rule["id"], f"matched {rule['id']}: {rule['when']}"
            )
    return RuleDecision(None, None, "no rule matched")
