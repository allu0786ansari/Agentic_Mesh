"""The Dispatcher: attribution → actions under the policy table (Proposal §8.5; ADR 0012 §5).

* **Low-impact actions are dispatched.**
* **High-impact actions are recorded ``pending_approval``.** They execute only
  through ``approvals.decide``.

**How each action is written.**

* It is first a row in ``actions``, unique per ``(session, action type)``.
* It is then published on ``actions.<site>`` with the action ID as its message ID.
* Both writes are idempotent, so a crash between them is repaired by running again.
  That makes dispatch exactly-once per (session, type), an outbox by idempotent keys.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from agmesh.bus.base import Bus
from agmesh.mesh.attribution import Attribution
from agmesh.mesh.sessions import SessionStore

POLICY = Path("configs/policy/actions.yaml")
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


@functools.lru_cache(maxsize=2)
def policy(path: Path = POLICY) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def proposed(
    attribution: Attribution, severity: str, table: dict[str, Any] | None = None
) -> list[tuple[str, str]]:
    """``(action type, impact)`` pairs the policy proposes for an attributed incident."""
    out = []
    attributed = attribution.technique_id is not None
    for name, rule in (table or policy())["actions"].items():
        when = rule["when"]
        if when.get("operator_initiated"):
            continue
        if when.get("attributed") and not attributed:
            continue
        if "min_severity" in when and SEVERITY_RANK[severity] < SEVERITY_RANK[when["min_severity"]]:
            continue
        if "min_confidence" in when and (attribution.confidence or 0.0) < float(
            when["min_confidence"]
        ):
            continue
        out.append((name, rule["impact"]))
    return out


def dispatch(
    store: SessionStore,
    bus: Bus,
    session_id: str,
    site: str,
    attribution: Attribution,
    severity: str,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Record and publish the proposed actions; returns them (new or already present)."""
    done = []
    for action_type, impact in proposed(attribution, severity):
        status = "dispatched" if impact == "low" else "pending_approval"
        payload = {
            "session_id": session_id,
            "site_id": site,
            "action_type": action_type,
            "technique_id": attribution.technique_id,
            "tactics": attribution.tactics,
            "mitigations": attribution.mitigations,
            "severity": severity,
            "attribution_mode": attribution.attribution_mode,
        }
        action_id, new = store.record_action(
            session_id,
            action_type,
            impact,
            status,
            payload,
            technique_id=attribution.technique_id,
            tactic=", ".join(attribution.tactics) or None,
            confidence=attribution.confidence,
            trace_id=trace_id,
        )
        bus.publish(
            f"actions.{site}", {**payload, "action_id": action_id, "status": status}, action_id
        )
        done.append(
            {
                "action_id": action_id,
                "action_type": action_type,
                "impact": impact,
                "status": status,
                "new": new,
            }
        )
    return done
