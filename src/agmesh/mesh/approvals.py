"""The approval gate for high-impact actions (Proposal §8.5; ADR 0012 §5).

A high-impact action leaves ``pending_approval`` only through :func:`decide`, which
requires a valid approval token:

* the token is an HMAC-SHA256 over ``action_id|decision|approver|expires_at`` with
  the key in ``MESH_APPROVAL_KEY``;
* an expired, malformed or tampered token is refused;
* so is a token for a different action or decision.

Every decision is published on the ``approvals`` topic. Week 11's authenticated HTTP
endpoint calls this function and adds nothing to the check.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Callable

from agmesh.bus.base import Bus
from agmesh.mesh.sessions import SessionStore

KEY_ENV = "MESH_APPROVAL_KEY"
DECISIONS = ("approved", "rejected")


class ApprovalError(PermissionError):
    pass


def _key(key: bytes | None) -> bytes:
    k = key if key is not None else os.environ.get(KEY_ENV, "").encode()
    if len(k) < 16:
        raise ApprovalError(f"{KEY_ENV} is not set (at least 16 bytes)")
    return k


def sign(
    action_id: str, decision: str, approver: str, expires_at: int, key: bytes | None = None
) -> str:
    msg = f"{action_id}|{decision}|{approver}|{expires_at}".encode()
    return hmac.new(_key(key), msg, hashlib.sha256).hexdigest()


def decide(
    store: SessionStore,
    bus: Bus,
    action_id: str,
    decision: str,
    approver: str,
    expires_at: int,
    token: str,
    *,
    key: bytes | None = None,
    clock: Callable[[], float] = time.time,
) -> str:
    """Approve or reject a pending high-impact action; returns its new status."""
    if decision not in DECISIONS:
        raise ApprovalError(f"unknown decision {decision!r}")
    if clock() > expires_at:
        raise ApprovalError("approval token expired")
    expected = sign(action_id, decision, approver, expires_at, key)
    if not hmac.compare_digest(expected, token):
        raise ApprovalError("invalid approval token")
    action = next((a for a in store.actions() if a["action_id"] == action_id), None)
    if action is None:
        raise ApprovalError(f"unknown action {action_id}")
    if action["impact"] != "high":
        raise ApprovalError("only high-impact actions need approval")
    if not store.set_action_status(action_id, "pending_approval", decision):
        raise ApprovalError(f"action {action_id} is not pending (it is {action['status']})")
    bus.publish(
        "approvals",
        {"action_id": action_id, "decision": decision, "approver": approver},
        f"{action_id}:{decision}",
    )
    return decision
