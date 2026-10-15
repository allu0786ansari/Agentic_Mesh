"""Test doubles for the mesh: a fake node API, a scripted LLM, a fixed retriever, records."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agmesh.common.schemas import EvidenceRecord
from agmesh.edge.evidence import iso, stable_id
from agmesh.kb.retriever import Hit
from agmesh.mesh.fusion import parse_iso
from agmesh.mesh.llm import LLMReply, LLMUnavailableError
from agmesh.mesh.nodeclient import NodeError, TransientNodeError

VERSION = "sha256:" + "cd" * 32
SUMMARY = {
    "read_ops": "1-10", "write_ops": ">100", "diagnostic_ops": "0", "exception_responses": "0",
    "request_rate": "high", "new_peer": True, "unseen_address": False, "oversized_quantity": False,
    "malformed_frames": False, "stacked_frames": False, "slow_responses": False,
}  # fmt: skip


def record(node: str, t0: float, feats: list[tuple[str, str, str]], **over: Any) -> EvidenceRecord:
    site = "site_a" if node.startswith("a") else "site_b"
    length = 60 if site == "site_a" else 10
    base: dict[str, Any] = {
        "evidence_id": stable_id("evidence", VERSION, node, t0), "site_id": site, "node_id": node,
        "window_start": iso(t0), "window_end": iso(t0 + length), "model_version": VERSION,
        "score": 3.0, "threshold": 1.0, "calibrated_prob": 0.95,
        "top_features": [{"name": n, "direction": d, "magnitude": m} for n, d, m in feats],
        "provenance": {"run_id": "test", "data_source": "fixture"},
    }  # fmt: skip
    if site == "site_b":
        base["protocol_summary"] = SUMMARY
    base.update(over)
    return EvidenceRecord.model_validate(base)


def alert_of(r: EvidenceRecord, latency_only: bool = False) -> dict[str, Any]:
    return {
        "alert_id": stable_id("alert", VERSION, r.node_id, parse_iso(r.window_start)),
        "evidence_id": r.evidence_id, "site_id": r.site_id, "node_id": r.node_id,
        "window_end": r.window_end, "calibrated_prob": r.calibrated_prob,
        "latency_only": latency_only, "run_id": "test", "data_source": "fixture",
    }  # fmt: skip


@dataclass
class FakeNodes:
    """Evidence per node; ``faults[node]`` scripts each call: 'ok', 'transient', 'error', 'slow'."""

    records: dict[str, list[EvidenceRecord]]
    faults: dict[str, list[str]] = field(default_factory=dict)
    advance: Callable[[float], None] | None = None  # a fake clock's advance, for 'slow'
    calls: list[str] = field(default_factory=list)

    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float
    ) -> list[EvidenceRecord]:
        self.calls.append(node)
        script = self.faults.get(node, [])
        fault = script.pop(0) if script else "ok"
        if fault == "transient":
            raise TransientNodeError(f"{node}: 503")
        if fault == "error":
            raise NodeError(f"{node}: 400")
        if fault == "slow" and self.advance is not None:
            self.advance(timeout_s + 0.01)
            raise TransientNodeError(f"{node}: read timeout")
        return [
            r for r in self.records.get(node, []) if start_s <= parse_iso(r.window_end) <= end_s
        ]


class ScriptedLLM:
    def __init__(self, reply: dict[str, Any] | str | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def choose(self, user_message: str, candidates: tuple[str, ...]) -> LLMReply:
        self.calls.append((user_message, candidates))
        if isinstance(self.reply, Exception):
            raise self.reply
        content = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return LLMReply(content, 0.01, "scripted")


def fixed_rank(order: list[str]) -> Callable[[str], list[Hit]]:
    def rank(_text: str) -> list[Hit]:
        return [Hit(t, 1.0 - 0.01 * i, f"{t}/summary") for i, t in enumerate(order)]

    return rank


UNAVAILABLE = LLMUnavailableError("timeout")
