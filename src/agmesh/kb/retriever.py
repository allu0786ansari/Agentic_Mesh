"""Technique retrieval: top-k ATT&CK for ICS techniques for a text (ADR 0010 §2, ADR 0011).

A technique's score is the highest cosine similarity among its passages (summary
and detection). The ranking covers every candidate. NIST context comes from a
separate collection and is never a technique candidate.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agmesh.evaluation import attack_ics
from agmesh.kb.embed import Encoder
from agmesh.kb.store import Store

VERSIONS = Path("configs/kb/retrieval.yaml")


@functools.lru_cache(maxsize=2)
def versions(path: Path = VERSIONS) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def version(name: str | None = None) -> dict[str, Any]:
    """A retrieval version (ADR 0010 v1, ADR 0011 v2); the default when ``name`` is None."""
    cfg = versions()
    key = name or cfg["default"]
    if key not in cfg["versions"]:
        raise ValueError(f"unknown retrieval version {key!r}")
    return {"name": key, **cfg["versions"][key]}


def excluded(only_tactics: Sequence[str]) -> frozenset[str]:
    """Active techniques whose tactics are all in ``only_tactics`` (ADR 0011 §2)."""
    if not only_tactics:
        return frozenset()
    keep = set(only_tactics)
    return frozenset(
        t
        for t, v in attack_ics.table().items()
        if v["status"] == "active" and v["tactics"] and set(v["tactics"]) <= keep
    )


@dataclass(frozen=True)
class Hit:
    technique_id: str
    score: float
    passage_id: str  # the passage that gave the score


class Retriever:
    def __init__(
        self, store: Store, encoder: Encoder, *, exclude_only_tactics: Sequence[str] = ()
    ) -> None:
        if store.dim("mitre_ics") != encoder.dim:
            raise ValueError(
                f"encoder dimension {encoder.dim} differs from mitre_ics {store.dim('mitre_ics')}"
            )
        self.store, self.encoder = store, encoder
        self.n_passages = store.count("mitre_ics")
        self.excluded = excluded(exclude_only_tactics)

    def rank(self, text: str) -> list[Hit]:
        """Every technique, best first (max over its passages; ties by technique ID)."""
        vector = self.encoder.encode([text])[0]
        best: dict[str, Hit] = {}
        for score, payload in self.store.search("mitre_ics", vector, self.n_passages):
            tid = str(payload["technique_id"])
            if tid in self.excluded:
                continue
            if tid not in best or score > best[tid].score:
                best[tid] = Hit(tid, score, str(payload["passage_id"]))
        return sorted(best.values(), key=lambda h: (-h.score, h.technique_id))

    def top(self, text: str, k: int = 5) -> list[Hit]:
        return self.rank(text)[:k]

    def nist_context(self, text: str, k: int = 3) -> list[tuple[float, str]]:
        if self.store.count("nist_80082") == 0:
            return []
        vector = self.encoder.encode([text])[0]
        return [(s, str(p["text"])) for s, p in self.store.search("nist_80082", vector, k)]
