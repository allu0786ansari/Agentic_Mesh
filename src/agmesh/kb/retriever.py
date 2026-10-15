"""Technique retrieval: top-k ATT&CK for ICS techniques for a text (ADR 0010 §2).

A technique's score is the highest cosine similarity among its passages (summary
and detection). The ranking covers every candidate. NIST context comes from a
separate collection and is never a technique candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from agmesh.kb.embed import Encoder
from agmesh.kb.store import Store


@dataclass(frozen=True)
class Hit:
    technique_id: str
    score: float
    passage_id: str  # the passage that gave the score


class Retriever:
    def __init__(self, store: Store, encoder: Encoder) -> None:
        if store.dim("mitre_ics") != encoder.dim:
            raise ValueError(
                f"encoder dimension {encoder.dim} differs from mitre_ics {store.dim('mitre_ics')}"
            )
        self.store, self.encoder = store, encoder
        self.n_passages = store.count("mitre_ics")

    def rank(self, text: str) -> list[Hit]:
        """Every technique, best first (max over its passages; ties by technique ID)."""
        vector = self.encoder.encode([text])[0]
        best: dict[str, Hit] = {}
        for score, payload in self.store.search("mitre_ics", vector, self.n_passages):
            tid = str(payload["technique_id"])
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
