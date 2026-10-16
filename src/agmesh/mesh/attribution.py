"""The Attribution agent: render → candidates → decide → validate (ADR 0012 §4).

| Mode | Candidates | Decision |
|---|---|---|
| ``full`` | retrieval v2 top ``k`` | one LLM call over the candidates |
| ``retrieval_only`` | retrieval v2 | top-1 |
| ``llm_no_retrieval`` | every v2 candidate technique (IDs and names) | one LLM call |
| ``rule_only`` | — | ``mesh.rules`` |

**Fallback.** Any invalid LLM output, a timeout, or an unreachable LLM falls back to
the top-1 retrieved technique (``retrieval_fallback``). The raw output and whether
it was grounded are kept for the grounding rate.

**What else is recorded.**

* The ranked list, used for top-3 (ADR 0012 §4).
* The tactic, which comes from the pinned extract.
* The mitigations, which come from the parsed bundle and are never generated.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agmesh.common.schemas import EvidenceRecord
from agmesh.evaluation import attack_ics
from agmesh.kb.render import render
from agmesh.kb.retriever import Hit, excluded
from agmesh.mesh import rules
from agmesh.mesh.llm import LLM, LLMUnavailableError, user_message, validate

MODES = ("full", "retrieval_only", "llm_no_retrieval", "rule_only")
TECHNIQUES_JSON = Path("data/kb/techniques.json")


@dataclass
class Attribution:
    mode: str
    attribution_mode: str  # mode actually used: the mode itself, retrieval_fallback or abstained
    technique_id: str | None
    tactics: list[str]
    confidence: float | None
    ranked: list[str]
    candidates: list[str]
    evidence_text: str
    mitigations: list[list[str]] = field(default_factory=list)
    llm: dict[str, Any] | None = None  # raw content, validation, seconds
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_mitigations(path: Path = TECHNIQUES_JSON) -> dict[str, list[list[str]]]:
    if not path.is_file():
        return {}
    return {
        t["technique_id"]: [list(m) for m in t["mitigations"]]
        for t in json.loads(path.read_text(encoding="utf-8"))
    }


def _failure(exc: LLMUnavailableError) -> dict[str, Any]:
    """What a failed call records: the message, the error class and the elapsed time."""
    return {"error": str(exc), "kind": exc.kind, "seconds": exc.seconds}


class AttributionAgent:
    def __init__(
        self,
        rank: Callable[[str], list[Hit]] | None,
        llm: LLM | None,
        *,
        k: int = 10,
        exclude_only_tactics: Sequence[str] = ("Impact",),
        min_confidence: float = 0.5,
        mitigations: dict[str, list[list[str]]] | None = None,
    ) -> None:
        self.rank, self.llm, self.k, self.min_confidence = rank, llm, k, min_confidence
        table = attack_ics.table()
        gone = excluded(exclude_only_tactics)
        self.universe = sorted(
            t for t, v in table.items() if v["status"] == "active" and t not in gone
        )
        self.names = {t: v["name"] for t, v in table.items()}
        self.mitigations = mitigations if mitigations is not None else load_mitigations()

    def line(self, tid: str) -> tuple[str, str, list[str]]:
        """A candidate's prompt line: ID, name and tactics."""
        return tid, self.names[tid], attack_ics.tactics(tid)

    def _result(
        self,
        mode: str,
        used: str,
        tid: str | None,
        conf: float | None,
        ranked: list[str],
        cands: list[str],
        text: str,
        llm: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> Attribution:
        return Attribution(
            mode,
            used,
            tid,
            attack_ics.tactics(tid) if tid else [],
            conf,
            ranked,
            cands,
            text,
            self.mitigations.get(attack_ics.canonical(tid), []) if tid else [],
            llm,
            reason,
        )

    def attribute(self, site: str, records: Sequence[EvidenceRecord], mode: str) -> Attribution:
        if mode not in MODES:
            raise ValueError(f"unknown attribution mode {mode!r}")
        if not records:
            return self._result(mode, "abstained", None, None, [], [], "", reason="no evidence")
        text = render(site, records)
        if mode == "rule_only":
            d = rules.decide(site, records)
            used = "rule_only" if d.technique else "abstained"
            return self._result(
                mode,
                used,
                d.technique,
                None,
                [d.technique] if d.technique else [],
                [],
                text,
                reason=d.basis,
            )
        if mode == "llm_no_retrieval":
            return self._no_retrieval(text)
        if self.rank is None:
            raise RuntimeError(f"mode {mode} needs a retriever")
        order = [h.technique_id for h in self.rank(text)]
        if mode == "retrieval_only":
            return self._result(mode, mode, order[0], None, order, order[: self.k], text)
        cands = order[: self.k]
        if self.llm is None:
            return self._result(
                mode,
                "retrieval_fallback",
                order[0],
                None,
                order,
                cands,
                text,
                reason="no LLM configured",
            )
        message = user_message(text, [self.line(t) for t in cands])
        try:
            reply = self.llm.choose(message, tuple(cands))
        except LLMUnavailableError as exc:
            return self._result(
                mode,
                "retrieval_fallback",
                order[0],
                None,
                order,
                cands,
                text,
                _failure(exc),
                reason="LLM unavailable or timed out",
            )
        v = validate(reply.content, tuple(cands), self.min_confidence)
        info = {
            "content": reply.content,
            "seconds": reply.seconds,
            "model": reply.model,
            "stats": reply.stats,
            "validation": v,
        }
        if not v["valid"]:
            return self._result(
                mode, "retrieval_fallback", order[0], None, order, cands, text, info, v["reason"]
            )
        chosen = str(v["technique_id"])
        ranked = [chosen] + [t for t in order if t != chosen]
        return self._result(mode, mode, chosen, float(v["confidence"]), ranked, cands, text, info)

    def _no_retrieval(self, text: str) -> Attribution:
        """The LLM over every candidate technique, with no retrieval at any step.

        An invalid output abstains: falling back to retrieval would mix retrieval into
        the ablation. The ranked list is the single choice.
        """
        mode, cands = "llm_no_retrieval", list(self.universe)
        if self.llm is None:
            return self._result(
                mode, "abstained", None, None, [], cands, text, reason="no LLM configured"
            )
        message = user_message(text, [self.line(t) for t in cands])
        try:
            reply = self.llm.choose(message, tuple(cands))
        except LLMUnavailableError as exc:
            return self._result(
                mode,
                "abstained",
                None,
                None,
                [],
                cands,
                text,
                _failure(exc),
                reason="LLM unavailable or timed out",
            )
        v = validate(reply.content, tuple(cands), self.min_confidence)
        info = {
            "content": reply.content,
            "seconds": reply.seconds,
            "model": reply.model,
            "stats": reply.stats,
            "validation": v,
        }
        if not v["valid"]:
            return self._result(mode, "abstained", None, None, [], cands, text, info, v["reason"])
        chosen = str(v["technique_id"])
        return self._result(mode, mode, chosen, float(v["confidence"]), [chosen], cands, text, info)
