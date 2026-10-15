"""How the Investigator asks a node for evidence (ADR 0012 §3).

``NodeClient.evidence(node, start_s, end_s, timeout_s)`` returns the node's minimised
evidence records whose window ends in ``[start_s, end_s]``, or raises:

* :class:`TransientNodeError`: a connection error, timeout or HTTP 503, worth one
  retry;
* :class:`NodeError`: anything else, never retried.

:class:`LocalNodeClient` reads an edge run's per-node evidence stores directly. That
is Week 10's backend. Week 11 swaps in the mTLS/JWT node API behind the same
interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agmesh.common.schemas import EvidenceRecord
from agmesh.edge.evidence import iso
from agmesh.edge.evidence_store import EvidenceStore


class NodeError(Exception):
    """A non-transient failure (refused, bad request, unknown node)."""


class TransientNodeError(NodeError):
    """Connection error, timeout or 503: one retry is allowed."""


class NodeClient(Protocol):
    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float
    ) -> list[EvidenceRecord]: ...


class LocalNodeClient:
    """Evidence stores of an edge run (``evidence_<node>.db``), opened per request."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float
    ) -> list[EvidenceRecord]:
        path = self.run_dir / f"evidence_{node}.db"
        if not path.is_file():
            raise NodeError(f"{node}: no evidence store in {self.run_dir}")
        store = EvidenceStore(path)
        try:
            return store.between(node, iso(start_s), iso(end_s))
        finally:
            store.close()
