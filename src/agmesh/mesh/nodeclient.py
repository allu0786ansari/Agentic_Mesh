"""How the Investigator asks a node for evidence (ADR 0012 §3; ADR 0014 §4).

``NodeClient.evidence(node, start_s, end_s, timeout_s, session_id=...)`` returns the
node's minimised evidence records whose window ends in ``[start_s, end_s]``, or
raises:

* :class:`TransientNodeError`: a connection error, timeout or HTTP 503, worth one
  retry;
* :class:`NodeError`: anything else, never retried. This includes 401 and 403 from
  a node API.

There are two backends:

* :class:`LocalNodeClient` reads an edge run's evidence stores in-process (Week 10).
* :class:`HttpsNodeClient` asks each node's API over mTLS, with one JWT per request
  (Week 11).
"""

from __future__ import annotations

import ssl
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from agmesh.common.schemas import EvidenceRecord
from agmesh.edge.evidence import iso
from agmesh.edge.evidence_store import EvidenceStore
from agmesh.security.jwt import TokenMinter

SITE_OF = {"a": "site_a", "b": "site_b"}
_RECORDS = TypeAdapter(list[EvidenceRecord])


class NodeError(Exception):
    """A non-transient failure (refused, bad request, unknown node)."""


class TransientNodeError(NodeError):
    """Connection error, timeout or 503: one retry is allowed."""


class NodeClient(Protocol):
    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float, session_id: str = ""
    ) -> list[EvidenceRecord]: ...


class LocalNodeClient:
    """Evidence stores of an edge run (``evidence_<node>.db``), opened per request."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float, session_id: str = ""
    ) -> list[EvidenceRecord]:
        path = self.run_dir / f"evidence_{node}.db"
        if not path.is_file():
            raise NodeError(f"{node}: no evidence store in {self.run_dir}")
        store = EvidenceStore(path)
        try:
            return store.between(node, iso(start_s), iso(end_s))
        finally:
            store.close()


class HttpsNodeClient:
    """Node APIs over mTLS (TLS 1.3, client certificate), one fresh JWT per request.

    * **Connections:** one keep-alive connection per node.
    * **Server identity:** SNI and the hostname check use ``node-<id>.agmesh``, so a
      node API cannot answer for another node.
    """

    def __init__(
        self,
        endpoints: Mapping[str, str],
        ctx: ssl.SSLContext,
        minter: TokenMinter,
        *,
        connect_timeout_s: float = 0.2,
    ) -> None:
        self.endpoints = dict(endpoints)
        self.minter = minter
        self.connect_timeout_s = connect_timeout_s
        self.http = httpx.Client(verify=ctx, timeout=httpx.Timeout(5.0, connect=connect_timeout_s))

    def close(self) -> None:
        self.http.close()

    def evidence(
        self, node: str, start_s: float, end_s: float, timeout_s: float, session_id: str = ""
    ) -> list[EvidenceRecord]:
        base = self.endpoints.get(node)
        if base is None:
            raise NodeError(f"{node}: no endpoint")
        token = self.minter.mint(node, SITE_OF[node[0]], session_id, start_s, end_s)
        try:
            response = self.http.get(
                f"{base}/v1/evidence",
                params={"start": repr(float(start_s)), "end": repr(float(end_s))},
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(timeout_s, connect=min(self.connect_timeout_s, timeout_s)),
                extensions={"sni_hostname": f"node-{node}.agmesh"},
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise TransientNodeError(f"{node}: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            raise NodeError(f"{node}: {type(exc).__name__}") from exc
        if response.status_code == 503:
            raise TransientNodeError(f"{node}: 503")
        if response.status_code != 200:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            raise NodeError(f"{node}: HTTP {response.status_code} {detail}")
        try:
            return _RECORDS.validate_json(response.content)
        except ValidationError as exc:
            raise NodeError(f"{node}: response is not a list of evidence records") from exc
