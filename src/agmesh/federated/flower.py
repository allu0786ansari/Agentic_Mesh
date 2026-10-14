"""Flower backend: the same server and clients over Flower's Message API (flwr 1.39).

``FlowerPool`` implements the server's :class:`~agmesh.federated.server.ClientPool`
by sending Flower ``Message`` objects through the ``Grid``. The ``ClientApp``
recreates a :class:`~agmesh.federated.client.FederatedClient` for every message, as
Flower's simulation does, and answers. So the round logic, the privacy accounting and
the evaluation are the same code as the in-process backend. A test checks that both
backends give identical results.

**Wire format.** Every protocol object is flattened into a ``RecordDict``. Its arrays
go into an ``ArrayRecord`` and everything else into a JSON string in a
``ConfigRecord``. Nothing is pickled, so a client cannot make the server execute
code.

**Node identity.** The server first sends a ``query`` to every Flower node. Each
client answers with its node name (from its ``partition-id``), and the server keeps
the mapping. The run settings reach the ClientApp through an environment variable,
because Ray workers inherit the driver's environment.

Requires the ``fl`` dependency group.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.federated import protocol as pr

SETTINGS_ENV = "AGMESH_FLOWER_SETTINGS"
_RESULT: dict[str, Any] = {}
_TYPES = {
    t.__name__: t
    for t in (
        pr.ModelSpec,
        pr.HelloReply,
        pr.Normalisation,
        pr.FitInstruction,
        pr.FitReply,
        pr.EvalInstruction,
        pr.EvalReply,
        pr.ThresholdReply,
    )
}


# ----------------------------------------------------------------- wire format
def _flatten(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return {"__array__": len(arrays) - 1}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": type(obj).__name__,
            "fields": {
                f.name: _flatten(getattr(obj, f.name), arrays) for f in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, dict):
        return {"__dict__": [[k, _flatten(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, list | tuple):
        return {"__list__": [_flatten(v, arrays) for v in obj], "tuple": isinstance(obj, tuple)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return {"__float__": repr(obj)}
    return obj


def _unflatten(raw: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(raw, dict):
        if "__array__" in raw:
            return arrays[int(raw["__array__"])]
        if "__type__" in raw:
            cls = _TYPES[raw["__type__"]]
            return cls(**{k: _unflatten(v, arrays) for k, v in raw["fields"].items()})
        if "__dict__" in raw:
            return {k: _unflatten(v, arrays) for k, v in raw["__dict__"]}
        if "__list__" in raw:
            items = [_unflatten(v, arrays) for v in raw["__list__"]]
            return tuple(items) if raw.get("tuple") else items
        if "__float__" in raw:
            return float(raw["__float__"])
    return raw


def pack(obj: Any) -> Any:
    from flwr.app import ArrayRecord, ConfigRecord, RecordDict

    arrays: list[np.ndarray] = []
    meta = json.dumps(_flatten(obj, arrays))
    return RecordDict(
        {
            "meta": ConfigRecord({"json": meta}),
            "arrays": ArrayRecord(numpy_ndarrays=[np.ascontiguousarray(a) for a in arrays]),
        }
    )


def unpack(content: Any) -> Any:
    arrays = content.array_records["arrays"].to_numpy_ndarrays()
    return _unflatten(json.loads(str(content.config_records["meta"]["json"])), list(arrays))


# ----------------------------------------------------------------- client side
@dataclasses.dataclass
class FlowerSettings:
    params_path: str
    site: str
    partitions: str
    run_id: str
    epsilon: float
    seed: int
    h1c_epsilon: float | None
    epochs: int | None
    privacy_root: str
    nodes: list[str]

    def to_env(self) -> str:
        return json.dumps(_flatten(self, []))


def _settings() -> FlowerSettings:
    raw = json.loads(os.environ[SETTINGS_ENV])["fields"]
    vals = {k: _unflatten(v, []) for k, v in raw.items()}
    return FlowerSettings(**vals)


def _client(context: Any) -> Any:
    from agmesh.common.config import load_params
    from agmesh.federated.client import FederatedClient

    s = _settings()
    index = int(context.node_config["partition-id"])
    params = load_params(Path(s.params_path))
    return FederatedClient(
        params,
        s.site,  # type: ignore[arg-type]
        s.nodes[index],
        partitions=Path(s.partitions),
        run_id=s.run_id,
        epsilon=s.epsilon,
        seed=s.seed,
        node_index=index,
        h1c_epsilon=s.h1c_epsilon,
        epochs=s.epochs,
        privacy_root=Path(s.privacy_root),
    ), s.nodes[index]


def build_client_app() -> Any:
    from flwr.app import Message
    from flwr.clientapp import ClientApp

    app = ClientApp()

    @app.query()
    def query(msg: Message, context: Any) -> Message:
        op = unpack(msg.content)
        client, node = _client(context)
        if op == "name":
            reply: Any = node
        elif op == "hello":
            reply = client.hello()
        elif op == "privacy":
            reply = client.privacy()
        else:  # ("threshold", EvalInstruction)
            reply = client.threshold(op[1])
        return Message(content=pack(reply), reply_to=msg)

    @app.train()
    def train(msg: Message, context: Any) -> Message:
        client, _node = _client(context)
        return Message(content=pack(client.fit(unpack(msg.content))), reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Any) -> Message:
        client, _node = _client(context)
        return Message(content=pack(client.evaluate(unpack(msg.content))), reply_to=msg)

    return app


# ----------------------------------------------------------------- server side
class FlowerPool:
    """The server's view of Flower nodes, addressed by our node names."""

    def __init__(self, grid: Any, nodes: list[str], timeout: float) -> None:
        self.grid, self.timeout = grid, timeout
        ids: list[int] = []
        deadline = time.monotonic() + 300.0
        while len(ids) < len(nodes):  # simulated nodes register asynchronously
            if time.monotonic() > deadline:
                raise RuntimeError(f"only {len(ids)} of {len(nodes)} Flower nodes came up")
            time.sleep(0.5)
            ids = list(grid.get_node_ids())
        names = {self._ask(i, "query", "name"): i for i in ids}
        if sorted(names) != sorted(nodes):
            raise RuntimeError(f"Flower nodes {sorted(names)} do not match {sorted(nodes)}")
        self.nodes = list(nodes)
        self.ids = names

    def _ask(self, node_id: int, kind: str, payload: Any) -> Any:
        from flwr.app import Message

        msg = Message(content=pack(payload), dst_node_id=node_id, message_type=kind)
        replies = list(self.grid.send_and_receive([msg], timeout=self.timeout))
        if not replies or replies[0].has_error():
            raise RuntimeError(f"node {node_id} did not answer {kind}")
        return unpack(replies[0].content)

    def hello(self, node: str) -> pr.HelloReply:
        out: pr.HelloReply = self._ask(self.ids[node], "query", "hello")
        return out

    def fit(self, node: str, ins: pr.FitInstruction) -> pr.FitReply:
        out: pr.FitReply = self._ask(self.ids[node], "train", ins)
        return out

    def evaluate(self, node: str, ins: pr.EvalInstruction) -> pr.EvalReply:
        out: pr.EvalReply = self._ask(self.ids[node], "evaluate", ins)
        return out

    def threshold(self, node: str, ins: pr.EvalInstruction) -> pr.ThresholdReply:
        out: pr.ThresholdReply = self._ask(self.ids[node], "query", ("threshold", ins))
        return out

    def privacy(self, node: str) -> dict[str, Any]:
        out: dict[str, Any] = self._ask(self.ids[node], "query", "privacy")
        return out


def run(
    settings: FlowerSettings, server_factory: Any, *, num_cpus_per_client: float, total_cpus: int
) -> Any:
    """Run one federation under Flower's simulation; returns the server's result."""
    from flwr.serverapp import Grid, ServerApp
    from flwr.simulation import run_simulation

    os.environ[SETTINGS_ENV] = settings.to_env()
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Any) -> None:
        pool = FlowerPool(grid, settings.nodes, timeout=3600.0)
        _RESULT["result"] = server_factory(pool).run()

    _RESULT.clear()
    run_simulation(
        server_app=server_app,
        client_app=build_client_app(),
        num_supernodes=len(settings.nodes),
        backend_config={
            "client_resources": {"num_cpus": num_cpus_per_client, "num_gpus": 0.0},
            "init_args": {"num_cpus": total_cpus, "include_dashboard": False},
        },
    )
    if "result" not in _RESULT:
        raise RuntimeError("the Flower server did not finish")
    return _RESULT["result"]
