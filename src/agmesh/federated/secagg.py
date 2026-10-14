"""SecAgg+ on Fed-A through Flower's own mechanism (Plan Week 7; experiment X8).

Secure aggregation is a **systems experiment**, separate from the detection
experiments (Proposal §7.4). It answers three questions:

1. **Correctness.** Is the SecAgg+ aggregate the plaintext mean, up to quantisation?
2. **Overhead.** What do time per round and bytes on the wire cost?
3. **Dropout tolerance.** Is the mean of the survivors still recovered when a client
   drops after the key exchange?

**Mechanism.** Flower 1.39's ``SecAggPlusWorkflow`` (server) with ``secaggplus_mod``
(client), through the legacy ``LegacyContext`` + ``DefaultWorkflow`` API, which is
where Flower 1.39 provides SecAgg+. Clients run the same
:class:`~agmesh.federated.client.FederatedClient` as everywhere else: DP-SGD, ledger,
guard. Each client reports ``num_examples = 1`` and the workflow uses
``max_weight = 1``, so the secure sum is the **uniform** mean (ADR 0007) at full
quantisation precision. A larger ``max_weight`` would scale updates down before
quantisation and lose precision.

**Measurement side channel (experimenter only).** To check correctness, each client
also writes its plaintext update to the run's record folder, and a client mod sums
the payload bytes of every message by stage. A deployment would do neither.

Requires the ``fl`` dependency group.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from agmesh.data.access import Site

SETTINGS_ENV = "AGMESH_SECAGG_SETTINGS"
_RESULT: dict[str, Any] = {}


@dataclasses.dataclass
class SecAggSettings:
    params_path: str
    site: str
    partitions: str
    run_id: str
    epsilon: float
    seed: int
    nodes: list[str]
    rounds: int
    record_dir: str
    privacy_root: str
    secure: bool
    drops: list[list[Any]]  # [[node, round], ...]: the client fails in that round's fit
    num_shares: int
    reconstruction_threshold: int
    clipping_range: float
    quantization_range: int

    def to_env(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_env(cls) -> SecAggSettings:
        return cls(**json.loads(os.environ[SETTINGS_ENV]))


def _payload_bytes(content: Any) -> int:
    """Payload bytes of a RecordDict: arrays plus config values (no framing)."""
    total = 0
    for rec in content.array_records.values():
        total += sum(int(np.asarray(a).nbytes) for a in rec.to_numpy_ndarrays())
    for rec in content.config_records.values():
        for value in rec.values():
            items = value if isinstance(value, list) else [value]
            for v in items:
                total += len(v) if isinstance(v, bytes | str) else 8
    return total


def size_mod(msg: Any, context: Any, call_next: Any) -> Any:
    """Client mod: add each message's payload bytes, in and out, to a per-node log."""
    s = SecAggSettings.from_env()
    node = s.nodes[int(context.node_config["partition-id"])]
    reply = call_next(msg, context)
    stage = msg.metadata.message_type
    records: list[Any] = list(msg.content.config_records.values())
    if records:
        stage = f"{stage}:{records[0].get('secaggplus_stage', '')}"
    path = Path(s.record_dir) / f"bytes_{node}.jsonl"
    with path.open("a", encoding="utf-8") as fh:  # experimenter side channel
        fh.write(
            json.dumps(
                {
                    "stage": stage,
                    "in": _payload_bytes(msg.content),
                    "out": _payload_bytes(reply.content),
                }
            )
            + "\n"
        )
    return reply


def _numpy_client(context: Any) -> Any:
    from flwr.client import NumPyClient

    from agmesh.common.config import load_params
    from agmesh.federated.client import FederatedClient
    from agmesh.federated.protocol import FitInstruction, Normalisation

    s = SecAggSettings.from_env()
    index = int(context.node_config["partition-id"])
    node = s.nodes[index]
    params = load_params(Path(s.params_path))
    record = Path(s.record_dir)
    norm_raw = np.load(record / "norm.npz")
    norm = Normalisation(norm_raw["centre"], norm_raw["scale"])

    class Client(NumPyClient):  # type: ignore[misc]  # untyped without the fl group
        def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> Any:
            rnd = int(config["round"])
            if [node, rnd] in s.drops:
                raise RuntimeError(f"{node} drops out in round {rnd} (injected)")
            client = FederatedClient(
                params,
                cast(Site, s.site),
                node,
                partitions=Path(s.partitions),
                run_id=s.run_id,
                epsilon=s.epsilon,
                seed=s.seed,
                node_index=index,
                epochs=s.rounds,
                privacy_root=Path(s.privacy_root),
            )
            ins = FitInstruction(
                rnd,
                [np.asarray(p, np.float32) for p in parameters],
                norm,
                params.federated.local_epochs,
                0.0,
                seed=s.seed * 100_000 + rnd * 100 + index,
            )
            reply = client.fit(ins)
            if reply.weights is None:
                raise RuntimeError(f"{node} declined: {reply.declined}")
            np.savez(record / f"update_r{rnd}_{node}.npz", *reply.weights)  # experimenter only
            return reply.weights, 1, {"seconds": reply.seconds}

    return Client().to_client()


def run(
    settings: SecAggSettings, initial: list[np.ndarray], *, num_cpus: float, total_cpus: int
) -> dict[str, Any]:
    """One federation of ``settings.rounds`` rounds; returns aggregates and timing."""
    from flwr.client import ClientApp
    from flwr.client.mod import secaggplus_mod
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import LegacyContext, ServerApp, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
    from flwr.simulation import run_simulation

    os.environ[SETTINGS_ENV] = settings.to_env()
    aggregates: dict[int, list[np.ndarray]] = {}
    marks: dict[int, float] = {}
    survivors: dict[int, int] = {}

    class Recording(FedAvg):  # type: ignore[misc]
        def configure_fit(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
            marks[server_round] = time.perf_counter()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> Any:
            out = super().aggregate_fit(server_round, results, failures)
            if out[0] is not None:
                aggregates[server_round] = parameters_to_ndarrays(out[0])
            survivors[server_round] = len(results)
            marks[server_round] = time.perf_counter() - marks[server_round]
            return out

    n = len(settings.nodes)
    strategy = Recording(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=n,
        min_available_clients=n,
        accept_failures=True,
        initial_parameters=ndarrays_to_parameters(initial),
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Any, context: Any) -> None:
        ctx = LegacyContext(
            context=context, config=ServerConfig(num_rounds=settings.rounds), strategy=strategy
        )
        fit = (
            SecAggPlusWorkflow(
                num_shares=settings.num_shares,
                reconstruction_threshold=settings.reconstruction_threshold,
                max_weight=1.0,
                clipping_range=settings.clipping_range,
                quantization_range=settings.quantization_range,
            )
            if settings.secure
            else None
        )
        DefaultWorkflow(fit_workflow=fit)(grid, ctx)

    mods: list[Any] = [size_mod, secaggplus_mod] if settings.secure else [size_mod]
    client_app = ClientApp(client_fn=_numpy_client, mods=mods)
    started = time.perf_counter()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n,
        backend_config={
            "client_resources": {"num_cpus": num_cpus, "num_gpus": 0.0},
            "init_args": {"num_cpus": total_cpus, "include_dashboard": False},
        },
    )
    return {
        "aggregates": aggregates,
        "round_seconds": marks,
        "survivors": survivors,
        "wall_seconds": time.perf_counter() - started,
    }
