"""Week 1 feasibility spike: Flower simulation memory and round time (Gate G0).

Flower's simulation runtime executes each client in a Ray actor, which is a separate
Python process that loads its own copy of torch. This benchmark runs a small
federation of dummy clients whose model has the same shape as the Site A VAE
(window 60 x features) and records:

* peak memory of the whole process tree (RSS and, on Linux, PSS);
* the change in system memory used;
* wall time per federated round;
* how many clients Ray runs concurrently for the chosen ``client_resources``.

Gate G0 uses these numbers to fix ``federated.client_resources`` in
``configs/params.yaml`` and to estimate how long the Week 12 experiment matrix takes.

Requires the ``fl`` dependency group: ``uv sync --group fl``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.common.memory import MemorySampler, process_tree

SETTINGS_ENV = "AGMESH_RAY_BENCH_SETTINGS"


@dataclass(frozen=True)
class BenchSettings:
    clients: int
    rounds: int
    num_cpus_per_client: float
    total_cpus: int
    backend: str  # "torch" or "numpy"
    input_dim: int
    hidden: tuple[int, ...]
    latent: int
    samples_per_client: int
    local_steps: int
    batch_size: int
    lr: float
    strategy: str  # "fedavg" or "fedprox"
    proximal_mu: float
    object_store_mb: int
    seed: int
    server_record_path: str

    def to_env(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_env(cls) -> BenchSettings:
        raw = json.loads(os.environ[SETTINGS_ENV])
        raw["hidden"] = tuple(raw["hidden"])
        return cls(**raw)


def layer_shapes(settings: BenchSettings) -> list[tuple[int, int]]:
    """Encoder and mirrored decoder of a dense autoencoder, as (fan_in, fan_out)."""
    dims = [settings.input_dim, *settings.hidden, settings.latent]
    encoder = list(itertools.pairwise(dims))
    decoder = [(b, a) for a, b in reversed(encoder)]
    return encoder + decoder


def initial_arrays(settings: BenchSettings) -> list[np.ndarray]:
    rng = np.random.default_rng(settings.seed)
    arrays: list[np.ndarray] = []
    for fan_in, fan_out in layer_shapes(settings):
        bound = 1.0 / math.sqrt(fan_in)
        arrays.append(rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float32))
        arrays.append(np.zeros(fan_out, dtype=np.float32))
    return arrays


def client_data(settings: BenchSettings, partition_id: int) -> np.ndarray:
    rng = np.random.default_rng(settings.seed + 1000 + partition_id)
    shift = 0.1 * partition_id  # mild per-client shift, standing in for non-IID data
    shape = (settings.samples_per_client, settings.input_dim)
    return (rng.standard_normal(shape) + shift).astype(np.float32)


def _train_torch(
    weights: list[np.ndarray], data: np.ndarray, settings: BenchSettings, mu: float
) -> tuple[list[np.ndarray], float]:
    import torch  # imported inside the Ray worker

    torch.set_num_threads(max(1, int(settings.num_cpus_per_client)))
    torch.manual_seed(settings.seed)
    layers: list[torch.nn.Module] = []
    shapes = layer_shapes(settings)
    for i, (fan_in, fan_out) in enumerate(shapes):
        layers.append(torch.nn.Linear(fan_in, fan_out))
        if i < len(shapes) - 1:
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers)
    params = list(model.parameters())
    with torch.no_grad():
        for param, value in zip(params, weights, strict=True):
            param.copy_(torch.from_numpy(value))
    global_params = [p.detach().clone() for p in params]
    optimiser = torch.optim.Adam(params, lr=settings.lr)
    tensor = torch.from_numpy(data)
    generator = torch.Generator().manual_seed(settings.seed)
    losses: list[float] = []
    for _ in range(settings.local_steps):
        idx = torch.randint(0, tensor.shape[0], (settings.batch_size,), generator=generator)
        batch = tensor[idx]
        loss = torch.nn.functional.mse_loss(model(batch), batch)
        if mu > 0:
            proximal = sum((p - g).pow(2).sum() for p, g in zip(params, global_params, strict=True))
            loss = loss + 0.5 * mu * proximal
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    return [p.detach().numpy().copy() for p in params], float(np.mean(losses))


def _train_numpy(
    weights: list[np.ndarray], data: np.ndarray, settings: BenchSettings
) -> tuple[list[np.ndarray], float]:
    """CPU-bound stand-in used when torch is unavailable: forward passes plus a small update."""
    rng = np.random.default_rng(settings.seed)
    losses: list[float] = []
    for _ in range(settings.local_steps):
        batch = data[rng.integers(0, data.shape[0], settings.batch_size)]
        activation = batch
        for i in range(0, len(weights), 2):
            activation = activation @ weights[i].T + weights[i + 1]
            if i < len(weights) - 2:
                activation = np.maximum(activation, 0.0)
        losses.append(float(np.mean((activation - batch) ** 2)))
    updated = [w - settings.lr * 0.01 * np.sign(w) for w in weights]
    return updated, float(np.mean(losses))


client_app = ClientApp()


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    settings = BenchSettings.from_env()
    partition_id = int(context.node_config["partition-id"])
    config = msg.content.config_records["config"]
    mu_value = config.get("proximal-mu", 0.0)
    mu = float(mu_value) if isinstance(mu_value, int | float) else 0.0
    weights = msg.content.array_records["arrays"].to_numpy_ndarrays()
    data = client_data(settings, partition_id)
    started = time.perf_counter()
    if settings.backend == "torch":
        updated, loss = _train_torch(weights, data, settings, mu)
    else:
        updated, loss = _train_numpy(weights, data, settings)
    metrics = MetricRecord(
        {
            "num-examples": int(data.shape[0]),
            "train-loss": loss,
            "train-time-s": time.perf_counter() - started,
            "worker-rss-mb": psutil.Process().memory_info().rss / 2**20,
        }
    )
    content = RecordDict({"arrays": ArrayRecord(numpy_ndarrays=updated), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # Evaluation is not part of this benchmark; reply with a trivial record so the
    # strategy's evaluate phase completes if it runs.
    metrics = MetricRecord({"num-examples": 1, "eval-loss": 0.0})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


server_app = ServerApp()


@server_app.main()
def server_main(grid: Grid, context: Context) -> None:
    settings = BenchSettings.from_env()
    common: dict[str, Any] = {
        "fraction_train": 1.0,
        "fraction_evaluate": 0.0,
        "min_train_nodes": settings.clients,
        "min_evaluate_nodes": 1,
        "min_available_nodes": settings.clients,
    }
    strategy = (
        FedProx(proximal_mu=settings.proximal_mu, **common)
        if settings.strategy == "fedprox"
        else FedAvg(**common)
    )
    marks: list[float] = []

    def mark_round(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        # Called once before round 1 and once after every round: the gaps are round times.
        marks.append(time.perf_counter())
        return None

    started = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(numpy_ndarrays=initial_arrays(settings)),
        num_rounds=settings.rounds,
        timeout=600,
        train_config=ConfigRecord({}),
        evaluate_fn=mark_round,
    )
    finished = time.perf_counter()
    train_metrics: dict[str, Any] = {}
    raw_metrics = getattr(result, "train_metrics_clientapp", None)
    if isinstance(raw_metrics, dict):
        train_metrics = {str(k): dict(v) for k, v in raw_metrics.items()}
    record = {
        "round_times_s": [round(b - a, 3) for a, b in itertools.pairwise(marks)],
        "server_wall_s": round(finished - started, 3),
        "train_metrics_by_round": train_metrics,
    }
    Path(settings.server_record_path).write_text(json.dumps(record), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_benchmark(settings: BenchSettings, *, sample_interval_s: float = 0.5) -> dict[str, Any]:
    """Run the simulation in this process and return the measurements."""
    os.environ[SETTINGS_ENV] = settings.to_env()
    record_path = Path(settings.server_record_path)
    record_path.unlink(missing_ok=True)
    baseline_used_mb = psutil.virtual_memory().used / 2**20
    backend_config: dict[str, Any] = {
        "client_resources": {"num_cpus": settings.num_cpus_per_client, "num_gpus": 0.0},
        "init_args": {
            "num_cpus": settings.total_cpus,
            "object_store_memory": settings.object_store_mb * 2**20,
            "include_dashboard": False,
        },
    }
    started = time.perf_counter()
    with MemorySampler(process_tree(), interval_s=sample_interval_s) as sampler:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=settings.clients,
            backend_config=backend_config,
        )
    wall = time.perf_counter() - started
    server_record: dict[str, Any] = {}
    if record_path.is_file():
        server_record = json.loads(record_path.read_text(encoding="utf-8"))
    round_times: list[float] = server_record.get("round_times_s", [])
    # Round 1 includes starting the Ray actors, so steady-state time uses later rounds.
    steady = round_times[1:] if len(round_times) > 1 else round_times
    steady_mean = float(np.mean(steady)) if steady else None
    estimate_50 = (
        round_times[0] + steady_mean * 49 if round_times and steady_mean is not None else None
    )
    concurrency = max(1, int(settings.total_cpus // settings.num_cpus_per_client))
    return {
        "settings": asdict(settings),
        "versions": {
            "flwr": _package_version("flwr"),
            "ray": _package_version("ray"),
            "torch": _package_version("torch"),
        },
        "completed": bool(round_times) and len(round_times) == settings.rounds,
        "wall_time_s": round(wall, 2),
        "round_times_s": round_times,
        "first_round_includes_actor_startup": True,
        "steady_round_time_s": round(steady_mean, 3) if steady_mean is not None else None,
        "estimated_50_round_run_s": round(estimate_50, 1) if estimate_50 is not None else None,
        "concurrent_clients": min(concurrency, settings.clients),
        "model_parameters": int(sum(a.size for a in initial_arrays(settings))),
        "memory": sampler.summary(baseline_system_used_mb=round(baseline_used_mb, 1)),
        "server_record": server_record,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower/Ray memory and round-time benchmark")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--num-cpus-per-client",
        type=float,
        default=None,
        help="default: federated.client_resources.num_cpus from params",
    )
    parser.add_argument("--total-cpus", type=int, default=psutil.cpu_count(logical=True) or 2)
    parser.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    parser.add_argument(
        "--features",
        type=int,
        default=79,
        help="HAI 21.03 has 79 features; input size = window length x features",
    )
    parser.add_argument("--samples-per-client", type=int, default=2000)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--object-store-mb", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/feasibility/ray_bench.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params = load_params(args.params)
    if args.backend == "torch" and _package_version("torch") is None:
        print(
            "ERROR: torch is not installed. Run `uv sync --group fl` (or use --backend numpy).",
            file=sys.stderr,
        )
        return 1
    window = params.data.windows.site_a.length_s * params.data.hai.sample_rate_hz
    record_path = args.output.with_suffix(".server.tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = BenchSettings(
        clients=args.clients,
        rounds=args.rounds,
        num_cpus_per_client=args.num_cpus_per_client or params.federated.client_resources.num_cpus,
        total_cpus=args.total_cpus,
        backend=args.backend,
        input_dim=window * args.features,
        hidden=tuple(params.models.vae.hidden),
        latent=params.models.vae.latent,
        samples_per_client=args.samples_per_client,
        local_steps=args.local_steps,
        batch_size=params.models.vae.batch,
        lr=params.models.vae.lr,
        strategy=args.strategy,
        proximal_mu=params.federated.proximal_mu,
        object_store_mb=args.object_store_mb,
        seed=params.project.seeds[0],
        server_record_path=str(record_path.resolve()),
    )
    print(
        f"Running {settings.clients} clients x {settings.rounds} rounds "
        f"({settings.backend}, input {settings.input_dim}, "
        f"{settings.num_cpus_per_client} CPU/client)"
    )
    report = run_benchmark(settings)
    record_path.unlink(missing_ok=True)
    path = write_artifact(
        args.output,
        kind="feasibility.ray_bench",
        data_source="synthetic",
        payload=report,
        params_path=args.params,
    )
    memory = report["memory"]
    print(f"\nWrote {path}")
    print(f"  completed: {report['completed']}, rounds: {report['round_times_s']}")
    print(
        f"  concurrent clients: {report['concurrent_clients']}, "
        f"peak PSS {memory.get('peak_pss_mb')} MB, peak RSS {memory.get('peak_rss_mb')} MB, "
        f"system delta {memory.get('peak_system_delta_mb')} MB"
    )
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
