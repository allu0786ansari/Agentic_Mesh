from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.metrics import log_event
from edge.bus import AsyncBus
from federated.client import NumPyFlowerClient
from federated.convergence import ConvergenceChecker
from federated.mlflow_callback import MLflowCallback
from federated.server import ConvergenceServer
from federated.strategy import ProjectFedAvgStrategy, ProjectFedProxStrategy
from storage.sqlite_schema import record_node_round, seed_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 7 Flower FedProx simulation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--stub", action="store_true", help="run the legacy scaffold smoke path")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--nodes", nargs="+", default=None)
    parser.add_argument("--proximal-mu", type=float, default=0.1)
    parser.add_argument("--strategy", choices=("fedprox", "fedavg"), default="fedprox")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="agentic-mesh-week7")
    parser.add_argument("--run-name", default="week7-fedprox")
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/metrics.jsonl"))
    parser.add_argument("--results-root", type=Path, default=Path("results/week7_federated"))
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--inject-attack", action="store_true")
    parser.add_argument("--alert-threshold", type=float, default=0.2)
    parser.add_argument("--database", type=Path, default=Path("agmesh.db"))
    return parser


def _node_ids(args: argparse.Namespace) -> list[str]:
    nodes = args.nodes or [f"node_{index:02d}" for index in range(1, args.clients + 1)]
    if len(nodes) != args.clients:
        raise ValueError("--nodes count must match --clients")
    return nodes

def _round_logger(callback: MLflowCallback, metrics_path: Path):
    def log_round(round_number: int, metrics: dict[str, float]) -> None:
        callback.log_round(round_number, metrics)

        log_event("fedprox_round", {"round": round_number, **metrics}, metrics_path)
        print(json.dumps({"round": round_number, **metrics}, sort_keys=True))

    return log_round


def _weighted_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {}
    keys = {key for _, values in metrics for key, value in values.items() if isinstance(value, (int, float))}
    return {
        key: sum(num_examples * float(values[key]) for num_examples, values in metrics if key in values) / total_examples
        for key in keys
    }

def _publish_global_alert(round_number: int, loss: float, *, inject_attack: bool, threshold: float, node_id: str) -> dict[str, Any] | None:
    if not inject_attack and loss <= threshold:
        return None
    alert = {
        "session_id": f"fl-round-{round_number}",
        "node_id": node_id,
        "score": max(float(loss), threshold),
        "global_loss": float(loss),
        "round": round_number,
        "source": "federated_evaluation",
        "embedding": [0.0] * 128,
    }
    asyncio.run(AsyncBus().publish("alerts.global", alert))
    return alert


def _run_flower(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("ray") is None:
        raise RuntimeError(
            "Flower simulation requires Ray. Install dependencies with "
            "python -m pip install -r requirements.txt, then rerun."
        )

    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig
    from flwr.server.client_manager import SimpleClientManager
    from flwr.simulation import start_simulation

    nodes = _node_ids(args)
    if not all((args.partitions_root / node / "train.parquet").exists() for node in nodes):
        raise FileNotFoundError("Every selected node must have data/partitions/<node>/train.parquet")
    seed_registry(args.database, [args.partitions_root / node / "meta.json" for node in nodes])

    def node_health(round_number: int, results: list[Any], failures: list[Any]) -> None:
        participating = {str(proxy.cid) for proxy, _ in results}
        failed = {
            str(item[0].cid)
            for item in failures
            if isinstance(item, tuple) and len(item) > 0 and hasattr(item[0], "cid")
        }
        for index, node_id in enumerate(nodes):
            if str(index) in participating:
                record_node_round(args.database, node_id, round_number, True)
            elif str(index) in failed:
                record_node_round(args.database, node_id, round_number, False)

    def make_client(client_id: str) -> Any:
        index = int(client_id)
        node_id = nodes[index]
        return NumPyFlowerClient(
            client_id=node_id,
            seed=args.seed + index,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            dataset_path=args.partitions_root / node_id / "train.parquet",
            window_size=args.window_size,
            stride=args.stride,
            max_windows=args.max_windows,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
        ).to_client()

    first_client = NumPyFlowerClient(
        client_id=nodes[0],
        seed=args.seed,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        dataset_path=args.partitions_root / nodes[0] / "train.parquet",
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    round_metrics_path = run_dir / "round_metrics.jsonl"
    callback = MLflowCallback(args.tracking_uri, args.experiment)
    strategy_class = ProjectFedProxStrategy if args.strategy == "fedprox" else ProjectFedAvgStrategy
    strategy_kwargs: dict[str, Any] = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": args.clients,
        "min_evaluate_clients": args.clients,
        "min_available_clients": args.clients,
        "initial_parameters": ndarrays_to_parameters(list(first_client.get_parameters())),
        "fit_metrics_aggregation_fn": _weighted_metrics,
        "evaluate_metrics_aggregation_fn": _weighted_metrics,
        "convergence_checker": ConvergenceChecker(args.patience, args.min_delta),
        "node_health_callback": node_health,
        "round_logger": _round_logger(callback, round_metrics_path),
        "on_fit_config_fn": lambda _round: {"local_epochs": args.local_epochs, "lr": args.learning_rate, "proximal_mu": args.proximal_mu if args.strategy == "fedprox" else 0.0},
    }
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
    strategy = strategy_class(
        **strategy_kwargs,
    )
    with callback.run(args.run_name):
        callback.log_parameters(
            {
                "flower_version": fl.__version__,
                "clients": args.clients,
                "nodes": ",".join(nodes),
                "rounds": args.rounds,
                "proximal_mu": args.proximal_mu,
                "strategy": args.strategy,
                "local_epochs": args.local_epochs,
                "max_windows": args.max_windows,
                "secure_aggregation_active": False,
            }
        )
        history = start_simulation(
            client_fn=make_client,
            num_clients=args.clients,
            client_resources={"num_cpus": 1, "num_gpus": 0},
            config=ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout),
            server=ConvergenceServer(client_manager=SimpleClientManager(), strategy=strategy),
            ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
        )
    emitted_alerts: list[dict[str, Any]] = []
    for round_number, loss in history.losses_distributed:
        alert = _publish_global_alert(
            int(round_number),
            float(loss),
            inject_attack=args.inject_attack,
            threshold=args.alert_threshold,
            node_id=nodes[0],
        )
        if alert is not None:
            emitted_alerts.append(alert)
    if strategy.final_parameters is None:
        raise RuntimeError("Flower completed without producing global parameters")
    np.savez_compressed(
        run_dir / "global_model.npz",
        **{f"parameter_{index}": value for index, value in enumerate(strategy.final_parameters)},
    )
    summary = {
        "run_id": run_id,
        "flower_version": fl.__version__,
        "nodes": nodes,
        "clients": args.clients,
        "rounds_requested": args.rounds,
        "rounds_completed": len(history.losses_distributed),
        "round_timeout_seconds": args.round_timeout,
        "proximal_mu": args.proximal_mu,
        "strategy": args.strategy,
        "local_epochs": args.local_epochs,
        "max_windows": args.max_windows,
        "secure_aggregation_active": False,
        "fit_metrics": {
            key: [(int(round_number), float(value)) for round_number, value in values]
            for key, values in history.metrics_distributed_fit.items()
        },
        "evaluation_loss": [(int(round_number), float(loss)) for round_number, loss in history.losses_distributed],
        "convergence": strategy.convergence_checker.summary() if strategy.convergence_checker else None,
        "emitted_alerts": emitted_alerts,
        "artifacts": {
            "round_metrics": str(round_metrics_path),
            "global_model": str(run_dir / "global_model.npz"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "result_dir": str(run_dir), "summary": str(run_dir / "summary.json")}))


def main() -> None:
    args = _build_parser().parse_args()
    if args.stub:
        print("Week 1 scaffold is operational")
        return
    if args.rounds < 1 or args.clients < 1:
        raise ValueError("--rounds and --clients must be positive")
    _run_flower(args)


if __name__ == "__main__":
    main()
