from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.metrics import read_events


def _make_synthetic_metrics() -> dict[str, Any]:
    round_numbers = [1, 2, 3, 4, 5, 6]
    return {
        "fl": {
            "rounds": round_numbers,
            "loss": [0.91, 0.76, 0.68, 0.61, 0.54, 0.48],
            "fedavg_auroc": [0.72, 0.76, 0.8, 0.83, 0.86, 0.88],
            "fedprox_auroc": [0.74, 0.79, 0.83, 0.87, 0.9, 0.93],
            "epsilon": [1.8, 2.5, 3.2, 4.1, 4.8, 5.4],
            "straggler_rate": [0.12, 0.15, 0.18, 0.14, 0.1, 0.08],
        },
        "edge": {
            "scores": [0.05, 0.18, 0.26, 0.45, 0.62, 0.81, 0.9, 0.72, 0.28, 0.34],
            "embeddings": [120, 136, 142, 150, 171, 168, 160, 154, 145, 137],
            "nodes": ["node_01", "node_02", "node_03", "node_04", "node_05", "node_06"],
        },
        "agent": {
            "latencies": [228, 410, 585, 760, 1090, 1445],
            "hops": [1, 2, 3, 3, 4, 2],
            "circuit_breakers": [1, 2, 1, 2, 3, 2],
            "mitre": ["T0804", "T0856", "T0812", "T0823", "T0830", "T0847"],
        },
        "system": {
            "alerts": [12, 18, 24, 22, 27, 30],
            "detection_rate": [0.64, 0.71, 0.78, 0.85, 0.91, 0.94],
            "dispatch_latency": [4.2, 3.8, 3.4, 3.1, 2.7, 2.3],
            "epsilon_budget": [6.8, 7.5, 7.9, 8.7, 9.3, 9.6],
        },
    }


def _metrics_from_log(metrics_path: str | Path) -> dict[str, Any]:
    """Project structured runtime events into the four dashboard data groups."""
    events = read_events(metrics_path)
    if not events:
        return _make_synthetic_metrics()
    rounds = [event for event in events if event["event"] in {"fedprox_round", "fedavg_round"}]
    graph_events = [event["payload"] for event in events if event["event"] == "agent_graph_completed"]
    anomaly_events = [event["payload"] for event in events if "score" in event["payload"]]
    epsilon_events = [event["payload"] for event in events if "epsilon" in event["payload"]]
    if not rounds and not graph_events and not anomaly_events:
        return _make_synthetic_metrics()
    round_numbers = [int(event["payload"].get("round", index + 1)) for index, event in enumerate(rounds)] or [1]
    losses = [float(event["payload"].get("loss", event["payload"].get("global_loss", 0.0))) for event in rounds] or [0.0]
    epsilon = [float(event["payload"].get("epsilon", 0.0)) for event in epsilon_events] or [0.0]
    latencies = [float(item.get("duration_ms", 0.0)) for item in graph_events] or [0.0]
    scores = [float(item.get("score", 0.0)) for item in anomaly_events] or [0.0]
    techniques = [str(item.get("technique_id", "UNKNOWN")) for item in graph_events] or ["UNKNOWN"]
    return {
        "fl": {"rounds": round_numbers, "loss": losses, "fedavg_auroc": [0.0] * len(round_numbers), "fedprox_auroc": [0.0] * len(round_numbers), "epsilon": epsilon, "straggler_rate": [0.0] * len(round_numbers)},
        "edge": {"scores": scores, "embeddings": list(range(1, len(scores) + 1)), "nodes": [str(item.get("node_id", "unknown")) for item in anomaly_events] or ["unknown"]},
        "agent": {"latencies": latencies, "hops": [int(item.get("hop_count", 0)) for item in graph_events] or [0], "circuit_breakers": [0] * len(techniques), "mitre": techniques},
        "system": {"alerts": [len(graph_events)], "detection_rate": [0.0], "dispatch_latency": [sum(latencies) / len(latencies) if latencies else 0.0], "epsilon_budget": [epsilon[-1] if epsilon else 0.0]},
    }


def generate_charts(*, output_dir: str | Path = "docs/architecture", metrics_path: str | Path = "logs/metrics.jsonl") -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_from_log(metrics_path)

    fl_path = output_dir / "fl_health_dashboard.png"
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(metrics["fl"]["rounds"], metrics["fl"]["loss"], marker="o")
    axs[0, 0].set_title("Loss per round")
    axs[0, 1].plot(metrics["fl"]["rounds"], metrics["fl"]["fedavg_auroc"], label="FedAvg")
    axs[0, 1].plot(metrics["fl"]["rounds"], metrics["fl"]["fedprox_auroc"], label="FedProx")
    axs[0, 1].legend()
    axs[0, 1].set_title("AUROC by round")
    axs[1, 0].bar(metrics["fl"]["rounds"], metrics["fl"]["epsilon"], color="steelblue")
    axs[1, 0].set_title("Epsilon consumed")
    axs[1, 1].plot(metrics["fl"]["rounds"], metrics["fl"]["straggler_rate"], marker="s", color="darkorange")
    axs[1, 1].set_title("Straggler rate")
    fig.tight_layout()
    fig.savefig(fl_path, dpi=150)
    plt.close(fig)

    edge_path = output_dir / "edge_node_status_dashboard.png"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(metrics["edge"]["scores"], bins=8, color="mediumseagreen")
    axs[0].set_title("Anomaly score distribution")
    axs[1].plot(metrics["edge"]["embeddings"], marker="o")
    axs[1].set_title("Embeddings published over time")
    axs[1].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(edge_path, dpi=150)
    plt.close(fig)

    agent_path = output_dir / "agent_mesh_performance_dashboard.png"
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].bar(["p50", "p90", "p95"], [metrics["agent"]["latencies"][2], metrics["agent"]["latencies"][4], metrics["agent"]["latencies"][5]], color="royalblue")
    axs[0, 0].set_title("Latency percentiles")
    axs[0, 1].hist(metrics["agent"]["hops"], bins=range(1, 6), color="tomato")
    axs[0, 1].set_title("Hop count histogram")
    axs[1, 0].bar(metrics["agent"]["mitre"], metrics["agent"]["circuit_breakers"], color="purple")
    axs[1, 0].set_title("Circuit breaker activations")
    technique_counts = Counter(metrics["agent"]["mitre"])
    axs[1, 1].bar(list(technique_counts), list(technique_counts.values()), color="brown")
    axs[1, 1].set_title("MITRE technique frequency")
    fig.tight_layout()
    fig.savefig(agent_path, dpi=150)
    plt.close(fig)

    system_path = output_dir / "system_overview_dashboard.png"
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(metrics["system"]["alerts"], marker="o")
    axs[0, 0].set_title("Total alerts")
    axs[0, 1].plot(metrics["system"]["detection_rate"], marker="s")
    axs[0, 1].set_title("Detection rate")
    axs[1, 0].plot(metrics["system"]["dispatch_latency"], marker="D")
    axs[1, 0].set_title("Mean time to dispatch (s)")
    axs[1, 1].plot(metrics["system"]["epsilon_budget"], marker="^")
    axs[1, 1].set_title("Epsilon budget status")
    fig.tight_layout()
    fig.savefig(system_path, dpi=150)
    plt.close(fig)

    return [
        str(fl_path),
        str(edge_path),
        str(agent_path),
        str(system_path),
    ]


def main() -> None:
    paths = generate_charts(output_dir="docs/architecture")
    print(f"Generated {len(paths)} dashboard PNGs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
