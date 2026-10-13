from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def plot(prox_summary: dict[str, Any], avg_summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    prox = [(int(round_number), float(loss)) for round_number, loss in prox_summary.get("evaluation_loss", [])]
    avg = [(int(round_number), float(loss)) for round_number, loss in avg_summary.get("evaluation_loss", [])]
    if not prox or not avg:
        raise ValueError("Both summaries must contain evaluation_loss history")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([item[0] for item in prox], [item[1] for item in prox], marker="o", label="FedProx")
    axes[0].plot([item[0] for item in avg], [item[1] for item in avg], marker="o", label="FedAvg")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Distributed evaluation loss")
    axes[0].set_title("Convergence comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].bar(["FedProx", "FedAvg"], [min(loss for _, loss in prox), min(loss for _, loss in avg)])
    axes[1].set_ylabel("Best distributed evaluation loss")
    axes[1].set_title("Best observed loss")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Week 9 FedAvg/FedProx convergence")
    parser.add_argument("--fedprox", type=Path, required=True)
    parser.add_argument("--fedavg", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/week9_comparison/convergence.png"))
    args = parser.parse_args()
    plot(_load(args.fedprox), _load(args.fedavg), args.output)
    print(json.dumps({"plot": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
