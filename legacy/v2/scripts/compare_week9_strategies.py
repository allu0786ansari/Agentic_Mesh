from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _losses(summary: dict[str, Any]) -> list[float]:
    return [float(item[1]) for item in summary.get("evaluation_loss", [])]


def _strategy(summary: dict[str, Any], fallback: str) -> str:
    return str(summary.get("strategy", fallback)).lower()


def compare(prox_summary: dict[str, Any], avg_summary: dict[str, Any]) -> dict[str, Any]:
    prox_losses = _losses(prox_summary)
    avg_losses = _losses(avg_summary)
    return {
        "strategy_runs": {
            "fedprox": prox_summary.get("run_id"),
            "fedavg": avg_summary.get("run_id"),
        },
        "fedprox": {
            "strategy": _strategy(prox_summary, "fedprox"),
            "rounds_completed": prox_summary.get("rounds_completed"),
            "best_global_loss": min(prox_losses) if prox_losses else None,
            "final_global_loss": prox_losses[-1] if prox_losses else None,
            "failed_clients": prox_summary.get("fit_metrics", {}).get("failed_clients", []),
        },
        "fedavg": {
            "strategy": _strategy(avg_summary, "fedavg"),
            "rounds_completed": avg_summary.get("rounds_completed"),
            "best_global_loss": min(avg_losses) if avg_losses else None,
            "final_global_loss": avg_losses[-1] if avg_losses else None,
            "failed_clients": avg_summary.get("fit_metrics", {}).get("failed_clients", []),
        },
        "delta": {
            "final_loss_fedprox_minus_fedavg": round(prox_losses[-1] - avg_losses[-1], 12) if prox_losses and avg_losses else None,
            "best_loss_fedprox_minus_fedavg": round(min(prox_losses) - min(avg_losses), 12) if prox_losses and avg_losses else None,
            "fedprox_final_loss_better": bool(prox_losses and avg_losses and prox_losses[-1] < avg_losses[-1]),
            "fedprox_best_loss_better": bool(prox_losses and avg_losses and min(prox_losses) < min(avg_losses)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved Week 9 FedProx and FedAvg runs")
    parser.add_argument("--fedprox", type=Path, required=True)
    parser.add_argument("--fedavg", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/week9_comparison/comparison.json"))
    args = parser.parse_args()
    result = compare(_load(args.fedprox), _load(args.fedavg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
