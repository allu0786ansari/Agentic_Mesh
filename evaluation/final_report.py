from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_final_report(
    *,
    evaluation_dir: str | Path = "evaluation",
    output_path: str | Path = "evaluation/final_report_data.json",
) -> dict[str, Any]:
    """Consolidate measured Week 14/15 evidence without inventing absent metrics."""
    root = Path(evaluation_dir)
    week14 = _load(root / "results.json")
    week15_ragas = _load(root / "ragas_results.json")
    week15_chaos = _load(root / "chaos_results.json")
    summary = week14.get("summary", {})
    retrieval = {
        key: week15_ragas[key]
        for key in ("n_queries", "mean_context_recall", "mean_context_precision")
        if key in week15_ragas
    }
    report = {
        "project": "Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics",
        "generated_by": "evaluation.final_report",
        "weeks": {
            "week14": {
                "detection": {
                    key: summary[key]
                    for key in ("detection_rate", "false_positive_rate", "root_cause_accuracy")
                    if key in summary
                },
                "latency_ms": {
                    key: summary[key]
                    for key in ("mean_latency_ms", "p95_latency_ms")
                    if key in summary
                },
                "windows": {
                    key: summary[key]
                    for key in ("total_windows", "attack_windows", "normal_windows")
                    if key in summary
                },
                "source": str(root / "results.json"),
            },
            "week15": {
                "chaos": {
                    "passed": week15_chaos.get("passed"),
                    "scenarios": week15_chaos.get("scenarios", {}),
                    "source": str(root / "chaos_results.json"),
                },
                "retrieval": {**retrieval, "source": str(root / "ragas_results.json")},
            },
        },
        "federated_learning": {"status": "not_measured", "reason": "No final FL comparison artifact was present."},
        "privacy": {"status": "not_measured", "reason": "Per-node cumulative epsilon evidence was not present in the Week 14 report."},
        "traceability": {
            "jwt": "implemented_in_agents.auth",
            "mtls": "implemented_in_agents.tls",
            "local_llm": "implemented_in_agents.llm",
            "phoenix": "implemented_in_agents.tracing",
            "reproducibility": "dvc.yaml and MLflow callback are present",
        },
        "limitations": [
            "Week 14 source data lacks stable scenario-level MITRE ground-truth IDs.",
            "Final FL AUROC/F1 and FedProx-vs-FedAvg comparison were not available as machine-readable artifacts.",
            "Runtime claims requiring Ollama, Phoenix, or a full DVC run require environment-backed validation.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_final_report(), indent=2))
