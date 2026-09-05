from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a persisted federated model bundle in MLflow")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--name", default="GlobalAnomalyDetector")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="agentic-mesh-week9-models")
    args = parser.parse_args()
    if not args.summary.exists():
        raise FileNotFoundError(args.summary)
    model_path = args.summary.parent / "global_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=f"register-{summary.get('strategy', 'federated')}-{summary.get('run_id', 'unknown')}") as run:
        mlflow.log_params(
            {
                "source_run_id": summary.get("run_id", "unknown"),
                "strategy": summary.get("strategy", "unknown"),
                "clients": summary.get("clients", 0),
                "rounds_completed": summary.get("rounds_completed", 0),
                "registration_type": "global_parameter_bundle",
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(args.summary), artifact_path="metadata")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, args.name)
    print(json.dumps({"model_name": args.name, "version": registered.version, "run_id": run.info.run_id}, indent=2))


if __name__ == "__main__":
    main()
