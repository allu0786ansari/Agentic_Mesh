from pathlib import Path

import mlflow
from qdrant_client import QdrantClient

from agents.metrics import log_event, read_events
from agents.tracing import configure_phoenix, get_tracer
from scripts.init_qdrant import COLLECTIONS, initialise
from storage.sqlite_schema import initialise as initialise_sqlite


def test_embedded_qdrant_collections_are_created(tmp_path: Path) -> None:
    names = initialise(tmp_path / "qdrant")
    assert set(COLLECTIONS).issubset(names)

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    listed = {item.name: item for item in client.get_collections().collections}
    assert listed["insight_embeddings"] is not None
    assert client.get_collection("insight_embeddings").config.params.vectors.size == 128


def test_sqlite_schema_is_reachable(tmp_path: Path) -> None:
    database_path = tmp_path / "agmesh.db"
    initialise_sqlite(database_path)
    import sqlite3

    with sqlite3.connect(database_path) as database:
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"node_registry", "alert_records", "fl_rounds"}.issubset(tables)


def test_mlflow_logs_round_metric(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("week3-test")
    with mlflow.start_run(run_name="infra-test") as run:
        mlflow.log_metric("global_loss", 0.25, step=1)
        run_id = run.info.run_id
    metric = mlflow.MlflowClient().get_metric_history(run_id, "global_loss")
    assert len(metric) == 1
    assert metric[0].value == 0.25


def test_structured_metrics_and_local_trace(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    log_event("week3.test", {"round": 1}, path)
    events = read_events(path)
    assert events[0]["event"] == "week3.test"
    configure_phoenix("http://127.0.0.1:6006/v1/traces")
    with get_tracer("week3-test").span("infra-span", component="test") as span:
        span.set_attribute("verified", True)
