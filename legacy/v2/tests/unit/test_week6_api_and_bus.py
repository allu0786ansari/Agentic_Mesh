import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from api.node_api import create_app
from edge.bus import AsyncBus, EmbeddingPublisher


def test_node_api_health_store_and_search(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "qdrant", "week6_test"))
    assert client.get("/health").json()["status"] == "ok"
    vector = [1.0] + [0.0] * 127
    response = client.post(
        "/embeddings",
        json={
            "embedding": vector,
            "payload": {
                "run_id": "test",
                "node_id": "node_04",
                "model_type": "isolation_forest",
                "score": 1.0,
                "threshold": 0.5,
                "index": 0,
            },
            "point_id": "test-point",
        },
    )
    assert response.status_code == 200
    matches = client.post("/embeddings/search", json={"embedding": vector, "limit": 1})
    assert matches.status_code == 200
    assert matches.json()[0]["payload"]["node_id"] == "node_04"


def test_embedding_publisher_delivers_topic_message() -> None:
    async def scenario() -> dict:
        topic = "embeddings.week6-test"
        await AsyncBus().drain(topic)
        await EmbeddingPublisher().publish("week6-test", {"score": 1.0})
        return await AsyncBus().get(topic, timeout=1)

    assert asyncio.run(scenario())["score"] == 1.0
