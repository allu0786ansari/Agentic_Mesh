import json
from pathlib import Path

from edge.config import Config


def test_config_loads_persistent_training_settings_from_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LOCAL_EPOCHS=7\nCHECKPOINT_ROOT=./saved-checkpoints\nRUN_MANIFEST_PATH=./results/run.json\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_EPOCHS", raising=False)
    monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("RUN_MANIFEST_PATH", raising=False)

    config = Config.from_env()

    assert config.local_epochs == 7
    assert config.checkpoint_root == "./saved-checkpoints"
    assert config.run_manifest_path == "./results/run.json"


def test_manifest_shape_is_json_serializable(tmp_path: Path) -> None:
    manifest = [{"node_id": "node_01", "epochs": 10, "losses": [0.5, 0.4], "epsilon": 2.0}]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))[0]["epochs"] == 10
