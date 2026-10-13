from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edge.models.isolation_forest import IsolationForestDetector


class ModelSelector:
    """Route node metadata to the correct local detector implementation."""

    def __init__(self, metadata_root: str | Path):
        self.metadata_root = Path(metadata_root)

    def metadata(self, node_id: str) -> dict[str, Any]:
        path = self.metadata_root / node_id / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing node metadata: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def model_type(self, node_id: str) -> str:
        model_type = self.metadata(node_id).get("model_type")
        if model_type not in {"vae", "isolation_forest"}:
            raise ValueError(f"Unsupported model type for {node_id}: {model_type}")
        return model_type

    def load(self, node_id: str, checkpoint_root: str | Path) -> Any:
        model_type = self.model_type(node_id)
        if model_type == "isolation_forest":
            detector, _ = IsolationForestDetector.load(Path(checkpoint_root) / f"{node_id}.joblib")
            return detector
        raise NotImplementedError("VAE loading is owned by evaluation.inference.load_checkpoint_model")
