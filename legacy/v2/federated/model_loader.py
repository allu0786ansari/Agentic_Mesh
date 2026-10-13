from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class GlobalModelLoader:
    """Load a persisted federated global model by run/version metadata."""

    def __init__(self, results_root: str | Path = "results") -> None:
        self.results_root = Path(results_root)

    def load(self, run_id: str) -> tuple[list[np.ndarray], dict[str, Any]]:
        run_dir = self._resolve_run(run_id)
        summary_path = run_dir / "summary.json"
        model_path = run_dir / "global_model.npz"
        if not summary_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Incomplete global model run: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        arrays = np.load(model_path)
        parameters = [arrays[name].copy() for name in sorted(arrays.files, key=lambda name: int(name.split("_")[-1]))]
        return parameters, summary

    def load_latest(self) -> tuple[list[np.ndarray], dict[str, Any]]:
        candidates = [path for path in self.results_root.rglob("summary.json") if (path.parent / "global_model.npz").exists()]
        if not candidates:
            raise FileNotFoundError(f"No persisted global model found under {self.results_root}")
        return self.load(max(candidates, key=lambda path: path.parent.name).parent.name)

    def _resolve_run(self, run_id: str) -> Path:
        direct = self.results_root / run_id
        if (direct / "summary.json").exists():
            return direct
        matches = list(self.results_root.rglob(run_id))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Unknown global model run: {run_id}")


__all__ = ["GlobalModelLoader"]
