"""The frozen global model and each node's local calibration (Proposal §9.3 T6).

**Global model** (released by the federation, the same on every node of a site):
architecture, weights and the pooled normalisation of round 0. Its
``model_version`` is the SHA-256 of all three, so an evidence record names exactly
which model produced it.

**Node calibration** (fitted by each node on its own validation data; never leaves
the node). It holds:

* the training-error references the scoring needs (ADR 0003 ``ErrorReference``;
  ADR 0005 ``ShiftReference`` median);
* the window threshold at the target false-positive rate on normal validation
  windows (Proposal §6.5);
* the isotonic calibrator from score to probability, fitted on all validation
  windows (normal and attack);
* the magnitude buckets of every feature: p90, p99 and p99.9 of its error on normal
  validation windows (Proposal §6.6);
* Site B only: the p1 and p99 of the request count, for the request-rate bucket.

This is the same per-node fitting the detection results use (``x1._evaluate``). The
edge runtime only adds the parts the evidence needs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.common.provenance import sha256_bytes, utc_now_iso
from agmesh.federated.protocol import ModelSpec, Normalisation, Weights

F64 = npt.NDArray[np.float64]
BUCKET_QUANTILES = (0.90, 0.99, 0.999)


def model_version(spec: ModelSpec, weights: Weights, norm: Normalisation) -> str:
    parts = [json.dumps(asdict(spec), sort_keys=True).encode()]
    parts += [np.ascontiguousarray(w, dtype=np.float32).tobytes() for w in weights]
    parts += [
        np.ascontiguousarray(a, dtype=np.float64).tobytes() for a in (norm.centre, norm.scale)
    ]
    return "sha256:" + sha256_bytes(b"".join(parts))


@dataclass(frozen=True)
class GlobalModel:
    site: str
    spec: ModelSpec
    weights: Weights
    norm: Normalisation
    source: dict[str, Any]  # run id, config, seed, ε, result artifact

    @property
    def version(self) -> str:
        return model_version(self.spec, self.weights, self.norm)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            f"w{i:03d}": np.asarray(w, np.float32) for i, w in enumerate(self.weights)
        }
        arrays["centre"], arrays["scale"] = self.norm.centre, self.norm.scale
        np.savez(directory / "global.npz", **arrays)
        meta = {
            "site": self.site,
            "spec": asdict(self.spec),
            "model_version": self.version,
            "n_weight_arrays": len(self.weights),
            "source": self.source,
            "saved_at": utc_now_iso(),
        }
        path = directory / "model.json"
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> GlobalModel:
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "global.npz") as z:
            weights = [z[f"w{i:03d}"] for i in range(int(meta["n_weight_arrays"]))]
            norm = Normalisation(z["centre"], z["scale"])
        spec = ModelSpec(
            meta["spec"]["kind"],
            int(meta["spec"]["n_inputs"]),
            tuple(meta["spec"]["hidden"]),
            int(meta["spec"]["latent"]),
        )
        model = cls(meta["site"], spec, weights, norm, meta["source"])
        if model.version != meta["model_version"]:
            raise ValueError(
                f"{directory}: weights do not match model_version {meta['model_version']}"
            )
        return model

    def network(self) -> Any:
        from agmesh.federated.client import set_weights
        from agmesh.models import nets

        s = self.spec
        net = (
            nets.build("vae", s.n_inputs, list(s.hidden), s.latent)
            if s.kind == "vae"
            else nets.build("ae", s.n_inputs, list(s.hidden))
        )
        set_weights(net, self.weights)
        net.eval()
        return net


@dataclass(frozen=True)
class NodeCalibration:
    """What one node fits locally on its own training and validation data."""

    node: str
    model_version: str
    features: list[str]  # base features, in model order
    threshold: float
    threshold_val_fpr: float
    calibrator: dict[str, list[float]]  # IsotonicCalibrator.to_dict()
    error_median: list[float] | None  # ErrorReference (Site B norm_top3)
    error_iqr: list[float] | None
    score_median: float  # ShiftReference median (R3 MAD floor)
    buckets: dict[str, list[float]]  # feature -> [p90, p99, p99.9] of normal-val error
    request_rate: list[float] | None  # Site B: [p1, p99] of req_recv on normal validation
    constants: dict[str, float]  # Site A: training-constant features and their value
    shift: str
    fitted_at: str

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> NodeCalibration:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def buckets_from(
    normal_feature_errors: npt.ArrayLike, features: list[str]
) -> dict[str, list[float]]:
    e = np.asarray(normal_feature_errors, dtype=np.float64)
    q = np.quantile(e, BUCKET_QUANTILES, axis=0)  # (3, features)
    return {name: [float(v) for v in q[:, j]] for j, name in enumerate(features)}
