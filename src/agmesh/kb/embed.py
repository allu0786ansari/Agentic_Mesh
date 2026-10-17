"""Text encoders: the reference sentence-transformer and a hashing encoder for tests.

* :class:`SentenceEncoder` is ``all-MiniLM-L6-v2`` with 384 dimensions and
  L2-normalised output, loaded lazily (dependency group ``kb``). The knowledge base
  and the queries are embedded with the same instance (Proposal §8.4).
  :meth:`identity` reports the Hugging Face snapshot and the weights' SHA-256, which
  ``configs/kb/encoder.lock.json`` pins after the first build (ADR 0010 §1).
* :class:`HashingEncoder` is deterministic, also 384-dimensional, and needs no model
  download. It is for CI and fixture tests only. ``retrieval_eval`` refuses to report
  with it (ADR 0010 §5).
"""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
DIM = 384
REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> F32: ...

    def identity(self) -> dict[str, Any]: ...


def _normalise(v: npt.NDArray[Any]) -> F32:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    out: F32 = (v / np.maximum(norms, 1e-12)).astype(np.float32)
    return out


class HashingEncoder:
    """Signed feature hashing of lower-cased word unigrams and bigrams (tests only)."""

    name = "hashing-384 (tests only)"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> F32:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for gram in words + [f"{a} {b}" for a, b in itertools.pairwise(words)]:
                h = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                j = int.from_bytes(h[:4], "little") % self.dim
                out[i, j] += 1.0 if h[4] & 1 else -1.0
        return _normalise(out)

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "reference": False}


class SentenceEncoder:
    """The reference encoder (``all-MiniLM-L6-v2``), on CPU, deterministic in eval mode."""

    def __init__(self, model: str = REFERENCE_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model, device="cpu")
        self.name = model
        # sentence-transformers renamed get_sentence_embedding_dimension to
        # get_embedding_dimension; use the new name when it exists (no FutureWarning).
        name = (
            "get_embedding_dimension"
            if hasattr(self._model, "get_embedding_dimension")
            else "get_sentence_embedding_dimension"
        )
        dim = getattr(self._model, name)()
        self.dim = int(dim) if dim is not None else DIM
        self.max_seq_length = int(self._model.max_seq_length)

    def encode(self, texts: list[str]) -> F32:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out: F32 = np.asarray(vectors, dtype=np.float32)
        return out

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "reference": True,
            "max_seq_length": self.max_seq_length,
            **snapshot(self.model_name),
        }


def snapshot(model: str) -> dict[str, Any]:
    """Hugging Face cache snapshot and weight hash of a downloaded model."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, local_files_only=True))
    weights = next(
        (p for p in (path / "model.safetensors", path / "pytorch_model.bin") if p.is_file()), None
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None
    return {
        "revision": path.name,
        "weights": weights.name if weights else None,
        "weights_sha256": digest,
    }
