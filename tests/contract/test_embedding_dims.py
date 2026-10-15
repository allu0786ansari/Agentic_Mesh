"""Query and knowledge-base vectors live in one 384-dimensional space (Plan Week 9; Proposal §8.4).

The rendered evidence and the knowledge base must be embedded by the same encoder.
The retriever refuses a store whose dimension differs from the encoder's. With the
``kb`` group installed and the reference model cached, the reference encoder is
checked as well; otherwise that part is skipped (CI has neither).
"""

from __future__ import annotations

import numpy as np
import pytest

from agmesh.kb.embed import DIM, HashingEncoder
from agmesh.kb.store import COLLECTIONS, MemoryStore


def test_hashing_encoder_matches_the_kb_dimension() -> None:
    enc = HashingEncoder()
    vec = enc.encode(["write requests >100 from a new peer", ""])
    assert vec.shape == (2, DIM) == (2, 384) and vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec[0]), 1.0)
    store = MemoryStore()
    for c in COLLECTIONS:
        store.create(c, enc.dim)
        assert store.dim(c) == 384
    with pytest.raises(ValueError, match="dimension"):
        store.upsert("mitre_ics", np.zeros((1, 128), np.float32), [{}])


def test_reference_encoder_is_384_dimensional() -> None:
    pytest.importorskip("sentence_transformers")
    from agmesh.kb.embed import SentenceEncoder

    try:
        enc = SentenceEncoder()
    except Exception as exc:  # model not downloaded (no network in CI)
        pytest.skip(f"reference encoder unavailable: {exc}")
    a, b = enc.encode(["setpoint changed", "setpoint changed"])
    assert enc.dim == 384 and a.shape == (384,) and np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
