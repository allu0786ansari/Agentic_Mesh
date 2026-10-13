import numpy as np
import torch

from edge.embedding.insight_embedding import InsightEmbeddingGenerator
from edge.models.vae import VAE


def test_vae_embedding_is_128_dimensional_and_normalized() -> None:
    model = VAE(input_dim=12, latent_dim=3, hidden_dim=8)
    embeddings = InsightEmbeddingGenerator.from_vae(model, torch.randn(4, 12))
    assert embeddings.shape == (4, 128)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_isolation_forest_embedding_is_128_dimensional() -> None:
    embeddings = InsightEmbeddingGenerator.from_isolation_forest(
        np.asarray([0.2, 0.5]), np.ones((2, 17), dtype=np.float32)
    )
    assert embeddings.shape == (2, 128)
    assert np.isfinite(embeddings).all()
