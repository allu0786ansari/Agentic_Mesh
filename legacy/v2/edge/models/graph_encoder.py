from __future__ import annotations

import torch
from torch import Tensor, nn


class SensorGraphEncoder(nn.Module):
    """Learn cross-sensor context independently at each time step."""

    def __init__(self, feature_count: int, embed_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        if feature_count < 1 or embed_dim < 1 or heads < 1 or embed_dim % heads:
            raise ValueError("feature_count, embed_dim, and heads are invalid")
        self.feature_count = feature_count
        self.embed_dim = embed_dim
        self.sensor_embeddings = nn.Parameter(torch.randn(feature_count, embed_dim) * 0.02)
        self.value_projection = nn.Linear(1, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.last_attention_weights: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.feature_count:
            raise ValueError("inputs must have shape [batch, time, feature_count]")
        batch, time, features = inputs.shape
        values = self.value_projection(inputs.reshape(batch * time, features, 1))
        tokens = values + self.sensor_embeddings.unsqueeze(0)
        attended, weights = self.attention(tokens, tokens, tokens, need_weights=True)
        self.last_attention_weights = weights.detach()
        pooled = self.output_projection(attended.mean(dim=1))
        return pooled.reshape(batch, time, self.embed_dim)
