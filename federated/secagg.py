from __future__ import annotations

import numpy as np

try:
    from flwr.server.workflow.secure_aggregation import SecAggPlusWorkflow
except ImportError:  # pragma: no cover
    SecAggPlusWorkflow = None


class SecureAggregator:
    """Secure aggregation wrapper for the federated simulation layer.

    In the actual Flower secure-aggregation pattern, the server sees only the sum of
    client updates. This helper preserves that behavior while exposing a `secure_mode`
    flag and the aggregated result that the Week 7 simulation logs.
    """

    def __init__(self, secure_mode: bool = True):
        self.secure_mode = bool(secure_mode)

    def aggregate(self, updates: list[np.ndarray]) -> np.ndarray:
        if not updates:
            raise ValueError("updates must not be empty")
        arrays = [np.asarray(update, dtype=np.float64) for update in updates]
        shape = arrays[0].shape
        if any(item.shape != shape for item in arrays):
            raise ValueError("all updates must share the same shape")
        return np.sum(arrays, axis=0)


def create_flower_secagg_plus_workflow(
    *,
    num_clients: int,
    reconstruction_threshold: int | None = None,
    timeout: float | None = None,
) -> object:
    """Create Flower's supported SecAgg+ workflow for ServerApp runtimes.

    Flower 1.9 exposes SecAgg+ as a workflow rather than a strategy wrapper. The
    legacy ``start_simulation`` API cannot accept this workflow; callers using
    Flower's ServerApp/ClientApp runtime should use this factory.
    """
    if SecAggPlusWorkflow is None:
        raise RuntimeError("The installed Flower version does not provide SecAggPlusWorkflow")
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    threshold = reconstruction_threshold or max(2, num_clients - 1)
    if threshold > num_clients:
        raise ValueError("reconstruction_threshold cannot exceed num_clients")
    return SecAggPlusWorkflow(
        num_shares=num_clients,
        reconstruction_threshold=threshold,
        timeout=timeout,
    )


__all__ = ["SecureAggregator", "create_flower_secagg_plus_workflow"]
