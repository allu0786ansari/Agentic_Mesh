from __future__ import annotations

import numpy as np

from federated.client import FlowerClient
from federated.secagg import SecureAggregator
from federated.strategy import FedProxStrategy


def test_fedprox_strategy_reduces_client_drift() -> None:
    strategy = FedProxStrategy(proximal_mu=0.25)
    global_params = np.asarray([0.0, 0.0], dtype=np.float64)
    client_updates = [
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([3.0, 3.0], dtype=np.float64),
    ]
    weights = np.asarray([0.5, 0.5], dtype=np.float64)

    aggregated = strategy.aggregate_fit(global_params, client_updates, weights)

    assert aggregated.shape == global_params.shape
    assert np.allclose(aggregated, np.asarray([2.0, 2.0], dtype=np.float64), rtol=1e-6)
    assert strategy.proximal_mu == 0.25


def test_secure_aggregator_sums_client_updates() -> None:
    aggregator = SecureAggregator()
    updates = [
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([3.0, 4.0], dtype=np.float64),
        np.asarray([5.0, 6.0], dtype=np.float64),
    ]

    aggregated = aggregator.aggregate(updates)

    assert np.allclose(aggregated, np.asarray([9.0, 12.0], dtype=np.float64), rtol=1e-6)
    assert aggregator.secure_mode is True


def test_flower_client_get_parameters_works() -> None:
    client = FlowerClient(client_id="node-1", seed=7)
    params = client.get_parameters()

    assert isinstance(params, tuple)
    assert len(params) == 2
    assert all(np.isfinite(np.asarray(arr)).all() for arr in params)
