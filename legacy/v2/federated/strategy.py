from __future__ import annotations

from typing import Any, Callable

import numpy as np

from federated.convergence import ConvergenceChecker

try:
    from flwr.common import parameters_to_ndarrays
    from flwr.server.strategy import FedAvg as FlowerFedAvg
    from flwr.server.strategy import FedProx as FlowerFedProx
except Exception:  # pragma: no cover
    FlowerFedAvg = None
    FlowerFedProx = None


class FedProxStrategy:
    """FedProx-compatible aggregation used by the project simulation.

    In Flower, the actual FedProx strategy is effectively weighted averaging at the
    server level; the proximal term is enforced in the local client objective. This
    keeps the project aligned with the real framework while allowing the project to
    emit a explicit proximal_mu parameter for client-side regularization.
    """

    def __init__(self, proximal_mu: float = 0.1):
        self.proximal_mu = float(proximal_mu)
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")

    def aggregate_fit(
        self,
        global_params: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        client_updates: list[np.ndarray],
        weights: np.ndarray | list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        global_params_array = np.asarray(global_params, dtype=np.float64)
        client_updates_array = [np.asarray(update, dtype=np.float64) for update in client_updates]
        weights_array = np.asarray(weights, dtype=np.float64)

        if len(client_updates_array) != len(weights_array):
            raise ValueError("client_updates and weights must have the same length")
        if client_updates_array and global_params_array.shape != client_updates_array[0].shape:
            raise ValueError("global params and client updates must share the same shape")

        weighted = np.zeros_like(global_params_array, dtype=np.float64)
        total_weight = float(np.sum(weights_array))
        if total_weight <= 0:
            raise ValueError("sum of weights must be positive")

        for update, weight in zip(client_updates_array, weights_array):
            weighted += weight * update

        return weighted / total_weight


__all__ = ["FedProxStrategy"]


if FlowerFedProx is not None:
    class InstrumentedFlowerStrategy:
        """Shared telemetry and persistence behavior for Flower strategies."""

        def _init_project(self, *, round_logger: Callable[[int, dict[str, float]], None] | None, convergence_checker: ConvergenceChecker | None, node_health_callback: Callable[[int, list[Any], list[Any]], None] | None) -> None:
            self.round_logger = round_logger
            self.convergence_checker = convergence_checker
            self.convergence_round: int | None = None
            self.node_health_callback = node_health_callback
            self.final_parameters: list[np.ndarray] | None = None

        def aggregate_fit(self, server_round: int, results: list[Any], failures: list[Any]) -> tuple[Any, dict[str, Any]]:
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            if self.node_health_callback:
                self.node_health_callback(server_round, results, failures)
            if parameters is not None:
                self.final_parameters = parameters_to_ndarrays(parameters)
            aggregate_metrics = {
                "client_count": float(len(results)),
                "successful_examples": float(sum(result.num_examples for _, result in results)),
                "failed_clients": float(len(failures)),
                "secure_aggregation_active": 0.0,
                "straggler_rate": float(len(failures) / max(1, len(results) + len(failures))),
            }
            aggregate_metrics.update({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
            if self.round_logger:
                self.round_logger(server_round, aggregate_metrics)
            return parameters, aggregate_metrics

        def aggregate_evaluate(self, server_round: int, results: list[Any], failures: list[Any]) -> tuple[float | None, dict[str, Any]]:
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            aggregate_metrics = {
                "global_loss": float(loss) if loss is not None else float("nan"),
                "evaluation_clients": float(len(results)),
                "evaluation_failures": float(len(failures)),
            }
            if loss is not None and self.convergence_checker is not None:
                converged = self.convergence_checker.update(float(loss))
                if converged:
                    self.convergence_round = server_round
                aggregate_metrics.update({
                    "convergence_stale_rounds": float(self.convergence_checker.stale_rounds),
                    "convergence_stop_requested": float(converged),
                })
            aggregate_metrics.update({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
            if self.round_logger:
                self.round_logger(server_round, aggregate_metrics)
            return loss, aggregate_metrics


    class ProjectFedProxStrategy(InstrumentedFlowerStrategy, FlowerFedProx):
        """Flower FedProx strategy with structured per-round telemetry."""

        def __init__(self, *, proximal_mu: float, round_logger: Callable[[int, dict[str, float]], None] | None = None, convergence_checker: ConvergenceChecker | None = None, node_health_callback: Callable[[int, list[Any], list[Any]], None] | None = None, **kwargs: Any) -> None:
            super().__init__(proximal_mu=proximal_mu, **kwargs)
            self._init_project(round_logger=round_logger, convergence_checker=convergence_checker, node_health_callback=node_health_callback)
            self.proximal_mu = float(proximal_mu)


    class ProjectFedAvgStrategy(InstrumentedFlowerStrategy, FlowerFedAvg):
        """Flower FedAvg baseline with the same telemetry as FedProx."""

        def __init__(self, *, round_logger: Callable[[int, dict[str, float]], None] | None = None, convergence_checker: ConvergenceChecker | None = None, node_health_callback: Callable[[int, list[Any], list[Any]], None] | None = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._init_project(round_logger=round_logger, convergence_checker=convergence_checker, node_health_callback=node_health_callback)

    __all__.append("ProjectFedProxStrategy")
    __all__.append("ProjectFedAvgStrategy")
