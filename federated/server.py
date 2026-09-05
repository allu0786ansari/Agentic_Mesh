from __future__ import annotations

import timeit

try:
    from flwr.server import Server
    from flwr.server.history import History
except ImportError:  # pragma: no cover
    Server = None
    History = None


if Server is not None:
    class ConvergenceServer(Server):
        """Flower server loop that stops after strategy convergence is requested."""

        def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
            history = History()
            self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
            initial = self.strategy.evaluate(0, parameters=self.parameters)
            if initial is not None:
                history.add_loss_centralized(server_round=0, loss=initial[0])
                history.add_metrics_centralized(server_round=0, metrics=initial[1])

            start_time = timeit.default_timer()
            for current_round in range(1, num_rounds + 1):
                fit_result = self.fit_round(server_round=current_round, timeout=timeout)
                if fit_result is not None:
                    parameters, fit_metrics, _ = fit_result
                    if parameters:
                        self.parameters = parameters
                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )

                centralized = self.strategy.evaluate(current_round, parameters=self.parameters)
                if centralized is not None:
                    history.add_loss_centralized(server_round=current_round, loss=centralized[0])
                    history.add_metrics_centralized(server_round=current_round, metrics=centralized[1])

                distributed = self.evaluate_round(server_round=current_round, timeout=timeout)
                if distributed is not None:
                    loss, metrics, _ = distributed
                    if loss is not None:
                        history.add_loss_distributed(server_round=current_round, loss=loss)
                        history.add_metrics_distributed(server_round=current_round, metrics=metrics)

                if getattr(self.strategy, "convergence_round", None) == current_round:
                    break

            return history, timeit.default_timer() - start_time
else:  # pragma: no cover
    class ConvergenceServer:  # type: ignore[no-redef]
        """Import-safe placeholder when Flower is not installed."""

        pass


__all__ = ["ConvergenceServer"]
