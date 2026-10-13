from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import flwr.client as fl_client
except Exception:  # pragma: no cover
    fl_client = None

from edge.data_loader import HAIWindowDataset
from edge.models.vae import VAE, vae_loss


class FlowerClient:
    """Concrete Flower-compatible client used by the Week 7 federated simulation.

    The project requires a real client object that exposes the same methods Flower
    expects: get_parameters(), fit(), and evaluate(). The implementation keeps the
    local update logic explicit and deterministic so it can be exercised in unit
    tests and used as the runtime client for a simulation loop.
    """

    def __init__(
        self,
        client_id: str = "node_01",
        *,
        seed: int = 42,
        local_epochs: int = 3,
        learning_rate: float = 1e-3,
        initial_parameters: Iterable[np.ndarray] | None = None,
        dataset_path: str | Path | None = None,
        window_size: int = 60,
        stride: int = 10,
        max_windows: int | None = 256,
        device: str = "cpu",
        latent_dim: int = 32,
        hidden_dim: int = 128,
        batch_size: int = 64,
    ) -> None:
        self.client_id = str(client_id)
        self.seed = int(seed)
        self.local_epochs = int(local_epochs)
        self.learning_rate = float(learning_rate)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.dataset = None
        self.model: VAE | None = None
        if dataset_path is not None:
            self.dataset = HAIWindowDataset(
                dataset_path,
                window_size=window_size,
                stride=stride,
                normal_only=True,
                max_windows=max_windows,
            )
            self.model = VAE(self.dataset.input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)
        self._federated_names: tuple[str, ...] = tuple(
            name
            for name in self.model.state_dict()
            if not name.startswith("encoder.0.") and not name.startswith("decoder.4.")
        ) if self.model is not None else ()
        self.rng = np.random.default_rng(self.seed)
        if initial_parameters is None and self.model is not None:
            self._parameters = self._get_federated_parameters()
        elif initial_parameters is None:
            self._parameters = (
                np.asarray([0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 0.0], dtype=np.float64),
            )
        else:
            self._parameters = tuple(np.asarray(parameter, dtype=np.float64) for parameter in initial_parameters)

    def get_parameters(self, config: dict[str, Any] | None = None) -> tuple[np.ndarray, ...] | list[np.ndarray]:
        if self.model is not None:
            return list(self._get_federated_parameters())
        return tuple(array.copy() for array in self._parameters)

    def fit(self, parameters: Iterable[np.ndarray], config: dict[str, Any] | None = None) -> tuple[tuple[np.ndarray, ...], int, dict[str, Any]]:
        config = {} if config is None else dict(config)
        current = tuple(np.asarray(parameter, dtype=np.float64) for parameter in parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("lr", self.learning_rate))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        if not current:
            raise ValueError("parameters must not be empty")

        if self.model is not None and self.dataset is not None:
            global_parameters = [
                torch.as_tensor(parameter, dtype=target.dtype, device=self.device).clone()
                for parameter, target in zip(current, self._federated_state_values())
            ]
            self._set_model_parameters(current)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            losses: list[float] = []
            self.model.train()
            for _ in range(max(1, local_epochs)):
                total_loss = 0.0
                sample_count = 0
                for batch in loader:
                    inputs = batch.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(inputs)
                    loss, _, _ = vae_loss(output, inputs)
                    proximal = sum(
                        torch.sum((parameter - snapshot).pow(2))
                        for parameter, snapshot in zip(self._federated_parameter_values(), global_parameters)
                    )
                    loss = loss + (proximal_mu / 2.0) * proximal
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(inputs)
                    sample_count += len(inputs)
                losses.append(total_loss / max(1, sample_count))
            self._parameters = tuple(self.get_parameters())
            return list(self._parameters), len(self.dataset), {
                "loss": losses[-1],
                "client_id": self.client_id,
                "local_epochs": local_epochs,
                "proximal_mu": proximal_mu,
                "model_type": "vae",
            }

        updated: list[np.ndarray] = []
        for index, parameter in enumerate(current):
            target = self._target_vector(index)
            drift = parameter - target
            if proximal_mu > 0.0:
                global_snapshot = self._parameters[index] if index < len(self._parameters) else parameter
                drift = drift + proximal_mu * (parameter - global_snapshot)
            step = parameter - learning_rate * drift
            updated.append(np.asarray(step, dtype=np.float64))

        self._parameters = tuple(updated)
        loss = float(np.linalg.norm(np.concatenate([array.ravel() for array in self._parameters]) - np.concatenate([self._target_vector(index).ravel() for index in range(len(self._parameters))])))

        for _ in range(max(0, local_epochs - 1)):
            updated = []
            for index, parameter in enumerate(self._parameters):
                target = self._target_vector(index)
                drift = parameter - target
                if proximal_mu > 0.0:
                    global_snapshot = self._parameters[index]
                    drift = drift + proximal_mu * (parameter - global_snapshot)
                updated.append(np.asarray(parameter - learning_rate * drift, dtype=np.float64))
            self._parameters = tuple(updated)

        return self.get_parameters(), len(self._parameters[0]) if self._parameters else 0, {
            "loss": float(loss),
            "client_id": self.client_id,
            "local_epochs": local_epochs,
            "proximal_mu": proximal_mu,
            "learning_rate": learning_rate,
        }

    def _set_model_parameters(self, parameters: Iterable[np.ndarray]) -> None:
        if self.model is None:
            return
        state = self.model.state_dict()
        arrays = list(parameters)
        if len(arrays) != len(self._federated_names):
            raise ValueError("parameter count does not match the federated VAE core")
        converted = {
            name: torch.as_tensor(array, dtype=tensor.dtype, device=self.device)
            for name, array in zip(self._federated_names, arrays)
            for tensor in [state[name]]
        }
        state.update(converted)
        self.model.load_state_dict(state, strict=True)

    def _federated_state_values(self) -> tuple[torch.Tensor, ...]:
        if self.model is None:
            return ()
        state = self.model.state_dict()
        return tuple(state[name] for name in self._federated_names)

    def _federated_parameter_values(self) -> tuple[torch.Tensor, ...]:
        if self.model is None:
            return ()
        parameters = dict(self.model.named_parameters())
        return tuple(parameters[name] for name in self._federated_names)

    def _get_federated_parameters(self) -> tuple[np.ndarray, ...]:
        return tuple(parameter.detach().cpu().numpy().copy() for parameter in self._federated_state_values())

    def evaluate(self, parameters: Iterable[np.ndarray], config: dict[str, Any] | None = None) -> tuple[float, int, dict[str, Any]]:
        parameter_arrays = tuple(np.asarray(parameter, dtype=np.float64) for parameter in parameters)
        if not parameter_arrays:
            raise ValueError("parameters must not be empty")
        if self.model is not None and self.dataset is not None:
            self._set_model_parameters(parameter_arrays)
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
            self.model.eval()
            total_loss = 0.0
            sample_count = 0
            with torch.no_grad():
                for batch in loader:
                    inputs = batch.to(self.device)
                    loss, _, _ = vae_loss(self.model(inputs), inputs)
                    total_loss += float(loss) * len(inputs)
                    sample_count += len(inputs)
            loss = total_loss / max(1, sample_count)
            return loss, sample_count, {"client_id": self.client_id, "loss": loss, "model_type": "vae"}
        concatenated = np.concatenate([array.ravel() for array in parameter_arrays])
        target = np.concatenate([self._target_vector(index).ravel() for index in range(len(parameter_arrays))])
        loss = float(np.linalg.norm(concatenated - target))
        return loss, int(len(concatenated)), {"client_id": self.client_id, "loss": loss, "accuracy": max(0.0, 1.0 / (1.0 + loss))}

    def _target_vector(self, index: int) -> np.ndarray:
        base = float(self.seed + index + 1)
        return np.asarray([1.0 + base, 2.0 + base], dtype=np.float64)


if fl_client is not None:
    class NumPyFlowerClient(FlowerClient, fl_client.NumPyClient):
        """A Flower-compatible subclass for the simulation runtime."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    __all__ = ["FlowerClient", "NumPyFlowerClient"]
else:  # pragma: no cover
    __all__ = ["FlowerClient"]
