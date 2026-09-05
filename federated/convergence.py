from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceChecker:
    """Patience-based stopping policy for distributed validation loss."""

    patience: int = 5
    min_delta: float = 1e-4
    history: list[float] = field(default_factory=list)
    best_loss: float | None = None
    stale_rounds: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def update(self, loss: float) -> bool:
        if loss < 0:
            raise ValueError("loss must be non-negative")
        loss = float(loss)
        self.history.append(loss)
        if self.best_loss is None or self.best_loss - loss >= self.min_delta:
            self.best_loss = loss
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return self.stale_rounds >= self.patience

    @property
    def rounds_seen(self) -> int:
        return len(self.history)

    def summary(self) -> dict[str, float | int | bool | None]:
        return {
            "best_loss": self.best_loss,
            "latest_loss": self.history[-1] if self.history else None,
            "rounds_seen": self.rounds_seen,
            "stale_rounds": self.stale_rounds,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "converged": self.should_stop,
        }


__all__ = ["ConvergenceChecker"]
