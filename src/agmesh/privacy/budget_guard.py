"""Budget guard: no release may take a client past its total ε (Proposal §7.3; flaw P2).

Before every release (each DP-SGD step, each one-off statistic), the guard projects
the client's ε as the history so far, plus this release, plus anything still
**reserved** for later in the plan (for example the H1c threshold histogram released
after training). If the projection exceeds the target it raises
:class:`BudgetExhausted` and records nothing. A federated client handles the
exception by declining ``fit``.

The guard has no reset path; it reads and appends to the persistent
:class:`~agmesh.privacy.accountant_store.AccountantStore` only.
"""

from __future__ import annotations

from agmesh.privacy.accountant_store import AccountantStore
from agmesh.privacy.calibrate import Mechanism, epsilon

SLACK = 1e-9


class BudgetExhausted(RuntimeError):  # noqa: N818 (name fixed by the Implementation Plan)
    """The next release would exceed the client's total privacy budget."""


class BudgetGuard:
    def __init__(self, store: AccountantStore, reserved: tuple[Mechanism, ...] = ()) -> None:
        self.store = store
        self.reserved = reserved

    def projected(self, release: Mechanism) -> float:
        return epsilon([*self.store.history, release, *self.reserved], self.store.delta)

    def check(self, release: Mechanism) -> None:
        eps = self.projected(release)
        if eps > self.store.target_epsilon + SLACK:
            raise BudgetExhausted(
                f"{self.store.node}: releasing {release.name} would bring ε to {eps:.4f} "
                f"> target {self.store.target_epsilon}"
            )

    def spend(self, release: Mechanism) -> float:
        """Check, then record the release; returns ε after it (reserved part excluded)."""
        self.check(release)
        self.store.record(release)
        return self.store.epsilon()

    def release_reserved(self, name: str) -> float:
        """Spend a reserved one-off release (it was already counted in every projection)."""
        matches = [m for m in self.reserved if m.name == name]
        if not matches:
            raise KeyError(f"no reserved release named {name!r}")
        rest = list(self.reserved)
        rest.remove(matches[0])
        self.reserved = tuple(rest)
        return self.spend(matches[0])
