"""Persistent, append-only privacy accounting per client (Proposal §7.3; flaw P2).

The accountant of each client lives in a JSON file,
``results/privacy/<run_id>/<node>.json``. It holds the client's δ, its target ε, the
plan, and the history of every mechanism it released. Every change is written
atomically before the release is used. So when the federation framework recreates
the client object, or the process restarts, the next object reads the same history
and ε keeps growing.

There is deliberately **no way to reset or shorten the history**: no ``reset``, no
``clear``, no truncating write. Opening an existing store with a different δ or
target ε is an error rather than a fresh start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agmesh.common.provenance import utc_now_iso
from agmesh.privacy.calibrate import Mechanism, epsilon

STORE_VERSION = 1
ROOT = Path("results/privacy")


class StoreMismatchError(ValueError):
    """An existing store was opened with a different δ, target ε or identity."""


def store_path(run_id: str, node: str, root: Path = ROOT) -> Path:
    return root / run_id / f"{node}.json"


class AccountantStore:
    def __init__(
        self,
        path: Path,
        *,
        node: str,
        delta: float,
        target_epsilon: float,
        plan: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key, want in (("node", node), ("delta", delta), ("target_epsilon", target_epsilon)):
                if raw[key] != want:
                    raise StoreMismatchError(f"{path}: {key} is {raw[key]!r}, not {want!r}")
            self._doc = raw
        else:
            self._doc = {
                "version": STORE_VERSION,
                "node": node,
                "delta": delta,
                "target_epsilon": target_epsilon,
                "plan": plan,
                "created_at": utc_now_iso(),
                "history": [],
            }
            self._write()

    # ----------------------------------------------------------------- reading
    @property
    def node(self) -> str:
        return str(self._doc["node"])

    @property
    def delta(self) -> float:
        return float(self._doc["delta"])

    @property
    def target_epsilon(self) -> float:
        return float(self._doc["target_epsilon"])

    @property
    def plan(self) -> dict[str, Any] | None:
        plan = self._doc.get("plan")
        return dict(plan) if plan else None

    @property
    def history(self) -> list[Mechanism]:
        return [Mechanism.from_dict(m) for m in self._doc["history"]]

    def epsilon(self) -> float:
        return epsilon(self.history, self.delta)

    def steps(self, name: str) -> int:
        return sum(m.steps for m in self.history if m.name == name)

    # ----------------------------------------------------------------- writing
    def record(self, mechanism: Mechanism) -> None:
        """Append a release; consecutive identical mechanisms are merged by step count."""
        history = self._doc["history"]
        last = history[-1] if history else None
        if (
            last is not None
            and last["name"] == mechanism.name
            and last["noise_multiplier"] == mechanism.noise_multiplier
            and last["sample_rate"] == mechanism.sample_rate
        ):
            last["steps"] = int(last["steps"]) + mechanism.steps
        else:
            history.append(mechanism.as_dict())
        self._doc["updated_at"] = utc_now_iso()
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def summary(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "delta": self.delta,
            "target_epsilon": self.target_epsilon,
            "epsilon": self.epsilon(),
            "history": [m.as_dict() for m in self.history],
            "plan": self.plan,
        }
