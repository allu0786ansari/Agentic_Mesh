"""The leakage guard: scoring code only ever sees label-free features (CC1).

:class:`LabelFreeFrame` holds a read-only float32 feature matrix and its column
names. It can only be built from an explicit feature list, and construction fails if
any feature is a label or label-derived column. Every model and scoring function in
the project accepts a ``LabelFreeFrame`` rather than a DataFrame, so a label cannot
reach a model by accident; ``tests/contract/test_label_free.py`` enforces this.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Column names that are labels, or derived from them, on either site.
LABEL_COLUMNS = frozenset(
    {"attack", "attack_P1", "attack_P2", "attack_P3", "label", "episode", "attacker_packets"}
)


class LabelLeakError(ValueError):
    """A label or label-derived column was offered as a model feature."""


def is_label_column(name: str) -> bool:
    lowered = name.lower()
    return name in LABEL_COLUMNS or lowered.startswith("attack") or lowered == "label"


@dataclass(frozen=True)
class LabelFreeFrame:
    columns: tuple[str, ...]
    values: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        leaked = [c for c in self.columns if is_label_column(c)]
        if leaked:
            raise LabelLeakError(f"label columns offered as features: {leaked}")
        if self.values.ndim != 2 or self.values.shape[1] != len(self.columns):
            raise ValueError("values must be 2-D with one column per feature")
        self.values.setflags(write=False)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, features: Iterable[str], forbidden: Iterable[str] = ()
    ) -> LabelFreeFrame:
        names = tuple(features)
        extra = sorted(set(names) & set(forbidden))
        if extra:
            raise LabelLeakError(f"label columns offered as features: {extra}")
        missing = [c for c in names if c not in frame.columns]
        if missing:
            raise KeyError(f"features missing from the frame: {missing}")
        values = np.ascontiguousarray(frame.loc[:, list(names)].to_numpy(dtype=np.float32))
        return cls(names, values)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.columns)

    def take(self, rows: npt.ArrayLike) -> LabelFreeFrame:
        return LabelFreeFrame(self.columns, np.ascontiguousarray(self.values[np.asarray(rows)]))
