"""Find features that act as proxies for the label.

A label proxy is a feature that reproduces the label exactly (``feature > 0`` equals
the label) or separates attack from normal windows perfectly. On Site B the label is
"the window holds traffic from the attacker's address", so a feature such as
``new_peers`` (a peer never seen in training) can reproduce it by construction. That
is not label leakage in the CC1 sense (the feature is computed from traffic, not
from labels), but a detector that sees it solves the task trivially, so proxies are
reported and handled explicitly in the model input decision (Gate G1).

Only validation partitions are examined; the test split stays sealed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.common.frames import LabelFreeFrame


@dataclass(frozen=True)
class Proxy:
    feature: str
    identical_to_label: bool
    perfect_split: bool
    attack_min: float
    attack_max: float
    normal_min: float
    normal_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label_proxies(features: LabelFreeFrame, target: npt.ArrayLike) -> list[Proxy]:
    """Features that reproduce or perfectly separate the label (empty if one class)."""
    y = np.asarray(target).astype(bool)
    if not y.any() or y.all():
        return []
    found: list[Proxy] = []
    for j, name in enumerate(features.columns):
        x = features.values[:, j].astype(np.float64)
        pos, neg = x[y], x[~y]
        identical = bool(np.array_equal(x > 0, y))
        split = bool(pos.min() > neg.max() or pos.max() < neg.min())
        if identical or split:
            found.append(
                Proxy(
                    name,
                    identical,
                    split,
                    float(pos.min()),
                    float(pos.max()),
                    float(neg.min()),
                    float(neg.max()),
                )
            )
    return found
