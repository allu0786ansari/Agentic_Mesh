from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def inject_anomalies(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    seed: int = 42,
    event_count: int = 8,
    event_length: int = 30,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject disclosed bias, freeze, ramp, and noise events into normal rows."""
    if event_count < 1 or event_length < 1:
        raise ValueError("event_count and event_length must be positive")
    result = frame.copy(deep=True)
    columns = list(feature_columns)
    if not columns or len(result) < event_length:
        raise ValueError("frame must contain enough rows and features")
    rng = np.random.default_rng(seed)
    labels = np.zeros(len(result), dtype=np.int8)
    normal = result[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    std = np.nanstd(normal, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    starts = rng.choice(max(1, len(result) - event_length), size=min(event_count, max(1, len(result) // event_length)), replace=False)
    for index, start in enumerate(np.sort(starts)):
        end = min(len(result), int(start) + event_length)
        column_index = int(rng.integers(0, len(columns)))
        if index % 4 == 0:
            normal[start:end, column_index] += 3.0 * std[column_index]
        elif index % 4 == 1:
            normal[start:end, column_index] = normal[start, column_index]
        elif index % 4 == 2:
            normal[start:end, column_index] += np.linspace(0.0, 4.0 * std[column_index], end - start)
        else:
            normal[start:end, column_index] += rng.normal(0.0, 4.0 * std[column_index], end - start)
        labels[start:end] = 1
    result.loc[:, columns] = normal
    return result, labels
