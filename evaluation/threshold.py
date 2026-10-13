from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def pot_threshold(scores: np.ndarray, *, risk: float = 1e-3, initial_quantile: float = 0.95) -> float:
    """Fit a generalized Pareto tail to normal validation scores."""
    if not 0 < risk < 1 or not 0 < initial_quantile < 1:
        raise ValueError("risk and initial_quantile must be between 0 and 1")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size < 20 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array with at least 20 values")
    threshold = float(np.quantile(values, initial_quantile))
    excess = values[values > threshold] - threshold
    if excess.size < 8 or np.allclose(excess, 0.0):
        return threshold
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    tail_probability = risk / max(float(np.mean(values > threshold)), 1e-12)
    if abs(shape) < 1e-8:
        quantile = -float(scale) * np.log(tail_probability)
    else:
        quantile = float(scale / shape * (tail_probability ** (-shape) - 1.0))
    result = threshold + max(0.0, quantile)
    return float(result) if np.isfinite(result) else threshold
