"""Statistics for multi-seed results (Proposal §10.6).

* :func:`bootstrap_ci` — percentile bootstrap CI of a mean (or any statistic).
* :func:`paired_bootstrap` — CI and two-sided p-value for the mean difference of
  paired results (same seeds, two configurations). This is the primary paired test.
* :func:`wilcoxon_signed_rank` — exact two-sided signed-rank test for paired
  results. With five seeds its smallest possible p-value is 2/32 = 0.0625, so it can
  never reject at 0.05 on its own; it is reported alongside the paired bootstrap.
* :func:`holm` — Holm step-down adjustment when several comparisons are made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Statistic = Callable[[npt.NDArray[np.float64]], float]


def _mean(x: npt.NDArray[np.float64]) -> float:
    return float(np.mean(x))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float


def bootstrap_ci(
    values: npt.ArrayLike,
    *,
    statistic: Statistic = _mean,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(resamples, x.size))
    stats = np.array([statistic(x[row]) for row in draws])
    tail = (1.0 - level) / 2.0
    return Interval(
        statistic(x), float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail)), level
    )


@dataclass(frozen=True)
class PairedResult:
    mean_difference: float
    ci: Interval
    p_value: float


def paired_bootstrap(
    a: npt.ArrayLike, b: npt.ArrayLike, *, resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> PairedResult:
    """Bootstrap over paired differences ``a - b`` (same seeds in the same order)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("a and b must be non-empty and paired")
    d = x - y
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, d.size, size=(resamples, d.size))
    means = d[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    ci = Interval(
        float(d.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)), level
    )
    # two-sided p: how often the resampled mean falls on the other side of zero
    below = float((means <= 0).mean())
    above = float((means >= 0).mean())
    p = min(1.0, 2.0 * min(below, above))
    return PairedResult(float(d.mean()), ci, p)


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float  # sum of positive ranks
    p_value: float
    n: int  # pairs with a non-zero difference


def wilcoxon_signed_rank(a: npt.ArrayLike, b: npt.ArrayLike) -> WilcoxonResult:
    """Exact two-sided Wilcoxon signed-rank test; zero differences are dropped.

    Ties get average ranks; the exact null distribution is computed on doubled ranks
    (integers) by dynamic programming, so ties are handled exactly.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return WilcoxonResult(0.0, 1.0, 0)
    magnitudes = np.abs(d)
    order = np.argsort(magnitudes, kind="mergesort")
    ranks = np.empty(n)
    sorted_mag = magnitudes[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_mag[j + 1] == sorted_mag[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    doubled = np.rint(ranks * 2).astype(np.int64)
    total = int(doubled.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in doubled:
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[: total + 1 - r]
        counts = counts + shifted
    probs = counts / counts.sum()
    w_plus = int(doubled[d > 0].sum())
    lower = probs[: w_plus + 1].sum()
    upper = probs[w_plus:].sum()
    p = min(1.0, 2.0 * min(lower, upper))
    return WilcoxonResult(w_plus / 2.0, float(p), n)


def holm(p_values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Holm-adjusted p-values, in the original order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def main(argv: list[str] | None = None) -> int:
    """``python -m agmesh.evaluation.stats --experiment X1|X2`` (ADR 0016 §4)."""
    import argparse
    from pathlib import Path

    from agmesh.common.artifacts import write_artifact
    from agmesh.common.config import DEFAULT_PARAMS_PATH
    from agmesh.experiments import x1x2_analysis as an

    parser = argparse.ArgumentParser(description="X1/X2 statistics (ADR 0016 §4)")
    parser.add_argument("--experiment", choices=("X1", "X2"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    args = parser.parse_args(argv)
    result = an.run(args.results, args.experiment, args.split, args.resamples)
    path = write_artifact(
        args.results / args.experiment / f"analysis_{args.split}.json",
        kind=f"{args.experiment}.analysis",
        data_source="runtime",
        payload=result,
        params_path=args.params,
    )
    md = an.markdown(result)
    path.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
