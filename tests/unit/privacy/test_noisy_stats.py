"""DP normalisation histograms: sensitivity, accuracy and composition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agmesh.privacy import bounds
from agmesh.privacy import noisy_stats as ns


def test_one_unit_changes_each_feature_by_one_count() -> None:
    rng = np.random.default_rng(0)
    lo, hi = np.zeros(4), np.full(4, 10.0)
    units = rng.uniform(0, 10, size=(100, 4))
    a = ns.exact_counts(units, lo, hi, 20)
    b = ns.exact_counts(units[:-1], lo, hi, 20)
    diff = a - b
    assert diff.sum(axis=1).tolist() == [1, 1, 1, 1]
    assert math.isclose(float(np.linalg.norm(diff)), 2.0)  # L2 sensitivity sqrt(d)
    clipped = ns.exact_counts(np.array([[-5.0, 50.0, 3.0, 9.99]]), lo, hi, 20)
    assert clipped[0, 0] == 1 and clipped[1, -1] == 1  # out of range goes to the end bins


def test_noise_free_moments_match_within_bin_error() -> None:
    rng = np.random.default_rng(1)
    units = np.column_stack([rng.normal(400, 20, 5000), rng.normal(5, 0.5, 5000)])
    lo, hi = np.array([0.0, 0.0]), np.array([720.0, 10.0])
    h = ns.release(units, lo, hi, bins=200, noise_multiplier=0.0, rng=rng)
    centre, scale = ns.centre_scale(h)
    np.testing.assert_allclose(centre, units.mean(axis=0), rtol=0.01)
    np.testing.assert_allclose(scale, units.std(axis=0), rtol=0.05)


def test_noisy_estimates_and_summed_histograms() -> None:
    rng = np.random.default_rng(2)
    lo, hi = np.zeros(3), np.full(3, 100.0)
    parts = [rng.normal(50, 5, size=(4000, 3)) for _ in range(4)]
    hs = [ns.release(u, lo, hi, bins=50, noise_multiplier=4.0, rng=rng) for u in parts]
    pooled = hs[0] + hs[1] + hs[2] + hs[3]
    assert pooled.noise_multiplier == pytest.approx(8.0)  # independent noise adds in variance
    centre, scale = ns.centre_scale(pooled)
    np.testing.assert_allclose(centre, 50.0, atol=1.0)
    np.testing.assert_allclose(scale, 5.0, rtol=0.2)
    with pytest.raises(ValueError):
        _ = hs[0] + ns.release(parts[0], lo, hi + 1, bins=50, noise_multiplier=4.0, rng=rng)
    assert ns.mechanism(4.0).sample_rate == 1.0 and ns.mechanism(4.0).steps == 1


def test_public_bounds_cover_every_modelled_feature() -> None:
    lo, hi = bounds.bounds_for("site_a", list(bounds.SITE_A))
    assert (hi > lo).all()
    assert lo[list(bounds.SITE_A).index("P3_LIT01")] == bounds.S7_RAW[0]  # raw S7 counts
    a, b = bounds.SITE_A["P1_B3005"]
    i = list(bounds.SITE_A).index("P1_B3005")
    assert (lo[i], hi[i]) == (a - 0.5 * (b - a), b + 0.5 * (b - a))
    with pytest.raises(KeyError):
        bounds.bounds_for("site_a", ["P9_UNKNOWN"])
    blo, bhi = bounds.bounds_for("site_b", ["req_recv", "tcp_rst"])
    assert (blo == 0).all() and np.allclose(bhi, math.log1p(10_000))


def test_rows_of_a_unit_share_one_unit_of_weight() -> None:
    lo, hi = np.zeros(2), np.full(2, 10.0)
    unit = np.array([[[1.0, 9.0], [1.0, 1.0], [5.0, 1.0], [9.0, 1.0]]])  # 1 unit, 4 rows
    counts = ns.exact_counts(unit, lo, hi, 10)
    assert counts.sum(axis=1).tolist() == [1.0, 1.0]
    assert float(np.linalg.norm(counts[1])) <= 1.0  # per-feature L2 contribution at most 1
    rng = np.random.default_rng(4)
    rows = rng.normal(5, 1, size=(2000, 60, 2))  # fast signal: rows vary within a window
    _, scale = ns.centre_scale(ns.release(rows, lo, hi, bins=200, noise_multiplier=0, rng=rng))
    np.testing.assert_allclose(scale, rows.reshape(-1, 2).std(axis=0), rtol=0.05)
