# ADR 0003 — Anomaly-score aggregation, selected on validation before G1

**Status:** Protocol declared (this section was committed before any variant was run).
Results and the decision are added below the line once the run is complete.
**Date:** 14 October 2026

## Why

The Week 4 baselines score a window by its *mean* squared reconstruction error over
all inputs. Two weaknesses are well known for multivariate ICS data:

1. **Feature scale.** Features with naturally large reconstruction errors dominate the
   mean; an attack that moves one quiet actuator is diluted by many noisy sensors.
2. **Single-window noise.** One noisy window raises a false alarm; attacks on HAI last
   17–422 s, so they span several 10 s inference windows.

Normalising each feature's error by its own normal error level, aggregating over the
most-deviating features, and smoothing scores causally over a few windows are
standard remedies in ICS anomaly detection. This ADR chooses among them **once**,
on validation data, with a grid fixed in advance, so that the choice cannot drift
towards the test set.

## Protocol (fixed before running)

**Variants (3 × 3 = 9 per site).**

| Factor | Levels |
|---|---|
| Aggregation | `mean` (current): mean squared error over inputs. `norm_mean`: per-feature error (averaged over a window's segments) standardised by that feature's median and inter-quartile range of **training** errors, then averaged over features. `norm_top3`: as `norm_mean`, but the mean of the three largest standardised feature errors |
| Causal smoothing | 1 (none), 3 or 6 windows: the mean of the current and previous windows' scores, only over consecutive windows (never across a run boundary or a gap) |

**Models.** The neural model of each site (VAE on Site A, autoencoder on Site B),
trained once per node (local) and once on pooled data (central), with the G1
inputs (ADR 0002). Each variant re-scores the same trained models; nothing is
retrained per variant.

**Thresholds.** For every variant, each node's threshold is recomputed on its own
normal-validation scores at 1% window FPR (unchanged policy).

**Selection criterion.** Per site, the variant with the highest mean of
(local AUPRC, central AUPRC), each averaged over the nodes where it is defined, on
attack validation. Ties within 0.005 are broken by the lower mean realised window
FPR, then by the simpler variant (`mean` < `norm_mean` < `norm_top3`; less
smoothing first). AUPRC is used because it is threshold-free and sensitive to the
rare attack class.

**Application.** The chosen variant becomes `scoring.<site>` in `configs/params.yaml`
and is applied to every reconstruction model on that site (neural model and PCA) in
X1–X3; Isolation Forest gets the chosen smoothing only. The grid is not extended or
re-run after this decision. Test data is not read.

**Reporting.** All nine variants per site are kept in `results/X1/scoring_selection.json`
and reported as a sensitivity analysis in the paper, whichever wins.

---

## Results and decision

*(added after the run)*
