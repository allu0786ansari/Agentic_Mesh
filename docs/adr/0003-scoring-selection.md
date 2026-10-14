# ADR 0003 — Anomaly-score aggregation, selected on validation before G1

**Status:** Accepted. The protocol section was committed (9081602) before any variant
was run; results and the decision are below the line.
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

**Status after the run:** Accepted. Run on 14 October 2026 with the G1 inputs
(ADR 0002), seed 11, validation only.

**Site A (VAE).** AUPRC local / central, and realised window FPR local / central:

| Variant | AUPRC | FPR |
|---|---|---|
| `mean`, no smoothing | **0.664 / 0.750** | 0.304 / 0.030 |
| `mean`, 3 windows | 0.642 / 0.727 | 0.362 / 0.033 |
| `mean`, 6 windows | 0.597 / 0.676 | 0.405 / 0.037 |
| `norm_mean`, no smoothing | 0.467 / 0.682 | 0.291 / 0.030 |
| `norm_mean`, 3 / 6 windows | 0.454 / 0.666; 0.422 / 0.617 | ≈0.29 / 0.03 |
| `norm_top3`, no smoothing | 0.303 / 0.624 | 0.319 / 0.035 |
| `norm_top3`, 3 / 6 windows | 0.303 / 0.612; 0.283 / 0.572 | ≈0.33 / 0.04 |

Chosen: **`mean`, no smoothing** (criterion 0.707). Normalising feature errors by
their training spread hurts on HAI. A likely reason, not yet tested: features that are
almost still in training get a tiny IQR, so ordinary between-period drift in them is
amplified.

**Site B (autoencoder).**

| Variant | AUPRC | FPR |
|---|---|---|
| `mean`, no smoothing | 0.994 / 1.000 | 0.059 / 0.052 |
| `norm_mean`, no smoothing | 0.994 / 1.000 | 0.053 / 0.052 |
| `norm_top3`, no smoothing | **0.994 / 1.000** | **0.052 / 0.049** |
| any variant with 3 windows | 0.36–0.42 | 0.10–0.12 |
| any variant with 6 windows | 0.31–0.34 | 0.17–0.21 |

Chosen: **`norm_top3`, no smoothing**. It ties `mean` on AUPRC (within 0.005) and has
the lower FPR. Smoothing is harmful on Site B because most attacker episodes touch
only one or two 10-second windows, so averaging dilutes them.

**Effect on the X1 baselines** (re-run with the G1 inputs and these choices). The
largest change is Site B PCA: local FPR 0.142 → 0.009 and AUROC 0.993 → 1.000.
Site A numbers equal the "without `P1_PCV02Z`" rows of ADR 0002.

**Consequences.** `models.scoring` in `configs/params.yaml` records the choice; the
grid is closed. All nine variants per site are kept in
`results/X1/scoring_selection.json` for the sensitivity analysis.

### Robustness across seeds (after ADR 0004; `results/X1/scoring_stability.json`)

The rule was repeated on every project seed with the ADR 0004 budget (VAE 200
epochs). It is a check of the decision, not a new selection; the configuration is not
changed by it.

- **Site A:** the rule picks `mean`, no smoothing, on **5/5** seeds and on the seed-mean
  criterion. The choice is stable.
- **Site B:** the rule picks `norm_top3` on 2/5 seeds and `mean` on 3/5. The three
  unsmoothed variants differ by at most 0.003 in AUPRC (0.997–0.999) and 0.003 in FPR
  (0.052–0.056), so the tie-break follows seed noise. Site B validation cannot tell
  these aggregations apart. The declared choice stays; the paper reports that the
  Site B aggregation is immaterial on validation and shows all three in the
  sensitivity table. The firm finding on both sites is that smoothing hurts, which
  holds on every seed.
