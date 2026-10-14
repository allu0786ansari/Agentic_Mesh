# ADR 0006 — Differential privacy design (Week 5)

**Status:** Accepted for Phase 2, 15 October 2026. Items marked **G2** are revisited
at Gate G2 using validation pilots.
**Implements:** Proposal §7.3; Plan Week 5; flaws P1, P2, P4, P5, F8.
**Code:** `src/agmesh/privacy/`, `src/agmesh/models/dp_train.py`,
`src/agmesh/experiments/dp_sanity.py`.

## 1. Privacy unit and neighbouring relation

- **Unit:** one non-overlapping training window of one client: 60 s on Site A, 10 s
  on Site B. The runner refuses to train if training windows overlap (P4).
- **Relation:** add or remove one unit (unbounded DP). This matches Poisson sampling
  in DP-SGD.
- **Guarantee:** (ε, δ) per client, for everything the client releases: model
  updates, normalisation statistics and, in the H1c arm, the score histogram.
- **What is not released:** the client's own threshold and its validation data stay
  on the client. Local early stopping on validation loss is not used under DP (§4).

## 2. Accounting: one RDP ledger per client

**Every release is one Gaussian mechanism.**

- A DP-SGD step is `(z, q, 1)`.
- A whole-data histogram is `(z, 1, 1)`, the plain Gaussian mechanism.

**Composition.** All of a client's releases compose in Rényi DP and are converted to
(ε, δ) once, at the client's δ. This is tighter than the basic composition the Plan
named, and still correct. Both are reported.

**Order grid.** Step 0.02 up to 11, integers up to 64, and large integer orders up
to 512. Fractional large orders were dropped because they gave negative ε in
Opacus's sampled-Gaussian bound. The grid is never looser than Opacus's default;
for example 9.9866 against 9.9868.

**Calibration.** The noise multiplier is calibrated once, up front, for:

- the planned DP-SGD steps (`federated.max_rounds × local_epochs` epochs, plus any
  warm-up; F8);
- the one-off releases, which are fixed first (P1).

**Persistent ledger.** The ledger is stored at `results/privacy/<run_id>/<node>.json`
(P2).

- It is append-only and written atomically before each release is used.
- It has no reset path; a test enforces that no such name exists anywhere in the
  package.
- The budget guard projects history + next release + reserved releases. If that
  would exceed the target it raises `BudgetExhausted` and records nothing.

**δ.** δ = min(10⁻⁵, 1/(10·N_c)), unchanged (P5). Example values: a1 has 3,060
windows, so δ = 10⁻⁵; b1 has 64,792, so δ = 1.5 × 10⁻⁶.

**What RDP composition buys:**

| Target ε | Site A: stats' marginal cost in RDP | Site A: total under basic composition | Site B: stats' marginal cost | Site B: total under basic composition |
|---|---|---|---|---|
| 1 | 0.060 | 1.29 | 0.058 | 1.29 |
| 3 | 0.025 | 3.40 | 0.022 | 3.38 |
| 10 | 0.011 | 10.62 | 0.008 | 10.64 |

The statistics release costs 0.3 on its own, but only 0.01–0.06 inside the RDP
ledger. Basic composition would overshoot every target by 0.3–0.6.

## 3. Normalisation statistics: noisy histograms over public ranges

**Mechanism.**

- Every unit adds total weight 1 to each feature's histogram. A Site A window's 60
  rows add 1/60 each, so the histogram describes rows, as the non-private scaler
  does.
- Bins are equal width over a public range. The L2 sensitivity is √d.
- Gaussian noise with standard deviation z√d is added to every bin.
- **Post-processing:** bins below 3 noise standard deviations are zeroed. The centre
  is the mean, and the scale is the standard deviation with Sheppard's correction,
  floored at bin width/√12.

**Public ranges** (`privacy/bounds.py`). These are the HAI documented point ranges,
widened by half the span on each side, with two exceptions:

- The points HAI 21.03 stores in raw Siemens S7 counts get the S7 raw range
  (−4,864 to 32,511).
- `P2_HILout` gets a wide declared range.
- Site B features (log1p counts) get [0, log1p(10⁴)].

> **Disclosure.** The need for the S7 convention and the margin was found by checking
> the documented ranges against the HAI training data: 6% of Site A window-feature
> values fell outside the documented ranges, for example `P3_LIT01`, which is
> documented in % but stored in raw counts. The bounds are dataset-level conventions
> fixed once and never fitted to any client's windows. The paper states that this
> check was made.

**Design choices, made on the clients' training data and validation before any
federated run:**

| Choice | Value | Evidence |
|---|---|---|
| Statistic | mean and SD from the histogram (not median and IQR) | Median/IQR gives scale 0 on Site B's zero-inflated counts |
| Row weighting within a window | yes | Window means understated the SD of fast signals by 2–6×; with row weighting the noise-free histogram scaler matches the exact scaler (a1 AUROC 0.957 against 0.959) |
| Threshold | 3 noise SDs | Lower thresholds let noise from empty bins accumulate: centre error 0.6–4 SD against 0.10–0.23 |
| Bins | Site A 50, Site B 200 | Site A with noisy statistics at ε_stats 0.3: a1 0.932 / 0.914 / 0.929 AUROC and a3 0.902 / 0.900 / 0.870 for 50 / 100 / 200 bins. Site B centre error is lowest at 200 |
| Standalone ε of the release | **0.3** (was 0.1) | At 0.1 a local Site A centre is off by 0.56 SD; at 0.3, 0.23 SD. Marginal cost in RDP: at most 0.06 |

## 4. DP-SGD

- **Implementation.** Opacus 1.6 `GradSampleModule` with `DPOptimizer` over Adam.
  Per-sample clipping at C = 1.0. Poisson sampling is implemented directly
  (`torch.rand < q` per step), so the accountant's assumption holds exactly. Empty
  batches are steps.
- **Planned steps, final model.** There is no early stopping or best-epoch selection
  on validation data, which would be an unaccounted use of the client's data.
  Training runs for 50 epochs (`federated.max_rounds` 50 × `local_epochs` 1), the
  same as a federated run.
- **Reproducible.** Seeded initial weights, Poisson masks and noise generator.
  `secure_mode=False` (torch PRNG) is recorded in artifacts.
- **Cost.** One VAE step with per-sample gradients takes 0.21 s, against 0.004 s
  plain (2 CPU threads). Site A local DP takes about 130 s for 600 steps; Site B
  about 80 s for 12,650 steps. **G2:** ghost clipping (Opacus fast gradient
  clipping) if the X1/X2 compute budget is tight.

## 5. Sanity check (Week 5 task 6): one client per site, seed 11, validation

ε = ∞ is the same private pipeline (noise-free histograms, fixed steps, final model)
without clipping or noise. "X1 non-private" is the Week 4 pipeline (exact scaler,
early stopping, best epoch). All rows use R3 scoring on Site A.

| Site A, client a1 | ε spent | z | AUROC | AUPRC | Event recall | Window FPR |
|---|---|---|---|---|---|---|
| ε = 1 | 1.000 | 8.88 | 0.802 | 0.378 | 0.50 | 0.001 |
| ε = 3 | 2.999 | 3.23 | 0.832 | 0.520 | 0.75 | 0.008 |
| ε = 10 | 10.000 | 1.32 | 0.915 | 0.598 | 0.75 | 0.028 |
| ε = ∞ (private pipeline, no noise) | — | — | 0.932 | 0.760 | 1.00 | 0.018 |
| X1 non-private | — | — | 0.968 | 0.826 | 1.00 | 0.013 |

| Site B, client b1 | ε spent | z | AUROC | AUPRC | Event recall | Window FPR |
|---|---|---|---|---|---|---|
| ε = 1 | 0.999 | 2.22 | 1.000 | 1.000 | 1.00 | 0.081 |
| ε = 3 | 3.000 | 0.99 | 0.999 | 0.994 | 1.00 | 0.059 |
| ε = 10 | 10.000 | 0.63 | 0.999 | 0.994 | 1.00 | 0.052 |
| ε = ∞ | — | — | 1.000 | 1.000 | 1.00 | 0.007 |

**Reading (one seed, one client: indicative only).**

- **Accounting works as designed.** Every run spends exactly its target (to 10⁻³),
  statistics included.
- **ε = 10 is close to the private pipeline without noise on Site A** (−0.017
  AUROC). The pipeline itself costs 0.036 against the Week 4 model. That cost is
  mostly the noise-free histogram scaler at 50 bins (−0.02); fixed-step training
  without early stopping costs about −0.01.
- **ε = 3 and ε = 1 are far below on one Site A client** (−0.10 and −0.13 AUROC; recall
  0.75 and 0.50). This is the regime where federation should help: averaging four
  clients' noisy updates cuts the noise variance by four. RQ1 is about exactly that.
- **Site B detection is unaffected by DP; its FPR is not.** It rises from 0.007 to
  0.05–0.08: the noise blurs the model's normal profile, and the threshold drawn from
  normal validation transfers worse.

**G2 items.**

- DP learning rate, clipping norm and epochs, tuned on validation pilots. DP-SGD
  usually prefers a larger learning rate and fewer, larger-batch epochs.
- Whether ε = 1 stays a headline value on Site A.
- Ghost clipping.
