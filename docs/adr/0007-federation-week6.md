# ADR 0007 — Federation design (Week 6)

**Status:** Accepted for Phase 2, 15 October 2026. Items marked **G2** are settled at
Gate G2 (Week 7) on validation pilots.
**Implements:** Plan Week 6; Proposal §7; flaws F1, F2, F3, F6, D4.
**Code:** `src/agmesh/federated/` and `tests/{contract,integration}/`.

## 1. What crosses the network

A client releases only four things:

| Release | Protection | Accounted |
|---|---|---|
| Noisy normalisation histograms (round 0) | Gaussian, ADR 0006 | yes |
| Model update after each round of DP-SGD | DP-SGD, ADR 0006 | yes, every step |
| Noisy normal-score histogram (H1c only) | Gaussian, §5 | yes, reserved up front |
| Evaluation summary (`EvalReply`) | not released by a deployment | — |

The evaluation summary is **the experimenter's measurement**: the validation loss each
round, and the metrics at the end. Nothing in training depends on it. The round count
is fixed in advance, with no early stopping on validation loss, because the privacy
plan is calibrated for R rounds. A deployed client keeps its threshold and metrics
local. The server imports no data loader and opens no file
(`tests/contract/test_server_no_data.py`: an import check in a fresh interpreter and
an AST scan of the server modules).

## 2. Aggregation and FedProx

- **FedAvg and FedProx average client models uniformly.** Weighting by window counts
  would make the aggregate depend on a data-dependent number outside the privacy
  analysis. The clients are of similar size, so this costs little.
- **FedProx's term** mu·(w − w_global) is added to the update **after** clipping and
  noise (Opacus `pre_step`, then the inner optimiser). The term does not depend on
  the data, so it is post-processing: it costs no privacy and is not distorted by
  per-sample clipping. Adding it to the per-sample loss, as a naive implementation
  would, inflates every per-sample gradient before clipping.
- Each client restarts Adam every round from the global weights.

## 3. Normalisation round

In round 0 every client releases its noisy histograms (ADR 0006). The server sums them
and derives one pooled scaler, which all clients then use for training and
evaluation, like the central model's pooled scaler in X1.

## 4. Rounds, stragglers and the registry

- **Fixed rounds:** R = `federated.max_rounds` (50), one local epoch per round. The
  heterogeneous arm (X2) draws 1, 2 or 3 local epochs per client per round; its
  privacy plan is calibrated for the maximum, 3 × R epochs.
- **Dropout injection:** with probability `dropout_prob`, seeded per round. A dropped
  client does not train and spends nothing.
- **Timeout:** a reply slower than `round_timeout_s` is discarded. Its spend is
  **not refunded**, which is conservative.
- **Minimum clients:** a round aggregates only with at least 3 of 4 (Fed-A) or 2 of 3
  (Fed-B) replies; otherwise the global model is kept.
- **Registry:** `node_registry` in SQLite, as in Plan §6.4. Three consecutive misses
  mark a node `degraded`; a success makes it `active` again. A node that declines
  because its budget is exhausted is marked `exhausted` and is not asked again.

## 5. H1c: the DP federated threshold, made precise

The Plan says clients release noisy histograms of normal-validation scores, and the
server takes the pooled 99th percentile. Before running it, two problems had to be
fixed. Both fixes are definitions, not results.

1. **The score must not depend on many units at once.** Site A scores use R3 (ADR
   0005). An R3 score depends on the past 30 minutes of scores through a median and a
   MAD. So one unit of validation data can move about 190 windows' scores, and the
   histogram's L2 sensitivity is about √(1 + 2·191) ≈ 19.6. Site B's `norm_top3`
   score is normalised by training-error quantiles, so one *training* window can
   shift every score. With either, the noise needed at ε = 0.5 would swamp the
   histogram; a Site A smoke run gave a meaningless threshold.
   **Definition:** H1c uses **R0 scores** (`mean` aggregation, no smoothing, no
   training statistic) of **non-overlapping** normal-validation windows. Each window
   is one unit, and its score depends only on its own rows and on the DP model, so
   the sensitivity is **1**. The runner refuses H1c for a site whose aggregation is
   not `mean`; that excludes Site B, which the hypothesis does not cover anyway.
2. **Noise must not set the tail.** With 1,000 bins, about one noise-only bin
   survives a 3-SD cut by chance. A single surviving bin high in the tail then
   decides the 99th percentile by itself; the smoke run gave a threshold of 11,600.
   **Definition:** the release uses **100 bins** (groups of 10 bins of the 1,000-bin
   score grid). Bins are cut at the **Bonferroni** level Φ⁻¹(1 − 0.05/100) ≈ 3.29
   noise SDs, so with probability 0.95 no noise-only bin survives. The threshold is
   the smallest bin upper edge with at most 1% of the remaining mass above it.

**Budget.** The histogram's standalone ε is `privacy.threshold_epsilon` = 0.5, composed
with everything else in RDP. It is reserved in the guard from the start, so training
cannot use it up.

**What H1c is compared against**, as the preregistration update after ADR 0005
requires: per-node R0 thresholds on the same model, and the main per-node R3
evaluation. All three are in every H1c artifact.

**G2:** whether H1c's ε (0.5) and bin count (100) stand. With about 2,400 pooled
normal windows on Site A, the 1% tail holds about 24 windows. That is near the limit
of what a DP histogram can resolve, and an honest outcome may be that H1c does not
beat per-node thresholds.

## 6. Two backends, one implementation

- **In-process** (default for experiments). Clients run in a fixed order in one
  process. A new client object is built for every call, as Flower does; only the
  on-disk ledger persists.
- **Flower 1.39** (`federated/flower.py`). The same server logic over the Message API,
  with `run_simulation` on Ray. Protocol objects travel as `ArrayRecord` plus a JSON
  `ConfigRecord`, never pickled. Week 7's SecAgg+ needs this backend.
- **Equivalence and determinism, tested.** On the fixture, both backends give
  bit-identical weights, metrics and ε (`tests/integration/test_flower_backend.py`).
  On real Site B data (2 rounds, ε = 3) the results are also identical. Two
  in-process runs with the same seed give identical weights, metrics and loss curves.

## 7. Convergence: a deviation from the Plan

The Plan measures rounds-to-target against the centralised Week 4 model's validation
loss. That loss is on the exact mean/SD scaler, while federated runs use the pooled
DP scaler, so the two losses are not on the same scale. **Rounds-to-target is
instead the first round within 2% (`rounds_to_target_tolerance`) of the run's own
best validation loss.** The per-round curve is stored, so any other definition can be
computed later.

## 8. Smoke tests (5 rounds, FedProx, ε = 10, seed 11, validation)

| Site | Validation loss by round | Window AUROC (macro) | Pooled AUROC (bins) | Max client ε | s / round |
|---|---|---|---|---|---|
| A (4 clients) | 1.238 → 1.201 → 1.140 → 1.077 → 1.068 | 0.859 | 0.862 ± 0.002 | 10.000 | 16.7 |
| B (3 clients) | 0.430 → 0.284 → 0.241 → 0.195 → 0.151 | 1.000 | 1.000 ± 0.000 | 10.000 | 6.9 |

The loss decreases on both sites. These are 5-round runs whose budget is calibrated
for 5 rounds, so their detection numbers say nothing about the 50-round headline
runs.

**Compute estimate** (2 CPUs, in-process). A 50-round run takes about 14 min on
Site A and 6 min on Site B. The X1 federated grid (2 sites × 2 strategies × ε ∈
{1, 3, 10, ∞} × 5 seeds = 80 runs) is about 13 CPU-hours. That is well inside the
Plan's 60 hours; the X2 arms and H1c add a few more. **G2:** confirm this on the full
pilot.
