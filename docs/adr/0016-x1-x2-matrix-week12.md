# ADR 0016 — X1 and X2: the run matrix and its analysis, fixed before any run (Week 12)

**Status:** Accepted, 17 October 2026. Written before any X1 or X2 federated run.

**Implements:** Plan Week 12, tasks 1–4; Proposal §10.5–10.6.

**Builds on:**

- ADR 0005: scoring and event-level inference;
- ADR 0006 and 0007: DP and the federation;
- ADR 0008: G2 settings (Site A DP lr 3e-3 with C = 1.0; μ = 0.01; ε = 3 headline;
  five seeds).

**Machine-readable form:** `configs/experiments/x1_x2.yaml`. The runner
(`make grid`) reads it, and the checker uses it to verify completeness.

## 1. Fixed for every run

- **Split:** validation only. Test evaluations happen after `g4-frozen`, in Week 14,
  with `--final`.
- **Seeds:** 11, 23, 37, 41, 53 for every configuration. The G2 compute rule applies
  (§5).
- **Training:** 50 rounds, in-process backend, uniform-mean aggregation.
- **Scoring:** R3 on Site A, R0 on Site B. Thresholds per node at 1% window FPR on
  its own normal validation.
- **Settings:** the `configs/params.yaml` values that produced G2 and G3. The only
  per-group changes are those named below.
- **One machine:** every run executes on the laptop, sequentially, with nothing else
  heavy running (stage isolation). Cross-machine agreement is only about 10⁻³ AUPRC
  (ADR 0008).

## 2. X1 (results/X1): detection utility, RQ1 and H1a

| Group | Sites | Mode | Strategy | ε | Extra | Configurations × seeds |
|---|---|---|---|---|---|---|
| `X1-fed` | A, B | federated | FedAvg, FedProx (μ 0.01) | 1, 3, 10, ∞ | — | 16 × 5 = 80 |
| `X1-local` | A, B | local-only through the same DP pipeline | — | 3, ∞ | each node trains alone | 4 × 5 = 20 |
| `X1-central` | A, B | centralised through the same DP pipeline | — | 3, ∞ | one curator holds all nodes' training windows | 4 × 5 = 20 |
| `X1-h1c` | A | federated | FedProx (μ 0.01) | 10 (preregistered), 3 | H1c threshold | 2 × 5 = 10 |

**Reused, not re-run:** the non-private Week 4/ADR 0005 baselines
(`site_*__local__*` and `site_*__central__*` for VAE/AE, PCA, Isolation Forest),
which already have five seeds. They are the **preregistered PT1/PT2 references**.

**The pipeline-matched baselines** (`X1-local`, `X1-central`) separate the two
costs:

- **The cost of DP:** central at ε = 3 against central at ε = ∞, through the same
  pipeline.
- **The cost of federation:** federated against central at the same ε.

**How the pipeline-matched baselines work:**

- **Same as the federation:** the same models, noisy normalisation, steps per epoch,
  DP learning rate and clipping, and 50 rounds of one local epoch.
- **Local:** one single-client federation per node. Each node is scored with its own
  model, and the macro average is taken over nodes.
- **Central:** a single training client holds every node's training windows. Its
  model is scored on each node's validation, with that node's own threshold. Its ε
  is per training window of the pooled data, like a trusted curator.
- **No re-tuning:** the DP learning rate is the G2 value. These baselines are
  references, not tuned competitors.
- **ε = ∞:** the same pipeline with exact histograms and no clipping or noise
  (ADR 0006).

## 3. X2 (results/X2): heterogeneity, RQ1 and H1b

All X2 groups run at ε = 3 (the headline), with μ = 0.01 unless a μ is named.

| Group | Sites | Arm | Strategy | Configurations × seeds |
|---|---|---|---|---|
| `X2-het` | A, B | systems-heterogeneous: local epochs drawn from {1, 2, 3} per client and round; dropout 0.2 | FedAvg, FedProx | 4 × 5 = 20 |
| `X2-iid` | A | IID partition (`scheme: iid`, 6 h blocks, `data/partitions_iid`) | FedAvg, FedProx | 2 × 5 = 10 |
| `X2-mu-het` | A | heterogeneous, as `X2-het` | FedProx μ 0.001, 0.1 | 2 × 5 = 10 |
| `X2-mu-homog` | A | homogeneous | FedProx μ 0.001, 0.1 | 2 × 5 = 10 |

**The homogeneous arm is X1's ε = 3 federated runs,** the same seeds. It is
referenced, not re-run.

**μ.** ADR 0008 chose μ = 0.01 for X1 and X2. So the **primary** H1b comparison is
FedProx μ 0.01 against FedAvg. The μ grid is a **secondary sensitivity analysis**
on Site A, where the partitions are non-IID. It is reported in full and never used
to pick a μ.

**IID partitions.** These are built from the same raw data with
`make partitions-iid`. Config IDs carry `__iid`.

- **What changes:** only the Site A training and normal-validation split. The
  `val_attack` and test partitions are byte-identical (same SHA-256).
- **Measured on the cloud copy of the data (same partitions as the laptop):**
  - each IID client gets 2,872 training windows from eight 6 h blocks, and 718
    normal-validation windows;
  - the time scheme gives 3,060–3,391 training and 539–597 validation windows.
- **Confound:** the IID split uses about 12% fewer training windows in total (11,488
  against 13,055), because whole blocks are dealt out evenly. So an IID/non-IID
  difference also includes this smaller training set, and the per-node thresholds
  come from different normal-validation windows. Both are reported with the result;
  neither is adjusted.

## 4. Analysis, fixed now

The analysis is computed by `make stats` (`agmesh.evaluation.stats`) from the
artifacts. Week 12 numbers are validation numbers, so they are development evidence.
The same code is applied once to the test split after G4, and the PT1/PT2 claims are
made there.

**Primary method:** the event-level two-stage block bootstrap (ADR 0005), with
B = 10 000, paired, and seeds averaged inside each draw. **Secondary:** seed-level
paired bootstrap and exact Wilcoxon.

**The RQ1 family** (Holm across these four):

| Test | Site | Comparison | Metric | Claim |
|---|---|---|---|---|
| PT1 | A, B | federated DP (headline: FedProx μ 0.01, ε = 3) against central non-private | AUROC | difference > −0.05 (non-inferiority, margin 0.05) |
| PT2 | A, B | the same federated DP model against local-only non-private (mean over nodes) | AUROC | difference ≥ 0 |

The headline is the configured strategy at the headline ε, which is also the edge
runtime's model (ADR 0009). ε = 10 and FedAvg are reported as secondary and are
not part of the family.

**The H1b family** (Holm across these four):

| Comparison | Arm | Sites | Metric |
|---|---|---|---|
| FedProx μ 0.01 against FedAvg, ε = 3 | homogeneous | A, B | AUROC |
| FedProx μ 0.01 against FedAvg, ε = 3 | heterogeneous | A, B | AUROC |

AUPRC, event recall and rounds-to-target are reported alongside.

**Reported, outside the families:**

- IID against non-IID (Site A, both strategies);
- the μ grid;
- the cost of DP (central ε 3 against central ∞);
- the cost of federation (federated against central, same ε);
- the privacy–utility curve over ε {1, 3, 10, ∞}.

**H1c** (Site A, ε = 10, five seeds) follows preregistration §3a. The DP pooled
threshold is compared with:

- per-node thresholds on R0;
- R3.

The metrics are window FPR at event recall within 0.05, with Holm across the two
comparators. The H1c runs store their per-window scores at the pooled threshold for
the event-level bootstrap. ε = 3 is reported as secondary.

**Privacy rule (CC4):** every private run has every client's final ε ≤ its target.
The checker rejects any run that does not.

## 5. Compute and order

Per-run times are from G2 (laptop): Site A DP about 9.6 min, Site B about 7 min;
heterogeneous runs take about 2× as long.

| Groups | Runs | Estimate |
|---|---|---|
| `X1-fed` | 80 | 11 h |
| `X1-local`, `X1-central` | 40 artifacts, the work of about 40 federations | 5.5 h |
| `X1-h1c` | 10 | 1.7 h |
| `X2-het` | 20 | 5.5 h |
| `X2-mu-het`, `X2-mu-homog` | 20 | 4.8 h |
| `X2-iid` | 10 | 1.6 h |
| **Total** | | **about 30 CPU-hours** |

This is under the G2 limit of 60 hours, so **every configuration keeps five seeds.**

**How it runs:**

- **Order:** the runs go in the YAML's order, X1 before X2.
- **Resumable:** a run whose artifact already exists is skipped.
- **Nightly limit:** `HOURS=<n>` stops starting new runs after n hours, so the work
  fits overnight batches.
- **Seeds are never cherry-picked:** a failed run is re-run with the same seed, and
  the failure is logged in `results/grid/log.jsonl`.

## 6. Figures (`make figures`)

All figures are drawn from artifacts only:

- the privacy–utility curve (AUROC and AUPRC against ε, both sites, both strategies,
  with the non-private and pipeline-matched baselines);
- convergence curves (validation loss per round, mean ± range over seeds);
- the H1b and heterogeneity panel.

The per-client privacy table is written alongside, with each client's ε, δ, noise
multiplier and steps.

## 7. Provenance rule for a lock-only dirty tree (17–18 October)

All 180 grid runs were made at commit `8b3d97a` with one uncommitted file:
`uv.lock`, re-locked for the Week 12 dependencies. The checker rejected them on
"uncommitted changes".

Every run's recorded `lockfile_sha256` (`1d8f5b63…`) equals the `uv.lock` committed
next (`e4cbb92`). So each run is fully pinned:

- **Code:** `git_sha`.
- **Environment:** a lock that is in the history.

The checker now accepts a dirty tree **only** when both hold:

- the sole dirty path is `uv.lock`;
- the recorded lock hash matches a committed `uv.lock`.

Any other dirty file is still rejected. With this rule, 250 of 250 X1/X2 artifacts
are accepted, the matrix has 0 missing runs, and CC4 has 0 violations.

## 8. Results on validation (18 October; development evidence)

These numbers are from the laptop, five seeds per configuration, 180 runs with no
failures. The runs took 22.7 h of compute against the 30 h estimate.

The analysis is `results/X{1,2}/analysis_val.{json,md}` (`make stats`). The figures
and the privacy table are in `results/figures/` (`make figures`).

All CIs are 95% event-level, and p-values are one-sided as declared in §4. These
are development evidence; **the claims are made only on the test split after G4.**

### RQ1 (family of four, Holm)

| Test | Site | Headline model | Reference | Difference [CI] | p | p (Holm) | Reading |
|---|---|---|---|---|---|---|---|
| PT1 | A | FedProx ε 3: AUROC 0.891 | central non-private: 0.948 | −0.057 [−0.100, −0.029] | 0.65 | 1.00 | non-inferiority within 0.05 **not shown** |
| PT2 | A | 0.891 | local-only non-private: 0.922 | −0.030 [−0.063, +0.004] | 0.96 | 1.00 | **not shown** |
| PT1 | B | 1.000 | 1.000 | 0.000 [0, 0] | 0.0001 | 0.0004 | holds at the ceiling |
| PT2 | B | 1.000 | 1.000 | 0.000 [0, 0.001] | 0.0001 | 0.0004 | holds at the ceiling |

**Site B.** Every model scores AUROC 1.000 on validation, so these "passes" carry
no information about the methods (ADR 0003, ADR 0005). What separates Site B
models is window FPR, for example 0.032 for FedProx ε 3 against 0.057 for central
non-private. That is not a preregistered claim.

### Where Site A's gap comes from (reported, not a family)

| Comparison (AUROC) | Difference [CI] | Reading |
|---|---|---|
| central ε ∞ against central ε 3 (same pipeline) | +0.038 [+0.016, +0.073] | **the cost of DP**, clearly above zero |
| central ε 3 against federated FedProx ε 3 | +0.002 [−0.008, +0.014] | **the cost of federation** under DP: about zero |
| central ε ∞ against FedProx ε ∞ | +0.002 [−0.012, +0.015] | about zero without DP too |
| FedProx ε 3 against local-only ε 3 (same pipeline) | +0.014 [−0.012, +0.046] | federation helps a little; CI includes 0 |

- **The pipeline's own cost:** central ε ∞ through the DP pipeline (0.931) against
  the Week 4 central non-private (0.948) is 0.017. It comes from 50 rounds of one
  epoch, against early stopping over 200 epochs, plus noisy normalisation.
- **Not converged:** Site A's federated runs reach their best validation loss at
  rounds 45–49 of 50, so they are still improving when the privacy plan ends them.
- **Secondary, ε 10:** −0.040 [−0.072, −0.017] against central non-private (p 0.25),
  so PT1 is not shown at ε 10 either.

**Privacy–utility (Site A, FedProx; AUROC / AUPRC):**

| ε | AUROC | AUPRC |
|---|---|---|
| 1 | 0.869 | 0.593 |
| 3 | 0.891 | 0.697 |
| 10 | 0.909 | 0.723 |
| ∞ | 0.930 | 0.746 |

FedAvg is within 0.001 AUROC and 0.007 AUPRC of FedProx at every ε.

### H1b (family of four, Holm)

| Arm | Site | FedProx − FedAvg [CI] | p (Holm) |
|---|---|---|---|
| homogeneous | A | −0.0002 [−0.0005, −0.0000] | 1.00 |
| heterogeneous | A | −0.0003 [−0.0008, +0.0000] | 1.00 |
| homogeneous | B | 0.000 [0, 0] | 1.00 |
| heterogeneous | B | +0.0014 [+0.0011, +0.0015] | 0.0004 |

- **No practical effect:** μ = 0.01 does not change Site A in either arm. The Site B
  "effect" is +0.001 AUROC at the ceiling.
- **Why it is significant anyway:** the event-level bootstrap fixes each seed's
  model, so a constant tiny difference is "significant" without mattering. This is
  reported as negligible.
- **Expected:** this matches G2. With about 12 steps per round, the proximal term
  barely acts.

**Systems heterogeneity** (epochs from {1, 2, 3}, dropout 0.2; 29–47 missed
client-rounds per run):

| Site | Strategy | Heterogeneous − homogeneous |
|---|---|---|
| A | FedAvg | +0.004 [+0.000, +0.008] |
| A | FedProx | +0.004 [+0.000, +0.008] |
| B | FedAvg | −0.001 |
| B | FedProx | 0.000 |

On Site A, the extra local epochs more than offset the dropouts.

**IID against non-IID** (Site A; unpaired, §3 confound):

| Strategy | IID − non-IID (AUROC) |
|---|---|
| FedAvg | −0.004 [−0.007, −0.002] |
| FedProx | −0.004 [−0.007, −0.001] |

- **Partly the smaller training set:** IID partitions use about 12% fewer training
  windows.
- **The non-IID cost is small anyway:** the time-based split costs no utility
  against IID.

**μ grid (Site A):**

| μ against 0.01 | Homogeneous | Heterogeneous |
|---|---|---|
| 0.001 | within ±0.0005 | within ±0.0005 |
| 0.1 | −0.002 | −0.003 |

The grid confirms G2's choice. It was not used to choose.

### H1c (Site A, FedProx ε 10; family of two, Holm)

| Comparator | Window FPR, pooled DP threshold against comparator | p (Holm) | Event recall difference [CI] | Recall within 0.05? |
|---|---|---|---|---|
| per-node R0 | 0.009 against 0.049 | 0.0002 | −0.012 [−0.037, 0.000] | yes (p 0.005) |
| per-node R3 | 0.009 against 0.014 | 0.019 | −0.062 [−0.188, 0.000] | **not shown** (p 0.69) |

- **Against fixed per-node thresholds:** H1c's pooled DP threshold sharply lowers FPR
  at nearly the same recall.
- **Against R3:** it lowers FPR again, but recall within 0.05 is not shown.
- **ε 3 (secondary)** is in the artifacts.

### Privacy (CC4)

In every private run, every client's final ε is at most its target. The maximum ε
over clients and seeds is, for example, 2.9992 at target 3. The per-client table is
`results/figures/privacy_table.md`.
