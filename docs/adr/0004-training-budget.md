# ADR 0004 — Training budget for the non-private models

**Status:** Accepted. The protocol was committed (ef6d574) before the run; results
and the decision are below the line.
**Date:** 14 October 2026

## Why

The Week 4 baselines train with `max_epochs` 100 (VAE) and 30 (autoencoder) and stop
early on the normal-validation loss (patience 10 and 5). In the G1 run, two of the
four Site A local VAEs (a3, a4) ran into the 100-epoch cap with their best epoch at
97–100: the cap, not convergence, ended their training. A model stopped by an
arbitrary cap is under-trained, and its detection numbers are not a fair baseline.

## Protocol (fixed before running)

**Rule (label-free).** For each site, choose the smallest `max_epochs` from
{100, 200, 400} for the VAE and {30, 60, 120} for the autoencoder such that early
stopping, not the cap, ends at least 95% of training runs. A run counts as capped if
its best epoch is within `patience` epochs of `max_epochs`. Patience is unchanged.

**Runs.** Local (every node) and central, five seeds (11, 23, 37, 41, 53), G1 inputs
(ADR 0002), validation only. The rule uses only the training curves (normal
validation loss), never attack labels or detection metrics.

**What is reported, not used for the choice.** Normal-validation loss at the best
epoch, and the X1 detection metrics under each budget, as a sensitivity analysis.

---

## Results and decision

**Status after the run:** Accepted (14 October 2026; seeds 11, 23, 37, 41, 53;
validation only).

| Site | `max_epochs` | Runs ended by the cap | Mean best epoch | Local AUROC / FPR | Central AUROC / FPR |
|---|---|---|---|---|---|
| A (VAE) | 100 | **32%** of 25 | 69 | 0.843 / 0.261 | 0.915 / 0.044 |
| A (VAE) | **200** | **0%** | 71 | 0.846 / 0.279 | 0.915 / 0.044 |
| A (VAE) | 400 | 0% | 71 | 0.846 / 0.279 | 0.915 / 0.044 |
| B (AE) | **30** | **0%** of 20 | 11 | 1.000 / 0.054 | 1.000 / 0.057 |
| B (AE) | 60, 120 | 0% | 11 | identical | identical |

**Decision.** Site A `models.vae.max_epochs` = **200**; Site B stays at **30**.

**What it shows.** Longer training removes every capped run, but detection barely
moves (local AUROC +0.003; local FPR within seed noise). The local models' high false
alarm rate is therefore not under-training; it is the recording-period drift that
federation is meant to address (RQ1, H1c). Training longer than 200 changes nothing
because early stopping already ends every run.

### Sensitivity check: training longer than early stopping allows (run after the decision)

The author asked whether more epochs would help. The rule above only moves the cap,
so this check also relaxes early stopping. Site A VAE, seeds 11 and 23, validation
only; the best normal-validation checkpoint is kept in every case. It was run after
the decision and does not change it.

| `max_epochs` / patience | Local AUROC | Local FPR | Central AUROC | Central FPR |
|---|---|---|---|---|
| 200 / 10 (chosen) | 0.841 | 0.284 | 0.914 | 0.038 |
| 400 / 50 | 0.835 | 0.351 | 0.926 | 0.033 |
| 400 / 400 (no early stopping) | 0.833 | 0.363 | 0.926 | 0.033 |

Training longer lowers the normal-validation loss, but on the local models it makes
detection **worse** (FPR up by about 0.07, AUROC down). On a3 the FPR rises from
0.30 to 0.63 for seed 11. Each local model fits its own recording period more tightly
and raises more alarms on the other recordings. The central model, trained on all four
periods, gains about 0.01 AUROC.

**Where the local false alarms come from** (seed 11, validation labels, a
diagnostic only). Every client's `val_normal` is the tail of its own training file,
but its `val_attack` windows come from the July test recordings. The local VAE's
false alarms are driven by variables that are almost constant inside one client's
training window and sit at a different level in other recordings. These are
setpoints (`P1_B4002`, `P1_B2004`, `P1_B4022`), turbine speed (`P2_SIT01/02`) and
vibration (`P2_VXT03`, `P2_VYT03`). For example, a3's `P1_B4002` has reconstruction
error 0.01 in training and 3.69 in its false alarms. That is a shift in operating
point between recordings, a form of non-IID data across clients. More epochs make it
worse; more diverse data (central, or federated) helps. PCA suffers less (local FPR
0.053) because it does not fit the local period as tightly.

**Consequence.** Training budget is not the lever. The fair local baseline still
needs a label-free answer to operating-point shift. That is proposed as ADR 0005
before Phase 2; it is not decided here.
