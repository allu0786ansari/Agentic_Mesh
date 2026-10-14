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
