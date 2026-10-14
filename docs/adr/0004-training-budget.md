# ADR 0004 — Training budget for the non-private models

**Status:** Protocol declared (committed before any run under it). Results and the
decision are added below the line.
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

*(added after the run)*
