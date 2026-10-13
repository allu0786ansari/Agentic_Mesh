# Upgrade Baseline Freeze

This document freezes the pre-upgrade reference for the industrial VAE work.

- Reference commit: `13811c1` (`phase 2 is updated`)
- Original HAI partitions: `data/partitions/`
- Original scaler artifacts: `results/scalers/`
- Original checkpoints: `checkpoints/20260821T024039Z/`
- Original comparative study: `results/comparative/20260821T072536Z/`
- Original evaluation summary: `results/evaluation/20260821T212655Z/`

The original artifacts are reference-only. Upgrade experiments must write to
separate directories such as `data/partitions_upgrade`, `results/scalers_upgrade`,
`checkpoints/upgrade`, and `results/comparative_upgrade`.

The original flattened VAE remains the default architecture. New preprocessing,
scoring, and recurrent-flow components are opt-in until an ablation demonstrates
an improvement on the frozen reference using the same node, seed, split, and
metrics.

## Active Upgrade Contract

The upgrade pipeline is now implemented independently of the frozen artifacts:

- HAI preparation uses synthetic-anomaly screening on a held-out normal slice
	and records candidate and selected features in `meta.json`.
- New partitions default to unscaled engineered values; reusable robust scaler
	artifacts perform the single scaling step during training and evaluation.
- Recurrent-flow checkpoints are evaluated with conditional flow scores, and
	comparative studies default to POT tail thresholds with quantile fallback.

Any partition or scaler generated before this contract was added must be
regenerated under the upgrade directories before its training metrics are used
as final results.
