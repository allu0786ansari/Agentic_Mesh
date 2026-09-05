# Research Results and Improvement Roadmap

**Project:** Decentralized privacy-preserving agentic intelligence for industrial control system anomaly detection

**Baseline status:** Training and first evaluation completed

**Baseline date:** 2026-08-22

## 1. Frozen First Evaluation

This section records the current result before any model improvement. These files are the first evaluation baseline and must not be overwritten:

- Training manifest: `results/week4_training_20260821T024039Z.json`
- Checkpoints: `checkpoints/20260821T024039Z/`
- Comparative study: `results/comparative/20260821T072536Z/summary.json`
- Evaluation summary: `results/evaluation/20260821T212655Z/summary.json`
- Scaler artifacts: `results/scalers/node_01.json`, `node_02.json`, `node_03.json`
- Evaluation plots: `results/evaluation/20260821T212655Z/`

This baseline used:

```text
VAE: fully connected encoder/decoder
Window size: 60 rows
Training stride: 10 rows
Epochs: 50
Batch size: 64
Learning rate: 0.001
Latent dimension: 32
Hidden dimension: 128
Target epsilon: 2.0
Delta: 1e-5
Seed: 42 for the primary run
Secure RNG: false on Windows
```

The Windows DP results are an engineering and utility baseline. They must not be presented as the final cryptographically secure DP result because `secure_rng=false`.

## 2. Dataset and Evaluation Scope

Each node uses a separate process-specific feature space:

| Node | Features | Raw train rows | Raw test rows | Training windows | Test windows | Attack windows |
|---|---:|---:|---:|---:|---:|---:|
| node_01 | 38 | 921,603 | 402,005 | 92,155 | 40,195 | 1,185 |
| node_02 | 22 | 921,603 | 402,005 | 92,155 | 40,195 | 1,185 |
| node_03 | 7 | 921,603 | 402,005 | 92,155 | 40,195 | 1,185 |

The attack-window prevalence is approximately 2.95%.

The current window configuration has 83.3% overlap because the window size is 60 and the default stride is 10. Window observations are therefore correlated and must not be treated as independent samples in statistical claims.

## 3. Training Results

| Node | Initial loss | Final loss | Improvement | Stable epoch | Runtime | Final epsilon |
|---|---:|---:|---:|---:|---:|---:|
| node_01 | 0.188697 | 0.074107 | 60.73% | 30 | 2.49 h | 1.9943 |
| node_02 | 0.076074 | 0.014933 | 80.37% | 32 | 1.54 h | 1.9943 |
| node_03 | 0.104931 | 0.082242 | 21.62% | 14 | 0.66 h | 1.9943 |

Interpretation:

- All nodes completed 50 epochs without numerical divergence.
- Node 01 and node 02 show strong training-loss reduction.
- Node 03 improves only modestly and is close to a plateau early.
- Training loss alone does not prove anomaly-detection quality.

## 4. First Detection Evaluation

Thresholds were selected from the final 20% of normal training windows. The attack-containing test partition was not used for threshold selection.

| Node | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| node_01 | 0.6010 | 0.1122 | 0.0721 | 0.1477 | 0.0969 | 0.0577 |
| node_02 | 0.5554 | 0.1726 | 0.0326 | 0.5274 | 0.0614 | 0.4757 |
| node_03 | 0.5499 | 0.0351 | 0.0408 | 0.0675 | 0.0509 | 0.0482 |

### Fixed-FPR recall

| Node | Recall at 1% FPR | Recall at 5% FPR | Recall at 10% FPR |
|---|---:|---:|---:|
| node_01 | 10.55% | 14.77% | 21.69% |
| node_02 | 42.45% | 52.74% | 57.55% |
| node_03 | 1.10% | 6.75% | 12.49% |

### Event-level results

| Node | Detected attack events | Total attack events | Event recall | False-positive row rate |
|---|---:|---:|---:|---:|
| node_01 | 10 | 50 | 20.00% | 6.18% |
| node_02 | 39 | 50 | 78.00% | 54.67% |
| node_03 | 21 | 50 | 42.00% | 10.28% |

### Bootstrap uncertainty

The current evaluation includes bootstrap 95% confidence intervals. The intervals remain concentrated around modest discrimination:

- node_01 ROC-AUC: 0.5827-0.6167
- node_02 ROC-AUC: 0.5365-0.5721
- node_03 ROC-AUC: 0.5339-0.5634

## 5. DP Versus Non-Private Multi-Seed Study

The comparative study completed 18 runs: three nodes, three seeds (`42`, `123`, `2024`), and two variants.

### Node 01

| Variant | ROC-AUC | PR-AUC | F1 | FPR |
|---|---:|---:|---:|---:|
| DP | 0.5941 +/- 0.0026 | 0.0991 +/- 0.0050 | 0.0879 +/- 0.0028 | 0.0464 +/- 0.0034 |
| Non-private | 0.7101 +/- 0.0023 | 0.1365 +/- 0.0015 | 0.1984 +/- 0.0149 | 0.0257 +/- 0.0035 |

### Node 02

| Variant | ROC-AUC | PR-AUC | F1 | FPR |
|---|---:|---:|---:|---:|
| DP | 0.5579 +/- 0.0025 | 0.1730 +/- 0.0021 | 0.0615 +/- 0.0005 | 0.4499 +/- 0.0137 |
| Non-private | 0.5542 +/- 0.0003 | 0.1703 +/- 0.0005 | 0.0627 +/- 0.0003 | 0.4547 +/- 0.0080 |

### Node 03

| Variant | ROC-AUC | PR-AUC | F1 | FPR |
|---|---:|---:|---:|---:|
| DP | 0.5495 +/- 0.0010 | 0.0354 +/- 0.0002 | 0.0504 +/- 0.0003 | 0.0483 +/- 0.0002 |
| Non-private | 0.5660 +/- 0.0003 | 0.0461 +/- 0.0025 | 0.0499 +/- 0.0019 | 0.0515 +/- 0.0003 |

## 6. Main Conclusions from the First Evaluation

1. **Node 01 is the strongest candidate for improvement.** It benefits substantially from the non-private baseline and has the clearest privacy-utility gap.
2. **Node 02 has a threshold/calibration problem.** Its recall is high, but the false-positive rate is operationally unacceptable.
3. **Node 03 has a representation or signal problem.** More epochs are unlikely to solve it because training stabilizes early and both DP and non-private results are weak.
4. **Privacy noise is not the only issue.** Node 02 and node 03 remain weak in the non-private comparison.
5. **The model is learning normal reconstruction, but reconstruction error alone is insufficient.** A better anomaly score and temporal aggregation are required.
6. **The current evidence supports continued research, not a final production or publication claim.**

## 7. Required Improvements

### Priority 1: Make preprocessing fully auditable

Already implemented:

- Per-node scaler artifacts
- Feature-order validation
- Finite-value validation
- Partition hashes
- Normal-training-only scaler fitting

Still required for the paper package:

- Record Python, PyTorch, Opacus, NumPy, and operating-system versions.
- Record the Git commit hash.
- Include scaler SHA-256 in every training and evaluation manifest.
- Record the exact fit-row range and feature statistics.
- Verify that all future experiments reuse the same scaler artifacts.

### Priority 2: Improve the anomaly score

Implement and compare:

1. Reconstruction MSE score.
2. Reconstruction MAE score.
3. Latent Mahalanobis distance using normal validation latent vectors.
4. Combined score:

```text
combined_score = normalized_reconstruction_error + alpha * normalized_latent_distance
```

Evaluate `alpha` using validation data only. Do not tune it on the attack test set.

### Priority 3: Add temporal/event aggregation

The current window-level alert is too noisy. Add:

- Consecutive-alert requirement.
- Minimum alert duration.
- Majority vote across neighboring windows.
- Cooldown period after an incident.
- Event-level precision, recall, detection delay, and false alarms per hour.

This is especially important for node 02, where window-level recall is high but false positives are excessive.

### Priority 4: Improve the model architecture

Compare the current fully connected VAE against:

- 1D convolutional VAE for local temporal patterns.
- GRU or LSTM VAE for temporal dependencies.
- Temporal convolutional autoencoder.
- Robust VAE loss using Huber or MAE reconstruction loss.
- Beta-VAE with a small beta sweep.

Keep the same temporal split, scaler, seeds, and evaluation protocol for every architecture.

### Priority 5: Tune DP hyperparameters

Run a controlled privacy-utility sweep:

```text
epsilon: 1, 2, 4, 8
max_grad_norm: 0.1, 0.5, 1.0, 2.0
batch_size: 64, 128, 256
learning_rate: 1e-4, 5e-4, 1e-3
```

Use validation PR-AUC or event-level F1 for selection. The attack test set must remain untouched until the final comparison.

### Priority 6: Complete the secure privacy run

The Windows baseline used `secure_rng=false`. For the final privacy claim:

- Run in the documented WSL/Linux environment with `torchcsprng`.
- Set `SECURE_RNG=true`.
- Keep the secure run in a separate run directory.
- Do not combine secure and non-secure results.
- Report the exact privacy accountant, epsilon, delta, sampling rate, clipping norm, and secure RNG status.

### Priority 7: Statistical and scientific reporting

For the final paper:

- Use at least three seeds for every selected configuration.
- Report mean +/- standard deviation.
- Add bootstrap confidence intervals.
- Report per-node results separately.
- Report event-level results separately from window-level results.
- Report stride sensitivity.
- Add ablation studies for score type, architecture, epsilon, stride, and threshold policy.
- Include confidence intervals and effect sizes for DP versus non-private comparisons.

## 8. Recommended Experiment Sequence

### Experiment A: Score improvement

Keep the existing trained checkpoints and evaluate MSE, MAE, latent distance, and combined scores.

### Experiment B: Temporal aggregation

Evaluate alert persistence, majority vote, and event-level metrics using existing scores.

### Experiment C: Architecture comparison

Train fully connected VAE, convolutional VAE, and recurrent VAE with the same split and seeds.

### Experiment D: DP hyperparameter sweep

Compare epsilon and clipping configurations using validation data only.

### Experiment E: Secure final run

Repeat only the selected best configuration with secure RNG enabled in WSL/Linux.

### Experiment F: Final statistical package

Run three seeds for the selected private and non-private configurations and freeze all artifacts.

## 9. Acceptance Criteria for Publication-Quality Results

A configuration should not be selected as the final method unless it satisfies most of the following:

- ROC-AUC clearly above 0.70 on the strongest nodes.
- PR-AUC substantially above the attack prevalence baseline.
- Useful recall at 1%-5% FPR.
- Event recall reported with acceptable false-positive row rate.
- Stable mean +/- standard deviation across three seeds.
- Confidence intervals reported.
- DP versus non-private comparison completed.
- Secure RNG enabled for the final private result.
- Preprocessing and model artifacts fully hashed and reproducible.
- No leakage from attack data into threshold or hyperparameter selection.
- Overlap sensitivity and event-level analysis included.

These are research acceptance targets, not claims about the current baseline.

## 10. Current Research Decision

The first evaluation is complete and preserved. The project should proceed with score engineering and temporal aggregation before launching another expensive full training run. Increasing epochs alone is not recommended: the current training curves already plateau, while detection quality remains weak. The next improvement should target anomaly-score quality and false-positive control.

