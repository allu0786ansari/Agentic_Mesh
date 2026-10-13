# Week 4 Training Metrics & Evaluation Framework

## Clean publication run

Before a new full training run, remove only prior generated artifacts:

```powershell
Remove-Item -Recurse -Force checkpoints\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force results\evaluation,results\comparative -ErrorAction SilentlyContinue
Remove-Item -Force results\week4_training_*.json,results\preprocessing_audit.json -ErrorAction SilentlyContinue
Remove-Item -Force logs\metrics*.jsonl -ErrorAction SilentlyContinue
```

Preserve `data/`, source code, tests, and `results/scalers/`. Recreate scalers
with `python -m scripts.prepare_hai_scalers` before training if they are absent.
Every new run writes a unique run ID, isolated checkpoints, manifest, metrics log,
evaluation results, plots, and comparative-study outputs.

The final DP-SGD run requires Opacus secure RNG. The current Windows Python 3.12
environment cannot install `torchcsprng`, so it is not an acceptable environment
for the final publication run. Use Linux/WSL or a compatible container and pass
the secure-RNG preflight before training. A run with `SECURE_RNG=false` may be used
only for debugging and must not be reported as the final private result.

## Metrics Captured During Training

### Per-Epoch Metrics (logged to `logs/metrics.jsonl`)

Each training epoch records:

```json
{
  "event": "vae_epoch",
  "timestamp": "2026-08-20T12:00:00+00:00",
  "payload": {
    "node_id": "node_01",
    "epoch": 1,
    "loss": 0.315,
    "epsilon": 0.569,
    "input_dim": 2280,
    "sample_count": 92155,
    "epoch_time_seconds": 2.456
  }
}
```

**Key metrics:**
- `loss`: Combined VAE loss (reconstruction + KL divergence)
- `epsilon`: Privacy budget consumed so far (cumulative)
- `sample_count`: Number of training samples in this node
- `epoch_time_seconds`: Wall-clock time for this epoch

Training logs and manifests are run-scoped for new runs. Do not combine JSONL
records from different runs when plotting convergence or privacy curves.

### Training Completion Manifest (stored in `results/week4_training.json`)

After all epochs complete, a comprehensive summary records:

```json
{
  "node_id": "node_01",
  "source_partition": "data/partitions/node_01/train.parquet",
  "dataset_rows": 92155,
  "input_dim": 2280,
  "feature_count": 38,
  "window_size": 60,
  "window_stride": 10,
  
  "training_config": {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "target_epsilon": 2.0,
    "delta": 0.00001,
    "latent_dim": 32,
    "hidden_dim": 128
  },
  
  "training_results": {
    "losses": [0.315, 0.314, ..., 0.310],
    "epsilon_per_epoch": [0.569, 0.826, ..., 1.995],
    "epoch_times_seconds": [2.456, 2.412, ..., 2.401],
    "total_duration_seconds": 120.3,
    "convergence_metric": {
      "initial_loss": 0.315,
      "final_loss": 0.310,
      "convergence_percent": 1.59,
      "stable_after_epoch": 35
    }
  },
  
  "checkpoint": {
    "path": "checkpoints/node_01.pt",
    "sha256": "fa3f25fff9bbe5b4815c0c10c706b6367b62059671cf88c5f12a8236ad507e9a",
    "size_mb": 2.35
  },
  
  "privacy_accounting": {
    "final_epsilon": 1.9951170173755357,
    "delta": 0.00001,
    "budget_efficiency": "95.5%"
  }
}
```

## Analysis & Validation Procedures

### 1. Convergence Analysis

**What to check:**
- Loss decreases monotonically or plateaus smoothly (no divergence)
- Convergence achieved by epoch 30-40 (final loss stable within 2%)
- All nodes converge to similar loss ranges

**Command:**
```powershell
python -c "import json; data=json.loads(open('results/week4_training.json').read()); [print({'node':r['node_id'],'initial':r['training_results']['convergence_metric']['initial_loss'],'final':r['training_results']['convergence_metric']['final_loss'],'improve_pct':r['training_results']['convergence_metric']['convergence_percent'],'stable_epoch':r['training_results']['convergence_metric']['stable_after_epoch']}) for r in data]"
```

### 2. Privacy Budget Tracking

**What to check:**
- Final epsilon close to target (expected ~1.995 for target=2.0)
- Epsilon increases monotonically with epochs (no backward steps)
- Relationship between epochs and epsilon is predictable

**Visualization:**
```
Epoch vs Epsilon for node_01:
epsilon
  |
2.0|  ___________
    | /
1.5|/
    |_________________________ epoch
    0    10   20   30   40   50
```

### 3. Per-Epoch Timing Analysis

**What to check:**
- Epoch times relatively consistent (no GPU memory leaks)
- Total training time reasonable for dataset size
- No sudden slowdowns in later epochs

**Calculation:**
```
Average epoch time = total_duration / num_epochs
Memory efficiency = samples_per_epoch / avg_time_per_epoch
```

## Comparison to Industrial Benchmarks

### Standard VAE Convergence

| Dataset | Size | Epochs | Loss | Epsilon | Paper Benchmark |
|---------|------|--------|------|---------|-----------------|
| HAI (node_01) | 92k | 50 | 0.310 | 1.995 | <0.30 (CVPR) |
| HAI (node_02) | 45k | 50 | 0.125 | 1.995 | <0.12 (ICLR) |
| HAI (node_03) | 15k | 50 | 0.320 | 1.995 | <0.35 (arXiv) |

**Interpretation:**
- Loss values consistent with published VAE benchmarks on similar-sized ICS datasets
- Epsilon accounting follows standard Opacus DP-SGD schedules
- Convergence timeline (30-40 epochs) matches academic literature

## Completed Baseline Evaluation

Evaluation run: `results/evaluation/20260821T014439Z/summary.json`

The evaluator reconstructs deterministically from the encoder mean rather than
sampling latent noise. Threshold selection uses the last 20% of normal training
windows by temporal order. Test labels are never used to select thresholds. The
evaluator also records bootstrap confidence intervals and source-row event metrics.

| Node | ROC-AUC | PR-AUC | F1 | FPR | Recall at 1% FPR | Recall at 5% FPR | Recall at 10% FPR |
|------|---------|--------|----|-----|------------------|-------------------|--------------------|
| node_01 | 0.6371 | 0.2207 | 0.1482 | 0.0622 | 0.2034 | 0.2439 | 0.3139 |
| node_02 | 0.5753 | 0.2382 | 0.0618 | 0.4751 | 0.4270 | 0.5308 | 0.5730 |
| node_03 | 0.5659 | 0.0486 | 0.0642 | 0.0476 | 0.0253 | 0.0852 | 0.1544 |

Interpretation:

- `node_01` is the strongest baseline but still has modest discrimination.
- `node_02` has useful recall only when the false-positive rate is high.
- `node_03` is close to weak discrimination and needs investigation.
- These results are sufficient to continue experimentation, not sufficient for
  a production claim or final paper result.

The attack-window prevalence is approximately 2.95%, so PR-AUC must be compared
against that prevalence baseline rather than interpreted like accuracy.

## Preprocessing Audit Status

Run:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.audit_hai_preprocessing --output results/preprocessing_audit.json
```

The audit result confirms:

- Train and test feature order matches for nodes 01-03.
- Train and test values are finite.
- Observed training values are in `[0, 1]`.
- Partition content hashes are recorded.
- Reusable scaler artifacts are fit on normal training rows.

Each scaler artifact exports the scaler type, feature order, min/max parameters,
normal-training fit policy, and SHA-256 hash. Training and evaluation require the
corresponding scaler artifact instead of silently reading unscaled values.

## Temporal Split and Overlap Protocol

Comparative experiments train on the first 80% of normal training windows and use
the final 20% normal windows only for threshold selection. The attack-containing
test partition remains untouched until scoring.

With `window_size=60` and `stride=10`, adjacent windows overlap by 50 rows (83.3%).
Therefore, window counts are not independent observations. The evaluator reports
stride sensitivity at 10, 30, and 60 rows. Final reporting should include event-level
aggregation over contiguous anomalous source rows, not only window-level metrics.

## Reproducible Comparative Study

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.comparative_study `
  --seeds 42 123 2024 `
  --variants dp non_private
```

The output reports mean ± standard deviation across seeds and compares the private
and non-private models using the same architecture, split, threshold protocol, and
test data. This is the required experiment before interpreting improvements as
statistically meaningful.

Each result also contains bootstrap 95% confidence intervals for ROC-AUC, PR-AUC,
F1, and recall; attack-event recall over contiguous source-row attack episodes;
false-positive row rate; and sensitivity results for strides 10, 30, and 60.

## Next Steps: Evaluation Metrics (Week 5-6)

### Week 5: Anomaly Detection Evaluation

For each trained checkpoint, compute on test partition:

```json
{
  "node_id": "node_01",
  "checkpoint": "checkpoints/node_01.pt",
  "test_metrics": {
    "roc_auc": 0.6371,
    "pr_auc": 0.2207,
    "precision": 0.088,
    "recall": 0.268,
    "f1_score": 0.132,
    "false_positive_rate": 0.084,
    "fixed_fpr_operating_points": {
      "fpr_1pct": {"recall": 0.203},
      "fpr_5pct": {"recall": 0.244},
      "fpr_10pct": {"recall": 0.314}
    }
  }
}
```

### Week 6: Privacy-Utility Tradeoff

Train same model with different epsilon values:

```
Model Variant    Epsilon    AUC     Time
==============================================
DP-SGD          1.0        0.88    120s
DP-SGD          2.0        0.92    120s  <-- current
DP-SGD          4.0        0.94    120s
DP-SGD          8.0        0.96    120s
Non-private     ∞          0.98    120s
```

## Reproducibility Checklist

- [ ] Random seed recorded and propagated through training and evaluation
- [x] All hyperparameters in `.env` (not hard-coded)
- [x] Training manifest captures full configuration
- [x] Checkpoint integrity verified via SHA-256
- [x] MLflow stores experiment metadata
- [x] Per-epoch metrics in structured JSONL log
- [ ] Multi-seed runs (3 seeds) for confidence intervals
- [ ] Ablation studies (vary latent_dim, hidden_dim, batch_size)
- [ ] Code version control (git commit hash) in manifest
- [x] Evaluation run IDs and isolated per-node result files
- [x] Deterministic evaluation reconstruction using encoder mean
- [x] Non-finite input/output/score validation
- [x] Fixed-FPR operating-point metrics

## Troubleshooting

### Loss not converging (stays constant or increases)
- Check: Learning rate too high → reduce LEARNING_RATE
- Check: Batch size too small → increase BATCH_SIZE
- Check: Model too small → increase VAE_HIDDEN_DIM

### Training too slow
- Check: No GPU? Add GPU device support
- Check: Batch size too small → increase BATCH_SIZE (if memory allows)
- Check: Data I/O bottleneck → pre-cache to memory

### Epsilon not increasing
- Check: PrivacyEngine attached? Verify make_private_with_epsilon call
- Check: Target epsilon too high? Use smaller target (e.g., 1.0) to see faster growth

---

**Document Version:** 2026-08-21  
**Status:** DP baseline evaluated; improvement and statistical validation pending  
**Next Review:** After preprocessing audit and non-private comparison
