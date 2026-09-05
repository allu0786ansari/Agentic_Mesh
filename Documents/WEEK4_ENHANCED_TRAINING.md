# Week 4 Enhanced Training: Ready for Full Real-Data Run

**Date**: 2026-08-20  
**Status**: ✅ Production-Ready Configuration  
**Next Action**: Run `python -m scripts.train_hai --partitions-root data/partitions`

## Summary of Updates

### 1. Epoch Count Increased to 50 (Industrial Standard)

**Why**: 
- 10 epochs is toy-level; standard VAE convergence studies use 50-100 epochs
- 50 epochs sufficient to validate convergence plateaus on real datasets
- Still computationally reasonable (~2-3 hours per full-data run)

**Changed Files**:
- `.env`: `LOCAL_EPOCHS=10` → `LOCAL_EPOCHS=50`
- `.env.example`: Updated with new epoch default
- `params.yaml`: Updated training section to `local_epochs: 50`

**Validation**: ✅ 50 epochs will be captured in manifest and MLflow logs

### 2. Per-Epoch Timing Added to Training Loop

**What**: Each epoch now records its wall-clock duration

**Changed Files**:
- `edge/training/dp_trainer.py`: 
  - Added `import time`
  - Added `epoch_times_seconds: list[float]` to TrainingResult
  - Track `time.time()` before/after each epoch

- `edge/training/trainer.py`:
  - Added `epoch_times_seconds` field to TrainingResult dataclass

**Benefit**: 
- Detect memory leaks (epoch times should be consistent)
- Estimate total training duration before starting
- Benchmark performance across different hardware

### 3. Enhanced Training Manifest with Convergence Analysis

**Changed Files**:
- `edge/main.py`: Restructured JSON output with:

```json
{
  "node_id": "node_01",
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
    "epoch_times_seconds": [2.456, 2.412, ..., 2.401],
    "total_duration_seconds": 120.3,
    "convergence_metric": {
      "initial_loss": 0.315,
      "final_loss": 0.310,
      "convergence_percent": 1.59,
      "stable_after_epoch": 35
    }
  },
  "privacy_accounting": {
    "final_epsilon": 1.9951170173755357,
    "delta": 0.00001
  },
  "checkpoint": {
    "path": "checkpoints/node_01.pt",
    "sha256": "fa3f25fff9bbe5b4815c0c10c706b6367b62059671cf88c5f12a8236ad507e9a",
    "size_mb": 2.35
  }
}
```

**Key Additions**:
- `convergence_metric`: automatic calculation of convergence %, stable-after epoch
- `epoch_times_seconds`: per-epoch wall-clock times
- `training_config`: organized configuration section
- `checkpoint`: includes file size in MB

**Benefit**: All information needed for paper-grade evaluation in one JSON record

### 4. Metrics Documentation (METRICS_AND_EVALUATION.md)

**Created**: `Documents/METRICS_AND_EVALUATION.md` with:
- Complete metric schemas (per-epoch, end-of-run)
- Convergence analysis procedures
- Privacy budget tracking explanation
- Comparison to academic benchmarks
- Reproducibility checklist
- Week 5-6 evaluation roadmap

## What Gets Recorded During Full-Data Training

### Per-Epoch JSONL Events (`logs/metrics.jsonl`)
Each epoch records to `logs/metrics.jsonl`:
```json
{
  "event": "vae_epoch",
  "timestamp": "2026-08-20T12:00:00+00:00",
  "payload": {
    "node_id": "node_01",
    "epoch": 1,
    "loss": 0.315,
    "epsilon": 0.569,
    "sample_count": 92155,
    "input_dim": 2280,
    "epoch_time_seconds": 2.456
  }
}
```

**50 epochs × 3 nodes = 150 epoch events**

### Training Completion Manifest (`results/week4_training.json`)
One comprehensive record per node with:
- Full loss history (all 50 losses)
- Full epsilon history (all 50 epsilon values)
- Full per-epoch timings (all 50 timings)
- Convergence metrics (auto-calculated)
- Privacy accounting
- Checkpoint integrity hash (SHA-256)

**Total: 3 records (one per node)**

### MLflow Tracking (`./mlruns/`)
For each node training run:
- Parameters: epochs=50, batch_size=64, target_epsilon=2.0, etc.
- Metrics: train_loss per epoch, final epsilon, duration
- Artifacts: trained checkpoint file
- Tags: node_id, timestamp

## How to Run the Full Real-Data Training

### Prerequisites
1. Ensure `.env` file exists with `LOCAL_EPOCHS=50`:
```powershell
Get-Content .env | grep LOCAL_EPOCHS
```
Expected: `LOCAL_EPOCHS=50`

2. Verify checkpoint directory exists:
```powershell
$null = mkdir -Force checkpoints
```

3. Verify results directory for manifest:
```powershell
$null = mkdir -Force results
```

### Run Command (One Line)
```powershell
python -m scripts.train_hai --partitions-root data/partitions
```

This will:
1. Train node_01 on 92,155 rows × 38 features (50 epochs)
2. Train node_02 on ~45k rows × 22 features (50 epochs)
3. Train node_03 on ~15k rows × 7 features (50 epochs)
4. Write manifest to `results/week4_training.json`
5. Append epoch events to `logs/metrics.jsonl`
6. Store checkpoint artifacts in MLflow

**Estimated Duration**: 1.5-3 hours (depending on CPU)

### Monitor Progress
While training runs, open new terminal to watch:
```powershell
# Watch manifest grow
Get-Content results/week4_training.json -Wait

# Watch metrics log (JSONL append-only)
Get-Content logs/metrics.jsonl -Tail 5 -Wait

# MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
# Then visit http://localhost:5000
```

## Validation After Training

### 1. Verify All Checkpoints Saved
```powershell
Get-ChildItem checkpoints/node_0[1-3].pt -ErrorAction Stop
```
Expected: 3 files

### 2. Analyze Convergence
```powershell
python -c "import json; data=json.loads(open('results/week4_training.json').read()); [print({'node':r['node_id'],'initial':r['training_results']['convergence_metric']['initial_loss'],'final':r['training_results']['convergence_metric']['final_loss'],'convergence_pct':r['training_results']['convergence_metric']['convergence_percent']}) for r in data]"
```

Expected: loss decreases 1-10% across nodes

### 3. Check Privacy Budget
```powershell
python -c "import json; data=json.loads(open('results/week4_training.json').read()); [print({'node':r['node_id'],'epsilon':r['privacy_accounting']['final_epsilon'],'target':2.0}) for r in data]"
```

Expected: epsilon ≈ 1.995 (close to target 2.0)

### 4. Review Per-Epoch Times
```powershell
python -c "import json; data=json.loads(open('results/week4_training.json').read()); node=data[0]; times=node['training_results']['epoch_times_seconds']; print({'node':node['node_id'],'avg_epoch_time':round(sum(times)/len(times),2),'min':round(min(times),2),'max':round(max(times),2)})"
```

Expected: epoch times consistent (no sudden slowdowns)

## Next Steps After Full-Data Training

### Week 5: Evaluation Harness (test set metrics)
- Load trained checkpoints
- Run inference on test partitions (contain attack data)
- Compute AUC, precision, recall, F1-score
- Generate ROC curves

### Week 6: Privacy-Utility Analysis
- Train with epsilon variants: 1.0, 2.0, 4.0, 8.0, ∞
- Compare AUC across epsilon values
- Generate privacy-utility tradeoff curves

### Week 7+: Multi-seed Studies & Threat Model
- Run 3 different random seeds for statistical rigor
- Report mean ± std of metrics
- Validate detection on real MITRE ATT&CK scenarios

## Industrial Checklist ✓

- [x] Epoch count: 50 (standard for VAE)
- [x] Per-epoch metrics: loss, epsilon, time
- [x] Convergence tracking: automatic stability detection
- [x] Privacy accounting: full epsilon budget curve
- [x] Configuration management: .env with all settings
- [x] Durable storage: manifest + JSONL + MLflow
- [x] Checkpoint integrity: SHA-256 hashing
- [x] Documentation: METRICS_AND_EVALUATION.md guide
- [x] Code quality: all files compile, no syntax errors
- [ ] Reproducibility: random seed control (coming next)
- [ ] Statistical rigor: multi-seed runs (Week 5+)
- [ ] Evaluation: test set metrics (Week 5+)

## File Changes Summary

| File | Change | Reason |
|------|--------|--------|
| `.env` | LOCAL_EPOCHS: 10→50 | Industrial standard convergence |
| `.env.example` | LOCAL_EPOCHS: 10→50 | Template update |
| `params.yaml` | epochs: 10→50 + evaluation section | DVC compatibility + params documentation |
| `edge/training/trainer.py` | Added epoch_times_seconds field | Timing tracking |
| `edge/training/dp_trainer.py` | Added time import + epoch timing loop | Wall-clock duration per epoch |
| `edge/main.py` | Restructured manifest + convergence metrics | Enhanced reporting |
| `Documents/instruction.md` | Updated Week 4 section (epochs 50, added analysis steps) | Runbook for full training |
| `Documents/METRICS_AND_EVALUATION.md` | NEW | Complete metrics guide |

---

**Ready to run?**
```powershell
python -m scripts.train_hai --partitions-root data/partitions
```

**Questions?**
- See `Documents/METRICS_AND_EVALUATION.md` for detailed metrics explanation
- See `Documents/instruction.md` Section "Week 4" for step-by-step validation
- Check `.env` for all configuration options
