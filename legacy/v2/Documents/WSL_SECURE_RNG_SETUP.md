# WSL Secure-RNG Publication Run

This document describes the later Linux/WSL run required for the final privacy-preserving publication experiment. The current Windows run may proceed with `SECURE_RNG=false` for engineering and baseline evaluation.

## 1. Open the repository in WSL

Install the VS Code WSL extension, then use **WSL: Reopen Folder in WSL**. In the WSL terminal:

```bash
cd /mnt/c/Users/LENOVO/OneDrive/Desktop/Agentic_Mesh
```

Do not reuse the Windows `agentvenv` environment.

## 2. Install Linux build tools and Python

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential
```

## 3. Create a WSL environment

```bash
python3.11 -m venv agentvenv-wsl
source agentvenv-wsl/bin/activate
python --version
```

Expected output is Python 3.11.x.

## 4. Install project dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 5. Install and verify secure RNG

```bash
python -m pip install torchcsprng
python -c "from opacus import PrivacyEngine; PrivacyEngine(secure_mode=True); print('secure RNG available')"
```

Do not continue if secure RNG initialization fails.

## 6. Configure the final run

In `.env`:

```text
RANDOM_SEED=42
SECURE_RNG=true
LOCAL_EPOCHS=50
EPSILON=2.0
DELTA=1e-5
```

## 7. Recreate scaler artifacts in WSL

```bash
python -m scripts.prepare_hai_scalers \
  --partitions-root data/partitions \
  --scalers-root results/scalers
```

## 8. Run the final secure training

```bash
python -m scripts.train_hai \
  --partitions-root data/partitions
```

The run creates a unique run ID and isolated checkpoints, manifest, and metrics.

## 9. Run publication evaluation

```bash
python -m scripts.evaluate_hai \
  --partitions-root data/partitions \
  --checkpoint-root checkpoints \
  --results-root results/evaluation \
  --scalers-root results/scalers \
  --overlap-strides 10 30 60
```

## 10. Publication evidence to retain

Keep the following for the final report:

- Run-specific training manifest
- Run-specific JSONL metrics
- Three checkpoint files
- Three scaler artifacts and their hashes
- Evaluation summary and per-node JSON files
- Distribution, ROC, and PR plots
- Bootstrap confidence intervals
- Event-level metrics
- DP versus non-private comparison
- Three-seed mean and standard deviation results
- Python, PyTorch, Opacus, and torchcsprng versions
- Git commit identifier

Never mix Windows `SECURE_RNG=false` results with the final WSL `SECURE_RNG=true` results.
