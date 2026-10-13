# Agentic Mesh Project Runbook

This file is the step-by-step command guide for developing and validating the
Decentralized Agentic Mesh project. Commands are recorded week by week. Do not
start the next week until the current week's validation commands pass.

All commands below assume Windows PowerShell and the repository root:

```powershell
cd C:\Users\LENOVO\OneDrive\Desktop\Agentic_Mesh
```

## Project rules

- Use the project virtual environment for every command.
- Run commands one at a time and confirm the expected result before continuing.
- Keep raw datasets under `data/raw/`; they are excluded from Git because they are large.
- Do not commit generated databases, Parquet partitions, model checkpoints, or vector-store data.
- Do not begin the next week until the current week's test suite passes.
- Record new commands and validation results in this file after each completed week.

## Initial environment setup

Run these commands once from the repository root:

```powershell
python -m venv agentvenv
```

```powershell
.\agentvenv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install -r requirements-dev.txt
```

Verify the active interpreter:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

## Week 1: Foundation

### 1. Verify the message bus

```powershell
python scripts/verify_messaging.py
```

Expected output:

```text
All messaging checks passed
```

### 2. Run the Flower scaffold

```powershell
python federated/simulate.py --rounds 1 --stub
```

Expected result: the Week 1 simulation scaffold starts successfully.

### 3. Run the Week 1 tests

```powershell
python -m pytest tests/unit/test_bus_and_replay.py tests/integration/test_infra_smoke.py -q
```

Expected result:

```text
3 passed
```

Use the project interpreter explicitly. The bare `pytest` command may select the
global Python installation and fail with `ModuleNotFoundError: No module named
'federated'`.

### 4. Compile the Week 1 packages

```powershell
python -m compileall -q edge federated
```

Week 1 is complete only when all commands above pass.

## Week 2: Real dataset preparation

Week 2 uses the following six-node allocation:

| Node | Dataset | Model |
|---|---|---|
| `node_01` | HAI 21.03 P1 | VAE |
| `node_02` | HAI 21.03 P2 | VAE |
| `node_03` | HAI 21.03 P3 | VAE |
| `node_04` | CIC Modbus IED1A | Isolation Forest |
| `node_05` | CIC Modbus IED1B | Isolation Forest |
| `node_06` | CIC Modbus SCADA HMI | Isolation Forest |

### 1. Confirm the raw datasets exist

```powershell
Test-Path data/raw/hai/hai-21.03
Test-Path data/raw/modbus2023
```

Both commands must return `True`.

Confirm the HAI files:

```powershell
Get-ChildItem data/raw/hai/hai-21.03 -Filter *.csv.gz
```

Confirm the Modbus captures:

```powershell
(Get-ChildItem data/raw/modbus2023 -Recurse -Filter *.pcap).Count
```

The expected Modbus capture count is 37 for the current download.

### 2. Confirm TShark is available

```powershell
Get-Command tshark.exe
```

The expected installation path on the development machine is:

```text
C:\Program Files\Wireshark\tshark.exe
```

### 3. Prepare all six real-data partitions

Run this as a single command:

```powershell
python -m scripts.prepare_week2_data --hai-root data/raw/hai/hai-21.03 --modbus-root data/raw/modbus2023 --output data/partitions --database agmesh.db --tshark "C:\Program Files\Wireshark\tshark.exe"
```

Expected output includes:

```json
{"nodes": 6, "output": "data\\partitions", "database": "agmesh.db"}
```

Use the module form exactly. Running `python scripts/prepare_week2_data.py`
directly may fail because repository-level imports such as `storage` are not
available when Python uses the `scripts` directory as its import root.

### 4. Verify the generated partition directories

```powershell
Get-ChildItem data/partitions -Directory | Sort-Object Name
```

Expected directories:

```text
node_01
node_02
node_03
node_04
node_05
node_06
```

Verify each node has the required artifacts:

```powershell
Get-ChildItem data/partitions -Recurse -File | Where-Object { $_.Name -in @('train.parquet', 'test.parquet', 'meta.json') } | Select-Object FullName
```

### 5. Verify real Parquet data and labels

```powershell
python -c "import json, sqlite3; from pathlib import Path; import pandas as pd; root=Path('data/partitions'); rows=[]; [rows.append((p.name, json.loads((p/'meta.json').read_text()), pd.read_parquet(p/'train.parquet'), pd.read_parquet(p/'test.parquet'))) for p in sorted(root.glob('node_*'))]; print('nodes', len(rows)); [print({'node':n,'source':m['source'],'features':m['feature_count'],'train':len(tr),'test':len(te),'train_labels':tr['label'].value_counts().to_dict(),'test_labels':te['label'].value_counts().to_dict()}) for n,m,tr,te in rows]; db=sqlite3.connect('agmesh.db'); print('registry_rows', db.execute('select count(*) from node_registry').fetchone()[0]); db.close()"
```

Expected result:

- `nodes 6`
- HAI training labels are normal only.
- Modbus training labels are benign only.
- Test partitions contain normal/benign and attack labels.
- `registry_rows 6`

### 6. Run all Week 2 tests

```powershell
python -m pytest tests/unit/test_week2_real_data.py tests/unit/test_week2_partitioning.py -q
```

Expected result:

```text
5 passed
```

### 7. Run the complete test suite

```powershell
python -m pytest tests/unit tests/integration -q
```

Expected result for the current implementation:

```text
8 passed
```

### 8. Compile the updated packages

```powershell
python -m compileall -q edge federated storage scripts
```

Week 2 is complete only when the real-data preparation succeeds, six partitions
and six registry rows exist, and the complete test suite passes.

## Week 3: Shared services

Status: complete.

Week 3 adds embedded Qdrant, SQLite integration checks, MLflow round tracking,
Phoenix-compatible tracing, and structured JSONL metrics.

### 1. Install Week 3 dependencies

The project pins `arize-phoenix==12.35.0` and `grpcio-tools==1.62.2` for the
Windows/Python 3.12 environment. Install the build compatibility dependency first:

```powershell
python -m pip install --upgrade "setuptools<81"
```

```powershell
python -m pip install --only-binary=:all: "grpcio-tools==1.62.2"
```

```powershell
python -m pip install --prefer-binary -r requirements.txt
```

```powershell
python -m pip install -r requirements-dev.txt
```

```powershell
python -m pip check
```

### 2. Create embedded Qdrant collections

```powershell
python -m scripts.init_qdrant --path qdrant_data
```

Expected collections:

```text
agent_memory
insight_embeddings
mitre_kb
```

### 3. Start MLflow for real experiment tracking

Run this in a separate PowerShell terminal and keep it running while training:

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open the tracking UI at:

```text
http://localhost:5000
```

The Week 4 training commands write to the local `./mlruns` store by default.

### 4. Start Phoenix for real tracing

Run this in another PowerShell terminal when agent or tracing workflows are active:

```powershell
python -c "from agents.tracing import launch_phoenix; launch_phoenix()"
```

If Phoenix is available and starts successfully, open its reported local URL.
The project tracing facade still provides local spans when the Phoenix process is
not running, so infrastructure tests do not require a persistent observability
daemon.

### 5. Run Week 3 infrastructure tests

```powershell
python -m pytest tests/integration/test_week3_infra.py -q
```

Expected result:

```text
4 passed
```

### 6. Run the complete test suite

The installed Phoenix distribution currently exposes a broken pytest entry point
that can fail before test collection. Disable third-party plugin auto-discovery and
load only pytest-asyncio explicitly:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -p pytest_asyncio.plugin tests/unit tests/integration -q
```

Expected result:

```text
12 passed
```

### 7. Compile the Week 3 packages

```powershell
python -m compileall -q agents edge federated scripts storage
```

Week 3 is complete when Qdrant collection creation succeeds, the focused
infrastructure tests pass, the cumulative suite passes, and compilation succeeds.

## Week 4: VAE and DP-SGD

Status: complete.

Week 4 adds the VAE model, lazy HAI sliding-window loader, standard training loop,
Opacus DP-SGD training, epsilon accounting, MLflow metrics, and checkpoints.

### 1. Run Week 4 unit tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -p pytest_asyncio.plugin tests/unit/test_vae.py tests/unit/test_dp_trainer.py tests/unit/test_hai_data_loader.py -q
```

Expected result:

```text
5 passed
```

### 2. Run bounded real HAI training

Use a small `--max-windows` value to validate the complete CLI without starting a
full laptop-scale training run:

```powershell
$env:NODE_ID='node_01'
python -m edge.main --mode train --max-windows 8 --checkpoint-root checkpoints_week4_smoke
```

Expected output includes:

```text
{'node_id': 'node_01', 'epochs': 1, 'samples': 8, 'epsilon': ..., 'checkpoint': 'checkpoints_week4_smoke\\node_01.pt'}
```

The command reads the real `data/partitions/node_01/train.parquet` HAI partition,
trains the VAE with DP-SGD, writes an MLflow run under the `agentic-mesh-week4`
experiment, logs structured epoch metrics, and saves the checkpoint.

### 3. Run full training on the real HAI dataset

The bounded command above is only a fast pipeline check. To train the actual
three HAI VAE nodes, run the following commands one at a time from the repository
root. Do not use `--max-windows`; without that option the loader uses the complete
normal training partition for the selected node.

All Week 4 settings are loaded from the repository `.env` file. Important settings
include `LOCAL_EPOCHS=50`, `BATCH_SIZE=64`, `EPSILON=2.0`, `DELTA=1e-5`, and the
checkpoint/metrics/manifest paths. Edit `.env` when changing a run configuration;
do not place the configuration values in the shell command.

The previous training outputs were cleared before this publication run. Source
partitions, code, tests, and `results/scalers` were preserved. Old checkpoints,
training manifests, evaluation directories, and append-only metrics were removed.
The next run writes only run-scoped artifacts and does not mix with historical data.

Fit the reusable scalers on normal training rows before the first full run:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.prepare_hai_scalers --partitions-root data/partitions --scalers-root results/scalers
```

The final `.env` should have `RANDOM_SEED=42` and `SECURE_RNG=true`. Secure RNG
is required for the final privacy-preserving experiment; tests may use it disabled.
The current Windows Python 3.12 environment cannot initialize secure RNG because
`torchcsprng` has no compatible distribution there. Run the final publication
training in the project Linux/WSL or container environment with a supported
`torchcsprng` build. Do not set `SECURE_RNG=false` for the final reported run.

Preflight the target environment:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -c "from opacus import PrivacyEngine; PrivacyEngine(secure_mode=True); print('secure RNG available')"
```

Proceed only when this prints `secure RNG available`.

Training metrics recorded per epoch:
- Loss (reconstruction + KL divergence)
- Epsilon (privacy budget consumed so far)
- Input dimension and feature count
- Elapsed time per epoch
- Cumulative gradient norm

Final training manifest records:
- Dataset metadata: row count, input dimension, feature count, source partition path
- Training configuration: epochs, batch size, learning rate, target epsilon, delta
- Complete loss history (all 50 epochs)
- Epsilon curve (privacy budget as function of epoch)
- Training duration in seconds
- Checkpoint path and SHA-256 integrity hash
- Model architecture: latent_dim, hidden_dim

The training manifest enables:
- Loss curve analysis (convergence validation)
- Privacy budget consumption tracking
- Reproducibility (checkpoint integrity verification)
- Cross-node comparison (loss dynamics across different feature spaces)

Train all three HAI nodes using one orchestrator command:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.train_hai --partitions-root data/partitions


```

The orchestrator reads `.env`, trains `node_01`, `node_02`, and `node_03`
sequentially on their complete real partitions, and writes:

```powershell
checkpoints/<run_id>/node_01.pt
checkpoints/<run_id>/node_02.pt
checkpoints/<run_id>/node_03.pt
results/week4_training_<run_id>.json
logs/metrics_<run_id>.jsonl
results/scalers/node_01.json  # prepared before training and reused
results/scalers/node_02.json  # prepared before training and reused
results/scalers/node_03.json  # prepared before training and reused
```

For a bounded pipeline check only, use:

```powershell
python -m scripts.train_hai --max-windows 8 --checkpoint-root checkpoints_persistence_smoke
```

The full command trains 50 epochs by default (standard for VAE convergence studies).
Each node summary is persisted in the run-specific manifest with dataset rows,
input dimensions, losses for every epoch, epsilon per epoch, delta, duration,
checkpoint path, and checkpoint SHA-256. Each epoch is also appended to
the run-specific JSONL file, and MLflow stores the same parameters, metrics, and
checkpoint artifact under `agentic-mesh-week4`.

The expected new checkpoint files are:

```text
checkpoints/<run_id>/node_01.pt
checkpoints/<run_id>/node_02.pt
checkpoints/<run_id>/node_03.pt
```

### 4. Verify all real HAI checkpoints and the training manifest

```powershell
Get-ChildItem checkpoints -Recurse -Filter 'node_0[1-3].pt' | Select-Object FullName,Length,LastWriteTime
```

Expected result: three non-empty checkpoint files.

Verify the saved privacy and architecture metadata:

```powershell
$latestCheckpointRoot = Get-ChildItem checkpoints -Directory | Sort-Object Name | Select-Object -Last 1
python -c "import torch; from pathlib import Path; root=Path(r'$($latestCheckpointRoot.FullName)'); [print({'file':str(p),'input_dim':torch.load(p,map_location='cpu',weights_only=False)['input_dim'],'epsilon':torch.load(p,map_location='cpu',weights_only=False)['epsilon'],'delta':torch.load(p,map_location='cpu',weights_only=False)['delta']}) for p in sorted(root.glob('node_0[1-3].pt'))]"
```

Expected result:

- `node_01`, `node_02`, and `node_03` are present.
- Each checkpoint has `epsilon` and `delta` metadata.
- Each checkpoint has an input dimension equal to `60 * feature_count` for its HAI process.

Verify the durable training manifest:

```powershell
$latestManifest = Get-ChildItem results -Filter 'week4_training_*.json' | Sort-Object Name | Select-Object -Last 1
Get-Content $latestManifest.FullName
```

Expected result: a JSON list containing one completed record for each of
`node_01`, `node_02`, and `node_03`, including a loss history with 50 values.
Loss should show steady convergence (decreasing trend or stable plateau by epoch 30+).

### 5. Verify MLflow records the real training runs

Start the local MLflow UI in a separate PowerShell terminal:

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open:

```text
http://localhost:5000
```

Select experiment `agentic-mesh-week4` and verify that there is one run for each
of `node_01`, `node_02`, and `node_03`, with `train_loss`, `epsilon`, `delta`,
and the node/window parameters recorded.

### 6. Run full project validation

Phoenix currently exposes a broken pytest auto-discovery entry point in this
environment. Disable third-party plugin discovery and load pytest-asyncio explicitly:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -p pytest_asyncio.plugin tests/unit tests/integration -q
```

The expected cumulative result after Week 4 is 19 passing tests.

### 7. Compile all Week 4 packages

```powershell
python -m compileall -q agents edge federated scripts storage
```

Opacus may emit secure-RNG and accounting warnings during development. The final
production training run must enable secure RNG; the current Week 4 test run is an
algorithm and pipeline validation run.

### 8. Analyze convergence and loss dynamics

After training completes, verify the loss curves converge properly:

```powershell
$latestManifest = Get-ChildItem results -Filter 'week4_training_*.json' | Sort-Object Name | Select-Object -Last 1
python -c "import json; from pathlib import Path; data=json.loads(Path(r'$($latestManifest.FullName)').read_text()); [print({'node':r['node_id'],'initial_loss':r['training_results']['losses'][0],'final_loss':r['training_results']['losses'][-1],'convergence_pct':r['training_results']['convergence_metric']['convergence_percent'],'epochs':r['training_config']['epochs']}) for r in data]"
```

Expected results:
- Initial loss: varies by node architecture (the completed baseline was node_01: 0.1834, node_02: 0.0705, node_03: 0.1050)
- Final loss: should be lower than initial (convergence evidence)
- Convergence: the completed baseline improved by 59.19%, 78.56%, and 21.56% for nodes 01, 02, and 03 respectively
- All nodes should complete 50 epochs without divergence

### 9. Record baseline metrics for evaluation comparison

These training runs establish the **DP-SGD baseline**. Later weeks will:
- Train **non-private VAE** (epsilon=∞) on same data for comparison
- Measure **anomaly detection performance** (AUC, precision, recall) using test partitions
- Generate **privacy-utility curves** across multiple epsilon values (1.0, 2.0, 4.0, 8.0, ∞)
- Validate **threat model effectiveness** on real MITRE ATT&CK scenarios

The current Week 4 checkpoints will serve as the privacy-constrained reference point.

## Week 5: Checkpoint evaluation and anomaly detection

Status: baseline evaluation complete; model improvement and statistical validation pending.

Run the evaluation in the project virtual environment:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.evaluate_hai `
	--partitions-root data/partitions `
	--checkpoint-root checkpoints `
	--results-root results/evaluation `
	--scalers-root results/scalers
```

Every evaluation receives a UTC `run_id` and writes isolated artifacts:

```text
results/evaluation/<run_id>/summary.json
results/evaluation/<run_id>/node_01.json
results/evaluation/<run_id>/node_02.json
results/evaluation/<run_id>/node_03.json
```

The evaluator automatically selects the newest complete checkpoint directory under
`checkpoints` when `--checkpoint-root checkpoints` is supplied. To evaluate a
specific training run, pass its explicit directory, for example:

```powershell
--checkpoint-root checkpoints/20260821T024039Z
```

The evaluator uses deterministic VAE reconstruction through the encoder mean,
selects thresholds from the last 20% of normal training windows, and records:

- Reconstruction-error distributions for validation and test windows
- ROC-AUC and PR-AUC
- Precision, recall, F1 score, and false-positive rate
- Confusion matrices and class counts
- Operating points at 1%, 5%, and 10% validation false-positive targets

Inspect the latest result:

```powershell
$latest = Get-ChildItem results/evaluation -Directory | Sort-Object Name | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'summary.json')
```

The 2026-08-21 baseline run was `20260821T014439Z`:

| Node | ROC-AUC | PR-AUC | F1 at validation 95th percentile | FPR |
|------|---------|--------|----------------------------------|-----|
| node_01 | 0.6371 | 0.2207 | 0.1482 | 0.0622 |
| node_02 | 0.5753 | 0.2382 | 0.0618 | 0.4751 |
| node_03 | 0.5659 | 0.0486 | 0.0642 | 0.0476 |

These are baseline results, not production or paper conclusions. Continue with
preprocessing verification, fixed-FPR analysis, a non-private comparison, and
three-seed confidence intervals before claiming improvement.

### Required evaluation gates

1. Confirm that train and test partitions use the same feature ordering and scaler.
2. Report results at fixed FPR values of 1%, 5%, and 10%, not only one percentile threshold.
3. Compare DP-SGD against a non-private VAE using the same partitions and architecture.
4. Repeat each configuration with at least three seeds and report mean and standard deviation.
5. Preserve each evaluation under a unique `run_id`; never append unrelated experiments to one result file.

### 10. Audit preprocessing and partition identity

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.audit_hai_preprocessing `
	--partitions-root data/partitions `
	--output results/preprocessing_audit.json
```

The audit verifies feature ordering, row counts, finite values, partition hashes,
and the observed value range. The current audit confirms all three nodes use the
same ordered features between train and test, contain no NaN/Inf values, and have
values in `[0, 1]`. It also reports that scaler provenance is not currently
available because `scaler_metadata_path` is null. Do not make a reproducibility
claim until the upstream scaler parameters are exported and hashed.

### 11. Generate distributions, ROC/PR curves, and overlap sensitivity

```powershell
& $py -m scripts.evaluate_hai `
	--partitions-root data/partitions `
	--checkpoint-root checkpoints `
	--results-root results/evaluation `
	--scalers-root results/scalers `
	--overlap-strides 10 30 60
```

Each node result now includes a PNG with normal/attack error distributions,
ROC and PR curves, plus metrics at strides 10, 30, and 60. Since `window_size=60`
and the default stride is 10, adjacent windows overlap by 83.3%; report event-level
results or stride sensitivity alongside window-level metrics.

### 12. Run the DP versus non-private three-seed study

This command trains on the first 80% of normal training windows, selects thresholds
from the final 20% normal validation windows, evaluates the untouched test partition,
and repeats DP and non-private models for seeds 42, 123, and 2024:

```powershell
& $py -m scripts.comparative_study `
	--partitions-root data/partitions `
	--results-root results/comparative `
	--seeds 42 123 2024 `
	--variants dp non_private
```

The command writes a unique run directory containing per-run checkpoints, plots,
raw records, and `summary.json` with mean and standard deviation for ROC-AUC,
PR-AUC, precision, recall, F1, and false-positive rate. This is a long-running
experiment: it retrains six configurations per node.

## Week 5: Isolation Forest, model selection, and embedded Qdrant

Status: implemented and validated.

The complete Week 5 implementation, command, saved artifacts, measured results,
limitations, and Week 6 handoff are maintained separately in:

`Documents/WEEK5_IMPLEMENTATION_AND_RESULTS.md`

## Week 6: Insight Embedding pipeline and local inference

Status: implemented and smoke-validated.

The implementation, commands, smoke-test result, artifacts, and follow-up work are
maintained separately in:

`Documents/WEEK6_IMPLEMENTATION_AND_RESULTS.md`

## Week 7: Federated learning

Status: implemented and five-round runtime validated; cryptographic SecAgg+ remains on the ServerApp path.

The implementation details, current validation evidence, commands, artifacts, and
remaining runtime work are maintained separately in:

`Documents/WEEK7_IMPLEMENTATION_AND_RESULTS.md`

### 1. Verify the Flower environment

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -c "import flwr, torch, opacus, mlflow; print('flwr', flwr.__version__)"
```

Expected Flower version:

```text
flwr 1.9.0
```


### 2. Install the missing Flower simulation backend package

```powershell
& $py -m pip install "ray==2.31.0"
```

### 3. Run the real three-node HAI FedProx simulation

```powershell
& $py -m federated.simulate `
	--rounds 5 `
	--clients 3 `
	--nodes node_01 node_02 node_03 `
	--partitions-root data/partitions `
	--max-windows 256 `
	--local-epochs 1 `
	--proximal-mu 0.1
```

Expected result: five Flower rounds, local VAE training on each HAI partition,
MLflow round metrics, `fedprox_round` events, and a timestamped directory under
`results/week7_federated/` containing `summary.json`, `round_metrics.jsonl`, and
`global_model.npz`.

### 4. Verify the SecAgg+ workflow factory

```powershell
& $py -c "from federated.secagg import create_flower_secagg_plus_workflow; print(type(create_flower_secagg_plus_workflow(num_clients=3)).__name__)"
```

Flower 1.9 provides SecAgg+ as a ServerApp workflow. It is not accepted by the
legacy `start_simulation()` API and is therefore not claimed active in the command
above.

The validated five-round run is saved under:

`results/week7_federated/20260822T061241Z/`

All three clients fit and evaluated successfully in every round. Weighted fit and
evaluation callbacks are configured, so the previous Flower metric-aggregation
warnings no longer occur.

### 5. Run the focused Week 7 tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week7_federated.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
3 passed
```

### 6. Compile the federated package

```powershell
& $py -m compileall -q federated
```

The full Flower `start_simulation()` path, MLflow round logging, real local model
training, and cryptographic SecAgg protocol remain explicit follow-up validation
items. Do not mark Week 7 complete until those checks pass.

## Week 8: Convergence and stragglers

Status: implemented and control-path smoke validated; full 50-round validation pending.

Implementation details and validation evidence are maintained separately in:

`Documents/WEEK8_IMPLEMENTATION_AND_RESULTS.md`

Run the focused Week 8 tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_convergence.py tests/unit/test_week8_node_health.py tests/unit/test_week7_federated.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
8 passed
```

Run the full Week 8 configuration:

```powershell
& $py -m federated.simulate `
	--rounds 50 `
	--clients 3 `
	--nodes node_01 node_02 node_03 `
	--partitions-root data/partitions `
	--results-root results/week8_federated `
	--local-epochs 5 `
	--max-windows 256 `
	--patience 5 `
	--min-delta 0.0001 `
	--round-timeout 300 `
	--database agmesh.db `
	--proximal-mu 0.1
```

Each run saves `summary.json`, `round_metrics.jsonl`, and `global_model.npz` under
`results/week8_federated/<run_id>/`. The control-path smoke result is saved under
`results/week8_validation/20260822T062814Z/`.

## Week 9: DVC and FedAvg/FedProx evaluation

Status: FedAvg/FedProx experiments and held-out AUROC evaluation completed; tuning gate pending.

Implementation details and commands are maintained separately in:

`Documents/WEEK9_IMPLEMENTATION_AND_RESULTS.md`

Run the Week 9 focused tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week9_comparison.py tests/unit/test_week7_federated.py tests/unit/test_convergence.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
8 passed
```

Use `--strategy fedprox` and `--strategy fedavg` with identical data and training
parameters, then compare the two saved `summary.json` files using
`scripts.compare_week9_strategies`.

Initialize DVC once before using the pipeline:

```powershell
& $py -m dvc init
& $py -m dvc dag
```

The latest comparison is saved under `results/week9_comparison/`, including
`comparison.json` and `convergence.png`. FedAvg currently has the better final loss;
FedProx has the slightly better best loss, so FedProx superiority is not established.
The selected FedAvg global parameter bundle is registered in MLflow as
`GlobalAnomalyDetector` version 1; the registration record is saved as
`results/week9_comparison/registration.json`.

Held-out global-core evaluation is saved under `results/week9_global_evaluation/`.
Macro ROC-AUC is `0.5710` for FedAvg and `0.5678` for FedProx, below the project target
of `0.92`; hyperparameter and architecture tuning is required before final sign-off.

## Week 10: Agentic mesh foundation, JWT, Ollama, and mTLS

Status: Python implementation complete; local Ollama/Phoenix service verification remains environment-dependent.

Implementation details and commands are maintained separately in:

`Documents/WEEK10_IMPLEMENTATION_AND_RESULTS.md`

Run Week 10 tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_auth.py tests/unit/test_agent_graph.py tests/unit/test_dev_mtls.py tests/unit/test_tls.py tests/unit/test_llm_adapter.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
13 passed
```

Generate development TLS files:

```powershell
& $py -m scripts.generate_dev_mtls --output certs/dev --common-name localhost
```

Run the API over HTTPS:

```powershell
& $py -m scripts.run_node_api `
	--host 127.0.0.1 `
	--port 8443 `
	--qdrant-path qdrant_data_week6 `
	--collection insight_embeddings_20260821T222733Z `
	--certfile certs/dev/dev-cert.pem `
	--keyfile certs/dev/dev-key.pem
```

Run the compiled graph:

```powershell
& $py -c "from agents.graph import app; print(app.invoke({'session_id':'week10-1','alert':{'node_id':'node_01','score':0.9}}))"
```

Install and verify Phi through Ollama separately:

```powershell
ollama create agmesh-phi3.5 -f Modelfile
ollama run agmesh-phi3.5 "Return JSON only: {\"ok\": true}"
ollama ps
```

Launch Phoenix through the project interpreter:

```powershell
$env:PHOENIX_HOST = '127.0.0.1'
$env:PHOENIX_PORT = '6006'
& $py -c "import phoenix; phoenix.launch_app()"
```

In a second terminal, emit and verify the graph spans:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -m scripts.run_agent_graph --endpoint $env:PHOENIX_OTEL_ENDPOINT
```

The graph sends spans to `POST /v1/traces` and the collector must return `200 OK`.
Open `http://127.0.0.1:6006` to inspect the `triage`, `investigator`, `security`,
and `dispatcher` child spans. If the root page reports the known Starlette/Jinja
compatibility error, OTLP ingestion is still verified but browser inspection remains
pending the Phoenix UI dependency repair.

## Week 11: Triage, Investigator, circuit breaker, and bounded hops

Status: real agent layer implemented; live service latency profiling pending.

Implementation details and commands are maintained separately in:

`Documents/WEEK11_IMPLEMENTATION_AND_RESULTS.md`

Run the Week 11 focused tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week11_agents.py tests/unit/test_auth.py tests/unit/test_agent_graph.py tests/unit/test_api_security.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
8 passed
```

Run the real agent graph after Phoenix, Ollama, and node APIs are available:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -c "from agents.tracing import configure_phoenix; configure_phoenix(); from agents.llm import LocalPhiModel; from agents.triage.triage_agent import TriageAgent; from agents.investigator.investigator_agent import InvestigatorAgent; from agents.graph import build_real_graph; import asyncio; triage=TriageAgent(llm=LocalPhiModel()); investigator=InvestigatorAgent(llm=LocalPhiModel(), node_urls={'node_01':'http://127.0.0.1:8443','node_02':'http://127.0.0.1:8444','node_03':'http://127.0.0.1:8445'}); print(asyncio.run(build_real_graph(triage_agent=triage, investigator_agent=investigator).ainvoke({'session_id':'week11-real','alert':{'node_id':'node_01','score':0.9,'embedding':[0.0]*128,'target_nodes':['node_01','node_02','node_03']}})))"
```

The Investigator enforces a maximum of three hops and a 500 ms timeout per node.
The live latency target is measured only when the real services are running.

## Week 12: MITRE knowledge base, Security Agent, and Dispatcher

Status: implemented; live KB indexing and service runtime validation pending.

Implementation details and commands are maintained separately in:

`Documents/WEEK12_IMPLEMENTATION_AND_RESULTS.md`

Run the focused Week 12 tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week12_agents.py tests/unit/test_week11_agents.py tests/unit/test_auth.py tests/unit/test_api_security.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
9 passed
```

Build the MITRE KB from the actual downloaded sources:

```powershell
& $py -m scripts.build_mitre_kb `
	--stix data/raw/mitre-ics/enterprise-attack.json `
	--nist data/raw/nist_800_82/nist-800-82.txt `
	--qdrant-path qdrant_data_week12 `
	--collection mitre_kb
```

## Week 13: FL-to-agent event runner and local stack

Status: graph runner and local-stack orchestration implemented; live FL trigger validation pending.

Implementation details and commands are maintained separately in:

`Documents/WEEK13_IMPLEMENTATION_AND_RESULTS.md`

Run the focused Week 13 tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_graph_runner.py tests/unit/test_local_stack.py tests/unit/test_week12_agents.py tests/unit/test_week11_agents.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
8 passed
```

Run five local alert sessions:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -m scripts.run_local_stack `
	--count 5 `
	--real-agents `
	--qdrant-path qdrant_data_week12 `
	--database agmesh.db `
	--node-id node_01 `
	--score 0.9 `
	--endpoint $env:PHOENIX_OTEL_ENDPOINT `
	--metrics-path results/week13_smoke/metrics.jsonl
```

The run persists five `agent_graph_completed` events and requires Phoenix OTLP to be
available because tracing has no local fallback.

The federated trigger is available with:

```powershell
& $py -m federated.simulate `
	--rounds 3 `
	--clients 3 `
	--nodes node_01 node_02 node_03 `
	--partitions-root data/partitions `
	--results-root results/week13_federated `
	--local-epochs 1 `
	--max-windows 256 `
	--inject-attack `
	--alert-threshold 0.2
```

## Week 14 and later: Evaluation and final packaging

## Weekly completion record

| Week | Scope | Status | Validation result |
|---|---|---|---|
| Week 1 | Repository foundation, async bus, telemetry replay, Flower scaffold | Complete | 3 tests passed |
| Week 2 | Real HAI/Modbus preparation, six partitions, SQLite registry | Complete | 8 tests passed |
| Week 3 | Shared services and observability | Complete | 12 tests passed |
| Week 4 | VAE and DP-SGD | Complete | 7 focused tests passed; three-node orchestrator smoke run passed |
| Week 5 | Isolation Forest and embeddings | Not started | Pending |
| Week 6 | Inference and local evaluation | Not started | Pending |
| Week 7 | Federated learning | Foundation implemented | 3 focused tests passed; full Flower runtime pending |
| Week 8 | Convergence and straggler handling | Implemented | 8 focused tests passed; control-path runtime validated |
| Week 9 | DVC and FedAvg/FedProx evaluation | Tooling implemented | 8 focused tests passed; experiments pending |
| Week 10 | Agentic mesh foundation, JWT, Ollama adapter, mTLS | Complete | 13 focused tests passed |
| Week 11 | Triage, Investigator, circuit breaker, bounded hops | Implemented | 8 focused tests passed; live profiling pending |
| Week 12 | MITRE security agent, dispatcher, full graph | Implemented | 9 focused tests passed; combined KB persisted at 741 points |
| Week 13 | FL-to-agent trigger, graph runner, local stack | Implemented | 8 focused tests passed; trigger helper validated; full stack pending |
| Week 12 | MITRE security agent, dispatcher, full graph | Implemented | 9 focused tests passed; live KB/runtime pending |
| Week 13+ | FL-to-agent trigger, Docker runtime, final evaluation | Not started | Pending |
