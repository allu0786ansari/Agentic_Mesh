# Week 9 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** FedAvg/FedProx experiments and held-out evaluation completed; AUROC target not met; tuning required

## 1. Objective

Week 9 makes the federated experiment reproducible and directly comparable:

- FedAvg and FedProx use the same clients, seeds, rounds, local data, and metrics.
- DVC records the partition, local training, federated training, and evaluation stages.
- Saved summaries can be compared without rerunning experiments.
- Persisted global models can be loaded by run ID.

## 2. Implemented Components

### Strategy switch

File: `federated/simulate.py`

Use `--strategy fedprox` or `--strategy fedavg`. Both strategies share the same
instrumentation, convergence policy, timeout, client-health updates, MLflow logging,
and result artifacts. FedProx passes `proximal_mu` to the local objective; FedAvg
passes zero for that parameter.

### Instrumented FedAvg baseline

File: `federated/strategy.py`

`ProjectFedAvgStrategy` subclasses Flower's `FedAvg` and uses the same round metrics
as `ProjectFedProxStrategy`, including client counts, failures, straggler rate,
training loss, and distributed evaluation loss.

### Saved-run comparison

File: `scripts/compare_week9_strategies.py`

Consumes the two `summary.json` files and writes a comparison JSON containing:

- Run IDs and strategy names.
- Completed rounds.
- Best and final global loss.
- Failed-client histories.
- FedProx minus FedAvg loss deltas.

### Versioned model loading

File: `federated/model_loader.py`

`GlobalModelLoader` loads `global_model.npz` and its `summary.json` by run ID or finds
the latest complete persisted model under a results root.

### DVC pipeline

File: `dvc.yaml`

Stages are defined for:

1. `partition_data`
2. `train_local_models`
3. `federated_train`
4. `evaluate`

Each stage declares source dependencies and generated outputs. The DVC commands use
repository-relative paths and can be reproduced from the project root.

## 3. Validation Commands

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week9_comparison.py tests/unit/test_week7_federated.py tests/unit/test_convergence.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
& $py -m compileall -q federated scripts
```

Validate the DVC graph:

```powershell
& $py -m dvc init
& $py -m dvc dag
```

`dvc init` is required once because this repository previously had `dvc.yaml` but no
`.dvc` metadata. After initialization, `dvc dag` validates the four-stage graph.

## 4. Run Controlled Experiments

Use identical runtime settings for both strategies. Run FedProx first:

```powershell
& $py -m federated.simulate `
  --strategy fedprox `
  --rounds 50 `
  --clients 3 `
  --nodes node_01 node_02 node_03 `
  --partitions-root data/partitions `
  --results-root results/week9_fedprox `
  --local-epochs 5 `
  --max-windows 256 `
  --patience 5 `
  --min-delta 0.0001 `
  --round-timeout 300 `
  --database agmesh.db `
  --proximal-mu 0.1
```

Run FedAvg with the same settings and a separate results root:

```powershell
& $py -m federated.simulate `
  --strategy fedavg `
  --rounds 50 `
  --clients 3 `
  --nodes node_01 node_02 node_03 `
  --partitions-root data/partitions `
  --results-root results/week9_fedavg `
  --local-epochs 5 `
  --max-windows 256 `
  --patience 5 `
  --min-delta 0.0001 `
  --round-timeout 300 `
  --database agmesh.db `
  --proximal-mu 0.0
```

Compare the resulting summaries:

```powershell
& $py -m scripts.compare_week9_strategies `
  --fedprox (Get-ChildItem results/week9_fedprox -Recurse -Filter summary.json | Sort-Object FullName | Select-Object -Last 1).FullName `
  --fedavg (Get-ChildItem results/week9_fedavg -Recurse -Filter summary.json | Sort-Object FullName | Select-Object -Last 1).FullName `
  --output results/week9_comparison/comparison.json
```

## 5. DVC Commands

Inspect the pipeline:

```powershell
& $py -m dvc dag
```

Reproduce it after data and parameter changes:

```powershell
& $py -m dvc repro
```

The DVC pipeline assumes the raw HAI source and existing script arguments are
available. Use the controlled experiment commands above when comparing strategies
with separate MLflow experiments and result roots.

## 6. Interpretation Policy

FedProx is considered better only when the comparison uses identical data, seed,
client set, local epochs, round limit, and evaluation policy. A lower training loss
alone is insufficient; compare best/final distributed evaluation loss, convergence
round, failed-client rate, and per-node variance.

The project target is global AUROC >= 0.92. The held-out global-core evaluator is now
implemented, but the current results are below that target.

## 7. Latest Saved Experiment Analysis

The three Week 9 result artifacts are:

- FedProx: `results/week9_fedprox/20260822T072743Z/summary.json`
- FedAvg: `results/week9_fedavg/20260822T072848Z/summary.json`
- Comparison: `results/week9_comparison/comparison.json`

Both experiments used the same three HAI clients, five local epochs, 256 windows per
client, a 50-round limit, and the same convergence policy. Both completed six rounds
with zero fit failures and zero evaluation failures before patience stopping.

| Metric | FedProx | FedAvg |
|---|---:|---:|
| Best global evaluation loss | 0.215190 | 0.217728 |
| Final global evaluation loss | 0.250907 | 0.232791 |
| Rounds completed | 6 | 6 |
| Fit failures | 0 | 0 |

FedProx achieved a slightly better best loss by `0.002539`, but FedAvg achieved a
better final loss by `0.018116`. The current evidence therefore does not demonstrate
that FedProx outperforms FedAvg. Both runs show improving local fit loss alongside
worsening distributed evaluation loss, indicating overfitting or model drift under
the current small-window experiment.

The comparison plot is saved at `results/week9_comparison/convergence.png`.
The comparison JSON records `fedprox_final_loss_better: false` and
`fedprox_best_loss_better: true`.

### Held-out attack-label evaluation

The global-core evaluator merges each saved federated core with the corresponding
node-local VAE input/output heads and evaluates the complete held-out test windows.
This avoids comparing incompatible full VAE states while still measuring attack-label
behavior.

Saved evaluations:

- FedAvg: `results/week9_global_evaluation/20260822T074153Z/summary.json`
- FedProx: `results/week9_global_evaluation/20260822T074245Z/summary.json`

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Macro ROC-AUC | 0.5710 | 0.5678 |
| Macro F1 | 0.0845 | 0.0906 |
| Node 01 ROC-AUC | 0.5189 | 0.4968 |
| Node 02 ROC-AUC | 0.6429 | 0.6598 |
| Node 03 ROC-AUC | 0.5513 | 0.5467 |

Each node evaluation used 40,195 test windows, including 1,185 attack windows, with
50 bootstrap samples. The results are substantially below the project AUROC target
of 0.92. FedProx has slightly better macro F1, but FedAvg has slightly better macro
ROC-AUC in this run. The global-core model is therefore not ready for production
detector use.

## 8. Model Bundle Registration

File: `scripts/register_global_model.py`

Register a persisted global parameter bundle and its metadata in MLflow:

```powershell
& $py -m scripts.register_global_model `
  --summary results/week9_fedavg/20260822T072848Z/summary.json `
  --name GlobalAnomalyDetector `
  --tracking-uri file:./mlruns
```

This registers the shared VAE core bundle. It is not an end-to-end attack detector,
because the node-specific input/output heads remain local.

Registration completed for the selected FedAvg run:

```text
Model: GlobalAnomalyDetector
Version: 1
Source run: 20260822T072848Z
MLflow run: 4d5b5af66b2d41138b501c83a7fe5c9a
```

The durable registration record is saved at
`results/week9_comparison/registration.json`.

## 9. Remaining Work

1. Tune `proximal_mu` across `0.001`, `0.01`, and `0.1`, local epochs, learning rate,
  and shared/local-head architecture because neither strategy approaches the target.
2. Improve the federated evaluation design by retaining and versioning each node's
  local heads alongside the shared core and validating on a strictly held-out split.
3. Add the held-out attack-label metrics directly to the federated run summary and
  MLflow run, rather than keeping them only in the evaluator output.
4. Repeat both strategies across multiple seeds and report confidence intervals.
5. Investigate the feature/schema mismatch across HAI nodes; a shared latent core may
  not be sufficient to transfer discriminative anomaly features.
6. Validate `dvc repro` on a clean checkout and resolve any dataset-specific stage
  arguments before declaring the pipeline reproducibility gate complete.

## 8. Week 10 Handoff

Week 10 begins the agentic mesh layer after the federated comparison and evaluator
are complete: anomaly dispatch, triage, investigation, MITRE retrieval, and secure
agent communication.
