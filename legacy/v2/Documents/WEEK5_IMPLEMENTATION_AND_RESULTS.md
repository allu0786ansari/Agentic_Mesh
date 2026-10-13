# Week 5 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Implemented and validated

**Run ID:** `20260821T214400Z`

## 1. Objective

Week 5 extends local anomaly detection from the HAI VAE nodes to the CIC Modbus network-flow nodes:

- Nodes 01-03: HAI time-series VAE
- Node 04: CIC Modbus IED1A Isolation Forest
- Node 05: CIC Modbus IED1B Isolation Forest
- Node 06: CIC Modbus SCADA HMI Isolation Forest

## 2. Implemented Features

### Tabular partition loader

File: `edge/tabular_loader.py`

- Loads train/test Parquet partitions.
- Validates the `label` column.
- Preserves feature-column order.
- Rejects non-numeric and non-finite values.
- Fits `StandardScaler` only on normal training rows.
- Reuses the fitted scaler for validation and test scoring.

### Isolation Forest detector

File: `edge/models/isolation_forest.py`

- Uses scikit-learn `IsolationForest`.
- Trains only on normal training rows.
- Uses anomaly score `-decision_function`, where higher means more anomalous.
- Saves model, scaler, feature order, and metadata in a `.joblib` checkpoint.
- Validates finite anomaly scores.

### Metadata-driven model selector

File: `edge/models/model_selector.py`

- Reads `data/partitions/<node_id>/meta.json`.
- Routes tabular nodes to Isolation Forest.
- Rejects unsupported model types.
- Prevents applying the wrong model family to a node.

### Embedded Qdrant insight store

File: `edge/embedding/insight_embedding.py`

- Stores fixed 128-dimensional insight vectors.
- Validates dimensions and finite values.
- Supports `store()` and `search()`.
- Normalizes arbitrary string IDs to deterministic UUID5 values for Qdrant compatibility.

### Week 5 runner

File: `scripts/train_modbus_iforest.py`

The runner performs:

1. Per-node normal-training scaler fitting.
2. Isolation Forest training.
3. Run-isolated checkpoint persistence.
4. Validation-only threshold selection.
5. Test scoring.
6. ROC-AUC, PR-AUC, precision, recall, F1, and FPR calculation.
7. Per-node JSON results and aggregate summary persistence.
8. Optional Qdrant insight storage.

## 3. Training Configuration

```text
Model: Isolation Forest
Nodes: node_04, node_05, node_06
Estimators: 100
Seed: 42
Scaler: StandardScaler fit on normal training rows only
Threshold: 95th percentile of final 20% normal validation rows
Parallel tree jobs: -1
```

## 4. Run Command

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.train_modbus_iforest `
  --partitions-root data/partitions `
  --results-root results/modbus_iforest `
  --checkpoint-root checkpoints `
  --nodes node_04 node_05 node_06 `
  --n-estimators 100
```

Optional Qdrant persistence:

```powershell
& $py -m scripts.train_modbus_iforest `
  --partitions-root data/partitions `
  --results-root results/modbus_iforest `
  --checkpoint-root checkpoints `
  --nodes node_04 node_05 node_06 `
  --n-estimators 100 `
  --store-insights
```

## 5. Saved Artifacts

```text
results/modbus_iforest/20260821T214400Z/summary.json
results/modbus_iforest/20260821T214400Z/node_04.json
results/modbus_iforest/20260821T214400Z/node_05.json
results/modbus_iforest/20260821T214400Z/node_06.json
checkpoints/20260821T214400Z/node_04.joblib
checkpoints/20260821T214400Z/node_05.joblib
checkpoints/20260821T214400Z/node_06.joblib
```

Each checkpoint includes a SHA-256 hash in the corresponding result record.

## 6. Dataset Scope

| Node | Feature count | Train rows | Test rows | Attack test rows |
|---|---:|---:|---:|---:|
| node_04 | 17 | 55,344 | 24,679 | 3,681 |
| node_05 | 17 | 59,106 | 15,843 | 3,681 |
| node_06 | 17 | 54,703 | 18,834 | 3,681 |

## 7. Evaluation Results

| Node | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| node_04 | 0.7251 | 0.2788 | 0.1432 | 0.0475 | 0.0714 | 0.0499 |
| node_05 | 0.8969 | 0.6687 | 0.2540 | 0.0690 | 0.1085 | 0.0613 |
| node_06 | 0.7278 | 0.3680 | 0.1871 | 0.0576 | 0.0881 | 0.0608 |

## 8. Interpretation

- Node 05 is the strongest Week 5 detector by ROC-AUC and PR-AUC.
- Nodes 04 and 06 have useful anomaly ranking quality.
- Recall is low for every node at the default 95th-percentile threshold.
- The model ranking quality is stronger than the thresholded alert quality.
- Threshold calibration, temporal aggregation, and event-level evaluation are required before operational use.

These results are a network-flow anomaly-ranking baseline, not a final production result.

### Latest persisted node 04-06 analysis

The latest run with 300 trees and Qdrant persistence is saved at:

`results/modbus_iforest/20260822T062046Z/summary.json`

| Node | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | TP | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node_04 | 0.7177 | 0.2738 | 0.1392 | 0.0486 | 0.0721 | 0.0527 | 179 | 1,107 |
| node_05 | 0.8970 | 0.6704 | 0.2328 | 0.0625 | 0.0985 | 0.0623 | 230 | 758 |
| node_06 | 0.7257 | 0.3656 | 0.1767 | 0.0551 | 0.0841 | 0.0624 | 203 | 946 |

The run evaluates 59,356 test rows and 11,043 attack rows across the three nodes.
Node 05 remains the strongest detector by a wide margin, with ROC-AUC near 0.90 and
PR-AUC near 0.67. Nodes 04 and 06 remain materially weaker, around ROC-AUC 0.72.
The 95th-percentile threshold keeps alert precision low and recall below 6.3% on all
nodes. This means the models rank attacks better than they produce usable alerts.

Qdrant persistence was enabled and each node result records a 128-dimensional insight
embedding collection. The result proves checkpointing, evaluation, and persistence;
it does not yet prove retrieval relevance or operational alert quality.

### Future improvements before moving beyond the Modbus baseline

1. Select thresholds on a held-out validation set using fixed FPR targets (1%, 5%,
  and 10%) and publish the complete precision-recall threshold curves.
2. Add event-level evaluation by grouping adjacent anomalous rows and report event
  recall, event precision, detection delay, and missed events.
3. Run seeds 42, 123, and 2024 with confidence intervals; node 04 and node 06 need
  stability analysis before being used downstream.
4. Remove or justify `window_start` as a feature because it is a temporal index and
  can create dataset-specific leakage or drift.
5. Compare StandardScaler with robust scaling and tune tree count, contamination,
  and feature subsets per node using validation data only.
6. Add feature-level explanations and attack/capture breakdowns for investigator
  agents, plus false alarms per hour and cooldown/consecutive-alert rules.
7. Validate Qdrant retrieval with known-neighbor tests, payload completeness checks,
  collection counts, and similarity relevance metrics rather than storage success
  alone.

## 9. Tests

Focused Week 5 tests:

```text
4 passed
```

Full repository suite after Week 5:

```text
23 passed
```

Test files:

- `tests/unit/test_week5_isolation_forest.py`
- `tests/unit/test_week5_qdrant_store.py`

## 10. Known Limitations

- Only one seed was used for this initial Week 5 run.
- Bootstrap confidence intervals are not yet included in the Isolation Forest runner.
- Event-level attack metrics are not yet included in the Week 5 runner.
- Qdrant persistence is implemented and tested, but the main run was executed without `--store-insights`.
- Threshold selection at the 95th percentile produces low recall.
- Feature-level explainability and attack-type breakdown are not yet recorded.

## 11. Next Week Handoff

Week 6 should implement:

1. Unified 128-dimensional Insight Embeddings for VAE and Isolation Forest.
2. Embedding generation from anomaly score, latent representation, and model metadata.
3. Async publisher through `edge/bus.py`.
4. Qdrant storage and similarity search for all six nodes.
5. Local inference mode for saved checkpoints.
6. FastAPI node endpoints: health, inference, and embedding query.
7. Event-level evaluation and detection-delay metrics.
8. Multi-seed validation for the selected Isolation Forest configuration.

## 12. Deferred Week 5 Improvements

These improvements are intentionally recorded for later implementation and are
not blockers for proceeding to Week 6.

### Threshold and alert quality

- Replace the single 95th-percentile threshold with validation-selected thresholds
  at fixed FPR targets of 1%, 5%, and 10%.
- Add precision-recall threshold curves.
- Report alert volume and false alarms per hour.
- Add configurable consecutive-alert and cooldown rules.

### Event-level evaluation

- Group adjacent anomalous rows into attack episodes.
- Report event recall, event precision, detection delay, and missed-event count.
- Compare row-level, window-level, and event-level metrics.

### Statistical validation

- Repeat nodes 04-06 with seeds 42, 123, and 2024.
- Report mean +/- standard deviation for ROC-AUC, PR-AUC, F1, recall, and FPR.
- Add bootstrap confidence intervals.
- Preserve every replicate under a unique run ID.

### Model and feature analysis

- Compare Isolation Forest estimators at 100, 300, and 500 trees.
- Evaluate contamination settings using validation data only.
- Add feature-importance and permutation-importance reports.
- Test whether `window_start` should be excluded as a temporal index feature.
- Compare StandardScaler with robust scaling for heavy-tailed network features.

### Qdrant integration

- Execute the Week 5 runner with `--store-insights` in a clean Qdrant directory.
- Verify stored vectors, payload fields, UUID stability, and similarity retrieval.
- Record the Qdrant collection count and run ID in the Week 5 summary.

### Publication evidence

- Record Python, scikit-learn, NumPy, pandas, and Qdrant versions.
- Store feature order, scaler parameters, partition hashes, and checkpoint hashes.
- Include attack-type and capture-level breakdowns.
- Keep the current `20260821T214400Z` run immutable as the first Week 5 baseline.
