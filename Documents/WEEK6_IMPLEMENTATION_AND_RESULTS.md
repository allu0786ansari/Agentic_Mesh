# Week 6 Implementation and Results

**Status:** Implemented and fully validated

## Scope

Week 6 unifies local detector outputs into fixed-size Insight Embeddings and provides local persistence, publishing, and query interfaces.

## Implemented Components

- `edge/embedding/insight_embedding.py`
  - VAE embeddings from encoder `mu` and `logvar`.
  - Isolation Forest embeddings from anomaly score and scaled tabular features.
  - Fixed 128 dimensions.
  - L2 normalization.
  - Finite-value validation.
- `edge/bus.py`
  - Asyncio topic bus used for `embeddings.<node_id>` topics.
  - `EmbeddingPublisher` with bounded exponential-backoff retries.
- `scripts/run_week6_inference.py`
  - Loads saved VAE and Isolation Forest checkpoints.
  - Scores test partitions.
  - Selects thresholds from normal validation data.
  - Generates embeddings only for anomalous records.
  - Stores vectors and payloads in embedded Qdrant.
  - Creates a unique Qdrant collection for every run.
  - Publishes embedding messages to the asyncio bus.
  - Writes run-scoped summary output with per-node counts and topic names.
- `api/node_api.py`
  - `GET /health`
  - `POST /embeddings`
  - `POST /embeddings/search`
- `tests/unit/test_week6_embeddings.py`
  - Embedding dimensions, normalization, and finite-value tests.
- `tests/unit/test_week6_api_and_bus.py`
  - API health, storage, search, and bus publication tests.

## Smoke Test

Command:

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.run_week6_inference `
  --vae-nodes node_03 `
  --iforest-nodes node_06 `
  --vae-checkpoint-root checkpoints/20260821T024039Z `
  --iforest-checkpoint-root checkpoints/20260821T214400Z `
  --results-root results/week6_smoke `
  --qdrant-path qdrant_data_week6_smoke
```

Result:

```text
Run ID: 20260821T215628Z
Generated anomaly embeddings: 3,093
Qdrant path: qdrant_data_week6_smoke
```

The smoke test confirms the end-to-end path from checkpoint loading to anomaly scoring, embedding generation, Qdrant persistence, and bus publication.

Final hardening smoke test:

```text
Run ID: 20260821T220815Z
Nodes: node_03 and node_06
Embeddings: 3,093
Qdrant collection: insight_embeddings_20260821T220815Z
Topics: embeddings.node_03, embeddings.node_06
```

## Full Six-Node Command

```powershell
$py = 'agentvenv\Scripts\python.exe'
& $py -m scripts.run_week6_inference `
  --vae-checkpoint-root checkpoints/20260821T024039Z `
  --iforest-checkpoint-root checkpoints/20260821T214400Z `
  --scalers-root results/scalers `
  --results-root results/week6_inference `
  --qdrant-path qdrant_data_week6
```

## Validation Status

- Week 6 modules compile successfully.
- Focused Week 6, API, bus, and Qdrant tests: 6 passed.
- Full repository regression suite: 27 passed.
- Run-scoped Qdrant collections and per-node summary counts validated.

## Analysis of Latest Week 6 Run

The latest saved run, `20260821T222733Z`, provides concrete evidence that the Week 6 pipeline is functionally working end-to-end. The system produced a run-scoped Qdrant collection named `insight_embeddings_20260821T222733Z` and published embeddings across all six nodes via topics such as `embeddings.node_01`, `embeddings.node_02`, `embeddings.node_03`, `embeddings.node_04`, `embeddings.node_05`, and `embeddings.node_06`.

The saved summary confirms:

- Total generated embeddings: 26,925
- Per-node totals: node_01 = 2,427; node_02 = 19,183; node_03 = 1,960; node_04 = 1,222; node_05 = 1,000; node_06 = 1,133
- Model mix: 23,570 VAE embeddings and 3,355 Isolation Forest embeddings
- Score range: 0.0132 to 6.3348
- Threshold range across generated records: 0.0132 to 0.1724

This is a meaningful result because it shows the inference pipeline is not simply generating a few test records; it is producing a large, structured embedding population across all nodes and storing it in Qdrant in a consistent, queryable form. The system is therefore successfully transitioning from local thresholded scores into a shared representation layer, which is exactly what the later triage and investigation agents need.

The actual label distribution inside the saved audit log is also informative:

- Total labeled positives (`label == 1`): 1,521
- Total labeled negatives (`label == 0`): 25,404
- The distribution is highly imbalanced, which is common in anomaly-detection settings but also means the run needs careful calibration.

Per-node behavior shows a major operational insight:

- node_02 generated the overwhelming majority of records (19,183), yet only 625 were labeled as positives; this suggests the thresholding rule is very permissive on that node and may be creating a large number of low-signal embeddings.
- node_05 had the highest positive ratio among the six nodes at 254/1000 = 25.4%, while node_02 was much lower at 625/19183 = 3.3%.
- node_01, node_03, and node_04 show moderate positive proportions, but the spread suggests thresholds are not yet balanced across models and nodes.

This is the main conclusion from the latest run: the pipeline is technically healthy, but the threshold policy is not yet calibrated well enough to produce a consistent anomaly signal quality across nodes. In other words, the system is generating embeddings successfully, but some nodes are likely producing too many weak or noisy anomalies, which could reduce retrieval quality and downstream investigation precision.

The saved run therefore validates the engineering pipeline, while identifying the largest remaining risk: anomaly threshold calibration and embedding relevance quality. The system is ready for retrieval workflow integration, but it is not yet proven that the generated embedding space is strongly discriminative enough for high-confidence triage decisions.

## Improvements Needed

1. Add quantitative embedding-quality evaluation using labeled normal vs attack samples.
   - Measure AUROC, F1, precision/recall, and retrieval separability.
   - Validate that embeddings cluster correctly for similar anomalies while remaining separated from normal traffic.

2. Strengthen the embedding payload schema.
   - Include explicit metadata such as node ID, timestamp, model type, anomaly score, threshold, and record index.
   - Standardize payload contracts between inference, API, and agent consumers.

3. Improve threshold calibration and anomaly selection logic.
   - Move beyond percentile-based heuristics by validating thresholds on a held-out calibration set.
   - Add adaptive thresholding and false-positive monitoring to reduce noisy embeddings.

4. Add retrieval quality checks beyond raw storage.
   - Evaluate whether nearest-neighbor queries return the expected historical anomalies and not unrelated events.
   - Measure similarity relevance across normal and attack windows.

5. Harden the API and operational workflow.
   - Add authentication and authorization for the node API before production use.
   - Add logging, retry metrics, and drift detection for Qdrant and bus health.

6. Improve explainability and downstream usability.
   - Attach feature-level or event-level explanation summaries to each embedding payload.
   - This will help investigators understand why an event was considered anomalous instead of just seeing a vector.

Overall, the Week 6 milestone is a meaningful technical success: the system has moved from isolated model outputs to a unified, searchable anomaly representation with end-to-end execution evidence. The most important next step is not to add more plumbing, but to validate that the embedding space is genuinely useful for distinguishing real anomalous patterns from normal industrial behavior at scale.

## Follow-up Work

- Add VAE and Isolation Forest embedding payload schemas to the API contract.
- Add embedding similarity quality evaluation across normal and attack events.
- Add authentication to the node API in the later security week.
- Add feature-level and event-level explanations to embedding payloads.
