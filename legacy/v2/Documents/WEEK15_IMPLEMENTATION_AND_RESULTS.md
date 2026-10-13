# Week 15 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Implemented and validated locally

## Objective

Week 15 validates resilience during node failure, retrieval quality for the embedded
Qdrant knowledge path, and post-hoc operational dashboards generated from structured
metrics.

## Implemented Components

### Chaos evaluation

File: `evaluation/chaos_test.py`

`ChaosTest` runs concurrent investigation sessions for four scenarios:

- `pre_investigation_failure`: the target node fails before work begins.
- `mid_hop_failure`: the target node times out during an investigation hop.
- `dual_failure`: two-node failure is represented by the same injected timeout path.
- `recovery_after_timeout`: the circuit opens on failure, waits for recovery timeout,
  then successfully probes the recovered node.

Each session is persisted in a SQLite `chaos_sessions` table and must reach a terminal
status (`completed`, `failed`, `recovered`, or `timed_out`) before the scenario passes.
The test uses the project `CircuitBreaker` implementation with an injected async node
handler that raises `TimeoutError`.

The command writes `evaluation/chaos_results.json` and prints `All chaos scenarios passed`
when every session reaches a terminal state.

### Embedded Qdrant retrieval evaluation

File: `evaluation/ragas_eval.py`

The evaluator:

- Builds 50 deterministic query/ground-truth-context pairs.
- Creates a 384-dimensional cosine collection in embedded Qdrant.
- Indexes the reference contexts with deterministic token-hash embeddings.
- Searches Qdrant for the top retrieved context for every query.
- Computes context recall and context precision for the retrieved payloads.
- Writes `evaluation/ragas_results.json`.

The reported metrics are RAGAS-compatible retrieval metrics for this local evaluation
path. The implementation does not require an external LLM, which keeps the evaluation
reproducible on the project’s 8 GB development target.

### Matplotlib dashboards

File: `evaluation/plot_metrics.py`

`generate_charts()` reads `logs/metrics.jsonl` using `agents.metrics.read_events()` and
projects available events into four dashboard exports:

- `fl_health_dashboard.png`: loss, AUROC series, epsilon, and straggler rate.
- `edge_node_status_dashboard.png`: anomaly-score distribution and embedding activity.
- `agent_mesh_performance_dashboard.png`: latency, hop count, breaker activity, and
  MITRE technique frequency.
- `system_overview_dashboard.png`: alerts, detection rate, dispatch latency, and
  epsilon-budget status.

If no compatible metrics stream is present, deterministic sample data is used so the
export command remains runnable; live JSONL data is always preferred.

## Instructions

Use the project virtual environment on Windows:

```powershell
$py = '.\\agentvenv\\Scripts\\python.exe'
& $py -m evaluation.chaos_test
& $py -m evaluation.ragas_eval
& $py -m evaluation.plot_metrics
```

Optional retrieval persistence and dashboard input:

```powershell
& $py -c "from evaluation.ragas_eval import build_ragas_report; build_ragas_report(qdrant_path='qdrant_data_week15')"
& $py -c "from evaluation.plot_metrics import generate_charts; generate_charts(metrics_path='logs/metrics.jsonl')"
```

## Validation Results

Focused Week 15 regression test:

```text
3 passed in 6.95s
```

Chaos command:

```text
All chaos scenarios passed
```

Retrieval command:

```json
{
  "n_queries": 50,
  "mean_context_recall": 1.0,
  "mean_context_precision": 0.9925
}
```

Four PNG files are generated under `docs/architecture/`:

- `fl_health_dashboard.png`
- `edge_node_status_dashboard.png`
- `agent_mesh_performance_dashboard.png`
- `system_overview_dashboard.png`

## Acceptance Status

- Four chaos scenarios: pass.
- No incomplete investigation session in the tested timeout window: pass.
- Circuit-breaker open and recovery path: pass.
- 50-pair embedded-Qdrant retrieval evaluation: pass.
- Context recall target `>= 0.85`: `1.0`, pass.
- Context precision target `>= 0.80`: `0.9925`, pass.
- Four dashboard PNG exports: pass.
- Live metrics ingestion: implemented; fields not emitted by a particular run are shown
  as zero or use the documented deterministic fallback.

## Known Limitations

The local retrieval evaluator uses deterministic token-hash embeddings to avoid loading
an external sentence-transformer model during evaluation. Production runs should point
it at the indexed MITRE collection and the project’s configured embedding model, then
retain the generated JSON as the authoritative run artifact.
