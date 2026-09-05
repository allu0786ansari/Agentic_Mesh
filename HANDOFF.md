# Agentic Mesh Handoff

## Scope

This repository is a local, 8 GB development implementation of privacy-preserving
industrial anomaly detection. Runtime services use Flower simulation, embedded Qdrant,
SQLite, an asyncio bus, local Ollama inference, and optional Phoenix tracing.

## Prerequisites

- Python 3.12
- A project environment at `agentvenv` or `.venv`
- Dependencies from `requirements.txt` and `requirements-dev.txt`
- Ollama for real-agent runs
- Optional Phoenix endpoint for trace collection

## Smoke Test

```powershell
$py = '.\\agentvenv\\Scripts\\python.exe'
& $py -m pytest tests/unit -q
& $py -m scripts.run_local_stack --count 5 --metrics-path logs/metrics.jsonl
```

The local stack uses the deterministic graph by default. Use `--real-agents` only when
Ollama, Qdrant data, and node API endpoints are available.

## Evaluation

```powershell
& $py -m evaluation.harness
& $py -m evaluation.chaos_test
& $py -m evaluation.ragas_eval
& $py -m evaluation.plot_metrics
& $py -m evaluation.check_success_criteria
```

The final checker intentionally returns `PASS_WITH_GAPS` when final FL or resource
artifacts are absent. It does not convert missing measurements into successful claims.

## Trigger a Federated Round

```powershell
& $py -m federated.simulate `
  --strategy fedprox `
  --rounds 1 `
  --clients 3 `
  --nodes node_01 node_02 node_03 `
  --partitions-root data/partitions `
  --results-root results/manual_fedprox `
  --local-epochs 1 `
  --max-windows 256
```

The input data is intentionally local or DVC-managed and is excluded from Git.

## Trigger an Investigation

For the deterministic local graph:

```powershell
& $py -m scripts.run_local_stack --count 1 --node-id node_01 --score 0.9
```

For real services, start the node APIs and Ollama, then pass `--real-agents`,
`--qdrant-path`, `--database`, and `--node-urls` to the same command.

## Add a Simulated Node

1. Add a node partition and a metadata JSON file under `data/partitions/<node_id>`.
2. Register the metadata with `storage.sqlite_schema.seed_registry()`.
3. Add the node to the `--nodes` list used by the federated simulation.
4. Select `vae` for time-series data or `isolation_forest` for tabular data.
5. Keep the node data in DVC or local storage; do not commit binaries to Git.

## Security and Operations

JWT validation is implemented in `agents/auth.py`; code-level TLS helpers are in
`agents/tls.py`; local model access is in `agents/llm.py`; Phoenix integration is in
`agents/tracing.py`. Never commit `.env`, private keys, certificates, databases, model
weights, Qdrant stores, or raw industrial telemetry.

## Known Limitations

- Final FL AUROC/F1, FedProx comparison, cumulative epsilon, and 8 GB resource
  benchmark require a fresh environment-backed run and are reported as gaps when absent.
- The available Week 14 data lacks stable scenario-level MITRE ground-truth IDs.
- The local RAGAS-compatible evaluator uses deterministic token-hash embeddings; a
  production run should use the indexed MITRE embedding model.
- The asyncio bus is process-local. Replace it with Kafka/NATS for multi-process scale.

## Production Scaling Path

For production, replace Flower simulation with a managed federated deployment, the
asyncio bus with Kafka or NATS, SQLite with PostgreSQL, embedded Qdrant with a managed
Qdrant service, and JSONL plus Phoenix with Prometheus/OpenTelemetry-backed observability.
Preserve JWT, mTLS, local inference, DVC lineage, and the evaluation gates during that
transition.
