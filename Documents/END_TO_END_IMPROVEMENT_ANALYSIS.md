# End-to-End Improvement Analysis

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics
**Analysis date:** 2026-08-24
**Scope:** Beginning-to-end review of data, edge inference, federated learning, agent orchestration, security, observability, evaluation, and delivery.

## Executive Finding

The repository has a functional foundational prototype. The current baseline is **65 tests
passing** and Ruff clean after mechanical unused-import cleanup. The highest opportunity is
not more scaffolding; it is replacing proxy evaluation evidence with measured end-to-end
results and making the runtime contracts production-like under failure.

The Week 16 gate currently reports `PASS_WITH_GAPS`. This is correct. Missing evidence is
not a software failure, but it prevents defensible claims for final FL quality, privacy
budget, and the 8 GB runtime target.

## Measured Baseline

| Area | Current evidence | Interpretation |
|---|---|---|
| Automated tests | 65 passed, 23 dependency warnings | Good regression baseline; integration depth is still limited |
| Static quality | Ruff clean after unused-import fixes | Formatting/type checks should remain CI gates |
| Week 14 detection | Detection rate 1.0; false-positive rate 0.0; mean latency 1297.5 ms | Useful smoke evidence, but score is heuristic and root-cause mapping is conservative |
| Week 15 chaos | Four scenarios passed | Local failure state machine works; real API cancellation/retry load is untested |
| Week 15 retrieval | 50 pairs; recall 1.0; precision 0.9925 | Embedded retrieval path is functional; evaluator uses deterministic token-hash embeddings |
| Local data footprint | 3,764,161,815 bytes across 87 data files | Data must remain DVC/object-storage managed, never Git-managed |
| Generated artifacts | 136,622,425 bytes across checkpoints/results/Qdrant/MLflow | Reproducible outputs need manifests and retention rules |

## Highest-Value Improvements

### 1. Establish a trustworthy model benchmark

**Problem:** Week 14 uses a label-informed heuristic score. Its perfect detection result
cannot validate the trained VAE or Isolation Forest models.

**Improvement:** Create one canonical evaluation pipeline per model type:

1. Load only the training-fitted scaler and checkpoint.
2. Score the untouched test partition without using labels in the score.
3. Select thresholds on a validation split only.
4. Report AUROC, AUPRC, F1, precision, recall, confusion matrix, and calibration.
5. Store dataset hash, checkpoint hash, parameters, seed, and environment in JSON.

**Acceptance:** Per-node and pooled HAI/Modbus results are reproducible from one command;
no scoring function reads the label column.

### 2. Produce the missing FL and privacy evidence

**Problem:** The final gate cannot measure global AUROC/F1, FedProx versus FedAvg, or
cumulative epsilon.

**Improvement:** Add a final comparison runner that executes both strategies on the same
node order, seed, rounds, timeout, and evaluation partition. Persist:

- Per-round train/evaluation loss
- Global AUROC/F1
- FedProx and FedAvg convergence curves
- Delta and confidence interval over repeated seeds
- Per-node and cumulative epsilon
- Straggler and dropped-client counts

**Acceptance:** `evaluation/federated_final_results.json` contains all fields needed for
criteria 1-3 and the checker passes those criteria only from that artifact.

### 3. Make the data lineage executable

**Problem:** `dvc.yaml` references scripts and paths that are not uniformly present or
replayed in the current local state. The 3.7 GB dataset is correctly excluded from Git,
but a reviewer cannot yet reproduce the exact result from a clean checkout without data
remote configuration.

**Improvement:**

- Add DVC remote documentation and a small public smoke fixture.
- Ensure every DVC stage command exists and runs from a clean checkout.
- Record source checksums, schema version, row counts, partition seed, and code revision.
- Separate public metadata from restricted industrial captures.
- Add a `make`/PowerShell equivalent for `prepare -> partition -> train -> evaluate`.

**Acceptance:** Clean checkout plus documented DVC remote reproduces metadata and metrics;
no raw industrial data is required in GitHub Actions.

### 4. Harden agent failure behavior

**Problem:** The current graph, bus, and chaos harness prove local state transitions, but
real investigator HTTP failures, cancellation, retry budgets, and durable session state
are not exercised together.

**Improvement:**

- Add an explicit investigation session table with idempotency key and terminal status.
- Bound every node call with connect/read/total timeouts.
- Retry only transient failures with exponential backoff and jitter.
- Persist each hop before advancing the graph.
- Make dispatcher actions idempotent using `session_id`.
- Add an integration test with a fake node API that fails before, during, and after a hop.

**Acceptance:** Replaying an alert cannot dispatch twice; every session reaches a terminal
state; retries and breaker transitions are visible in metrics.

### 5. Close the security evidence loop

**Problem:** JWT, TLS helpers, and local Ollama configuration exist, but criteria 7-9 are
currently mostly source-presence claims rather than runtime evidence.

**Improvement:** Add a security integration suite that verifies expired/wrong-audience JWT,
certificate hostname mismatch, plaintext endpoint rejection, external LLM URL rejection,
redacted logs, and Phoenix trace correlation by session ID. Add a secret scanner to CI.

**Acceptance:** CI proves authenticated encrypted calls and fails on credentials or private
keys in tracked files.

### 6. Make metrics authoritative

**Problem:** Dashboard generation has a deterministic fallback when logs are absent. This
is useful for a demo but can be mistaken for measured operational data.

**Improvement:** Put a `data_source` field in every dashboard/report (`runtime_log` or
`fallback_fixture`), validate required metric schemas, and export a metrics manifest with
event counts and time range. Add event producers for dispatch latency, circuit-breaker
activation, embedding publication, AUROC, epsilon, and stragglers.

**Acceptance:** A report using fallback data is visibly marked and cannot satisfy a final
production evidence gate.

### 7. Improve operational packaging

**Problem:** The local asyncio architecture is appropriate for an 8 GB prototype but the
process boundary and deployment story are still manual.

**Improvement:** Keep the local mode, then add a production profile with Kafka/NATS,
PostgreSQL, managed Qdrant, OpenTelemetry/Prometheus, health/readiness endpoints, bounded
queues, and graceful shutdown. Document resource limits and service ownership.

**Acceptance:** Local and production profiles share message schemas and evaluation
contracts; transport changes do not change agent behavior.

## Recommended Execution Order

### Phase A: Evidence integrity

- Replace label-informed scoring with checkpoint-backed scoring.
- Add final FL comparison and epsilon artifact.
- Add dataset/checkpoint/environment manifests.
- Make the final checker require these artifacts.

### Phase B: Runtime correctness

- Add durable investigation sessions and idempotent dispatch.
- Add fake-node integration tests for timeout, retry, breaker, cancellation, and recovery.
- Add metrics schema validation and session trace correlation.

### Phase C: Security and reproducibility

- Add JWT/mTLS/external-LLM negative tests.
- Configure DVC remote and public smoke fixture.
- Run clean-checkout replay and store signed/hash-linked result manifests.

### Phase D: Performance and production path

- Profile memory and latency on the 8 GB target.
- Tune batching, model size, Qdrant payloads, and queue bounds.
- Add transport-adapter interfaces for Kafka/NATS/PostgreSQL/managed Qdrant.
- Repeat the full evaluation under realistic concurrent load.

## Definition of Improved Results

The project should be considered meaningfully improved when:

- Model metrics are produced without label leakage.
- FL comparison and epsilon accounting are machine-readable and reproducible.
- The final checker returns `PASS`, not `PASS_WITH_GAPS`.
- Every investigation is durable, idempotent, authenticated, traceable, and terminal.
- Dashboards identify measured versus fixture data.
- A clean checkout can reproduce the published metrics using DVC-managed data.
- The 8 GB resource claim is backed by a recorded benchmark, not an assumption.
