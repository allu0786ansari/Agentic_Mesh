# Phase 3, Week 8 — Laptop Runbook (edge runtime, evidence, bus, fusion)

Week 8 is built and tested in the cloud workspace. The design is in
`docs/adr/0009-edge-runtime-week8.md`.

**This round on your laptop:**

1. Train and save the two development global models (FedProx, ε = 3, seed 11).
2. Replay each site's validation split through the edge runtime.
3. Record detection latency.

**What you'll need:** about 25 minutes on 2 CPU cores. Do not run Ollama at the same
time (WR-4).

**What you'll commit:**

- the two saved models;
- each edge run's `summary.json` and its calibration files;
- the latency report;
- the training artifacts and privacy ledgers.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup-fl
make check            # expect 245 passed
git status --short    # must print nothing
```

## Step 2 — Train and save the development models (about 20 minutes)

```bash
make edge-model SITE=site_b     # about 6 min -> results/models/site_b__fedprox__eps3/seed_11/
make edge-model SITE=site_a     # about 11 min -> results/models/site_a__fedprox__eps3/seed_11/
```

Each prints one line. In the cloud the Site B line was `AUROC 1.000 ... max ε
3.000`. Check that max ε is 3.000 on both.

## Step 3 — Replay validation through the edge runtime (about 2 minutes)

```bash
make edge SITE=site_b SPLIT=val
make edge SITE=site_a SPLIT=val
make latency
```

Each `make edge` prints the alert count, detection latency p50 and p95, and the
path of its `summary.json`. Look for these in each summary:

- **`streams.*.window_fpr_normal`:** about 0.01 on every node's `val_normal`.
- **`stream_vs_batch`:** `scored_mask_mismatches` 0. `alert_decision_mismatches`
  should be 0 or a handful, at the threshold (ADR 0009 §8).
- **`detection_latency_ms.p95`:** far below the 1,000 ms target. The cloud gave
  1.2 ms on Site B.
- **Site B `fusion`:** a descriptive table only; nothing is chosen from it.

## Step 4 — Commit and push

`results/` is ignored by default, so add the files with `-f`:

```bash
git add -f results/models results/federated/week8_dev results/privacy results/latency \
  results/edge/*/summary.json results/edge/*/calibration_*.json
git commit -m "Week 8: dev global models, edge replay on validation (both sites), detection latency"
git push origin main
```

The SQLite files (`storage.db`, `evidence_*.db`) stay local. They can be rebuilt by
running `make edge` again.
