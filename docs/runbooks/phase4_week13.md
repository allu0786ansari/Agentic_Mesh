# Phase 4, Week 13 — Laptop Runbook (two short jobs; the analysis runs in the cloud)

Read `docs/adr/0017-x3-x6-week13.md`.

X3, X4, X5, X6 and the robustness suite are computed in the cloud from committed
artifacts. The laptop provides two things.

**Time needed:** about 80 minutes. **Stage isolation:** stop Ollama first.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make check
```

## Step 2 — Headline models, five seeds, saved (about 75 minutes)

```bash
pkill -f "ollama serve"; pkill llama-server
make x3-models
```

- **What it does:** it retrains FedProx ε 3 on both sites for seeds 11, 23, 37, 41
  and 53, one process per run.
- **What it saves:** each model goes to `results/models/x3/`.
- **The check:** it then verifies each rerun reproduces its X1 artifact exactly.
- **Expect** it to end with `verify: 0 problem(s) over 10 models`. If it lists
  problems, send me the output and do not go on.

## Step 3 — Export the mesh sessions for X4 (seconds)

```bash
make x4-export
```

It writes `sessions.jsonl` next to every `results/mesh/<run>/summary.json`. The file
holds no evidence text or prompt.

## Step 4 — Commit and push

```bash
git add -f results/models/x3 results/x13_rerun results/mesh/*/sessions.jsonl
git commit -m "Week 13: five-seed headline models (verified) and mesh session exports"
git push origin main
```
