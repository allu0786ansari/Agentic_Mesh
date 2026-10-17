# Phase 3, Week 11 — Laptop Runbook (multi-process stack over mTLS, JWT, traces; G3 inputs)

Read `docs/adr/0014-process-boundaries-security-tracing-week11.md` first.

In the cloud, `rule_only` already ran across process boundaries on both sites, with
results identical to the in-process runs. On the laptop you run the two modes that
need the encoder and Ollama, over mTLS. These runs are Gate G3's inputs (ADR 0014
§9).

**Time needed:** about 1.5 hours, most of it Site B `full`.

**Stage isolation:** nothing else heavy runs at the same time.

## Step 0 — Finish Week 10d first

If Week 10d's Step 5 (`llm_no_retrieval` on both sites, about 4 hours) has not run
and been pushed yet, do it before this runbook.

## Step 1 — Update, re-lock, check

Week 11 adds core dependencies: FastAPI, uvicorn, PyJWT, cryptography, OpenTelemetry
and OpenInference.

```bash
cd ~/Agentic_Mesh
git pull origin main
uv lock
make setup-kb
make check                 # expect 339 passed
```

## Step 2 — Development certificates (once)

```bash
make certs                 # -> certs/dev (git-ignored; private keys mode 0600)
git status --short certs   # must print nothing
```

## Step 3 — A lean Ollama with the prompt cache off (as in Week 10d)

```bash
pkill -f "ollama serve"; pkill llama-server; sleep 3
free -h                    # "available" at least 2 GB
LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
  ollama serve > ~/ollama.log 2>&1 &
sleep 3
```

## Step 4 — Site B over mTLS (about 65 minutes)

```bash
make stack-up SITE=site_b          # three node API processes; prints "up" for b1, b2, b3
make mesh SITE=site_b MODE=retrieval_only NODES=https
make mesh SITE=site_b MODE=full NODES=https
make stack-down
```

## Step 5 — Site A over mTLS (about 15 minutes)

Restart Ollama as in Step 3 first, then:

```bash
make stack-up SITE=site_a          # four node API processes
make mesh SITE=site_a MODE=retrieval_only NODES=https
make mesh SITE=site_a MODE=full NODES=https
make stack-down
```

**What to expect:**

- every run ends with `220 sessions` (Site B) or `45 sessions` (Site A);
- the summary has `"nodes": "https"` and `"traces": {... "problem_count": 0}`.

If a node API fails to start, its log is `results/stack/<node>.log`.

## Step 6 — Check the traces

```bash
uv run python -m agmesh.evaluation.checker --traces $(ls -d results/mesh/*__val__* | tail -4)
```

There should be four `OK` lines.

## Step 7 — Commit and push

```bash
git add uv.lock
git add -f results/mesh/*/summary.json results/mesh/*/events.jsonl results/traces
git status --short | grep -E 'certs/|\.env$|\.key$'     # must print nothing
git commit -m "Week 11: retrieval_only and full over mTLS (both sites), traces"
git push origin main
```

After you push, I check the runs against ADR 0014 §9 and write Gate G3 (ADR 0015).
