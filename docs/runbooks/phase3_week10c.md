# Phase 3, Week 10c — Laptop Runbook (ADR 0013 §6: lean Ollama, Site A `full`, `llm_no_retrieval` once more)

The 17 October runs ran out of memory: `llama-server` held 5.0 GiB of WSL's
5.8 GiB, and only 160 MiB was left. ADR 0013 §6 sets the conditions below and a
stopping rule. Your machine stays as it is.

Each run now prints one line per incident, ending with the available memory and the
LLM server's memory. Watch those two numbers.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make check                 # expect 292 passed
```

## Step 2 — A fresh, lean Ollama (repeat before every run below)

The systemd service is already disabled.

```bash
pkill -f "ollama serve"; pkill llama-server; sleep 3
free -h                    # "available" should be at least 2 GB; close other programs if not
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > ~/ollama.log 2>&1 &
sleep 3; ollama list | grep llama3.2:3b
```

## Step 3 — Site A `full` (about 15 minutes)

```bash
make mesh SITE=site_a MODE=full
```

You should see 45 lines, then the summary.

## Step 4 — `llm_no_retrieval`: re-measure once (repeat step 2 first)

```bash
make llm-calibrate MODE=llm_no_retrieval ARGS=--force
```

There are two possible results:

- **It prints a timeout:** run Step 5.
- **It prints `STOPPED by ADR 0013 §6`:** a call decoded under 1 token/s, or p95 was
  above 120 s. The ablation then does not run on this laptop, and that becomes a
  reported finding. Skip Step 5.

## Step 5 — Only if Step 4 gave a timeout (repeat step 2 before each site)

This takes about 4 hours; overnight is fine.

```bash
make mesh SITE=site_b MODE=llm_no_retrieval
make mesh SITE=site_a MODE=llm_no_retrieval
```

## Step 6 — Commit and push

```bash
cp ~/ollama.log results/llm_calibration/ollama_$(date +%Y%m%dT%H%M).log
git add configs/mesh/llm_timeouts.lock.json
git add -f results/llm_calibration results/mesh/*/summary.json results/mesh/*/events.jsonl
git status --short | grep -v '^??'      # .env must NOT appear
git commit -m "ADR 0013 §6: Site A full; llm_no_retrieval re-calibration (and runs, if not stopped)"
git push origin main
```
