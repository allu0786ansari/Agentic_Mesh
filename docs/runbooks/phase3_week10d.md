# Phase 3, Week 10d — Laptop Runbook (ADR 0013 §7: prompt cache off; Site B `full`; final `llm_no_retrieval` attempt)

The server inside Ollama (`llama-server`) saves every prompt in a RAM cache with an
8 GiB limit. On your 5.8 GiB WSL, that cache filled memory. Turning it off is one
environment variable.

Each run now also warms the model up before incident 1.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make check                 # expect 294 passed
```

## Step 2 — A fresh Ollama with the prompt cache off (repeat before every run below)

```bash
[ -s ~/ollama.log ] && cp ~/ollama.log results/llm_calibration/ollama_$(date +%Y%m%dT%H%M%S).log   # keep the previous log
pkill -f "ollama serve"; pkill llama-server; sleep 3
free -h                    # "available" at least 2 GB
LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
  ollama serve > ~/ollama.log 2>&1 &
sleep 3
```

## Step 3 — Site B `full` (about 1 hour); this also checks that the cache is off

```bash
make mesh SITE=site_b MODE=full
```

**Check the cache right after the first progress line,** in a second terminal:

```bash
grep -i "prompt cache" ~/ollama.log | head -3
```

- **Good:** it says the cache is disabled, or shows no "enabled, size limit" line.
- **Bad:** it shows `prompt cache is enabled, size limit: 8192 MiB`. Stop the run
  (Ctrl-C) and send me that output. The setting did not reach the server, and we
  need another route.

**While it runs:**

- the runner's own `WARNING ... not visibly off` line should not appear;
- the `llm server … MB` figure on each line should stay roughly flat (about
  2,500–2,800 MB) instead of climbing.

## Step 4 — Final `llm_no_retrieval` attempt (repeat step 2 first)

```bash
make llm-calibrate MODE=llm_no_retrieval ARGS=--force
```

There are two possible results, and either one is final:

- **It prints a timeout:** go to Step 5.
- **It prints `STOPPED`:** skip Step 5.

## Step 5 — Only if Step 4 gave a timeout (repeat step 2 before each run)

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
git commit -m "ADR 0013 §7: Site B full (cache off); final llm_no_retrieval attempt"
git push origin main
```
