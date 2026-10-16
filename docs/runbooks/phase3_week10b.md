# Phase 3, Week 10b — Laptop Runbook (ADR 0013: LLM timeouts measured, two LLM modes re-run)

ADR 0013 (`docs/adr/0013-llm-path-fix.md`) was written before this re-run. It
does four things:

- **G0's setting, restored:** it sends `num_predict` 256 again.
- **Context per mode:** `num_ctx` is 2048 for `full` and 4096 for
  `llm_no_retrieval`.
- **A measured timeout:** the LLM timeout is re-measured with G0's rule
  (1.5 × p95) on 30 real validation prompts per mode.
- **Abandoned requests stop:** a timed-out request now stops on the server, and the
  runner waits for Ollama to recover before the next incident.

Your first runs stay in the repository as the ADR 0012 result. Only `full` and
`llm_no_retrieval` are re-run.

**Time needed:** about 4 hours, almost all of it Ollama. You can run it overnight.

- **Calibration:** about 10 minutes for `full` and 20 for `llm_no_retrieval`.
- **Re-runs:** about 70 minutes for `full` (both sites) and about 2.5 hours for
  `llm_no_retrieval`.

**Stage isolation:** do not train anything at the same time, and close other heavy
programs.

## Step 0 — Secrets in `.env` (once)

```bash
cd ~/Agentic_Mesh
git pull origin main
cp .env.example .env
nano .env        # HF_TOKEN=hf_...  (read-only token from huggingface.co/settings/tokens)
                 # MESH_APPROVAL_KEY=<output of: python3 -c "import secrets; print(secrets.token_hex(32))">
```

`.env` is git-ignored; never `git add` it.

- **What reads it:** `make kb`, `make retrieval-eval`, `make mesh` and
  `make llm-calibrate` read it at start-up.
- **What wins:** a variable you export in the shell wins over `.env`.

With `HF_TOKEN` set, the encoder loads without the "unauthenticated requests"
warning.

## Step 1 — Check

```bash
make setup-kb
make check                 # expect 287 passed
```

## Step 2 — Ollama, with its log kept

The first `llm_no_retrieval` run had 35 `ConnectError`s, which means Ollama stopped
listening for a while. This time keep its log, so we can see why if it happens
again:

```bash
pkill ollama; sleep 2
ollama serve > ~/ollama.log 2>&1 &
ollama list | grep llama3.2:3b
```

## Step 3 — Calibrate the two LLM modes (once each)

```bash
make llm-calibrate MODE=full
make llm-calibrate MODE=llm_no_retrieval
cat configs/mesh/llm_timeouts.lock.json
```

**What it does:**

1. It takes 15 sessions per site from your first run of each mode (the
   `storage.db` files are still on your laptop).
2. It sends each prompt once, with no client timeout.
3. It sets `timeout_s = ceil(1.5 × p95)`.

**What it prints:** one line per call, with seconds and prompt/output tokens, then
the timeout.

**Once per mode:** a second run of the same mode is refused. Measuring until a value
looks good would be tuning.

## Step 4 — Re-run the two LLM modes, both sites

```bash
for site in site_b site_a; do
  make mesh SITE=$site MODE=full
  make mesh SITE=$site MODE=llm_no_retrieval
done
```

**What to expect:**

- **Sessions:** 220 on Site B and 45 on Site A.
- **Each summary now shows:**
  - `llm.failures` by error class;
  - prompt and output tokens;
  - `llm_recoveries`;
  - `llm_settings`, the timeout used.

If something breaks, send me the last 50 lines of `~/ollama.log`.

## Step 5 — Commit and push

```bash
git add configs/mesh/llm_timeouts.lock.json
git add -f results/llm_calibration results/mesh/*/summary.json results/mesh/*/events.jsonl
git status --short | grep -v '^??'      # .env must NOT appear
git commit -m "ADR 0013: LLM timeouts calibrated; full and llm_no_retrieval re-run on validation (both sites)"
git push origin main
```
