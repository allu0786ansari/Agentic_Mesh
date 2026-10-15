# Phase 3, Week 10 — Laptop Runbook (retrieval v2, agent mesh in four modes)

Week 10 is built and tested in the cloud workspace. The decisions are in:

- `docs/adr/0011-retrieval-v2.md`: retrieval v2, declared after the PT6 miss;
- `docs/adr/0012-agent-mesh-week10.md`: the mesh.

The cloud ran `rule_only` on the validation alerts. The other three modes need the
real encoder, and two of them need Ollama, so they run on your laptop.

**Time needed:** about 2–3 hours, mostly Ollama calls. Site B has 220 incidents and
`llm_no_retrieval` sends 92 candidates per prompt. You can run it overnight.

**Before the long runs:** do not train anything at the same time (stage isolation),
and close other heavy programs.

## Step 1 — Update, lock and check

`langgraph` is now a core dependency, so the lock changes again, as in Week 9.

```bash
cd ~/Agentic_Mesh
git pull origin main
uv lock
make setup-kb
make check                 # expect 271 passed
```

## Step 2 — Retrieval v2 on validation (exploratory, about 1 minute)

```bash
make retrieval-eval VERSION=v2     # -> results/retrieval/val__v2__exploratory__<time>.json
```

It prints Recall@5, Recall@10 and MRR. The result is exploratory; v2 is confirmed
only on the test incidents at G4 (PT6-v2).

## Step 3 — Ollama

```bash
ollama serve &              # if it is not already running
ollama pull llama3.2:3b     # already there since G0
```

## Step 4 — The mesh in all four modes, both sites

Run them in this order, so the fast modes catch any problem first:

```bash
for site in site_b site_a; do
  make mesh SITE=$site MODE=rule_only
  make mesh SITE=$site MODE=retrieval_only
  make mesh SITE=$site MODE=full
  make mesh SITE=$site MODE=llm_no_retrieval
done
```

Each run prints its session count and statuses. Expect:

- **Site B:** 220 sessions in every mode.
- **Site A:** 45 sessions in every mode.
- **Statuses:**
  - `COMPLETED` in every mode;
  - `ABSTAINED` is possible in `llm_no_retrieval`, where invalid output abstains;
  - `COMPLETED_DEGRADED` should not appear, because the stores are local.

The runner checks Ollama before an LLM mode starts. It stops with a clear message
if Ollama is not running or the model is missing.

## Step 5 — Commit and push

```bash
git add uv.lock
git add -f results/retrieval results/mesh/*/summary.json results/mesh/*/events.jsonl
git commit -m "Week 10: retrieval v2 (exploratory), mesh in four modes on validation (both sites)"
git push origin main
```

The run directories' `storage.db` files stay local. Each one can be rebuilt by
running `make mesh` again.
