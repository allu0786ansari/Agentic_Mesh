# Phase 4, Week 14 — Laptop Runbook (X7, reproduction, Gate G4, the single test run)

Read `docs/adr/0019-g4-final-test-protocol-week14.md` first. There are two parts.

- **Part 1:** X7 (the reproduction is waived, ADR 0019 §2a). Push.
- **Part 2:** tag `g4-frozen`, then run every test evaluation once.

**Everything that reads test data runs on this laptop**, so that
`results/test_access.log` has one writer. The cloud only analyses committed
artifacts.

---

## Part 1 — before the gate (about 1.5 h)

### Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
uv sync --locked --group dev --group ml --group fl --group kb --group figures
make check
git status --short        # must print nothing outside results/
```

Plain `uv sync --locked` installs only the default `dev` group, and it removes
torch, Flower and the knowledge-base packages. Always name the groups, as above.

### Step 2 — X7, training stage (about 15 min; Ollama stopped)

```bash
pkill -f "ollama serve"; pkill llama-server
make x7-training
```

It runs one Site A FedProx ε 3 training (seed 11) and samples its memory every
0.5 s. The run writes to `results/x7_scratch/`, which is not committed.

### Step 3 — X7, operations stage (about 20 min)

**Terminal 1** starts the sampler and leaves it running:

```bash
cd ~/Agentic_Mesh && make x7-watch
```

**Terminal 2** runs the operations, in order.

1. Start Ollama:

   ```bash
   cd ~/Agentic_Mesh
   LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve &
   ```

2. Run the edge replay of Site A validation, into scratch:

   ```bash
   uv run python -m agmesh.edge.runtime --site site_a \
       --model results/models/site_a__fedprox__eps3/seed_11 --out results/x7_scratch/edge
   EDGE=$(ls -d results/x7_scratch/edge/site_a__val__* | tail -1)
   ```

3. Start the stack (7 node API processes over mTLS, plus the approval API) and run
   one `full` mesh run against it:

   ```bash
   make stack-up SITE=site_a ARGS="--edge-run $EDGE"
   uv run --group kb python -m agmesh.mesh.runner --site site_a --mode full --nodes https \
       --edge-run $EDGE --out results/x7_scratch/mesh --traces results/x7_scratch/traces
   make stack-down
   ```

Then press **Ctrl-C in Terminal 1**, and write the report:

```bash
make x7-report
```

**Expected:** `X7 passed`, with a peak used memory at most 6000 MB in both stages.
It prints the peak PSS per role (node API, mesh, LLM, edge). Send me the output
either way.

### Step 4 — Clean-checkout reproduction: not run (author's decision)

On 20 October 2026 you decided not to run it. ADR 0019 §2a records the decision,
and `results/reproduce/waiver.json` marks CC7 as **waived**, not met. Nothing to do
here.

### Step 5 — Commit and push X7

```bash
cd ~/Agentic_Mesh
git add -f results/X7/resources.json results/X7/samples
git commit -m "Week 14: X7 resources"
git push origin main
```

`results/` is git-ignored, so its files need `git add -f`. Listing a file that
does not exist makes `git add` stage nothing at all.

Then go on to Part 2.

---

## Part 2 — Gate G4 and the single test run

**Rules for all of Part 2:**

- **No code change after the tag.** `git status --short` may only list `results/`.
- **Every `--final` command refuses to run twice.** If one fails part-way, send me
  the output before doing anything else. A crashed run may be run again only if it
  wrote no artifact.
- **Commit and push results after each step**, `results/` only. That is allowed
  after the tag.

### Step 6 — Tag the gate

```bash
cd ~/Agentic_Mesh
git pull origin main
git status --short                 # nothing outside results/
git tag -a g4-frozen -m "Gate G4: configurations frozen (ADR 0019)"
git push origin g4-frozen
```

### Step 7 — Short training-stage runs (about 1 h; Ollama stopped)

```bash
pkill -f "ollama serve"; pkill llama-server
make baselines-final               # 14 configurations x 5 seeds
make x3-final                      # X3 on test, five Site B seeds
make edge-final SITE=site_a
make edge-final SITE=site_b
git add -f results/test results/test_access.log results/edge/*__test__*/summary.json \
    results/edge/*__test__*/events.jsonl results/edge/*__test__*/calibration_*.json
git commit -m "G4 test: baselines, X3, edge replays"
git push origin main
```

### Step 8 — The mesh on the test incidents (about 4–7 h; Ollama running)

Start Ollama as in Step 3. Then, for **site_a** first and then **site_b**:

```bash
S=site_a                                           # then S=site_b
make mesh-final SITE=$S MODE=rule_only
make stack-up SITE=$S ARGS="--split test"
make mesh-final SITE=$S MODE=retrieval_only NODES=https
make mesh-final SITE=$S MODE=full NODES=https
make stack-down
make mesh-final SITE=$S MODE=llm_no_retrieval
```

Then score them:

```bash
make x4-final        # exports sessions.jsonl and scores X4 on test
make pt6-final       # PT6-v2 on the test rendered queries
make x5-final        # latency on the test runs
git add -f results/test results/test_access.log results/mesh/*__test__*/summary.json \
    results/mesh/*__test__*/sessions.jsonl results/mesh/*__test__*/events.jsonl
git commit -m "G4 test: mesh runs, X4, PT6-v2, latency"
git push origin main
```

### Step 9 — The X1/X2 matrix on test (about 22–30 h, over several nights; Ollama stopped)

```bash
pkill -f "ollama serve"; pkill llama-server
make grid-final HOURS=9
git add -f results/test results/test_access.log results/grid/final_log.jsonl
git commit -m "G4 test: X1/X2 matrix (batch)"
git push origin main
```

- **Repeat each night** until it prints `180/180 runs done`. The order puts the
  configurations the preregistered tests use first.
- **Before the IID runs**, check once that the IID scheme's test partitions equal
  the time scheme's:

  ```bash
  python3 -c "import json;a=json.load(open('data/partitions/site_a/manifest.json'));b=json.load(open('data/partitions_iid/site_a/manifest.json'));print(all(a['nodes'][n]['partitions']['test']['sha256']==b['nodes'][n]['partitions']['test']['sha256'] for n in a['nodes']))"
  ```

### Step 10 — Completion check

```bash
make completion
git add -f results/test/completion_check.json && git commit -m "G4: completion check" && git push
```

Send me the output. I then run every test analysis from the committed artifacts.
