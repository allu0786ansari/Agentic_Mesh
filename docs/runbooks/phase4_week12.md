# Phase 4, Week 12 — Laptop Runbook (X1 and X2 on validation: about 30 CPU-hours)

The matrix and its analysis are fixed in
`docs/adr/0016-x1-x2-matrix-week12.md` and `configs/experiments/x1_x2.yaml`:
180 runs, five seeds each.

`make grid` runs the matrix one run per process, in a fixed order:

- it skips runs that are already done;
- it stops starting new runs when its time budget is used;
- you can run it in nightly batches and stop it any time with Ctrl-C (the run in
  progress is lost and redone next time).

**Stage isolation:** while the grid runs, nothing else heavy runs. Stop Ollama, the
stack and any mesh run, and close VS Code and browsers if you can.

## Step 1 — Update, re-lock, check (once)

Week 12 adds the `figures` group (matplotlib), so the lock changes.

```bash
cd ~/Agentic_Mesh
git pull origin main
uv lock
make setup-kb              # now also installs the figures group
make check                 # expect 352 passed
```

## Step 2 — IID partitions for X2 (once, a few minutes)

```bash
make partitions-iid        # -> data/partitions_iid (git-ignored)
```

It prints `Site A: 56 features; events 50 (validation 15, test 35)`.

## Step 3 — See what will run

```bash
pkill -f "ollama serve"; pkill llama-server     # stage isolation
make grid ARGS=--dry-run
```

It should print `180 runs in scope, 0 done, 180 to go (about 30.0 h)` and one line
per group.

## Step 4 — Run in batches (repeat until all 180 are done)

```bash
make grid HOURS=9          # e.g. overnight; prints one line per run
```

Run it again the next night, or whenever the laptop is free. It resumes where it
stopped. When it prints `180/180 runs done`, go to Step 5.

**Pushing after each batch is a good idea,** so progress is safe and I can check it
early:

```bash
git add -f results/X1 results/X2 results/grid results/privacy
git commit -m "Week 12: X1/X2 grid batch"
git push origin main
```

**If a run prints `FAILED`:** leave it, and it runs again, with the same seed, next
time. If the same run fails twice, send me the last lines it printed and the end of
`results/grid/log.jsonl`.

## Step 5 — Check completeness and privacy

```bash
uv run python -m agmesh.evaluation.checker --experiments X1 X2
```

Expect:

- `0 expected run(s) missing`;
- `privacy rule (CC4): 0 violation(s)`;
- all result artifacts accepted.

Then push as above. I'll then build the statistics and figures from the artifacts
(ADR 0016 §4 and §6).
