# Phase 1, Gate G1 — Laptop Runbook (ADR 0005: shift-robust scoring, event-level inference)

This round replaces `phase1_g1_seeds.md`. If you have not run that one yet, skip it
and run this one; it covers everything. It does four things, all on validation, with
the test set still sealed:

1. checks the training budget (ADR 0004);
2. runs the ADR 0005 selection once (Site A chose R3, the robust adaptive
   threshold);
3. re-runs the X1 baselines on five seeds with R3. Every result now has a
   per-window score file beside it;
4. writes the summary, which now has event-level 95% CIs and the paired tests with
   Holm correction.

**Time needed:** about 75–90 minutes on 2 CPU cores. Close other heavy programs and
do not run Ollama at the same time.
**You will commit:** everything under `results/X1/`. That is 70 `seed_<n>.json` files
with 70 `.scores.npz` files (about 4 MB), plus `training_budget.json`,
`shift_selection.json`, `scoring_stability.json` and `summary_val.json`.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup
make check            # expect 161 passed, 1 skipped
git status --short    # must print nothing: results are rejected from a dirty tree
```

Do not edit any file in the repository until Step 6. An uncommitted change while a
run is writing results makes the checker reject them.

## Step 2 — Training budget (ADR 0004), about 25 minutes

```bash
make training-budget
```

Expected last lines: `site_a: chosen max_epochs 200` and `site_b: chosen max_epochs 30`.

## Step 3 — Shift selection (ADR 0005), about 15 minutes

```bash
make shift-selection
```

Expected: `site_a: chosen r3`, and in the summary lines,
`site_a vae r3: auroc 0.935 auprc 0.767 event_recall 0.969 window_fpr 0.019`.

## Step 4 — Baselines on five seeds with R3, about 20 minutes

```bash
make baselines
make check-results    # expect 70/70 result artifacts accepted
```

The checker now also verifies each score file against the hash in its artifact.

## Step 5 — Summary and scoring stability, about 15 minutes

```bash
make summarise
uv run python -m agmesh.experiments.scoring_selection --stability
```

`make summarise` prints two tables. The seed-level table is secondary. The
event-level table is primary, followed by the five pre-declared comparisons with
Holm-adjusted p-values. Expected (from the cloud run):

| Config | AUROC | Window FPR |
|---|---|---|
| `site_a__local__vae` | about 0.922 | about 0.021 |
| `site_a__central__vae` | about 0.948 | about 0.016 |
| `site_a__local__pca` | about 0.935 | about 0.013 |

## Step 6 — Commit and push

```bash
git add results/X1
git commit -m "X1 under ADR 0005: five seeds, R3 scoring, score files, event-level summary"
git push origin main
```
