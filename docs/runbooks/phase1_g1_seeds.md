# Phase 1, Gate G1 — Laptop Runbook (five seeds, training budget)

The Week 4 baselines were run with one seed. This round re-runs them with all five
project seeds (11, 23, 37, 41, 53) under the training budget chosen in ADR 0004
(VAE `max_epochs` 200). It also writes the seed summary (mean and 95% CI, paired
tests) and checks that the ADR 0003 scoring choice holds on every seed. Everything is
on validation; the test set stays sealed.

**Time needed:** about 45–60 minutes of running on 2 CPU cores. Close other heavy
programs and do not run Ollama at the same time.
**You will commit:** `results/X1/` (70 seed files), `results/X1/training_budget.json`,
`results/X1/summary_val.json`, `results/X1/scoring_stability.json`.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup
make check            # expect 145 passed, 1 skipped
git status --short    # must print nothing: results are rejected from a dirty tree
```

## Step 2 — Training budget (ADR 0004), about 25 minutes

```bash
make training-budget
```

Expected last lines: `site_a: chosen max_epochs 200` and `site_b: chosen max_epochs 30`.
At 100 epochs about 32% of Site A runs are capped; at 200, none.

## Step 3 — Baselines on five seeds, about 15 minutes

```bash
make baselines
make check-results    # expect 70/70 result artifacts accepted
```

14 configurations × 5 seeds. `seed_11.json` files are replaced; seeds 23–53 are new.

## Step 4 — Seed summary and scoring stability, about 10 minutes

```bash
make summarise
uv run python -m agmesh.experiments.scoring_selection --stability
```

Expected from the summary (mean over five seeds):

| Config | AUROC | Window FPR |
|---|---|---|
| `site_a__central__vae` | about 0.915 | about 0.044 |
| `site_a__local__vae` | about 0.846 | about 0.279 |
| `site_a__central__pca` | 0.903 | 0.008 |
| `site_a__local__pca` | 0.887 | 0.053 |

The stability lines report how many of the five seeds the rule agrees with the
configured scoring (`mean__s1` on Site A, `norm_top3__s1` on Site B).

## Step 5 — Commit and push

```bash
git add results/X1
git commit -m "X1: five seeds, training budget (ADR 0004), seed summary, scoring stability"
git push origin main
```
