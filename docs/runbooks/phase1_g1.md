# Phase 1, Gate G1 — Laptop Runbook (freeze the evaluation)

This round applies the accepted G1 decisions (ADR 0002), the scoring choice (ADR
0003), and produces the first-pass ground truth for your review. After your two
review passes we tag `g1-frozen` and Phase 2 starts.

**Time needed:** about 10 minutes of running, plus your review time.
**You will commit:** `results/X1/` (refreshed), `results/X1/scoring_selection.json`,
`evaluation/ground_truth/mapping_site_{a,b}.json`, `results/test_access.log`.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup
make check            # expect 140 passed, 1 skipped
```

## Step 2 — Scoring selection (ADR 0003), about 1 minute

```bash
make scoring-selection
```

The last two lines should read `site_a: chosen mean__s1` and
`site_b: chosen norm_top3__s1`.

## Step 3 — Baselines with the G1 configuration, about 3 minutes

```bash
make baselines
make check-results    # expect 14/14 result artifacts accepted
```

Site A rows should now match the "without `P1_PCV02Z`" table in ADR 0002 (for
example `site_a__central__vae` AUROC about 0.914, FPR about 0.030), and
`site_b__local__pca` FPR about 0.009.

## Step 4 — First-pass ground truth

This step reads the HAI test events' labels and variable values to build the
attribution ground truth. It is the one logged exception to the sealed test set
(`docs/preregistration.md` §5); no model is involved.

```bash
make ground-truth-draft
```

**Expected result:** `50 events; primary {'T0855': 33, 'T0856': 13, 'T0836': 1, 'None': 3} …`
and `7 events; …` for Site B. Four lines are appended to `results/test_access.log`.

## Step 5 — Commit and push

```bash
git add results/X1 evaluation/ground_truth results/test_access.log
git commit -m "G1: scoring selection, baselines with G1 inputs, first-pass ground truth"
git push origin main
```

## Step 6 — Your review of the ground truth (two passes, different days)

Open `evaluation/ground_truth/mapping_site_a.json`. For each event:

1. Read `processes`, `basis` and `evidence` (the variables that moved, in order).
2. Decide whether `primary` and `acceptable` are right under `docs/mapping_protocol.md` §2.
   Pay attention to the "Uncertain" names in the variable-type table, especially
   `P2_CO_rpm` (nine events depend on it).
3. Fill in the three events whose `primary` is `null`.
4. Edit the JSON directly, then add an entry to `reviews`:
   `{"pass": 1, "date": "2026-10-..", "reviewer": "Allaudin Ansari", "changed": [...]}`.

Do the same for `mapping_site_b.json` (seven episodes). On another day, do pass 2
without looking at your pass-1 notes, then record any disagreements in
`docs/mapping_protocol.md` §6. Push after each pass and tell me; I will then set
`"reviewed": true`, finalise `docs/preregistration.md`, and tag `g1-frozen`.
