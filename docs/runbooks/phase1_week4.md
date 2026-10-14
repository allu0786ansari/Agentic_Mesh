# Phase 1, Week 4 — Laptop Runbook (models and validation baselines)

Week 4 adds the detection models and runs the non-private baselines on **validation**
data only. The results feed the Gate G1 decisions in `docs/adr/0002-g1-proposals.md`.

**Time needed:** about 10 minutes.
**You will commit:** `results/X1/` (14 small JSON files).

> Close Ollama and the browser first (`sudo systemctl stop ollama`). Training and
> operations never run at the same time (WR-4).

## Step 1 — Update and check the code

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup          # now also installs the ml group (torch CPU)
make check
```

**Expected result:** mypy `Success`; pytest ends with `132 passed, 1 skipped`.

## Step 2 — Run the baselines

```bash
make baselines
```

It prints one line per configuration (14 in all) and takes about 3–5 minutes. You
should see numbers close to these (small differences between machines are normal):

```
config                             AUROC  AUPRC  evRec  rngF1    FPR    TTD   FA/h
site_a__local__vae                 0.844  0.656  1.000  0.081  0.637 17.875  6.997
site_a__central__vae               0.916  0.758  1.000  0.206  0.051 33.375  4.338
...
site_b__local__rule_new_peer       1.000  1.000  1.000  0.196  0.000 10.000  0.000
```

## Step 3 — Check the result artifacts

```bash
make check-results
```

**Expected result:** `14/14 result artifacts accepted`. If some are rejected with
"uncommitted changes", run `git status`, commit or discard the local changes, and
run Step 2 again.

## Step 4 — Commit and push

```bash
git add results/X1
git commit -m "Week 4: X1 validation baselines"
git push origin main
```

## Step 5 — Gate G1 decisions

Read `docs/adr/0002-g1-proposals.md` and tell me your choice for each decision (or
"accept the recommendations"). I then update `params.yaml`, re-run the baselines with
the chosen inputs, build the ground-truth mappings with you, and we tag `g1-frozen`.
