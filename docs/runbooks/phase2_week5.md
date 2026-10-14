# Phase 2, Week 5 — Laptop Runbook (DP training and persistent accounting)

This round does three things. It publishes the `g1-frozen` tag, which this cloud
session cannot push. It adds Opacus to the locked environment, which the cloud
cannot do because it can't reach the PyTorch package index. Then it runs the Week 5
DP sanity check. The design and the cloud results are in
`docs/adr/0006-dp-design-week5.md`.

**Time needed:** about 15 minutes of running on 2 CPU cores.
**You will commit:** `pyproject.toml`, `uv.lock`, and `results/privacy/`.

## Step 1 — Update, and publish the G1 tag

```bash
cd ~/Agentic_Mesh
git pull origin main
git tag -a g1-frozen f5bb470 -m "Gate G1: detection evaluation frozen (ADRs 0002-0005, preregistration Status part 1). Attribution ground truth is frozen separately at gt-frozen."
git push origin g1-frozen
```

`f5bb470` is the commit "Preregistration: two-part freeze". Check it with
`git show --stat f5bb470`.

## Step 2 — Add Opacus to the locked environment

```bash
uv add --group ml "opacus>=1.6,<1.7"
make setup
make check            # expect 188 passed, 1 skipped
git add pyproject.toml uv.lock
git commit -m "Add Opacus 1.6 to the ml group (Week 5 DP-SGD)"
git push origin main
git status --short    # must print nothing before Step 3
```

Without Opacus, the privacy tests skip. With it, all 26 run: accountant persistence,
budget guard, total budget, δ rule, DP histograms and DP training.

## Step 3 — DP sanity check, about 12 minutes

```bash
make dp-sanity
```

Expected (the cloud run; your numbers should match to about 0.01 because seeds are
fixed):

| Site, client | ε = 1 | ε = 3 | ε = 10 | ε = ∞ |
|---|---|---|---|---|
| Site A a1, AUROC | 0.802 | 0.832 | 0.915 | 0.932 |
| Site B b1, AUROC | 1.000 | 0.999 | 0.999 | 1.000 |

Every line reports `spent` equal to its ε, to three decimals. That spend includes
the normalisation statistics.

## Step 4 — Commit and push

```bash
git add results/privacy
git commit -m "Week 5: DP sanity check (single client, eps 1/3/10/inf) with accountant stores"
git push origin main
```
