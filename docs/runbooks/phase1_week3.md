# Phase 1, Week 3 — Laptop Runbook (evaluation harness)

Week 3 adds the evaluation harness: metrics, thresholds, calibration, statistics,
the label-leakage guard, the sealed test set and the result checker. It trains no
model, so the laptop part is short.

**Time needed:** about 5 minutes.
**You will commit:** an updated `results/data/partition_check.json`.

## Step 1 — Update and check the code

```bash
git pull origin main
make setup
make check
```

**Expected result:** ruff and mypy report no issues; pytest ends with about
`125 passed, 1 skipped`.

## Step 2 — Refresh the partition check

The check now also reports *label proxies*: features that reproduce the label by
construction.

```bash
make check-data
```

**Expected result:**

```
Partition checks: 37/37 passed -> results/data/partition_check.json
  note: site_b/b1 label proxies: identical ['new_peers', 'unseen_addr', 'oversized_qty'], perfect split ['peers']
```

The note is expected (Proposal Appendix B, MB-15). It is about Site B's attack
validation episodes only; the test split is not read.

## Step 3 — Confirm the checker runs

```bash
make check-results
```

**Expected result:** `0/0 result artifacts accepted` and `no result artifacts found`.
There are no results yet; the first ones arrive in Week 4.

## Step 4 — Commit and push

```bash
git add results/data/partition_check.json
git commit -m "Week 3: partition check with label-proxy report"
git push origin main
```
