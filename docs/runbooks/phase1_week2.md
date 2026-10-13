# Phase 1, Week 2 — Laptop Runbook (data pipeline)

Week 2 turns the raw datasets into the per-node partitions that every later week
reads. The code is on `main`. Your job is to run it on your copy of the data, confirm
the checks pass, and commit the small evidence files.

**Time needed:** about 15 minutes, most of it the Modbus parsing.
**You will commit:** `dvc.lock`, `results/data/partition_check.json`, and
`data/raw_checksums.json` (left over from Week 1).

> Run everything inside WSL2 Ubuntu, from the repository folder (`~/Agentic_Mesh`).
> Close the browser and Ollama first (`sudo systemctl stop ollama`); the partition
> step peaks at about 2.2 GB.

---

## Step 1 — Update and check the code

```bash
git pull origin main
make setup
make check
```

**Expected result:** ruff and mypy report no issues; pytest ends with about
`96 passed, 1 skipped`.

## Step 2 — Run the pipeline on the synthetic fixture

This is what CI runs. It takes a few seconds and shows the pipeline works before it
touches your real data.

```bash
make fixture-smoke
```

**Expected result:** the last line is `Partition checks: 26/26 passed`.

## Step 3 — Run the real pipeline with DVC

```bash
dvc repro
```

DVC runs two stages:

1. **`prepare`** parses all 37 Modbus captures (about 40 million packets) into
   `data/interim/modbus/`. It prints one line per capture and takes about 4 minutes.
2. **`partition`** builds the Site A and Site B partitions in `data/partitions/`
   (about 30 seconds). It ends with:

```
Site A: 56 features; events 50 (validation 15, test 35)
Site B: 7 attacker episodes; attack windows {'ied1a': 51, 'ied1b': 0, 'scada-hmi': 0}
```

If DVC refuses to run (for example, a message about `data/raw` or `.gitignore`), run
`make prepare partition` instead and tell me the DVC message; the outputs are the same.

## Step 4 — Run the checks

```bash
make check-data
```

**Expected result:** `Partition checks: 37/37 passed`. If any check fails, the lines
starting with `FAILED` say which value differed. Don't change
`configs/partition_expectations.yaml`; send me the output instead.

These were the counts when I ran the same code on the analysed data:

| Node | train | val_normal | val_attack (attack windows) | test (attack windows) |
|---|---|---|---|---|
| a1 (`train1`) | 3,060 windows | 539 | 320 (9) | 1,015 (31) |
| a2 (`train2`) | 3,213 | 566 | 326 (20) | 1,023 (39) |
| a3 (`train3`, first half) | 3,391 | 597 | 623 (17) | 1,457 (32) |
| a4 (`train3`, second half) | 3,391 | 597 | 303 (13) | 1,561 (34) |
| b1 (IED1A) | 64,792 | 11,433 | 147 (12) | 3,532 (39) |
| b2 (IED1B) | 60,596 | 10,692 | 147 (0) | 3,531 (0) |
| b3 (SCADA HMI) | 59,419 | 10,485 | 147 (0) | 3,532 (0) |

Site A windows are 60 s; Site B windows are 10 s. The full numbers are in
`results/data/partition_check.json` under `counts`.

## Step 5 — Confirm the partitions are deterministic

```bash
sha256sum data/partitions/site_a/manifest.json data/partitions/site_b/manifest.json > /tmp/m1
make partition
sha256sum data/partitions/site_a/manifest.json data/partitions/site_b/manifest.json | diff /tmp/m1 - && echo identical
```

**Expected result:** `identical`.

## Step 6 — Record the Week 1 checksums (missing from the last push)

```bash
make checksums
```

**Expected result:** `Recorded 45 files`.

## Step 7 — Commit and push

```bash
make check
git add dvc.lock results/data/partition_check.json data/raw_checksums.json
git commit -m "Week 2: partitions built and checked on the real data"
git push origin main
dvc push   # only if you configured a DVC remote in Week 1
```

Then tell me it's pushed. I'll read the check artifact, and we'll do the manual
spot-check of the episode boundaries (`docs/labeling_protocol.md` §5) together before
moving on to Week 3.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `missing capture folder data/raw/modbus2023/...` | The folder layout differs from `configs/params.yaml` (`paths.modbus_dir`); fix the path, not the folders |
| `no HAI train and test files found` | Check `paths.hai_dir` points at the folder holding `train1.csv.gz` … `test5.csv.gz` |
| The laptop slows down during `partition` | Close other applications; it needs about 2.2 GB for a minute |
| `Partition checks` shows fewer than 37 passed | Send me the `FAILED` lines |
