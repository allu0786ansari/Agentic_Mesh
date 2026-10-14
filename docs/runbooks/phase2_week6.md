# Phase 2, Week 6 — Laptop Runbook (federations)

The federated server and clients, both backends, stragglers and H1c are built and
tested; design in `docs/adr/0007-federation-week6.md`. This round installs the
Flower stack, runs the test suite including the backend-equivalence test, and runs
the Week 6 smoke tests on your data.

**Time needed:** about 10 minutes of running on 2 CPU cores. Close other heavy
programs and do not run Ollama at the same time (WR-4).
**You will commit:** `results/federated/smoke/` and `results/privacy/`.

## Step 1 — Update, install the federated stack, check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup-fl         # dev + ml + fl: flwr 1.39 and Ray
make check            # expect 207 passed
git status --short    # must print nothing
```

If you skip `make setup-fl`, `make check` shows 201 passed and 2 skipped; the Flower
tests are the ones skipped.

## Step 2 — 5-round smoke tests, both sites, about 3 minutes

```bash
make train SITE=site_a ROUNDS=5
make train SITE=site_b ROUNDS=5
```

Each prints one line ending in the artifact path. Check two things:

- **Max client ε is 10.000.** The default run is FedProx at ε = 10.
- **The validation loss falls.** The cloud run gave 1.238 → 1.068 on Site A and
  0.430 → 0.151 on Site B. To see your values:

```bash
python3 -c "import json,glob; [print(f, [round(r['val_loss'],3) for r in json.load(open(f))['federation']['round_log']]) for f in sorted(glob.glob('results/federated/smoke/*.json'))]"
```

## Step 3 — The same run on Flower, about 1 minute

```bash
make train SITE=site_b ROUNDS=2 EPS=3 BACKEND=flower
make train SITE=site_b ROUNDS=2 EPS=3
```

The two lines must report the same AUROC, FPR and max ε; the cloud gave identical
numbers. Ray may print warnings about accelerators and the dashboard; they are
harmless.

## Step 4 — Commit and push

```bash
git add results/federated results/privacy
git commit -m "Week 6: federated smoke tests (5 rounds, both sites) and Flower equivalence"
git push origin main
```
