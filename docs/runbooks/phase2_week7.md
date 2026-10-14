# Phase 2, Week 7 — Laptop Runbook (SecAgg+, G2 pilots and tuning)

This round runs the Gate G2 work:

- **X8**, the SecAgg+ systems experiment;
- **the four 50-round pilots** at ε = 10;
- **the tuning runs declared in ADR 0008** before any result: the DP learning rate
  and clipping norm on Site A, and FedProx's μ.

The cloud ran the same commands; its results are in ADR 0008, and your numbers
should match them to about 0.01.

**Time needed:** about 3 hours of running on 2 CPU cores. It can run overnight.
Close other heavy programs and do not run Ollama (WR-4).
**You will commit:** `results/X8/`, the four pilot runs under `results/X1/`,
`results/federated/g2_tuning/`, `results/federated/g2_mu/` and `results/privacy/`.

## Step 1 — Update and check

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup-fl
make check            # expect all passed
git status --short    # must print nothing; keep the tree untouched until Step 5
```

## Step 2 — X8, SecAgg+, about 3 minutes

```bash
make x8
```

The printed summary should show:

- `"error_within_step": 1.0`: the SecAgg+ aggregate equals the plaintext mean within
  one quantisation step;
- `"dropout_recovered": 1.0`: with a4 dropping in round 2, the three survivors' mean
  is recovered;
- the time and bytes overhead ratios.

## Step 3 — Pilots, about 40 minutes

```bash
make g2-pilot
make check-results    # the four new runs must be accepted
```

## Step 4 — Tuning grid (ADR 0008), about 2 hours

```bash
make g2-tuning
```

Six Site A FedAvg runs at ε = 3 (learning rate × clipping norm), then two Site A
FedProx runs at ε = 10 (μ = 0.001 and 0.1).

## Step 5 — Commit and push

```bash
git add results/X8 results/X1 results/federated results/privacy
git commit -m "Week 7: X8 SecAgg+, G2 pilots and tuning grid"
git push origin main
```

Then tell me. I will check the runs against ADR 0008's rule, record the decisions,
set them in `configs/params.yaml`, and close Gate G2.
