# Phase 2, Week 7 — Laptop Runbook (Gate G2)

Week 7 is done in the cloud workspace, and Gate G2 has passed. All the evidence is
committed: X8 (SecAgg+) in `results/X8/`, and the four pilots and the
tuning grid in `results/G2/`. The decisions are in
`docs/adr/0008-g2-decisions.md`. The G2 setting, Site A DP learning rate 3e-3, is in
`configs/params.yaml`.

**Required (about 5 minutes):** update and check.
**Optional (about 3 hours):** reproduce the G2 runs on your laptop. Earlier rounds
matched across machines bit for bit, so this is a confirmation, not a dependency.

## Step 1 — Update and check (required)

```bash
cd ~/Agentic_Mesh
git pull origin main
make setup-fl
make check            # expect 208 passed
make check-results    # expect 71/71 result artifacts accepted (70 X1 + X8)
```

## Step 2 — Reproduce G2 (optional)

```bash
make x8               # about 3 min
make g2-pilot         # about 40 min, -> results/federated/g2_pilot/
make g2-tuning        # about 2 h,    -> results/federated/g2_tuning/, g2_mu/
git add results/X8 results/federated results/privacy
git commit -m "Week 7: laptop reproduction of X8 and the G2 runs"
git push origin main
```

The pilots and the μ runs pass the pre-G2 learning rate (`--lr 0.001`) explicitly,
so they reproduce the ADR 0008 runs rather than the new default. X8 should report
`error_within_step` 1.0 and `dropout_recovered` 1.0. The tuning table should match
ADR 0008 to about 0.01.

## Still open from Phase 1

Your verification pass of the attribution ground truth (`docs/mapping_protocol.md`
§6), after which we tag `gt-frozen`. Phase 3 (Week 8) cannot start without it.
