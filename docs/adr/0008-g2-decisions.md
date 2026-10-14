# ADR 0008 — Gate G2: pilots, tuning protocol and decisions

**Status:** Protocol accepted on 15 October 2026 and committed before any pilot or
tuning result was seen; results and decisions go below the line. (The Plan names this
file `0002-g2-decisions.md`; 0002 was already taken by G1.)
**Implements:** Plan Week 7 tasks 3 and 4 and Gate G2 (§7).

## Protocol (fixed before running)

All runs are on validation only, 50 rounds, in-process (ADR 0007), with the frozen
detection evaluation (ADR 0005, R3 on Site A).

**1. Pilot (Plan Week 7 task 3).** FedAvg and FedProx (μ = 0.01) at ε = 10, seed 11,
on both sites: 4 runs. They give per-run time and a first look at federated utility.
Nothing is selected on them except as stated below.

**2. DP hyperparameters (Site A only).** The G2 fallback in Plan §7 applies: the
single-client sanity check (ADR 0006) lost 0.10 AUROC at ε = 3.

- Grid: DP learning rate ∈ {1e-3 (current), 3e-3} × clipping norm C ∈ {0.5, 1.0
  (current), 2.0}.
- Runs: FedAvg, ε = 3, seed 11, 50 rounds. Six runs.
- Criterion: the highest macro AUPRC. This is the same criterion as ADR 0003.
- Ties: values within 0.01 AUPRC count as tied, and the configuration nearer the
  defaults wins (first fewer changed values, then C before lr).
- The choice applies to Site A for every ε and both strategies.
- Site B keeps the defaults. Its validation set cannot rank methods (ADR 0003, ADR
  0005).

**3. FedProx μ (Site A).** μ ∈ {0.001, 0.01, 0.1} at ε = 10, seed 11, using the
pilot's μ = 0.01 run plus two more. Criterion: macro AUPRC; ties within 0.01 go to
0.01, the current value. The chosen μ is used for FedProx in X1 and X2 on both
sites.

**4. Headline ε and seeds.**

- ε ∈ {1, 3, 10, ∞} is kept for the X1 curve. Dropping a point after seeing it would
  be selective reporting.
- **ε = 3 is the headline privacy level**, fixed now. ε = 10 is the "weak privacy"
  end, and ε = 1 shows where utility breaks.
- Seeds: 5 for every X1 configuration if the measured compute for X1 + X2 is at most
  60 CPU-hours. Otherwise, 3 seeds for configurations other than ε = 3 and ∞ (Plan
  §7).

**5. X8 scope.** X8 stays in scope if SecAgg+ aggregates within one quantisation
step and recovers the survivors' mean after a dropout (`tests/integration/test_secagg.py`
and the X8 artifact). If it fails, it is reported as a negative systems result and
not fixed by tuning.

---

## Results and decisions

### Addendum to the protocol (written after the grid finished, before the run it governs)

The grid's best learning rate, 3e-3, is its **upper edge**. A best value at the edge
of a grid is not known to be the best, so one boundary extension is declared here,
before it is run:

- **lr = 1e-2 with C = 1.0**: FedAvg, ε = 3, seed 11.
- It replaces the grid's choice only if its AUPRC is higher by **more than 0.01**,
  the tie margin.
- No further extension is made either way.

*(Results and decisions follow below.)*
