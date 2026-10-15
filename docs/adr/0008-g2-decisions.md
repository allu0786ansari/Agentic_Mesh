# ADR 0008 — Gate G2: pilots, tuning protocol and decisions

**Status:** Accepted, 15 October 2026. The protocol was committed (9e05551) before any
pilot or tuning result was seen; results and decisions are below the line. (The Plan names this
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

### Runs

All runs below were done in the cloud workspace on the same partitions, seed 11,
validation only, from clean commits (one exception is noted). The cloud G2 artifacts
are under `results/G2/`. (They sat in the ignored `results/` tree until the
reproduction commit below; they are now committed unchanged.) The cloud X8 artifact
was never committed, and the laptop's X8 artifact replaced it; the X8 table below
gives the cloud numbers, and the reproduction section gives the laptop's. The laptop
reproduced everything with `make x8 g2-pilot g2-tuning` (see "Laptop reproduction").

**1. Pilots, ε = 10, 50 rounds, defaults before G2 (Site A DP lr 1e-3, C 1.0):**

| Run | AUROC | AUPRC | Event recall | Window FPR | Pooled AUROC (bins) | s / run |
|---|---|---|---|---|---|---|
| Site A FedAvg | 0.865 | 0.607 | 0.875 | 0.011 | 0.861 ± 0.002 | 576 |
| Site A FedProx (μ 0.01) | 0.865 | 0.606 | 0.875 | 0.011 | 0.862 ± 0.002 | 574 |
| Site B FedAvg | 1.000 | 1.000 | 1.000 | 0.007 | 1.000 ± 0.000 | 332 |
| Site B FedProx | 1.000 | 1.000 | 1.000 | 0.026 | 0.999 ± 0.001 | 484 |

The Site B FedAvg pilot ran while `pyproject.toml` had an uncommitted mypy-only
change; its artifact records that. The change cannot affect a run.

**Site A was not converged.** Its validation loss was still falling at round 50
(0.585 at round 40, 0.451 at round 45, 0.389 at round 50). At lr 1e-3 the private federation is limited
by optimisation, not by the data.

**2. DP learning rate × clipping norm, Site A FedAvg, ε = 3:**

| lr | C | AUPRC | AUROC | Event recall | Window FPR | Validation loss, round 50 |
|---|---|---|---|---|---|---|
| 1e-3 | 0.5 | 0.509 | 0.845 | 0.875 | 0.015 | 0.783 |
| 1e-3 | 1.0 | 0.529 | 0.845 | 0.812 | 0.012 | 0.795 |
| 1e-3 | 2.0 | 0.549 | 0.858 | 0.812 | 0.005 | 1.056 |
| 3e-3 | 0.5 | 0.700 | 0.888 | 0.938 | 0.011 | 0.308 |
| **3e-3** | **1.0** | **0.703** | **0.888** | **0.938** | **0.010** | **0.313** |
| 3e-3 | 2.0 | 0.636 | 0.880 | 0.875 | 0.013 | 0.418 |
| 1e-2 (extension) | 1.0 | 0.669 | 0.884 | 0.938 | 0.009 | 0.512 (plateau from round 14) |

- **Rule outcome:** the best AUPRC is 0.703 (lr 3e-3, C 1.0). C 0.5 is within the
  0.01 tie margin (0.700), but it changes two values, so C 1.0 with one change wins.
- **Extension:** lr 1e-2 does not beat 0.703 by more than 0.01. It is worse: its loss
  stops falling at a noise floor by round 14.

**Decision:** `privacy.dp_lr.site_a = 0.003`, C = 1.0 unchanged. Site B keeps its
defaults.

**3. FedProx μ, Site A, ε = 10, lr 1e-3:**

| μ | 0.001 | 0.01 | 0.1 |
|---|---|---|---|
| AUPRC | 0.607 | 0.606 | 0.591 |

All three are within 0.016, and 0.001 and 0.01 are tied. **Decision: μ = 0.01
stays.** With one local epoch, clients drift little from the global model in 12
steps, so the proximal term barely acts. FedProx can only differ from FedAvg in the
heterogeneous arm X2, which is where it is meant to matter.

**4. X8: SecAgg+ on Fed-A** (`results/X8/secagg/seed_11.json`; 3 rounds, ε = 10;
241,770 parameters):

| | Plaintext | SecAgg+ | SecAgg+, a4 drops in round 2 |
|---|---|---|---|
| Max abs error against the plaintext mean | 0 | 3.18 × 10⁻⁶ | 3.41 × 10⁻⁶ (3 survivors) |
| Quantisation step 2c/(Q−1) | — | 3.81 × 10⁻⁶ | 3.81 × 10⁻⁶ |
| Parameters outside ±8 (clipped) | — | 0 | 0 |
| Seconds per round (1st round includes Ray start) | 38.6 / 10.6 / 11.4 | 38.4 / 11.2 / 11.0 | 39.1 / 9.0 / 12.2 |
| Payload bytes, 3 rounds, all clients | 23.2 MB | 42.6 MB (×1.84) | 38.8 MB |

- **Correct:** the error is within one quantisation step in every round.
- **Tolerant:** the dropout round recovers the survivors' mean.
- **Cheap in time:** no measurable overhead per round (×1.00), because DP-SGD
  dominates the round.
- **Costs bandwidth:** payload grows ×1.84, from the masked integer vectors and the
  key-exchange stages.

**Decision: X8 stays in scope**, and SecAgg+ is reported as compatible with the DP
federation at negligible time cost.

**5. Compute plan.** A 50-round run takes about 9.6 min on Site A (DP) and 5.5–8 min
on Site B, on 2 CPUs.

| Experiment | Runs | Estimated time |
|---|---|---|
| X1 federated (2 sites × 2 strategies × ε {1, 3, 10, ∞} × 5 seeds) | 80 | about 11 h |
| X2 (heterogeneous epochs and dropout, both sites, ε = 3, 5 seeds); heterogeneous runs take about 2× longer | — | about 11 h |
| H1c arm (Site A, ε {3, 10}, 5 seeds) | 10 | about 2 h |
| **Total** | | **about 24 CPU-hours** |

This is under 60 hours, so **every configuration gets 5 seeds.**

**6. Unchanged:** ε ∈ {1, 3, 10, ∞} with ε = 3 as the headline; H1c at ε 0.5 with 100
bins, since no new evidence bears on them.

### Reading for the papers (validation, one seed)

- **Federation at ε = 3, after tuning, beats a single client at ε = 3.**
  Federated Site A reaches AUROC 0.888 and AUPRC 0.703. A single client (a1) reached
  0.832 / 0.520 (ADR 0006). Averaging four clients' noisy updates helps, as RQ1
  predicts.
- **It is still below the non-private references.** Local VAE with R3: 0.922;
  central: 0.948 (ADR 0005). The gap is the price of ε = 3, and X1 will put CIs on it.
- **Tuning on validation was decisive.** The default DP learning rate understated
  private federated utility by 0.17 AUPRC. Reporting untuned DP baselines would have
  exaggerated the cost of privacy. The tuning used validation only, under a rule
  fixed in advance, and the same setting applies to every ε and strategy.

### Laptop reproduction (commit 7dfe55f, code at be1289e, clean tree)

All 13 G2 runs and X8 were rerun on the author's laptop (WSL2, 2 physical cores) and
are committed under `results/federated/g2_{pilot,tuning,mu}/` and `results/X8/`.

- **Pilots, μ runs, and five of the seven tuning cells:** the same metrics to three
  decimals.
- **Two cells differ:**
  - lr 3e-3, C 1.0: AUPRC 0.7024 against 0.7026 in the cloud.
  - lr 1e-2, C 1.0: 0.668 against 0.669.
  - Cause: validation losses differ from round 1 at the 10⁻⁵ level. This is
    floating-point summation order in CPU kernels on different hardware. It is not
    a code or data difference:
    - The lockfile hashes match.
    - The parameters files differ only by the `dp_lr` default that G2 introduced.
      These runs set `--lr` explicitly, so the default does not apply.
    - Five of seven cells agree to three decimals on the same partitions.
  - Earlier comparisons were bit-identical because they used shorter or non-DP runs.
    Over 50 DP rounds, a larger learning rate amplifies these last-bit differences.
- **Decisions are unchanged.** On the laptop numbers:
  - lr 3e-3 / C 1.0 (0.702) and C 0.5 (0.700) are still tied, and C 1.0 still wins
    on fewer changes.
  - lr 1e-2 still does not beat 3e-3.
  - μ is unchanged.
- **X8 on the laptop:**
  - Max error 3.36 × 10⁻⁶, within one quantisation step (3.81 × 10⁻⁶).
  - Dropout recovered.
  - Bytes ×1.84.
  - Time ratio 0.87: SecAgg+ was not slower, and the difference is round-to-round
    noise.

**Consequence for X1–X2:** cross-machine agreement is to about 10⁻³ AUPRC, not
bitwise. All X1/X2 seeds for a comparison are run on one machine, the laptop. Five
seeds' spread (ADR 0005: 0.01–0.03) is far larger than this.

**Gate G2: passed.** The X1/X2 grid can start (Phase 3 planning, Week 8, also needs
`gt-frozen`).
