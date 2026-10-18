# X3 — fusion on Site B validation

Seeds [11, 23, 37, 41, 53]; Δt 60 s, latency-only excluded: True. Benign span 29.1 h in 30 blocks of 60 min.

## H2a — false alarms per hour (benign validation, seeds averaged)

| Detector | FA/h |
|---|---|
| b1 | 2.877 |
| b2 | 3.151 |
| b3 | 2.856 |
| 2of3 | 1.394 |
| noisy_or | 0.000 |

| Seed | τ | Best single node | its FA/h | thresholds' val FPR |
|---|---|---|---|---|
| 11 | inf | b1 | 2.609 | b1 0.0100, b2 0.0099, b3 0.0099 |
| 23 | inf | b2 | 2.643 | b1 0.0100, b2 0.0099, b3 0.0099 |
| 37 | inf | b1 | 2.849 | b1 0.0100, b2 0.0099, b3 0.0099 |
| 41 | inf | b3 | 2.334 | b1 0.0100, b2 0.0099, b3 0.0099 |
| 53 | inf | b1 | 2.849 | b1 0.0085, b2 0.0099, b3 0.0099 |

**PT4 (validation):** 2-of-3 reduction of FA/h against the best single node 0.475 [0.358, 0.593]; one-sided p (H0: reduction ≤ 0.30) = 0.0024. Noisy-OR (descriptive, matched by construction): 1.000 [1.000, 1.000].

## H2b — real attack day, validation episodes

| Detector | Recall [95% CI] | Median TTD (s) | Detected seeds per episode |
|---|---|---|---|
| b1 | 1.000 [1.000, 1.000] | 10 | {'0': 5, '1': 5, '2': 5, '3': 5} |
| b2 | 0.000 [0.000, 0.000] | — | {'0': 0, '1': 0, '2': 0, '3': 0} |
| b3 | 0.250 [0.000, 0.500] | 130 | {'0': 0, '1': 3, '2': 2, '3': 0} |
| 2of3 | 0.400 [0.100, 0.700] | 10 | {'0': 1, '1': 3, '2': 4, '3': 0} |
| noisy_or | 0.000 [0.000, 0.000] | — | {'0': 0, '1': 0, '2': 0, '3': 0} |

Recall loss of 2-of-3 against b1: 0.600 [0.300, 0.900].

## H2c — synthetic cross-device bursts (SYNTHETIC)

### W=5 (primary)

| Detector | Events | Recall [95% CI] | Median TTD (s) |
|---|---|---|---|
| b1 | 16 | 0.438 [0.338, 0.537] | 20 |
| b2 | 16 | 0.887 [0.838, 0.938] | 10 |
| b3 | 16 | 0.637 [0.600, 0.700] | 10 |
| 2of3 | 24 | 0.525 [0.433, 0.625] | 10 |
| noisy_or | 24 | 0.000 [0.000, 0.000] | — |

### W=3 (sensitivity)

| Detector | Events | Recall [95% CI] | Median TTD (s) |
|---|---|---|---|
| b1 | 16 | 0.362 [0.263, 0.475] | 10 |
| b2 | 16 | 0.887 [0.838, 0.938] | 10 |
| b3 | 16 | 0.637 [0.600, 0.700] | 10 |
| 2of3 | 24 | 0.492 [0.400, 0.583] | 10 |
| noisy_or | 24 | 0.000 [0.000, 0.000] | — |

### W=10 (sensitivity)

| Detector | Events | Recall [95% CI] | Median TTD (s) |
|---|---|---|---|
| b1 | 16 | 0.800 [0.762, 0.838] | 10 |
| b2 | 16 | 0.912 [0.863, 0.963] | 10 |
| b3 | 16 | 0.825 [0.800, 0.863] | 10 |
| 2of3 | 24 | 0.783 [0.733, 0.833] | 10 |
| noisy_or | 24 | 0.000 [0.000, 0.000] | — |

Benign `req_write` maximum per node: {'b1': 2.0, 'b2': 2.0, 'b3': 4.0}.
