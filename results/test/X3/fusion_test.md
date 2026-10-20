# X3 — fusion on Site B, TEST

Seeds [11, 23, 37, 41, 53]; benign test time 9.33 h in 10 blocks of 60 min (episodes excluded, widened by 60 s). Thresholds, τ and the best single node from validation.

| Detector | FA/h (seed mean) |
|---|---|
| b1 | 3.044 |
| b2 | 3.130 |
| b3 | 2.272 |
| 2of3 | 1.265 |
| noisy_or | 0.000 |

**PT4:** 2-of-3 reduction 0.614 [0.477, 0.733], one-sided p (H0: ≤ 0.30) = 0.0001 -> met.

## H2b — test episodes

| Detector | Recall [95% CI] | Median TTD (s) | Detected seeds per episode |
|---|---|---|---|
| b1 | 1.000 [1.000, 1.000] | 10 | {'4': 5, '5': 5, '6': 5} |
| b2 | 0.667 [0.000, 1.000] | 325 | {'4': 5, '5': 5, '6': 0} |
| b3 | 0.333 [0.000, 0.800] | 180 | {'4': 4, '5': 1, '6': 0} |
| 2of3 | 0.667 [0.000, 1.000] | 190 | {'4': 5, '5': 5, '6': 0} |
| noisy_or | 0.000 [0.000, 0.000] | — | {'4': 0, '5': 0, '6': 0} |

Recall loss of 2-of-3 against b1: 0.333 [0.000, 1.000].
