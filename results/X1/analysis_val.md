### X1 (development evidence (validation))

| Family | Test | Metric | A | B | A - B [95% CI] | p (one-sided) | p Holm |
|---|---|---|---|---|---|---|---|
| RQ1 | PT1 site_a | auroc | 0.891 | 0.948 | -0.057 [-0.100, -0.029] | 0.6462 | 1.0000 |
| RQ1 | PT2 site_a | auroc | 0.891 | 0.922 | -0.030 [-0.063, 0.004] | 0.9625 | 1.0000 |
| — | PT1 site_a, eps10 (secondary) | auroc | 0.909 | 0.948 | -0.040 [-0.072, -0.017] | 0.2504 | — |
| — | PT2 site_a, FedAvg eps3 (secondary) | auroc | 0.892 | 0.922 | -0.030 [-0.063, 0.004] | 0.9620 | — |
| — | cost of DP, central, site_a | auroc | 0.931 | 0.894 | +0.038 [0.016, 0.073] | 0.0001 | — |
| — | cost of federation at eps3, site_a | auroc | 0.894 | 0.891 | +0.002 [-0.008, 0.014] | 0.3101 | — |
| — | cost of federation at eps inf, site_a | auroc | 0.931 | 0.930 | +0.002 [-0.012, 0.015] | 0.4022 | — |
| — | federation vs local-only at eps3, site_a | auroc | 0.891 | 0.878 | +0.014 [-0.012, 0.046] | 0.1649 | — |
| RQ1 | PT1 site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | 0.0004 |
| RQ1 | PT2 site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.001] | 0.0001 | 0.0004 |
| — | PT1 site_b, eps10 (secondary) | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | — |
| — | PT2 site_b, FedAvg eps3 (secondary) | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.001] | 0.0001 | — |
| — | cost of DP, central, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | cost of federation at eps3, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | cost of federation at eps inf, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | federation vs local-only at eps3, site_b | auroc | 1.000 | 0.998 | +0.002 [0.001, 0.002] | 0.0001 | — |
| H1c | H1c FPR: pooled vs per-node R0 | window_fpr | 0.009 | 0.049 | -0.040 [-0.054, -0.026] | 0.0001 | 0.0002 |
| — | H1c recall: pooled vs per-node R0 (within 0.05) | event_recall | 0.875 | 0.887 | -0.012 [-0.037, 0.000] | 0.0045 | — |
| H1c | H1c FPR: pooled vs per-node R3 | window_fpr | 0.009 | 0.014 | -0.005 [-0.011, -0.000] | 0.0193 | 0.0193 |
| — | H1c recall: pooled vs per-node R3 (within 0.05) | event_recall | 0.875 | 0.938 | -0.062 [-0.188, 0.000] | 0.6859 | — |
