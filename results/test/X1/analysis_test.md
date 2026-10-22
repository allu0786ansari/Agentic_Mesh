### X1 (confirmatory (test))

| Family | Test | Metric | A | B | A - B [95% CI] | p (one-sided) | p Holm |
|---|---|---|---|---|---|---|---|
| RQ1 | PT1 site_a | auroc | 0.865 | 0.929 | -0.064 [-0.100, -0.037] | 0.8335 | 1.0000 |
| RQ1 | PT2 site_a | auroc | 0.865 | 0.926 | -0.061 [-0.101, -0.030] | 1.0000 | 1.0000 |
| — | PT1 site_a, eps10 (secondary) | auroc | 0.881 | 0.929 | -0.048 [-0.078, -0.024] | 0.4465 | — |
| — | PT2 site_a, FedAvg eps3 (secondary) | auroc | 0.865 | 0.926 | -0.061 [-0.101, -0.030] | 1.0000 | — |
| — | cost of DP, central, site_a | auroc | 0.896 | 0.872 | +0.024 [0.001, 0.048] | 0.0187 | — |
| — | cost of federation at eps3, site_a | auroc | 0.872 | 0.865 | +0.007 [-0.006, 0.021] | 0.1392 | — |
| — | cost of federation at eps inf, site_a | auroc | 0.896 | 0.882 | +0.014 [-0.005, 0.034] | 0.0735 | — |
| — | federation vs local-only at eps3, site_a | auroc | 0.865 | 0.841 | +0.024 [0.005, 0.046] | 0.0032 | — |
| RQ1 | PT1 site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | 0.0004 |
| RQ1 | PT2 site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | 0.0004 |
| — | PT1 site_b, eps10 (secondary) | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | — |
| — | PT2 site_b, FedAvg eps3 (secondary) | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 0.0001 | — |
| — | cost of DP, central, site_b | auroc | 1.000 | 1.000 | +0.000 [-0.000, 0.000] | 0.6011 | — |
| — | cost of federation at eps3, site_b | auroc | 1.000 | 1.000 | -0.000 [-0.000, 0.000] | 1.0000 | — |
| — | cost of federation at eps inf, site_b | auroc | 1.000 | 1.000 | +0.000 [-0.000, 0.000] | 0.2536 | — |
| — | federation vs local-only at eps3, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| H1c | H1c FPR: pooled vs per-node R0 | window_fpr | 0.006 | 0.031 | -0.025 [-0.036, -0.016] | 0.0001 | 0.0002 |
| — | H1c recall: pooled vs per-node R0 (within 0.05) | event_recall | 0.710 | 0.743 | -0.033 [-0.089, 0.000] | 0.2655 | — |
| H1c | H1c FPR: pooled vs per-node R3 | window_fpr | 0.006 | 0.008 | -0.001 [-0.005, 0.002] | 0.1946 | 0.1946 |
| — | H1c recall: pooled vs per-node R3 (within 0.05) | event_recall | 0.710 | 0.738 | -0.028 [-0.106, 0.044] | 0.2930 | — |
