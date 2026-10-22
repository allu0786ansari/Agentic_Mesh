### X2 (confirmatory (test))

| Family | Test | Metric | A | B | A - B [95% CI] | p (one-sided) | p Holm |
|---|---|---|---|---|---|---|---|
| H1b | FedProx vs FedAvg, homogeneous, site_a | auroc | 0.865 | 0.865 | -0.000 [-0.000, -0.000] | 0.9998 | 1.0000 |
| H1b | FedProx vs FedAvg, heterogeneous, site_a | auroc | 0.868 | 0.869 | -0.000 [-0.000, 0.000] | 0.8215 | 1.0000 |
| — | heterogeneous vs homogeneous, fedavg, site_a | auroc | 0.869 | 0.865 | +0.003 [0.000, 0.008] | 0.0211 | — |
| — | heterogeneous vs homogeneous, fedprox, site_a | auroc | 0.868 | 0.865 | +0.004 [0.000, 0.008] | 0.0155 | — |
| H1b | FedProx vs FedAvg, homogeneous, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | 1.0000 |
| H1b | FedProx vs FedAvg, heterogeneous, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | 1.0000 |
| — | heterogeneous vs homogeneous, fedavg, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | heterogeneous vs homogeneous, fedprox, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | IID vs non-IID, fedavg, site_a (unpaired) | auroc | 0.864 | 0.865 | -0.001 [-0.004, 0.002] | 0.7582 | — |
| — | IID vs non-IID, fedprox, site_a (unpaired) | auroc | 0.864 | 0.865 | -0.001 [-0.004, 0.003] | 0.7276 | — |
| — | mu 0.001 vs 0.01, homogeneous, site_a | auroc | 0.865 | 0.865 | +0.000 [0.000, 0.000] | 0.0007 | — |
| — | mu 0.001 vs 0.01, heterogeneous, site_a | auroc | 0.869 | 0.868 | +0.000 [-0.000, 0.000] | 0.2500 | — |
| — | mu 0.1 vs 0.01, homogeneous, site_a | auroc | 0.863 | 0.865 | -0.002 [-0.003, -0.000] | 0.9962 | — |
| — | mu 0.1 vs 0.01, heterogeneous, site_a | auroc | 0.866 | 0.868 | -0.002 [-0.005, -0.000] | 0.9922 | — |
