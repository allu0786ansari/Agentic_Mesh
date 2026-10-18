### X2 (development evidence (validation))

| Family | Test | Metric | A | B | A - B [95% CI] | p (one-sided) | p Holm |
|---|---|---|---|---|---|---|---|
| H1b | FedProx vs FedAvg, homogeneous, site_a | auroc | 0.891 | 0.892 | -0.000 [-0.000, -0.000] | 0.9937 | 1.0000 |
| H1b | FedProx vs FedAvg, heterogeneous, site_a | auroc | 0.896 | 0.896 | -0.000 [-0.001, 0.000] | 0.9610 | 1.0000 |
| — | heterogeneous vs homogeneous, fedavg, site_a | auroc | 0.896 | 0.892 | +0.004 [0.000, 0.008] | 0.0175 | — |
| — | heterogeneous vs homogeneous, fedprox, site_a | auroc | 0.896 | 0.891 | +0.004 [0.000, 0.008] | 0.0195 | — |
| H1b | FedProx vs FedAvg, homogeneous, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | 1.0000 |
| H1b | FedProx vs FedAvg, heterogeneous, site_b | auroc | 1.000 | 0.999 | +0.001 [0.001, 0.001] | 0.0001 | 0.0004 |
| — | heterogeneous vs homogeneous, fedavg, site_b | auroc | 0.999 | 1.000 | -0.001 [-0.001, -0.001] | 1.0000 | — |
| — | heterogeneous vs homogeneous, fedprox, site_b | auroc | 1.000 | 1.000 | +0.000 [0.000, 0.000] | 1.0000 | — |
| — | IID vs non-IID, fedavg, site_a (unpaired) | auroc | 0.888 | 0.892 | -0.004 [-0.007, -0.002] | 0.9996 | — |
| — | IID vs non-IID, fedprox, site_a (unpaired) | auroc | 0.888 | 0.891 | -0.004 [-0.007, -0.001] | 0.9995 | — |
| — | mu 0.001 vs 0.01, homogeneous, site_a | auroc | 0.892 | 0.891 | +0.000 [0.000, 0.000] | 0.0011 | — |
| — | mu 0.001 vs 0.01, heterogeneous, site_a | auroc | 0.896 | 0.896 | +0.000 [-0.000, 0.001] | 0.0438 | — |
| — | mu 0.1 vs 0.01, homogeneous, site_a | auroc | 0.889 | 0.891 | -0.002 [-0.004, -0.001] | 1.0000 | — |
| — | mu 0.1 vs 0.01, heterogeneous, site_a | auroc | 0.893 | 0.896 | -0.003 [-0.006, 0.000] | 0.9707 | — |
