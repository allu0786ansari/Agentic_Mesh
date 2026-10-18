# Robustness suite — synthetic perturbations of normal validation

Seeds per site (only models that reproduce X1, ADR 0017 §1): site_a [11, 53], excluded ['23', '37', '41']; site_b [11, 23, 37, 41, 53].

20 injections per kind and node (placement seed 2026). Descriptive; nothing is selected or tuned on these numbers. SYNTHETIC.

Per kind (seeds averaged): detection rate / the same rule on the unperturbed stream at the same placements (chance, from benign alerts at the 1% window FPR) / window FPR on untouched windows. Clean window FPR for reference.

| Site | Node | spike | drift | stuck | replay | clean FPR |
|---|---|---|---|---|---|---|
| site_a | a1 | 0.47 / 0.00 / 0.0092 | 0.05 / 0.00 / 0.0100 | 0.20 / 0.15 / 0.0096 | 0.23 / 0.15 / 0.0105 | 0.0097 |
| site_a | a2 | 0.35 / 0.15 / 0.0078 | 0.12 / 0.00 / 0.0092 | 0.05 / 0.05 / 0.0099 | 0.05 / 0.05 / 0.0119 | 0.0099 |
| site_a | a3 | 0.53 / 0.00 / 0.0096 | 0.15 / 0.10 / 0.0091 | 0.15 / 0.15 / 0.0101 | 0.12 / 0.05 / 0.0112 | 0.0099 |
| site_a | a4 | 0.35 / 0.10 / 0.0091 | 0.03 / 0.03 / 0.0103 | 0.03 / 0.03 / 0.0111 | 0.05 / 0.10 / 0.0105 | 0.0099 |
| site_b | b1 | 0.33 / 0.03 / 0.0097 | 0.65 / 0.48 / 0.0098 | 0.66 / 0.62 / 0.0094 | 0.37 / 0.40 / 0.0097 | 0.0097 |
| site_b | b2 | 0.41 / 0.13 / 0.0099 | 0.50 / 0.29 / 0.0100 | 0.38 / 0.28 / 0.0102 | 0.63 / 0.42 / 0.0100 | 0.0099 |
| site_b | b3 | 0.42 / 0.10 / 0.0098 | 0.71 / 0.45 / 0.0099 | 0.58 / 0.49 / 0.0099 | 0.22 / 0.31 / 0.0100 | 0.0099 |
