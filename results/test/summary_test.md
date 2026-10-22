# Test-split summary (after g4-frozen)

| Target | Test result | Verdict |
|---|---|---|
| PT1 site_a | -0.064 [-0.100, -0.037], Holm p 1.0000 | **not met** |
| PT2 site_a | -0.061 [-0.101, -0.030], Holm p 1.0000 | **not met** |
| PT1 site_b | +0.000 [+0.000, +0.000], Holm p 0.0004 | met |
| PT2 site_b | +0.000 [+0.000, +0.000], Holm p 0.0004 | met |
| PT3 site_a | recall 0.676 [0.523, 0.819] (pooled over 35: 0.674); window FPR 0.0083 | **not met** |
| PT4 site_b | reduction 0.614 [0.477, 0.733], p 0.0001 | met |
| PT5 site_a (`full`) | top-3 0.400 [0.229, 0.571]; majority on these events 0.629 | **not met** |
| PT6-v2 | Recall@10 0.889 on 27 queries (11 incidents without a query; 0.632 if counted as misses) | met |
| PT7 | every segment within its p95 target | met |

Validation re-check inside the final runs: 249/250 within the Plan's tolerance (192 exact); largest |dAUROC| 0.0056.

- not within tolerance: results/test/X1/site_a__central__vae__epsinf/seed_41.json (|dAUROC| 0.0056)
