# X4 — attribution on validation incidents (development evidence)

Protocol: docs/adr/0017-x3-x6-week13.md §4. Tolerance 60 s; bootstrap B = 10000, seed 2026.

## Site A (15 events)

| Mode | Run | Matched | Abstained | Top-1 (95% CI) | Top-3 (95% CI) | Top-1 answered | Tactic answered | FP incidents | Raw grounding | Distinct answers (modal share) |
|---|---|---|---|---|---|---|---|---|---|---|
| full | site_a__full__val__20261017T193228 | 14/15 | 0 | 46.7% [20.0%, 73.3%] | 53.3% [26.7%, 80.0%] | 50.0% | 57.1% | 31/45 | 100.0% | 5 (T0836 35.7%) |
| retrieval_only | site_a__retrieval_only__val__20261017T193158 | 14/15 | 0 | 0.0% [0.0%, 0.0%] | 33.3% [13.3%, 60.0%] | 0.0% | 0.0% | 31/45 | — | 5 (T0801 35.7%) |
| llm_no_retrieval | site_a__llm_no_retrieval__val__20261017T171042 | 14/15 | 0 | 73.3% [46.7%, 93.3%] | 73.3% [46.7%, 93.3%] | 78.6% | 100.0% | 31/45 | 100.0% | 1 (T0836 100.0%) **constant** |
| rule_only | site_a__rule_only__val__20261016T015627 | 14/15 | 0 | 53.3% [26.7%, 80.0%] | 53.3% [26.7%, 80.0%] | 57.1% | 100.0% | 31/45 | — | 3 (T1692.001 78.6%) |

Majority-class baseline (T0836): 73.3% [46.7%, 93.3%]. Constant answers (all 50 events / these 15): T0806 20.0% / 13.3%; T0836 64.0% / 73.3%; T1692.001 64.0% / 66.7%; T1692.002 38.0% / 26.7%.

`full` - `retrieval_only`, top-3 (paired event bootstrap): +0.200 [+0.000, +0.400], p = 0.067.

## Site B (case study, no accuracy)

| Episode | Acceptable | full | retrieval_only | llm_no_retrieval | rule_only |
|---|---|---|---|---|---|
| 0 | T0888, T0861, T0846, T0814 | T0806 | T0814 | T0836 | T0814 |
| 1 | T0861, T0888, T0814 | T0806 | T0814 | T0836 | T0814 |
| 2 | T0861, T0888 | T0806 | T0814 | T0836 | T0814 |
| 3 | T0861, T0801 | T0806 | T0814 | T0836 | T0814 |

| Mode | Incidents | LLM calls | Schema valid | Raw grounding | Distinct answers over incidents (modal share) |
|---|---|---|---|---|---|
| full | 220 | 220 | 100.0% | 100.0% | 6 (T0814 61.8%) |
| retrieval_only | 220 | 0 | — | — | 1 (T0814 100.0%) **constant** |
| llm_no_retrieval | 220 | 220 | 100.0% | 100.0% | 4 (T0835 79.5%) |
| rule_only | 220 | 0 | — | — | 3 (T0861 78.6%) |

## Checks

passed
