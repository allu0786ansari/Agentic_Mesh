# X4 — attribution on TEST incidents

Protocol: docs/adr/0017-x3-x6-week13.md §4; docs/adr/0019-g4-final-test-protocol-week14.md §5. Tolerance 60 s; bootstrap B = 10000, seed 2026.

## Site A (35 events)

| Mode | Run | Matched | Abstained | Top-1 (95% CI) | Top-3 (95% CI) | Top-1 answered | Tactic answered | FP incidents | Raw grounding | Distinct answers (modal share) |
|---|---|---|---|---|---|---|---|---|---|---|
| full | site_a__full__test__20261020T205536 | 24/35 | 0 | 37.1% [22.9%, 54.3%] | 40.0% [22.9%, 57.1%] | 54.2% | 70.8% | 30/55 | 100.0% | 4 (T0836 54.2%) |
| retrieval_only | site_a__retrieval_only__test__20261020T205050 | 24/35 | 0 | 2.9% [0.0%, 8.6%] | 28.6% [14.3%, 42.9%] | 4.2% | 8.3% | 30/55 | — | 6 (T0801 50.0%) |
| llm_no_retrieval | site_a__llm_no_retrieval__test__20261020T211156 | 24/35 | 0 | 42.9% [25.7%, 60.0%] | 42.9% [25.7%, 60.0%] | 62.5% | 100.0% | 30/55 | 100.0% | 1 (T0836 100.0%) **constant** |
| rule_only | site_a__rule_only__test__20261020T205023 | 24/35 | 0 | 48.6% [31.4%, 65.7%] | 48.6% [31.4%, 65.7%] | 70.8% | 100.0% | 30/55 | — | 2 (T1692.001 91.7%) |

Majority-class baseline (T1692.001): 62.9% [45.7%, 77.1%]. Constant answers (all 50 events / these 35): T0806 20.0% / 22.9%; T0836 64.0% / 60.0%; T1692.001 64.0% / 62.9%; T1692.002 38.0% / 42.9%.

`full` - `retrieval_only`, top-3 (paired event bootstrap): +0.114 [-0.029, +0.257], p = 0.136.

## Site B (case study, no accuracy)

| Episode | Acceptable | full | retrieval_only | llm_no_retrieval | rule_only |
|---|---|---|---|---|---|
| 4 | T0801, T0861 | T0814 | T0814 | T0835 | T1692.001 |
| 5 | T0806, T1692.001, T0814 | T0814 | T0814 | T0835 | T0806 |
| 6 | T1692.001 | T0801 | T0814 | T0835 | T1692.001 |

| Mode | Incidents | LLM calls | Schema valid | Raw grounding | Distinct answers over incidents (modal share) |
|---|---|---|---|---|---|
| full | 66 | 66 | 100.0% | 100.0% | 5 (T0814 63.6%) |
| retrieval_only | 66 | 0 | — | — | 1 (T0814 100.0%) **constant** |
| llm_no_retrieval | 66 | 66 | 100.0% | 100.0% | 3 (T0835 86.4%) |
| rule_only | 66 | 0 | — | — | 4 (T0861 77.3%) |

Site B, every incident matching any of the seven episodes (0-3 validation, 4-6 test):

- full: 10 incidents, schema valid 100.0%, raw grounding 100.0%
- llm_no_retrieval: 10 incidents, schema valid 100.0%, raw grounding 100.0%

## PT5 (`full`)

- top-3 40.0% [22.9%, 57.1%] against 0.70: not met
- above the preregistered majority baseline 0.64: no (on these events: 62.9%)
- `full` - `retrieval_only` top-3: +0.114 [-0.029, +0.257]: met
- Site B JSON validity over the seven episodes: 100.0%: met
- **PT5: not met**

## Checks

passed
