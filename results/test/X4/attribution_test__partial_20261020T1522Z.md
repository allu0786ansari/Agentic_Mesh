# X4 — attribution on TEST incidents

Protocol: docs/adr/0017-x3-x6-week13.md §4; docs/adr/0019-g4-final-test-protocol-week14.md §5. Tolerance 60 s; bootstrap B = 10000, seed 2026.

## Site A (35 events)

| Mode | Run | Matched | Abstained | Top-1 (95% CI) | Top-3 (95% CI) | Top-1 answered | Tactic answered | FP incidents | Raw grounding | Distinct answers (modal share) |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval_only | site_a__retrieval_only__test__20261020T205050 | 24/35 | 0 | 2.9% [0.0%, 8.6%] | 28.6% [14.3%, 42.9%] | 4.2% | 8.3% | 30/55 | — | 6 (T0801 50.0%) |
| rule_only | site_a__rule_only__test__20261020T205023 | 24/35 | 0 | 48.6% [31.4%, 65.7%] | 48.6% [31.4%, 65.7%] | 70.8% | 100.0% | 30/55 | — | 2 (T1692.001 91.7%) |

Majority-class baseline (T1692.001): 62.9% [45.7%, 77.1%]. Constant answers (all 50 events / these 35): T0806 20.0% / 22.9%; T0836 64.0% / 60.0%; T1692.001 64.0% / 62.9%; T1692.002 38.0% / 42.9%.

## Checks

FAILED
- site_a: no full run with sessions.jsonl
- site_a: no llm_no_retrieval run with sessions.jsonl
- site_b: no full run with sessions.jsonl
- site_b: no retrieval_only run with sessions.jsonl
- site_b: no llm_no_retrieval run with sessions.jsonl
- site_b: no rule_only run with sessions.jsonl
