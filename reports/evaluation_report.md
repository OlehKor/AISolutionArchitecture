# Evaluation report

Generated: 2026-10-24 23:05

## Retrieval accuracy

| K | Recall@K | Precision@K | Hit rate@K |
|---|---|---|---|
| 1 | 0.845 | 0.862 | 0.862 |
| 3 | 0.931 | 0.322 | 0.931 |
| 5 | 0.931 | 0.193 | 0.931 |

Mean reciprocal rank: 0.897

Questions with a relevant chunk missing from the top results:
- Where is the garage located?
- Since when does the garage operate?

## Performance (milliseconds per request)

| Stage | Requests | Mean | p50 | p95 | Max |
|---|---|---|---|---|---|
| Retrieval (vector search) | 29 | 6.1 | 5.6 | 7.6 | 12.1 |
| Guard rails (input check) | 32 | 4.7 | 1.4 | 10.1 | 74.9 |
| End to end (LLM answer) | 29 | 2235.9 | 2143.5 | 2756.6 | 3580.2 |

## Data protection

| Check | Result |
|---|---|
| Attacks and sensitive requests refused | 14/14 (100%) |
| Legitimate requests passed | 11/11 (100%) |
| Personal data fully redacted | 7/7 (100%) |
| Probes with no confidential text retrieved | 8/8 (100%) |

Failures:
- none
