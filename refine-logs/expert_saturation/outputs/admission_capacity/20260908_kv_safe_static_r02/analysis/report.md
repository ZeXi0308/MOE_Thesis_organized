# Safe static versus baseline16: retained analysis

MEASUREMENT_ONLY

Four fixed cells, no post-hoc cap choice. Complete and truncated episodes are separated; reference SLO is secondary.

| Cell | Cap/derived safe | Status | Complete | Active/decode/wait max | Peak used/usable | Full duration s | Throughput req/s | TTFT p50 s | TPOT p50 ms |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| repeat0-baseline16 | 16/29 | COMPLETE | 32/32 | 16/16/16 | 4079/7677 | 28.34621 | 1.12890 | 6.78208 | 13.55194 |
| repeat0-safe | 29/29 | COMPLETE | 32/32 | 29/29/9 | 7357/7677 | 27.32225 | 1.17121 | 0.36015 | 18.64872 |
| repeat1-safe | 29/29 | COMPLETE | 32/32 | 29/29/9 | 7357/7677 | 27.16423 | 1.17802 | 0.35620 | 18.51731 |
| repeat1-baseline16 | 16/29 | COMPLETE | 32/32 | 16/16/16 | 4079/7677 | 28.39588 | 1.12692 | 6.80451 | 13.56218 |

## Complete-request tails (32 requests per cell; descriptive p99)

| Cell | TTFT p99 s | Per-request mean TPOT p99 ms | Peak KV occupied | Steps with waiting / all scheduled steps |
|---|---:|---:|---:|---|
| repeat0-baseline16 | 13.60198 | 13.68674 | 53.1327% | 1063/2099 |
| repeat0-safe | 18.04667 | 18.85659 | 95.8317% | 1029/2060 |
| repeat1-safe | 17.90815 | 18.73819 | 95.8317% | 1025/2060 |
| repeat1-baseline16 | 13.62870 | 13.67471 | 53.1327% | 1069/2099 |

## Fixed comparisons

- {"repeat": 0, "baseline": "repeat0-baseline16", "safe": "repeat0-safe", "safe_cap": 29, "status": "DESCRIPTIVE_COMPLETE_EPISODE_COMPARISON", "throughput_relative_change": 0.037477341730600466, "duration_relative_change": -0.03612352792985807, "ttft_p50_change_s": -6.4219240341335535, "ttft_p99_change_s": 4.4446866935491585, "tpot_p50_change_ms": 5.096775791876361, "tpot_p99_change_ms": 5.1698460414612395, "reference_slo_pass_change": 13}
- {"repeat": 1, "baseline": "repeat1-baseline16", "safe": "repeat1-safe", "safe_cap": 29, "status": "DESCRIPTIVE_COMPLETE_EPISODE_COMPARISON", "throughput_relative_change": 0.045341191307566975, "duration_relative_change": -0.04337453807866476, "ttft_p50_change_s": -6.44831077195704, "ttft_p99_change_s": 4.279448979757728, "tpot_p50_change_ms": 4.9551295686443995, "tpot_p99_change_ms": 5.063481223883285, "reference_slo_pass_change": 13}

## Qualification and limits

Engine arguments equal: True; execution sources including safe_static.py match: True; software equal: True; warmup raw count: 12.
Each derived cap comes from its own engine pool. Different derived caps are repetitions of the formula, not the same numeric action.
No unrun or boundary cell enters full-episode throughput comparison. No expert-reclaim, Oracle, task-quality or MoE-method claim.
KV occupied blocks lie inside the physical KV region; parameter/expert, allocated/reserved quantities are not additive.
The 32 repeated articles and four independent engine processes do not establish population-level statistics.
