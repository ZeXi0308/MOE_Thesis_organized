# Safe static versus baseline16: retained analysis

INCOMPLETE_OR_UNRUN

Four fixed cells, no post-hoc cap choice. Complete and truncated episodes are separated; reference SLO is secondary.

| Cell | Cap/derived safe | Status | Complete | Active/decode/wait max | Peak used/usable | Full duration s | Throughput req/s | TTFT p50 s | TPOT p50 ms |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| repeat0-baseline16 | — | UNRUN | — | — | — | — | — | — | — |
| repeat0-safe | — | MISSING | — | — | — | — | — | — | — |
| repeat1-safe | — | MISSING | — | — | — | — | — | — | — |
| repeat1-baseline16 | — | MISSING | — | — | — | — | — | — | — |

## Fixed comparisons

- {"repeat": 0, "baseline": "repeat0-baseline16", "safe": "repeat0-safe", "safe_cap": null, "status": "UNRUN_OR_UNQUALIFIED"}
- {"repeat": 1, "baseline": "repeat1-baseline16", "safe": "repeat1-safe", "safe_cap": null, "status": "UNRUN_OR_UNQUALIFIED"}

## Qualification and limits

Engine arguments equal: False; execution sources including safe_static.py match: True; software equal: True; warmup raw count: 0.
Each derived cap comes from its own engine pool. Different derived caps are repetitions of the formula, not the same numeric action.
No unrun or boundary cell enters full-episode throughput comparison. No expert-reclaim, Oracle, task-quality or MoE-method claim.
KV occupied blocks lie inside the physical KV region; parameter/expert, allocated/reserved quantities are not additive.
The 32 repeated articles and four independent engine processes do not establish population-level statistics.
