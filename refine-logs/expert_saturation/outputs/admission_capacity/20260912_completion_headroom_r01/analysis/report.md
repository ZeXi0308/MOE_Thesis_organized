# Native versus retained-KV completion headroom

MEASUREMENT_ONLY

| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Max ITL s | Preemptions / recomputed positions | Held requests / longest held steps | Decision s |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| repeat0-native | COMPLETE | 32/32 | 22.913784 | 1.396539 | 20.556992 | 4.468961 | 2 / 7685 | 0 / 0 | 0.013645 |
| repeat0-headroom | COMPLETE | 32/32 | 23.828437 | 1.342933 | 22.259208 | 1.381322 | 0 / 0 | 31 / 227 | 0.455992 |
| repeat1-headroom | COMPLETE | 32/32 | 23.803814 | 1.344322 | 22.240772 | 1.381332 | 0 / 0 | 31 / 227 | 0.487366 |
| repeat1-native | COMPLETE | 32/32 | 22.875082 | 1.398902 | 20.506041 | 4.448064 | 2 / 7685 | 0 / 0 | 0.012745 |

All request identities, metrics, paired differences and same-policy repeats are retained in analysis.json. Pairing requires all four cells COMPLETE.

Single-model native in-process fixed-pool pilot. Host request time includes waiting, holding, recomputation and instrumentation; work counts and decision time are not pure GPU time. Repeated 32-request cohorts do not establish production tails, quality equivalence, generalization or a method GO. Reference TTFT/mean-TPOT SLO does not constrain maximum ITL.
