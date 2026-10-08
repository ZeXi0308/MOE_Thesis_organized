# Native versus retained-KV completion headroom

MEASUREMENT_ONLY

| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Max ITL s | Preemptions / recomputed positions | Held requests / longest held steps | Decision s |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| repeat0-native | COMPLETE | 32/32 | 22.892435 | 1.397842 | 20.537725 | 4.487516 | 2 / 7685 | 0 / 0 | 0.014793 |
| repeat0-headroom | COMPLETE | 32/32 | 23.660591 | 1.352460 | 22.100533 | 1.386702 | 0 / 0 | 31 / 227 | 0.221457 |
| repeat1-headroom | COMPLETE | 32/32 | 23.588630 | 1.356586 | 22.030503 | 1.385728 | 0 / 0 | 31 / 227 | 0.219039 |
| repeat1-native | COMPLETE | 32/32 | 22.913837 | 1.396536 | 20.544998 | 4.473709 | 2 / 7685 | 0 / 0 | 0.013802 |

All request identities, metrics, paired differences and same-policy repeats are retained in analysis.json. Pairing requires all four cells COMPLETE.

Single-model native in-process fixed-pool pilot. Host request time includes waiting, holding, recomputation and instrumentation; work counts and decision time are not pure GPU time. Repeated 32-request cohorts do not establish production tails, quality equivalence, generalization or a method GO. Reference TTFT/mean-TPOT SLO does not constrain maximum ITL.
