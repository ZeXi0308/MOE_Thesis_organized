# Native / safe29 / headroom / absence rotation

MEASUREMENT_ONLY

| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Max ITL s | Preemptions / recomputed positions | Held requests / longest held steps | Decision s |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| repeat0-native | COMPLETE | 32/32 | 22.804285 | 1.403245 | 20.443663 | 4.443723 | 2 / 7685 | 0 / 0 | 0.013327 |
| repeat0-safe29 | COMPLETE | 32/32 | 26.832109 | 1.192601 | 19.810509 | 0.308060 | 0 / 0 | 0 / 0 | 0.014588 |
| repeat0-headroom | COMPLETE | 32/32 | 23.488375 | 1.362376 | 21.934057 | 1.381525 | 0 / 0 | 31 / 227 | 0.213042 |
| repeat0-rotate | COMPLETE | 32/32 | 22.929957 | 1.395554 | 20.902513 | 1.006017 | 10 / 38561 | 0 / 0 | 0.055014 |
| repeat1-rotate | COMPLETE | 32/32 | 22.902337 | 1.397237 | 20.867643 | 1.006155 | 10 / 38561 | 0 / 0 | 0.053548 |
| repeat1-headroom | COMPLETE | 32/32 | 23.415711 | 1.366604 | 21.863351 | 1.370199 | 0 / 0 | 31 / 227 | 0.208757 |
| repeat1-safe29 | COMPLETE | 32/32 | 26.563681 | 1.204652 | 19.543185 | 0.095501 | 0 / 0 | 0 / 0 | 0.015074 |
| repeat1-native | COMPLETE | 32/32 | 22.914865 | 1.396473 | 20.539101 | 4.483895 | 2 / 7685 | 0 / 0 | 0.013729 |

| Cell | Natural preemptions | Forced preemptions |
|---|---:|---:|
| repeat0-native | 2 | 0 |
| repeat0-safe29 | 0 | 0 |
| repeat0-headroom | 0 | 0 |
| repeat0-rotate | 2 | 8 |
| repeat1-rotate | 2 | 8 |
| repeat1-headroom | 0 | 0 |
| repeat1-safe29 | 0 | 0 |
| repeat1-native | 2 | 0 |

All request identities, metrics, paired differences and same-policy repeats are retained in analysis.json. All eight cells must be COMPLETE; compare all six within-repeat pairs and each policy repeat. Safe29 alone changes admission cap; engine maximum and physical KV pool stay fixed.

Single-model native in-process fixed-pool pilot. Host request time includes waiting, holding, recomputation and instrumentation; work counts and decision time are not pure GPU time. Repeated 32-request cohorts do not establish production tails, quality equivalence, generalization or a method GO. Reference TTFT/mean-TPOT SLO does not constrain maximum ITL.
