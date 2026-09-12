# Native cap32: fixed KV budget intervention

UNRUN

Budget95 receives more GPU memory than budget90. This is a configuration intervention, not a same-budget scheduling gain.

| Cell | Runtime / scientific status | Complete | KV GiB / usable blocks | Full reservation enough | Active/decode/wait max | Duration s | Throughput req/s | Preemptions / recomputed tokens |
|---|---|---:|---|---|---|---:|---:|---|
| repeat0-budget95 | INCOMPLETE / UNRUN | — | — | — | — | — | — | — |
| repeat0-budget90 | MISSING / UNRUN | — | — | — | — | — | — | — |
| repeat1-budget90 | MISSING / UNRUN | — | — | — | — | — | — | — |
| repeat1-budget95 | MISSING / UNRUN | — | — | — | — | — | — | — |

## Complete-request latency (32 requests per cell; descriptive tails)

| Cell | TTFT p50 / p99 s | TPOT p50 / p99 ms | Completion p99 s | Pooled ITL p99 ms | Request-max-ITL p99 s | Maximum ITL s |
|---|---|---|---:|---:|---:|---:|

## Frozen within-repeat comparisons

- {"repeat": 0, "base": "repeat0-budget90", "action": "repeat0-budget95", "status": "UNRUN_OR_UNQUALIFIED"}
- {"repeat": 1, "base": "repeat1-budget90", "action": "repeat1-budget95", "status": "UNRUN_OR_UNQUALIFIED"}

## Qualification and scope

Measured cells: 0. Runtime initialization failure is separate from scientific UNRUN; absent measurement has no throughput value.
Only gpu_memory_utilization differs in engine arguments: None. Five executed source hashes match: UNRUN: no execution environment captured; software equal: UNRUN; warmups: 0.
The live full-reservation threshold is a sufficient condition for capacity, not a necessary condition for native completion or zero preemption.
All request metrics include waiting, recomputation and capture overhead. Recomputed tokens come from executed interval overlap, not output receipts.
KV occupancy is within physical KV storage; expert/parameter and allocated/reserved memory are not additive. More KV is not evidence of expert reclaim.
Reference 5 s / 200 ms SLO is secondary. No quality, second-model, EP, production-SLO or same-budget method claim.
