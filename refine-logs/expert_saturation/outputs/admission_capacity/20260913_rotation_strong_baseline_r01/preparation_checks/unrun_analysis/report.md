# Strong-baseline rotation controls

UNRUN

| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|---:|
| cohort2-block0-native | UNRUN | — | — | — | — | — |
| cohort2-block0-headroom | UNRUN | — | — | — | — | — |
| cohort2-block0-most_output | UNRUN | — | — | — | — | — |
| cohort2-block0-native_aa | UNRUN | — | — | — | — | — |
| cohort2-block1-native_aa | UNRUN | — | — | — | — | — |
| cohort2-block1-most_output | UNRUN | — | — | — | — | — |
| cohort2-block1-headroom | UNRUN | — | — | — | — | — |
| cohort2-block1-native | UNRUN | — | — | — | — | — |

## Primary paired observations

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → headroom | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/0 | native → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/0 | headroom → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/1 | native → headroom | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/1 | native → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/1 | headroom → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |

## Native A/A observed drift

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → native_aa | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/1 | native → native_aa | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |

All eight cells must qualify with common engine/config before numeric pairing. Per block: native→headroom, native→most_output, headroom→most_output. Native A/A and same-role repeats are separate; primary native is never replaced or drift-corrected. Only completion_policy and verified rotation_victim_order differ; cap stays 32.

One document cohort, two reverse-order blocks and four roles; eight engines are not eight independent workloads. Primary native is the sole native baseline; native A/A is retained observed drift, never a replacement baseline, correction or noise bound. No significance, quality equivalence, noninferiority or method GO is inferred. Reference 5 s TTFT / 0.2 s mean-TPOT SLO does not constrain maximum ITL.

wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is included in scheduler time. Disjoint work classes retain prefill/recompute, new decode, held work and every completion/failure. Recovery spans can contain useful concurrent decode. Width/path changes are observed costs, not independent causal savings.

Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.
