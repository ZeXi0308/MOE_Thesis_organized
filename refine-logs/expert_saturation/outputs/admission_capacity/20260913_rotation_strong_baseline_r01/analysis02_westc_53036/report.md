# Strong-baseline rotation controls

MEASUREMENT_ONLY

| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|---:|
| cohort2-block0-native | COMPLETE | 32/32 | 23.982332 | 1.334316 | 21.423385 | 4.571520 |
| cohort2-block0-headroom | COMPLETE | 32/32 | 25.189647 | 1.270363 | 23.580178 | 1.542319 |
| cohort2-block0-most_output | COMPLETE | 32/32 | 24.313487 | 1.316142 | 22.700631 | 1.519854 |
| cohort2-block0-native_aa | COMPLETE | 32/32 | 23.843607 | 1.342079 | 21.330607 | 4.698591 |
| cohort2-block1-native_aa | COMPLETE | 32/32 | 23.833761 | 1.342633 | 21.372369 | 4.690422 |
| cohort2-block1-most_output | COMPLETE | 32/32 | 23.342345 | 1.370899 | 21.824907 | 1.020476 |
| cohort2-block1-headroom | COMPLETE | 32/32 | 24.580290 | 1.301856 | 22.974216 | 1.566409 |
| cohort2-block1-native | COMPLETE | 32/32 | 24.098925 | 1.327860 | 21.602143 | 4.760450 |

## Primary paired observations

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -4.793% | +10.067% | -3.029202 |
| cohort2/0 | native → most_output | DESCRIPTIVE_MATCHED_PAIR | -1.362% | +5.962% | -3.051666 |
| cohort2/0 | headroom → most_output | DESCRIPTIVE_MATCHED_PAIR | +3.604% | -3.730% | -0.022465 |
| cohort2/1 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -1.958% | +6.352% | -3.194041 |
| cohort2/1 | native → most_output | DESCRIPTIVE_MATCHED_PAIR | +3.241% | +1.031% | -3.739974 |
| cohort2/1 | headroom → most_output | DESCRIPTIVE_MATCHED_PAIR | +5.303% | -5.003% | -0.545933 |

## Native A/A observed drift

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | +0.582% | -0.433% | +0.127070 |
| cohort2/1 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | +1.113% | -1.064% | -0.070028 |

All eight cells must qualify with common engine/config before numeric pairing. Per block: native→headroom, native→most_output, headroom→most_output. Native A/A and same-role repeats are separate; primary native is never replaced or drift-corrected. Only completion_policy and verified rotation_victim_order differ; cap stays 32.

One document cohort, two reverse-order blocks and four roles; eight engines are not eight independent workloads. Primary native is the sole native baseline; native A/A is retained observed drift, never a replacement baseline, correction or noise bound. No significance, quality equivalence, noninferiority or method GO is inferred. Reference 5 s TTFT / 0.2 s mean-TPOT SLO does not constrain maximum ITL.

wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is included in scheduler time. Disjoint work classes retain prefill/recompute, new decode, held work and every completion/failure. Recovery spans can contain useful concurrent decode. Width/path changes are observed costs, not independent causal savings.

Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.
