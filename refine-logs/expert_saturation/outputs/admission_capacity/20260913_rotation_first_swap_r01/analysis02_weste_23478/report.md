# First-successful-swap diagnostic

MEASUREMENT_ONLY

| Cell | Status | Forced / natural | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress | COMPLETE | 8 / 2 | 1.404776 | 20.747718 | 1.009063 |
| cohort0-block0-first_most_then_least | COMPLETE | 8 / 2 | 1.407102 | 20.741040 | 1.010542 |
| cohort0-block0-most_output | COMPLETE | 9 / 2 | 1.426094 | 20.964272 | 0.981989 |
| cohort0-block1-most_output | COMPLETE | 9 / 2 | 1.423462 | 21.002803 | 0.979500 |
| cohort0-block1-first_most_then_least | COMPLETE | 8 / 2 | 1.401923 | 20.824178 | 1.009927 |
| cohort0-block1-least_progress | COMPLETE | 8 / 2 | 1.407910 | 20.690988 | 1.009009 |

## Within-block treatment comparisons

| Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s | Forced count Δ |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress → cohort0-block0-first_most_then_least | DESCRIPTIVE_MATCHED_PAIR | +0.166% | -0.032% | +0.001479 | +0 |
| cohort0-block0-most_output → cohort0-block0-first_most_then_least | DESCRIPTIVE_MATCHED_PAIR | -1.332% | -1.065% | +0.028553 | -1 |
| cohort0-block0-least_progress → cohort0-block0-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.518% | +1.044% | -0.027075 | +1 |
| cohort0-block1-least_progress → cohort0-block1-first_most_then_least | DESCRIPTIVE_MATCHED_PAIR | -0.425% | +0.644% | +0.000918 | +0 |
| cohort0-block1-most_output → cohort0-block1-first_most_then_least | DESCRIPTIVE_MATCHED_PAIR | -1.513% | -0.850% | +0.030427 | -1 |
| cohort0-block1-least_progress → cohort0-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.105% | +1.507% | -0.029510 | +1 |

## Same-role block observations

| Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s | Forced count Δ |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress → cohort0-block1-least_progress | DESCRIPTIVE_MATCHED_PAIR | +0.223% | -0.273% | -0.000054 | +0 |
| cohort0-block0-first_most_then_least → cohort0-block1-first_most_then_least | DESCRIPTIVE_MATCHED_PAIR | -0.368% | +0.401% | -0.000615 | +0 |
| cohort0-block0-most_output → cohort0-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | -0.185% | +0.184% | -0.002489 | +0 |

All six newly executed cells must be COMPLETE and qualified before numeric comparisons. Within each block compare A→C, B→C, A→B, then identical roles across blocks. A=least_progress; C=first_most_then_least; B=most_output. Only verified rotation_victim_order differs; subsequent action times and policy-specific states need not match.

Same-document exploratory diagnostic on the previously observed cohort0; six fresh engines are one reused workload, two order blocks and three roles, not six independent workloads or a new holdout. Prior victim-order runs do not replace the newly executed A/B controls. Same-role block differences are observations, not a noise bound. No significance, fairness guarantee, quality equivalence, noninferiority or method GO is inferred. Reference 5 s TTFT / 0.2 s mean-TPOT SLO does not constrain maximum ITL.

wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is included in scheduler time. Work classes are disjoint calls. Recompute positions are not newly produced outputs; recovery spans may include useful concurrent decode. Observed width/path changes are not independent savings or counterfactual bounds.

Full per-request changes, metric vectors, work costs and actual recovery accounting are in analysis.json.
