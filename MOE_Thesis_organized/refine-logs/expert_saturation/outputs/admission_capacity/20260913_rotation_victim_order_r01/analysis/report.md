# Victim-order ablation

MEASUREMENT_ONLY

| Cell | Status | Forced / natural | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress | COMPLETE | 8 / 2 | 1.399387 | 20.823713 | 1.010667 |
| cohort0-block0-most_output | COMPLETE | 9 / 2 | 1.419977 | 21.062662 | 0.988988 |
| cohort1-block0-most_output | COMPLETE | 9 / 2 | 1.426733 | 20.962926 | 0.989056 |
| cohort1-block0-least_progress | COMPLETE | 8 / 2 | 1.405240 | 20.730090 | 1.008693 |
| cohort0-block1-most_output | COMPLETE | 9 / 2 | 1.424990 | 20.988725 | 0.981825 |
| cohort0-block1-least_progress | COMPLETE | 8 / 2 | 1.409016 | 20.684496 | 1.005855 |
| cohort1-block1-least_progress | COMPLETE | 8 / 2 | 1.399966 | 20.823135 | 1.009128 |
| cohort1-block1-most_output | COMPLETE | 9 / 2 | 1.421806 | 21.032113 | 0.988171 |

## Within-block treatment comparisons

| Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s | Forced count Δ |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress → cohort0-block0-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.471% | +1.147% | -0.021679 | +1 |
| cohort0-block1-least_progress → cohort0-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.134% | +1.471% | -0.024030 | +1 |
| cohort1-block0-least_progress → cohort1-block0-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.530% | +1.123% | -0.019638 | +1 |
| cohort1-block1-least_progress → cohort1-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | +1.560% | +1.004% | -0.020957 | +1 |

## Same-role block observations

| Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s | Forced count Δ |
|---|---|---:|---:|---:|---:|
| cohort0-block0-least_progress → cohort0-block1-least_progress | DESCRIPTIVE_MATCHED_PAIR | +0.688% | -0.669% | -0.004811 | +0 |
| cohort0-block0-most_output → cohort0-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | +0.353% | -0.351% | -0.007163 | +0 |
| cohort1-block0-least_progress → cohort1-block1-least_progress | DESCRIPTIVE_MATCHED_PAIR | -0.375% | +0.449% | +0.000435 | +0 |
| cohort1-block0-most_output → cohort1-block1-most_output | DESCRIPTIVE_MATCHED_PAIR | -0.345% | +0.330% | -0.000884 | +0 |

All eight cells must be COMPLETE and qualified before numeric comparisons. Compare least_progress to most_output within each cohort/block and identical roles across blocks within each cohort. No cross-cohort request pairs. Only the verified rotation_victim_order treatment key is additionally excluded from config equality; recovery paths need not match.

Same-document exploratory ablation on the two previously observed cohorts; this is not a new text holdout. Eight fresh engines represent two cohorts, two order blocks and two victim orders, not eight independent workloads. Same-role block differences describe observed drift, not a noise bound. No significance, noninferiority, quality or method GO is inferred. Fixed 5 s TTFT / 0.2 s mean-TPOT reference SLO does not constrain maximum ITL; all-pass goodput equals throughput.

wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is a subset of scheduler time. Work classes are disjoint calls, not disjoint GPU kernels. Recovery spans can include concurrent useful decode and host overhead. Actual policy-specific work is not a counterfactual bound.

Full per-request changes, metric vectors, work costs and actual recovery accounting are in analysis.json.
