# Waiting order: retained measurements

PARTIAL_OR_INVALID_MEASUREMENT

Host receipt timing; pooled ITL is descriptive. Every raw retained, warmups excluded from comparisons. Completion/SLO require completed status; group rates use group first arrival, wall uses shared episode origin. No Oracle, route counterfactual, significance claim or quality guarantee from output equality.

Latencies: ms. SLO: TTFT 5 s / mean TPOT 0.2 s. Warmups excluded.

| Block/cohort/order | Group | Complete/arrived/planned; SLO pass | Wall s | TTFT mean/p95 | TPOT mean | ITL p95/p99 | Request mean/p95/max |
|---|---|---|---:|---|---:|---|---|
| forward/all_short/fcfs | all | 16/16/16; 16 | 2.131 | 299.251 / 584.210 | 7.164 | 8.041 / 13.195 | 1209.127 / 1500.848 / 1527.697 |
| forward/all_short/fcfs | short128 | 16/16/16; 16 | 2.131 | 299.251 / 584.210 | 7.164 | 8.041 / 13.195 | 1209.127 / 1500.848 / 1527.697 |
| forward/all_short/fcfs | long2048 | 0/0/0; 0 | 2.131 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/short_prompt_first | all | 16/16/16; 16 | 2.145 | 301.388 / 589.414 | 7.190 | 7.750 / 13.613 | 1214.550 / 1501.999 / 1528.830 |
| forward/all_short/short_prompt_first | short128 | 16/16/16; 16 | 2.145 | 301.388 / 589.414 | 7.190 | 7.750 / 13.613 | 1214.550 / 1501.999 / 1528.830 |
| forward/all_short/short_prompt_first | long2048 | 0/0/0; 0 | 2.145 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/mixed/fcfs | all | 16/16/16; 16 | 2.565 | 437.830 / 852.370 | 8.700 | 13.181 / 20.698 | 1542.707 / 1967.644 / 1967.861 |
| forward/mixed/fcfs | short128 | 8/8/8; 8 | 2.565 | 419.455 / 824.012 | 8.805 | 13.482 / 20.698 | 1537.747 / 1950.260 / 1967.861 |
| forward/mixed/fcfs | long2048 | 8/8/8; 8 | 2.565 | 456.205 / 856.613 | 8.594 | 12.998 / 20.624 | 1547.667 / 1953.786 / 1967.572 |
| forward/mixed/short_prompt_first | all | 16/16/16; 16 | 2.577 | 429.858 / 929.885 | 8.759 | 13.386 / 20.663 | 1542.259 / 2002.951 / 2041.376 |
| forward/mixed/short_prompt_first | short128 | 8/8/8; 8 | 2.577 | 379.830 / 816.822 | 8.978 | 20.222 / 20.663 | 1520.046 / 1963.536 / 1985.542 |
| forward/mixed/short_prompt_first | long2048 | 8/8/8; 8 | 2.577 | 479.886 / 932.891 | 8.540 | 12.870 / 20.516 | 1564.472 / 2023.444 / 2041.376 |

forward/all_short/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 295, 'queue_multiple_steps': 110, 'kv_adjustments': 0}; all-short no-op=True.


forward/all_short/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 296, 'queue_multiple_steps': 110, 'kv_adjustments': 0}; all-short no-op=True.


forward/mixed/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 286, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.


forward/mixed/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 286, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.


SPT minus FCFS, each block separately; negative latency delta is lower.

| Block/cohort | Group | Request mean Δ% | Request p95 Δ% | Request max Δ% | Wall Δ% |
|---|---|---:|---:|---:|---:|
| forward/all_short | all | 0.449 | 0.077 | 0.074 | 0.656 |
| forward/all_short | short128 | 0.449 | 0.077 | 0.074 | 0.656 |
| forward/all_short | long2048 | NA | NA | NA | 0.656 |
| forward/mixed | all | -0.029 | 1.794 | 3.736 | 0.461 |
| forward/mixed | short128 | -1.151 | 0.681 | 0.898 | 0.461 |
| forward/mixed | long2048 | 1.086 | 3.565 | 3.751 | 0.461 |

forward/all_short: valid=True; first-schedule order changed=False; equal outputs=7/16; exclusion=None.


forward/mixed: valid=True; first-schedule order changed=True; equal outputs=6/16; exclusion=None.


Campaign issues: ['missing_block:reverse', 'missing_or_duplicate_pair:reverse/all_short', 'missing_or_duplicate_pair:reverse/mixed']. Retained warmups: 4; details and all metric deltas in summary.json.
