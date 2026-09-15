# Waiting order: retained measurements

DESCRIPTIVE_MEASUREMENT_ONLY

Host receipt timing; pooled ITL is descriptive. Every raw retained, warmups excluded from comparisons. Completion/SLO require completed status; group rates use group first arrival, wall uses shared episode origin. No Oracle, route counterfactual, significance claim or quality guarantee from output equality.

Latencies: ms. SLO: TTFT 5 s / mean TPOT 0.2 s. Warmups excluded.

| Block/cohort/order | Group | Complete/arrived/planned; SLO pass | Wall s | TTFT mean/p95 | TPOT mean | ITL p95/p99 | Request mean/p95/max |
|---|---|---|---:|---|---:|---|---|
| forward/all_short/fcfs | all | 16/16/16; 16 | 2.134 | 282.638 / 579.010 | 7.097 | 7.670 / 13.016 | 1184.002 / 1457.923 / 1462.987 |
| forward/all_short/fcfs | short128 | 16/16/16; 16 | 2.134 | 282.638 / 579.010 | 7.097 | 7.670 / 13.016 | 1184.002 / 1457.923 / 1462.987 |
| forward/all_short/fcfs | long2048 | 0/0/0; 0 | 2.134 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/short_prompt_first | all | 16/16/16; 16 | 2.149 | 293.937 / 587.812 | 7.195 | 7.871 / 13.462 | 1207.726 / 1484.361 / 1486.379 |
| forward/all_short/short_prompt_first | short128 | 16/16/16; 16 | 2.149 | 293.937 / 587.812 | 7.195 | 7.871 / 13.462 | 1207.726 / 1484.361 / 1486.379 |
| forward/all_short/short_prompt_first | long2048 | 0/0/0; 0 | 2.149 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/mixed/fcfs | all | 16/16/16; 16 | 2.577 | 441.495 / 863.690 | 8.749 | 14.311 / 20.691 | 1552.637 / 1981.338 / 1981.839 |
| forward/mixed/fcfs | short128 | 8/8/8; 8 | 2.577 | 422.643 / 831.220 | 8.857 | 19.900 / 20.691 | 1547.427 / 1964.105 / 1981.839 |
| forward/mixed/fcfs | long2048 | 8/8/8; 8 | 2.577 | 460.348 / 868.126 | 8.642 | 13.090 / 20.645 | 1557.846 / 1965.611 / 1981.171 |
| forward/mixed/short_prompt_first | all | 16/16/16; 16 | 2.544 | 418.901 / 926.272 | 8.618 | 12.993 / 20.622 | 1513.450 / 1972.149 / 2009.933 |
| forward/mixed/short_prompt_first | short128 | 8/8/8; 8 | 2.544 | 363.221 / 777.553 | 8.836 | 20.145 / 20.624 | 1485.449 / 1909.624 / 1925.526 |
| forward/mixed/short_prompt_first | long2048 | 8/8/8; 8 | 2.544 | 474.581 / 926.726 | 8.401 | 12.489 / 20.605 | 1541.452 / 1992.300 / 2009.933 |
| reverse/mixed/short_prompt_first | all | 16/16/16; 16 | 2.558 | 418.559 / 933.490 | 8.671 | 13.304 / 20.643 | 1519.815 / 1983.726 / 2020.945 |
| reverse/mixed/short_prompt_first | short128 | 8/8/8; 8 | 2.558 | 360.338 / 761.045 | 8.888 | 20.273 / 20.649 | 1489.150 / 1901.724 / 1911.702 |
| reverse/mixed/short_prompt_first | long2048 | 8/8/8; 8 | 2.558 | 476.779 / 934.099 | 8.454 | 13.009 / 20.598 | 1550.480 / 2003.577 / 2020.945 |
| reverse/mixed/fcfs | all | 16/16/16; 16 | 2.557 | 424.997 / 844.489 | 8.647 | 13.271 / 20.722 | 1523.218 / 1922.065 / 1936.592 |
| reverse/mixed/fcfs | short128 | 8/8/8; 8 | 2.557 | 402.070 / 795.457 | 8.737 | 13.840 / 20.725 | 1511.624 / 1894.192 / 1898.285 |
| reverse/mixed/fcfs | long2048 | 8/8/8; 8 | 2.557 | 447.923 / 845.457 | 8.558 | 13.114 / 20.715 | 1534.812 / 1929.812 / 1936.592 |
| reverse/all_short/short_prompt_first | all | 16/16/16; 16 | 2.149 | 298.584 / 585.462 | 7.196 | 7.790 / 13.378 | 1212.501 / 1495.884 / 1507.279 |
| reverse/all_short/short_prompt_first | short128 | 16/16/16; 16 | 2.149 | 298.584 / 585.462 | 7.196 | 7.790 / 13.378 | 1212.501 / 1495.884 / 1507.279 |
| reverse/all_short/short_prompt_first | long2048 | 0/0/0; 0 | 2.149 | NA / NA | NA | NA / NA | NA / NA / NA |
| reverse/all_short/fcfs | all | 16/16/16; 16 | 2.147 | 301.942 / 590.591 | 7.203 | 7.685 / 13.373 | 1216.727 / 1504.229 / 1531.077 |
| reverse/all_short/fcfs | short128 | 16/16/16; 16 | 2.147 | 301.942 / 590.591 | 7.203 | 7.685 / 13.373 | 1216.727 / 1504.229 / 1531.077 |
| reverse/all_short/fcfs | long2048 | 0/0/0; 0 | 2.147 | NA / NA | NA | NA / NA | NA / NA / NA |

forward/all_short/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 315, 'queue_multiple_steps': 109, 'kv_adjustments': 0}; all-short no-op=True.


forward/all_short/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 304, 'queue_multiple_steps': 110, 'kv_adjustments': 0}; all-short no-op=True.


forward/mixed/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 286, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.


forward/mixed/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 290, 'queue_multiple_steps': 117, 'kv_adjustments': 0}; all-short no-op=None.


reverse/mixed/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 294, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.


reverse/mixed/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 294, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.


reverse/all_short/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 301, 'queue_multiple_steps': 109, 'kv_adjustments': 0}; all-short no-op=True.


reverse/all_short/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 296, 'queue_multiple_steps': 110, 'kv_adjustments': 0}; all-short no-op=True.


SPT minus FCFS, each block separately; negative latency delta is lower.

| Block/cohort | Group | Request mean Δ% | Request p95 Δ% | Request max Δ% | Wall Δ% |
|---|---|---:|---:|---:|---:|
| forward/all_short | all | 2.004 | 1.813 | 1.599 | 0.728 |
| forward/all_short | short128 | 2.004 | 1.813 | 1.599 | 0.728 |
| forward/all_short | long2048 | NA | NA | NA | 0.728 |
| forward/mixed | all | -2.524 | -0.464 | 1.418 | -1.282 |
| forward/mixed | short128 | -4.005 | -2.774 | -2.841 | -1.282 |
| forward/mixed | long2048 | -1.052 | 1.358 | 1.452 | -1.282 |
| reverse/all_short | all | -0.347 | -0.555 | -1.554 | 0.073 |
| reverse/all_short | short128 | -0.347 | -0.555 | -1.554 | 0.073 |
| reverse/all_short | long2048 | NA | NA | NA | 0.073 |
| reverse/mixed | all | -0.223 | 3.208 | 4.356 | 0.048 |
| reverse/mixed | short128 | -1.487 | 0.398 | 0.707 | 0.048 |
| reverse/mixed | long2048 | 1.021 | 3.822 | 4.356 | 0.048 |

forward/all_short: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


forward/mixed: valid=True; first-schedule order changed=True; equal outputs=5/16; exclusion=None.


reverse/all_short: valid=True; first-schedule order changed=False; equal outputs=7/16; exclusion=None.


reverse/mixed: valid=True; first-schedule order changed=True; equal outputs=8/16; exclusion=None.


Campaign issues: []. Retained warmups: 8; details and all metric deltas in summary.json.
