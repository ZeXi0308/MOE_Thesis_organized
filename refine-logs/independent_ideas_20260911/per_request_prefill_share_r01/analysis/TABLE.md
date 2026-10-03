# Global versus per-request prefill budgets: retained measurements

DESCRIPTIVE_MEASUREMENT_ONLY

Only current-campaign same-engine native1024/global512/per_request512 comparisons; no cross-GPU or old missing-block substitution. FCFS stays fixed. All-short verifies ordering/threshold no-op only; global512 may still bind aggregate budget. Threshold-hit rows with remaining global budget and same-step short admissions describe actual execution, not a same-state causal counterfactual. Host receipt timing; pooled ITL is descriptive. Every raw and per-request measurement retained, warmups excluded from comparisons. Completion/SLO require completed status; group rates use group first arrival, wall uses shared episode origin. queue_s is host submission lag, not native queue wait. arrival_to_first_schedule_s includes both. Queue reorder timing uses action end-start; ledger_end also includes native schedule and is not pure policy tax. All ledger overhead is inside request/wall metrics. Bypass counts use actual first-admission prefix order, including same-step order, not a wall-clock waiting guarantee. No Oracle, route counterfactual, significance claim or quality guarantee from output equality.

Latencies: ms. SLO: TTFT 5 s / mean TPOT 0.2 s. Warmups excluded.

| Block/cohort/prefill policy | Group | Complete/arrived/planned; SLO pass | Wall s | TTFT mean/p95 | TPOT mean | ITL p95/p99 | Request mean/p95/max |
|---|---|---|---:|---|---:|---|---|
| forward/all_short/native1024 | all | 16/16/16; 16 | 2.075 | 262.565 / 537.093 | 6.872 | 7.455 / 11.793 | 1135.279 / 1399.798 / 1407.557 |
| forward/all_short/native1024 | short128 | 16/16/16; 16 | 2.075 | 262.565 / 537.093 | 6.872 | 7.455 / 11.793 | 1135.279 / 1399.798 / 1407.557 |
| forward/all_short/native1024 | long2048 | 0/0/0; 0 | 2.075 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/global512 | all | 16/16/16; 16 | 2.084 | 262.662 / 542.233 | 6.895 | 7.824 / 11.935 | 1138.365 / 1406.096 / 1412.320 |
| forward/all_short/global512 | short128 | 16/16/16; 16 | 2.084 | 262.662 / 542.233 | 6.895 | 7.824 / 11.935 | 1138.365 / 1406.096 / 1412.320 |
| forward/all_short/global512 | long2048 | 0/0/0; 0 | 2.084 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/per_request512 | all | 16/16/16; 16 | 2.129 | 275.932 / 571.005 | 7.069 | 8.063 / 13.218 | 1173.749 / 1455.599 / 1459.062 |
| forward/all_short/per_request512 | short128 | 16/16/16; 16 | 2.129 | 275.932 / 571.005 | 7.069 | 8.063 / 13.218 | 1173.749 / 1455.599 / 1459.062 |
| forward/all_short/per_request512 | long2048 | 0/0/0; 0 | 2.129 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/mixed/native1024 | all | 16/16/16; 16 | 2.490 | 411.196 / 805.508 | 8.430 | 12.176 / 20.687 | 1481.760 / 1886.161 / 1888.265 |
| forward/mixed/native1024 | short128 | 8/8/8; 8 | 2.490 | 392.794 / 769.878 | 8.537 | 13.125 / 20.687 | 1476.942 / 1869.783 / 1888.265 |
| forward/mixed/native1024 | long2048 | 8/8/8; 8 | 2.490 | 429.598 / 805.618 | 8.323 | 11.930 / 20.673 | 1486.579 / 1874.015 / 1885.459 |
| forward/mixed/global512 | all | 16/16/16; 16 | 2.526 | 418.869 / 834.301 | 8.396 | 13.620 / 14.078 | 1485.223 / 1892.285 / 1909.901 |
| forward/mixed/global512 | short128 | 8/8/8; 8 | 2.526 | 387.611 / 784.819 | 8.455 | 13.670 / 14.186 | 1461.369 / 1851.680 / 1859.901 |
| forward/mixed/global512 | long2048 | 8/8/8; 8 | 2.526 | 450.127 / 834.819 | 8.338 | 13.596 / 14.062 | 1509.078 / 1901.680 / 1909.901 |
| forward/mixed/per_request512 | all | 16/16/16; 16 | 2.518 | 424.110 / 840.350 | 8.511 | 14.917 / 16.372 | 1505.013 / 1913.880 / 1929.493 |
| forward/mixed/per_request512 | short128 | 8/8/8; 8 | 2.518 | 395.089 / 780.343 | 8.648 | 14.999 / 16.384 | 1493.378 / 1890.144 / 1908.676 |
| forward/mixed/per_request512 | long2048 | 8/8/8; 8 | 2.518 | 453.131 / 841.996 | 8.374 | 14.829 / 16.305 | 1516.649 / 1914.816 / 1929.493 |
| reverse/mixed/per_request512 | all | 16/16/16; 16 | 2.524 | 411.061 / 836.010 | 8.477 | 15.035 / 16.511 | 1487.611 / 1895.985 / 1913.898 |
| reverse/mixed/per_request512 | short128 | 8/8/8; 8 | 2.524 | 374.220 / 747.078 | 8.577 | 15.256 / 16.511 | 1463.467 / 1840.616 / 1847.813 |
| reverse/mixed/per_request512 | long2048 | 8/8/8; 8 | 2.524 | 447.902 / 836.911 | 8.377 | 14.980 / 16.504 | 1511.755 / 1905.539 / 1913.898 |
| reverse/mixed/global512 | all | 16/16/16; 16 | 2.547 | 428.026 / 845.728 | 8.476 | 13.924 / 15.442 | 1504.491 / 1915.779 / 1938.752 |
| reverse/mixed/global512 | short128 | 8/8/8; 8 | 2.547 | 399.207 / 791.934 | 8.556 | 14.005 / 16.034 | 1485.813 / 1878.427 / 1888.752 |
| reverse/mixed/global512 | long2048 | 8/8/8; 8 | 2.547 | 456.845 / 845.893 | 8.396 | 13.745 / 15.442 | 1523.169 / 1928.031 / 1938.752 |
| reverse/mixed/native1024 | all | 16/16/16; 16 | 2.508 | 406.769 / 821.846 | 8.451 | 12.580 / 20.814 | 1480.047 / 1874.474 / 1884.371 |
| reverse/mixed/native1024 | short128 | 8/8/8; 8 | 2.508 | 378.578 / 772.512 | 8.527 | 16.195 / 20.918 | 1461.485 / 1829.752 / 1834.371 |
| reverse/mixed/native1024 | long2048 | 8/8/8; 8 | 2.508 | 434.960 / 822.512 | 8.375 | 11.923 / 20.584 | 1498.609 / 1879.752 / 1884.371 |
| reverse/all_short/per_request512 | all | 16/16/16; 16 | 2.089 | 272.934 / 553.442 | 6.943 | 7.639 / 12.473 | 1154.721 / 1415.174 / 1416.128 |
| reverse/all_short/per_request512 | short128 | 16/16/16; 16 | 2.089 | 272.934 / 553.442 | 6.943 | 7.639 / 12.473 | 1154.721 / 1415.174 / 1416.128 |
| reverse/all_short/per_request512 | long2048 | 0/0/0; 0 | 2.089 | NA / NA | NA | NA / NA | NA / NA / NA |
| reverse/all_short/global512 | all | 16/16/16; 16 | 2.096 | 283.351 / 554.626 | 7.004 | 7.776 / 12.177 | 1172.861 / 1440.213 / 1462.446 |
| reverse/all_short/global512 | short128 | 16/16/16; 16 | 2.096 | 283.351 / 554.626 | 7.004 | 7.776 / 12.177 | 1172.861 / 1440.213 / 1462.446 |
| reverse/all_short/global512 | long2048 | 0/0/0; 0 | 2.096 | NA / NA | NA | NA / NA | NA / NA / NA |
| reverse/all_short/native1024 | all | 16/16/16; 16 | 2.085 | 267.554 / 547.110 | 6.912 | 7.570 / 12.340 | 1145.382 / 1418.197 / 1424.768 |
| reverse/all_short/native1024 | short128 | 16/16/16; 16 | 2.085 | 267.554 / 547.110 | 6.912 | 7.570 / 12.340 | 1145.382 / 1418.197 / 1424.768 |
| reverse/all_short/native1024 | long2048 | 0/0/0; 0 | 2.085 | NA / NA | NA | NA / NA | NA / NA / NA |

forward/all_short/native1024: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 317, 'queue_multiple_steps': 109, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 317, 'p50': 3.1171366572380066e-05, 'p95': 4.724971950054168e-05, 'p99': 5.6144483387470205e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.356303819910585e-05, 'max': 0.00010935869067907333}.


forward/all_short/global512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 320, 'queue_multiple_steps': 108, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 320, 'p50': 3.176834434270859e-05, 'p95': 4.790667444467545e-05, 'p99': 5.584940314292908e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.348382597323507e-05, 'max': 0.00010910723358392715}.


forward/all_short/per_request512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 317, 'queue_multiple_steps': 110, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 317, 'p50': 3.557372838258743e-05, 'p95': 5.3715892136096954e-05, 'p99': 6.763417273759842e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.733819391731208e-05, 'max': 0.00010831840336322784}.


forward/mixed/native1024: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 289, 'queue_multiple_steps': 115, 'budget_exhausted_steps': 16, 'prefill_steps': 29, 'prefill_request_steps': 32, 'partial_prefill_with_new_request_steps': 3, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 3, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 289, 'p50': 3.575161099433899e-05, 'p95': 5.189739167690276e-05, 'p99': 7.347114384174353e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.856378623564763e-05, 'max': 9.955931454896927e-05}.


forward/mixed/global512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 305, 'queue_multiple_steps': 118, 'budget_exhausted_steps': 32, 'prefill_steps': 42, 'prefill_request_steps': 48, 'partial_prefill_with_new_request_steps': 6, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 6, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 305, 'p50': 3.327522426843643e-05, 'p95': 5.346685647964478e-05, 'p99': 0.00010083209723234163, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.7416547048287314e-05, 'max': 0.0001064324751496315}.


forward/mixed/per_request512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 290, 'queue_multiple_steps': 115, 'budget_exhausted_steps': 0, 'prefill_steps': 35, 'prefill_request_steps': 40, 'partial_prefill_with_new_request_steps': 3, 'threshold_limited_prefill_rows': 24, 'short_first_admitted_with_preceding_long_prefill': 5, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 290, 'p50': 3.5682227462530136e-05, 'p95': 5.821795202791691e-05, 'p99': 0.00010107798501849178, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.919509913900803e-05, 'max': 0.00012200325727462769}.


reverse/mixed/per_request512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 304, 'queue_multiple_steps': 114, 'budget_exhausted_steps': 0, 'prefill_steps': 36, 'prefill_request_steps': 40, 'partial_prefill_with_new_request_steps': 3, 'threshold_limited_prefill_rows': 24, 'short_first_admitted_with_preceding_long_prefill': 4, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 304, 'p50': 3.709504380822182e-05, 'p95': 7.422482594847678e-05, 'p99': 0.0001112921722233293, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.35167488544003e-05, 'max': 0.0008340245112776756}.


reverse/mixed/global512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 299, 'queue_multiple_steps': 119, 'budget_exhausted_steps': 32, 'prefill_steps': 42, 'prefill_request_steps': 48, 'partial_prefill_with_new_request_steps': 6, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 6, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 299, 'p50': 3.930088132619858e-05, 'p95': 5.852673202753066e-05, 'p99': 0.00010447870939970009, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.1670321695581326e-05, 'max': 0.00011051446199417114}.


reverse/mixed/native1024: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 303, 'queue_multiple_steps': 115, 'budget_exhausted_steps': 16, 'prefill_steps': 29, 'prefill_request_steps': 32, 'partial_prefill_with_new_request_steps': 3, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 3, 'kv_adjustments': 0}; all-short ordering/threshold no-op=None; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 303, 'p50': 3.529898822307587e-05, 'p95': 6.648385897278783e-05, 'p99': 0.00010816814377903939, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.060117361864241e-05, 'max': 0.00013201963156461716}.


reverse/all_short/per_request512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 312, 'queue_multiple_steps': 108, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 312, 'p50': 3.391830250620842e-05, 'p95': 5.141841247677803e-05, 'p99': 6.108525209128853e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.643124961318114e-05, 'max': 0.00010442174971103668}.


reverse/all_short/global512: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 301, 'queue_multiple_steps': 109, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 301, 'p50': 3.449153155088425e-05, 'p95': 5.882978439331055e-05, 'p99': 0.00010718032717704773, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.098453308359729e-05, 'max': 0.0008542193099856377}.


reverse/all_short/native1024: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 317, 'queue_multiple_steps': 109, 'budget_exhausted_steps': 0, 'prefill_steps': 16, 'prefill_request_steps': 16, 'partial_prefill_with_new_request_steps': 0, 'threshold_limited_prefill_rows': 0, 'short_first_admitted_with_preceding_long_prefill': 0, 'kv_adjustments': 0}; all-short ordering/threshold no-op=True; global512 budget may differ=False.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 317, 'p50': 3.275927156209946e-05, 'p95': 4.937462508678436e-05, 'p99': 6.078686565160748e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.481873131888898e-05, 'max': 0.00010580476373434067}.


Intervention minus baseline, each block separately; negative latency delta is lower.

| Block/cohort/intervention vs baseline | Group | Request mean Δ% | Request p95 Δ% | Request max Δ% | Wall Δ% |
|---|---|---:|---:|---:|---:|
| forward/all_short/global512 vs native1024 | all | 0.272 | 0.450 | 0.338 | 0.477 |
| forward/all_short/global512 vs native1024 | short128 | 0.272 | 0.450 | 0.338 | 0.477 |
| forward/all_short/global512 vs native1024 | long2048 | NA | NA | NA | 0.477 |
| forward/all_short/per_request512 vs native1024 | all | 3.389 | 3.986 | 3.659 | 2.637 |
| forward/all_short/per_request512 vs native1024 | short128 | 3.389 | 3.986 | 3.659 | 2.637 |
| forward/all_short/per_request512 vs native1024 | long2048 | NA | NA | NA | 2.637 |
| forward/all_short/per_request512 vs global512 | all | 3.108 | 3.521 | 3.310 | 2.150 |
| forward/all_short/per_request512 vs global512 | short128 | 3.108 | 3.521 | 3.310 | 2.150 |
| forward/all_short/per_request512 vs global512 | long2048 | NA | NA | NA | 2.150 |
| forward/mixed/global512 vs native1024 | all | 0.234 | 0.325 | 1.146 | 1.465 |
| forward/mixed/global512 vs native1024 | short128 | -1.054 | -0.968 | -1.502 | 1.465 |
| forward/mixed/global512 vs native1024 | long2048 | 1.513 | 1.476 | 1.296 | 1.465 |
| forward/mixed/per_request512 vs native1024 | all | 1.569 | 1.470 | 2.183 | 1.130 |
| forward/mixed/per_request512 vs native1024 | short128 | 1.113 | 1.089 | 1.081 | 1.130 |
| forward/mixed/per_request512 vs native1024 | long2048 | 2.023 | 2.177 | 2.335 | 1.130 |
| forward/mixed/per_request512 vs global512 | all | 1.332 | 1.141 | 1.026 | -0.330 |
| forward/mixed/per_request512 vs global512 | short128 | 2.190 | 2.077 | 2.622 | -0.330 |
| forward/mixed/per_request512 vs global512 | long2048 | 0.502 | 0.691 | 1.026 | -0.330 |
| reverse/all_short/global512 vs native1024 | all | 2.399 | 1.552 | 2.645 | 0.536 |
| reverse/all_short/global512 vs native1024 | short128 | 2.399 | 1.552 | 2.645 | 0.536 |
| reverse/all_short/global512 vs native1024 | long2048 | NA | NA | NA | 0.536 |
| reverse/all_short/per_request512 vs native1024 | all | 0.815 | -0.213 | -0.606 | 0.169 |
| reverse/all_short/per_request512 vs native1024 | short128 | 0.815 | -0.213 | -0.606 | 0.169 |
| reverse/all_short/per_request512 vs native1024 | long2048 | NA | NA | NA | 0.169 |
| reverse/all_short/per_request512 vs global512 | all | -1.547 | -1.739 | -3.167 | -0.366 |
| reverse/all_short/per_request512 vs global512 | short128 | -1.547 | -1.739 | -3.167 | -0.366 |
| reverse/all_short/per_request512 vs global512 | long2048 | NA | NA | NA | -0.366 |
| reverse/mixed/global512 vs native1024 | all | 1.652 | 2.204 | 2.886 | 1.549 |
| reverse/mixed/global512 vs native1024 | short128 | 1.665 | 2.660 | 2.965 | 1.549 |
| reverse/mixed/global512 vs native1024 | long2048 | 1.639 | 2.568 | 2.886 | 1.549 |
| reverse/mixed/per_request512 vs native1024 | all | 0.511 | 1.148 | 1.567 | 0.642 |
| reverse/mixed/per_request512 vs native1024 | short128 | 0.136 | 0.594 | 0.733 | 0.642 |
| reverse/mixed/per_request512 vs native1024 | long2048 | 0.877 | 1.372 | 1.567 | 0.642 |
| reverse/mixed/per_request512 vs global512 | all | -1.122 | -1.033 | -1.282 | -0.893 |
| reverse/mixed/per_request512 vs global512 | short128 | -1.504 | -2.013 | -2.168 | -0.893 |
| reverse/mixed/per_request512 vs global512 | long2048 | -0.749 | -1.167 | -1.282 | -0.893 |

forward/all_short/global512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=7/16; exclusion=None.


forward/all_short/per_request512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=12/16; exclusion=None.


forward/all_short/per_request512 vs global512: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


forward/mixed/global512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=2/16; exclusion=None.


forward/mixed/per_request512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=5/16; exclusion=None.


forward/mixed/per_request512 vs global512: valid=True; first-schedule order changed=False; equal outputs=1/16; exclusion=None.


reverse/all_short/global512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


reverse/all_short/per_request512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=5/16; exclusion=None.


reverse/all_short/per_request512 vs global512: valid=True; first-schedule order changed=False; equal outputs=10/16; exclusion=None.


reverse/mixed/global512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=3/16; exclusion=None.


reverse/mixed/per_request512 vs native1024: valid=True; first-schedule order changed=False; equal outputs=5/16; exclusion=None.


reverse/mixed/per_request512 vs global512: valid=True; first-schedule order changed=False; equal outputs=2/16; exclusion=None.


Campaign issues: []. Retained warmups: 12; details and all metric deltas in summary.json.
