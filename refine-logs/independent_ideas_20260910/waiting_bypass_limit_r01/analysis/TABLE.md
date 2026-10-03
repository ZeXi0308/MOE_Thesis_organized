# Waiting order with one-bypass limit: retained measurements

DESCRIPTIVE_MEASUREMENT_ONLY

Only current-campaign same-engine arm comparisons; no cross-GPU or old missing-block substitution. Host receipt timing; pooled ITL is descriptive. Every raw and per-request measurement retained, warmups excluded from comparisons. Completion/SLO require completed status; group rates use group first arrival, wall uses shared episode origin. queue_s is host submission lag, not native queue wait. arrival_to_first_schedule_s includes both. Queue reorder timing uses action end-start; ledger_end also includes native schedule and is not pure policy tax. All ledger overhead is inside request/wall metrics. Bypass counts use actual first-admission prefix order, including same-step order, not a wall-clock waiting guarantee. No Oracle, route counterfactual, significance claim or quality guarantee from output equality.

Latencies: ms. SLO: TTFT 5 s / mean TPOT 0.2 s. Warmups excluded.

| Block/cohort/order | Group | Complete/arrived/planned; SLO pass | Wall s | TTFT mean/p95 | TPOT mean | ITL p95/p99 | Request mean/p95/max |
|---|---|---|---:|---|---:|---|---|
| forward/all_short/fcfs | all | 16/16/16; 16 | 2.098 | 279.339 / 561.124 | 7.007 | 7.640 / 13.384 | 1169.247 / 1433.226 / 1434.297 |
| forward/all_short/fcfs | short128 | 16/16/16; 16 | 2.098 | 279.339 / 561.124 | 7.007 | 7.640 / 13.384 | 1169.247 / 1433.226 / 1434.297 |
| forward/all_short/fcfs | long2048 | 0/0/0; 0 | 2.098 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/short_prompt_first | all | 16/16/16; 16 | 2.080 | 282.517 / 551.883 | 6.956 | 7.571 / 12.404 | 1165.963 / 1436.549 / 1454.871 |
| forward/all_short/short_prompt_first | short128 | 16/16/16; 16 | 2.080 | 282.517 / 551.883 | 6.956 | 7.571 / 12.404 | 1165.963 / 1436.549 / 1454.871 |
| forward/all_short/short_prompt_first | long2048 | 0/0/0; 0 | 2.080 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/all_short/bounded_bypass_once | all | 16/16/16; 16 | 2.111 | 287.228 / 569.001 | 7.045 | 7.710 / 12.661 | 1181.907 / 1443.865 / 1455.645 |
| forward/all_short/bounded_bypass_once | short128 | 16/16/16; 16 | 2.111 | 287.228 / 569.001 | 7.045 | 7.710 / 12.661 | 1181.907 / 1443.865 / 1455.645 |
| forward/all_short/bounded_bypass_once | long2048 | 0/0/0; 0 | 2.111 | NA / NA | NA | NA / NA | NA / NA / NA |
| forward/mixed/fcfs | all | 16/16/16; 16 | 2.488 | 400.679 / 801.880 | 8.394 | 12.071 / 20.606 | 1466.654 / 1856.841 / 1873.506 |
| forward/mixed/fcfs | short128 | 8/8/8; 8 | 2.488 | 375.438 / 752.063 | 8.475 | 12.661 / 20.614 | 1451.794 / 1817.664 / 1823.506 |
| forward/mixed/fcfs | long2048 | 8/8/8; 8 | 2.488 | 425.920 / 802.063 | 8.312 | 11.955 / 20.564 | 1481.513 / 1865.729 / 1873.506 |
| forward/mixed/short_prompt_first | all | 16/16/16; 16 | 2.504 | 402.015 / 897.871 | 8.461 | 12.469 / 20.409 | 1476.513 / 1929.898 / 1963.762 |
| forward/mixed/short_prompt_first | short128 | 8/8/8; 8 | 2.504 | 346.836 / 736.949 | 8.662 | 19.991 / 20.470 | 1446.934 / 1850.668 / 1863.571 |
| forward/mixed/short_prompt_first | long2048 | 8/8/8; 8 | 2.504 | 457.194 / 898.143 | 8.259 | 11.531 / 20.385 | 1506.093 / 1947.959 / 1963.762 |
| forward/mixed/bounded_bypass_once | all | 16/16/16; 16 | 2.527 | 412.906 / 841.123 | 8.575 | 14.505 / 21.084 | 1501.973 / 1906.442 / 1922.692 |
| forward/mixed/bounded_bypass_once | short128 | 8/8/8; 8 | 2.527 | 380.396 / 757.039 | 8.720 | 20.081 / 21.084 | 1487.831 / 1882.542 / 1901.025 |
| forward/mixed/bounded_bypass_once | long2048 | 8/8/8; 8 | 2.527 | 445.416 / 845.081 | 8.431 | 12.341 / 20.625 | 1516.115 / 1913.465 / 1922.692 |
| reverse/mixed/bounded_bypass_once | all | 16/16/16; 16 | 2.507 | 399.670 / 817.501 | 8.496 | 12.397 / 20.639 | 1478.667 / 1883.387 / 1902.845 |
| reverse/mixed/bounded_bypass_once | short128 | 8/8/8; 8 | 2.507 | 365.808 / 718.122 | 8.625 | 19.919 / 20.639 | 1461.246 / 1843.569 / 1851.562 |
| reverse/mixed/bounded_bypass_once | long2048 | 8/8/8; 8 | 2.507 | 433.531 / 818.213 | 8.367 | 11.892 / 20.492 | 1496.087 / 1893.764 / 1902.845 |
| reverse/mixed/short_prompt_first | all | 16/16/16; 16 | 2.500 | 398.574 / 899.324 | 8.449 | 12.280 / 20.381 | 1471.645 / 1929.767 / 1963.760 |
| reverse/mixed/short_prompt_first | short128 | 8/8/8; 8 | 2.500 | 339.352 / 714.886 | 8.647 | 19.869 / 20.381 | 1437.479 / 1828.288 / 1833.327 |
| reverse/mixed/short_prompt_first | long2048 | 8/8/8; 8 | 2.500 | 457.796 / 902.798 | 8.252 | 10.781 / 20.240 | 1505.811 / 1947.897 / 1963.760 |
| reverse/mixed/fcfs | all | 16/16/16; 16 | 2.495 | 412.658 / 810.146 | 8.448 | 12.047 / 20.529 | 1485.503 / 1880.969 / 1885.078 |
| reverse/mixed/fcfs | short128 | 8/8/8; 8 | 2.495 | 393.886 / 770.800 | 8.547 | 15.162 / 20.572 | 1479.344 / 1864.017 / 1879.600 |
| reverse/mixed/fcfs | long2048 | 8/8/8; 8 | 2.495 | 431.431 / 813.720 | 8.348 | 11.832 / 20.494 | 1491.662 / 1877.381 / 1885.078 |
| reverse/all_short/bounded_bypass_once | all | 16/16/16; 16 | 2.081 | 270.865 / 546.401 | 6.905 | 7.511 / 12.023 | 1147.803 / 1405.564 / 1408.341 |
| reverse/all_short/bounded_bypass_once | short128 | 16/16/16; 16 | 2.081 | 270.865 / 546.401 | 6.905 | 7.511 / 12.023 | 1147.803 / 1405.564 / 1408.341 |
| reverse/all_short/bounded_bypass_once | long2048 | 0/0/0; 0 | 2.081 | NA / NA | NA | NA / NA | NA / NA / NA |
| reverse/all_short/short_prompt_first | all | 16/16/16; 16 | 2.059 | 269.755 / 533.863 | 6.857 | 7.534 / 11.737 | 1140.540 / 1397.358 / 1407.029 |
| reverse/all_short/short_prompt_first | short128 | 16/16/16; 16 | 2.059 | 269.755 / 533.863 | 6.857 | 7.534 / 11.737 | 1140.540 / 1397.358 / 1407.029 |
| reverse/all_short/short_prompt_first | long2048 | 0/0/0; 0 | 2.059 | NA / NA | NA | NA / NA | NA / NA / NA |
| reverse/all_short/fcfs | all | 16/16/16; 16 | 2.074 | 271.071 / 541.690 | 6.904 | 7.471 / 11.946 | 1147.841 / 1403.059 / 1405.217 |
| reverse/all_short/fcfs | short128 | 16/16/16; 16 | 2.074 | 271.071 / 541.690 | 6.904 | 7.471 / 11.946 | 1147.841 / 1403.059 / 1405.217 |
| reverse/all_short/fcfs | long2048 | 0/0/0; 0 | 2.074 | NA / NA | NA | NA / NA | NA / NA / NA |

forward/all_short/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 306, 'queue_multiple_steps': 109, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 306, 'p50': 3.197183832526207e-05, 'p95': 5.177082493901253e-05, 'p99': 9.303418919444061e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 5.4809045703972084e-05, 'max': 0.0060814740136265755}.


forward/all_short/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 301, 'queue_multiple_steps': 110, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 301, 'p50': 3.360956907272339e-05, 'p95': 5.464721471071243e-05, 'p99': 6.649736315011978e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.6115878790716e-05, 'max': 9.50722023844719e-05}.


forward/all_short/bounded_bypass_once: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 304, 'queue_multiple_steps': 109, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 304, 'p50': 3.416091203689575e-05, 'p95': 6.700004450976847e-05, 'p99': 7.921765558421598e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.144096563227082e-05, 'max': 8.886866271495819e-05}.


forward/mixed/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 299, 'queue_multiple_steps': 115, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 299, 'p50': 3.254879266023636e-05, 'p95': 4.9606151878833754e-05, 'p99': 5.7282857596874074e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.59883070938962e-05, 'max': 0.00010403431951999664}.


forward/mixed/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 293, 'queue_multiple_steps': 115, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=6, affected requests=3, max=3, requests bypassed >1=2. Queue reorder duration: {'n': 293, 'p50': 3.484264016151428e-05, 'p95': 5.486942827701568e-05, 'p99': 7.180072367191262e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.853893231714952e-05, 'max': 0.00011520553380250931}.


forward/mixed/bounded_bypass_once: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 292, 'queue_multiple_steps': 115, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=3, affected requests=3, max=1, requests bypassed >1=0. Queue reorder duration: {'n': 292, 'p50': 3.77860851585865e-05, 'p95': 7.286719046533107e-05, 'p99': 9.019397199153828e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.689707994869311e-05, 'max': 0.00016201846301555634}.


reverse/mixed/bounded_bypass_once: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 296, 'queue_multiple_steps': 114, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=3, affected requests=3, max=1, requests bypassed >1=0. Queue reorder duration: {'n': 296, 'p50': 3.7764664739370346e-05, 'p95': 6.919447332620621e-05, 'p99': 8.856537751853492e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.422693903482443e-05, 'max': 0.0001169322058558464}.


reverse/mixed/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 3, 'action_calls': 297, 'queue_multiple_steps': 116, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=6, affected requests=3, max=3, requests bypassed >1=2. Queue reorder duration: {'n': 297, 'p50': 3.408733755350113e-05, 'p95': 5.630347877740858e-05, 'p99': 7.552206516265872e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.8135680762003565e-05, 'max': 9.557977318763733e-05}.


reverse/mixed/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 290, 'queue_multiple_steps': 115, 'kv_adjustments': 0}; all-short no-op=None.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 290, 'p50': 3.436114639043808e-05, 'p95': 5.456903018057348e-05, 'p99': 9.658108465373525e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.764529808841902e-05, 'max': 0.00011096056550741196}.


reverse/all_short/bounded_bypass_once: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 313, 'queue_multiple_steps': 108, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 313, 'p50': 3.472995012998581e-05, 'p95': 6.535127758979797e-05, 'p99': 7.742401212453842e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 4.013113903637511e-05, 'max': 9.580980986356735e-05}.


reverse/all_short/short_prompt_first: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 305, 'queue_multiple_steps': 109, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 305, 'p50': 3.3054500818252563e-05, 'p95': 5.1266700029373166e-05, 'p99': 5.999263375997536e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.8547523808283886e-05, 'max': 0.0009674541652202606}.


reverse/all_short/fcfs: comparison eligible=True; issues=[]; error=None; diagnostics={'decode_checks': 2032, 'decode_skips': 0, 'preemptions': 0, 'actual_reorders': 0, 'action_calls': 309, 'queue_multiple_steps': 108, 'kv_adjustments': 0}; all-short no-op=True.

Actual first-admission bypasses: total=0, affected requests=0, max=0, requests bypassed >1=0. Queue reorder duration: {'n': 309, 'p50': 3.173854202032089e-05, 'p95': 4.916265606880187e-05, 'p99': 5.641032010316849e-05, 'small_sample': False, 'small_sample_threshold': 100, 'percentile_method': 'linear_interpolation', 'mean': 3.4213364485976766e-05, 'max': 9.379815310239792e-05}.


Intervention minus baseline, each block separately; negative latency delta is lower.

| Block/cohort/intervention vs baseline | Group | Request mean Δ% | Request p95 Δ% | Request max Δ% | Wall Δ% |
|---|---|---:|---:|---:|---:|
| forward/all_short/short_prompt_first vs fcfs | all | -0.281 | 0.232 | 1.434 | -0.843 |
| forward/all_short/short_prompt_first vs fcfs | short128 | -0.281 | 0.232 | 1.434 | -0.843 |
| forward/all_short/short_prompt_first vs fcfs | long2048 | NA | NA | NA | -0.843 |
| forward/all_short/bounded_bypass_once vs fcfs | all | 1.083 | 0.742 | 1.488 | 0.615 |
| forward/all_short/bounded_bypass_once vs fcfs | short128 | 1.083 | 0.742 | 1.488 | 0.615 |
| forward/all_short/bounded_bypass_once vs fcfs | long2048 | NA | NA | NA | 0.615 |
| forward/all_short/bounded_bypass_once vs short_prompt_first | all | 1.368 | 0.509 | 0.053 | 1.470 |
| forward/all_short/bounded_bypass_once vs short_prompt_first | short128 | 1.368 | 0.509 | 0.053 | 1.470 |
| forward/all_short/bounded_bypass_once vs short_prompt_first | long2048 | NA | NA | NA | 1.470 |
| forward/mixed/short_prompt_first vs fcfs | all | 0.672 | 3.934 | 4.818 | 0.628 |
| forward/mixed/short_prompt_first vs fcfs | short128 | -0.335 | 1.816 | 2.197 | 0.628 |
| forward/mixed/short_prompt_first vs fcfs | long2048 | 1.659 | 4.407 | 4.818 | 0.628 |
| forward/mixed/bounded_bypass_once vs fcfs | all | 2.408 | 2.671 | 2.625 | 1.551 |
| forward/mixed/bounded_bypass_once vs fcfs | short128 | 2.482 | 3.569 | 4.251 | 1.551 |
| forward/mixed/bounded_bypass_once vs fcfs | long2048 | 2.336 | 2.559 | 2.625 | 1.551 |
| forward/mixed/bounded_bypass_once vs short_prompt_first | all | 1.724 | -1.215 | -2.091 | 0.917 |
| forward/mixed/bounded_bypass_once vs short_prompt_first | short128 | 2.827 | 1.722 | 2.010 | 0.917 |
| forward/mixed/bounded_bypass_once vs short_prompt_first | long2048 | 0.665 | -1.771 | -2.091 | 0.917 |
| reverse/all_short/short_prompt_first vs fcfs | all | -0.636 | -0.406 | 0.129 | -0.744 |
| reverse/all_short/short_prompt_first vs fcfs | short128 | -0.636 | -0.406 | 0.129 | -0.744 |
| reverse/all_short/short_prompt_first vs fcfs | long2048 | NA | NA | NA | -0.744 |
| reverse/all_short/bounded_bypass_once vs fcfs | all | -0.003 | 0.179 | 0.222 | 0.318 |
| reverse/all_short/bounded_bypass_once vs fcfs | short128 | -0.003 | 0.179 | 0.222 | 0.318 |
| reverse/all_short/bounded_bypass_once vs fcfs | long2048 | NA | NA | NA | 0.318 |
| reverse/all_short/bounded_bypass_once vs short_prompt_first | all | 0.637 | 0.587 | 0.093 | 1.069 |
| reverse/all_short/bounded_bypass_once vs short_prompt_first | short128 | 0.637 | 0.587 | 0.093 | 1.069 |
| reverse/all_short/bounded_bypass_once vs short_prompt_first | long2048 | NA | NA | NA | 1.069 |
| reverse/mixed/short_prompt_first vs fcfs | all | -0.933 | 2.594 | 4.174 | 0.218 |
| reverse/mixed/short_prompt_first vs fcfs | short128 | -2.830 | -1.917 | -2.462 | 0.218 |
| reverse/mixed/short_prompt_first vs fcfs | long2048 | 0.949 | 3.756 | 4.174 | 0.218 |
| reverse/mixed/bounded_bypass_once vs fcfs | all | -0.460 | 0.129 | 0.943 | 0.484 |
| reverse/mixed/bounded_bypass_once vs fcfs | short128 | -1.223 | -1.097 | -1.492 | 0.484 |
| reverse/mixed/bounded_bypass_once vs fcfs | long2048 | 0.297 | 0.873 | 0.943 | 0.484 |
| reverse/mixed/bounded_bypass_once vs short_prompt_first | all | 0.477 | -2.403 | -3.102 | 0.265 |
| reverse/mixed/bounded_bypass_once vs short_prompt_first | short128 | 1.653 | 0.836 | 0.995 | 0.265 |
| reverse/mixed/bounded_bypass_once vs short_prompt_first | long2048 | -0.646 | -2.779 | -3.102 | 0.265 |

forward/all_short/short_prompt_first vs fcfs: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


forward/all_short/bounded_bypass_once vs fcfs: valid=True; first-schedule order changed=False; equal outputs=5/16; exclusion=None.


forward/all_short/bounded_bypass_once vs short_prompt_first: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


forward/mixed/short_prompt_first vs fcfs: valid=True; first-schedule order changed=True; equal outputs=7/16; exclusion=None.


forward/mixed/bounded_bypass_once vs fcfs: valid=True; first-schedule order changed=True; equal outputs=9/16; exclusion=None.


forward/mixed/bounded_bypass_once vs short_prompt_first: valid=True; first-schedule order changed=True; equal outputs=8/16; exclusion=None.


reverse/all_short/short_prompt_first vs fcfs: valid=True; first-schedule order changed=False; equal outputs=6/16; exclusion=None.


reverse/all_short/bounded_bypass_once vs fcfs: valid=True; first-schedule order changed=False; equal outputs=10/16; exclusion=None.


reverse/all_short/bounded_bypass_once vs short_prompt_first: valid=True; first-schedule order changed=False; equal outputs=4/16; exclusion=None.


reverse/mixed/short_prompt_first vs fcfs: valid=True; first-schedule order changed=True; equal outputs=8/16; exclusion=None.


reverse/mixed/bounded_bypass_once vs fcfs: valid=True; first-schedule order changed=True; equal outputs=5/16; exclusion=None.


reverse/mixed/bounded_bypass_once vs short_prompt_first: valid=True; first-schedule order changed=True; equal outputs=7/16; exclusion=None.


Campaign issues: []. Retained warmups: 12; details and all metric deltas in summary.json.
