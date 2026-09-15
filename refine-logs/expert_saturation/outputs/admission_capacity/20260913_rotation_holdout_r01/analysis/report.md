# Independent-cohort rotation controls

MEASUREMENT_ONLY

| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|---:|
| cohort1-block1-native | COMPLETE | 32/32 | 22.681486 | 1.410842 | 20.305374 | 4.465285 |
| cohort1-block1-safe29 | COMPLETE | 32/32 | 26.300305 | 1.216716 | 19.295048 | 0.114482 |
| cohort1-block1-native_aa | COMPLETE | 32/32 | 22.739833 | 1.407222 | 20.370378 | 4.482028 |
| cohort1-block1-rotate | COMPLETE | 32/32 | 22.681231 | 1.410858 | 20.653219 | 1.004141 |
| cohort1-block1-headroom | COMPLETE | 32/32 | 23.311968 | 1.372685 | 21.755639 | 1.382521 |
| cohort0-block1-headroom | COMPLETE | 32/32 | 23.295388 | 1.373662 | 21.739796 | 1.373627 |
| cohort0-block1-rotate | COMPLETE | 32/32 | 22.851350 | 1.400355 | 20.814232 | 1.013891 |
| cohort0-block1-native | COMPLETE | 32/32 | 22.710009 | 1.409070 | 20.349501 | 4.446978 |
| cohort0-block1-safe29 | COMPLETE | 32/32 | 26.523859 | 1.206461 | 19.335921 | 0.121338 |
| cohort0-block1-native_aa | COMPLETE | 32/32 | 22.738236 | 1.407321 | 20.359544 | 4.444982 |
| cohort1-block0-safe29 | COMPLETE | 32/32 | 26.299534 | 1.216752 | 19.306072 | 0.121883 |
| cohort1-block0-native_aa | COMPLETE | 32/32 | 22.749321 | 1.406635 | 20.374634 | 4.494480 |
| cohort1-block0-native | COMPLETE | 32/32 | 22.650577 | 1.412768 | 20.288051 | 4.455265 |
| cohort1-block0-rotate | COMPLETE | 32/32 | 22.711666 | 1.408968 | 20.673723 | 1.005360 |
| cohort1-block0-headroom | COMPLETE | 32/32 | 23.277566 | 1.374714 | 21.721678 | 1.379451 |
| cohort0-block0-native | COMPLETE | 32/32 | 22.844354 | 1.400784 | 20.479039 | 4.480222 |
| cohort0-block0-rotate | COMPLETE | 32/32 | 22.736469 | 1.407430 | 20.706190 | 1.012792 |
| cohort0-block0-safe29 | COMPLETE | 32/32 | 26.534416 | 1.205981 | 19.374513 | 0.093393 |
| cohort0-block0-headroom | COMPLETE | 32/32 | 23.291138 | 1.373913 | 21.733793 | 1.376210 |
| cohort0-block0-native_aa | COMPLETE | 32/32 | 22.667184 | 1.411732 | 20.311025 | 4.432977 |

## Primary paired observations

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort0/0 | native → safe29 | DESCRIPTIVE_MATCHED_PAIR | -13.907% | -5.393% | -4.386829 |
| cohort0/0 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -1.918% | +6.127% | -3.104012 |
| cohort0/0 | native → rotate | DESCRIPTIVE_MATCHED_PAIR | +0.475% | +1.109% | -3.467430 |
| cohort0/0 | safe29 → headroom | DESCRIPTIVE_MATCHED_PAIR | +13.925% | +12.177% | +1.282817 |
| cohort0/0 | safe29 → rotate | DESCRIPTIVE_MATCHED_PAIR | +16.704% | +6.873% | +0.919399 |
| cohort0/0 | headroom → rotate | DESCRIPTIVE_MATCHED_PAIR | +2.440% | -4.728% | -0.363418 |
| cohort0/1 | native → safe29 | DESCRIPTIVE_MATCHED_PAIR | -14.379% | -4.981% | -4.325640 |
| cohort0/1 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -2.513% | +6.832% | -3.073351 |
| cohort0/1 | native → rotate | DESCRIPTIVE_MATCHED_PAIR | -0.619% | +2.284% | -3.433087 |
| cohort0/1 | safe29 → headroom | DESCRIPTIVE_MATCHED_PAIR | +13.859% | +12.432% | +1.252289 |
| cohort0/1 | safe29 → rotate | DESCRIPTIVE_MATCHED_PAIR | +16.071% | +7.645% | +0.892553 |
| cohort0/1 | headroom → rotate | DESCRIPTIVE_MATCHED_PAIR | +1.943% | -4.257% | -0.359736 |
| cohort1/0 | native → safe29 | DESCRIPTIVE_MATCHED_PAIR | -13.875% | -4.840% | -4.333382 |
| cohort1/0 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -2.694% | +7.066% | -3.075814 |
| cohort1/0 | native → rotate | DESCRIPTIVE_MATCHED_PAIR | -0.269% | +1.901% | -3.449905 |
| cohort1/0 | safe29 → headroom | DESCRIPTIVE_MATCHED_PAIR | +12.982% | +12.512% | +1.257568 |
| cohort1/0 | safe29 → rotate | DESCRIPTIVE_MATCHED_PAIR | +15.797% | +7.084% | +0.883477 |
| cohort1/0 | headroom → rotate | DESCRIPTIVE_MATCHED_PAIR | +2.492% | -4.824% | -0.374091 |
| cohort1/1 | native → safe29 | DESCRIPTIVE_MATCHED_PAIR | -13.760% | -4.976% | -4.350803 |
| cohort1/1 | native → headroom | DESCRIPTIVE_MATCHED_PAIR | -2.705% | +7.142% | -3.082764 |
| cohort1/1 | native → rotate | DESCRIPTIVE_MATCHED_PAIR | +0.001% | +1.713% | -3.461144 |
| cohort1/1 | safe29 → headroom | DESCRIPTIVE_MATCHED_PAIR | +12.819% | +12.752% | +1.268039 |
| cohort1/1 | safe29 → rotate | DESCRIPTIVE_MATCHED_PAIR | +15.956% | +7.039% | +0.889659 |
| cohort1/1 | headroom → rotate | DESCRIPTIVE_MATCHED_PAIR | +2.781% | -5.067% | -0.378380 |

## Native A/A observed drift

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort0/0 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | +0.782% | -0.820% | -0.047245 |
| cohort0/1 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | -0.124% | +0.049% | -0.001996 |
| cohort1/0 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | -0.434% | +0.427% | +0.039215 |
| cohort1/1 | native → native_aa | DESCRIPTIVE_MATCHED_PAIR | -0.257% | +0.320% | +0.016743 |

All twenty cells must be COMPLETE and qualified before numeric comparisons. Six primary pairs per cohort/block; native_aa is compared to primary native, never selected as a replacement baseline. No cross-cohort request pairing.

Two distinct document cohorts, each run in two order blocks; twenty engine executions are not twenty independent workloads. Request/token observations are not independent experimental repeats. Four native A/A pairs describe observed drift only, not a statistical noise bound. No significance, noninferiority or method GO is inferred. Fixed 5 s TTFT / 0.2 s mean-TPOT reference SLO does not constrain maximum ITL; all-pass goodput equals throughput.

wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is a subset of scheduler time. Work classes are disjoint calls; recompute/prefill calls may also produce new decode. Engine remainder includes host overhead, sampling, synchronization and model execution, not pure GPU time. Width differences are observed policy-specific paths, not counterfactual savings.

Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.
