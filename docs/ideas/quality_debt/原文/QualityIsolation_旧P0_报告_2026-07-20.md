# Per-request Quality Isolation P0

## Part A: oracle-vs-random upgrade quota ceiling (worst-case KL reduction)

| model | base_strategy | quota_frac | quota_m | baseline_p95_kl | oracle_p95_kl | random_p95_kl_mean | oracle_p95_reduction_pct | random_p95_reduction_pct | oracle_advantage_over_random_pp | proxy_signal_strategy | proxy_realized_p95_kl | proxy_realized_p95_reduction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe | fixed_tail4 | 0.1000 | 3 | 0.0149 | 0.0087 | 0.0140 | 41.6215 | 6.1760 | 35.4455 | gate_mass_profile_3_5 | 0.0087 | 41.6215 |
| olmoe | fixed_tail4 | 0.1500 | 5 | 0.0149 | 0.0085 | 0.0135 | 43.0558 | 9.4223 | 33.6335 | gate_mass_profile_3_5 | 0.0087 | 41.6969 |
| olmoe | fixed_tail4 | 0.2500 | 8 | 0.0149 | 0.0078 | 0.0127 | 47.5163 | 14.9085 | 32.6077 | gate_mass_profile_3_5 | 0.0087 | 41.6969 |
| olmoe | fixed_tail4 | 0.3500 | 11 | 0.0149 | 0.0069 | 0.0117 | 53.6940 | 21.3567 | 32.3373 | gate_mass_profile_3_5 | 0.0085 | 43.1888 |
| olmoe | fixed_tail4 | 0.5000 | 16 | 0.0149 | 0.0059 | 0.0104 | 60.6729 | 30.0570 | 30.6159 | gate_mass_profile_3_5 | 0.0077 | 48.4880 |
| llmjp | fixed_tail8 | 0.1000 | 3 | 0.0419 | 0.0323 | 0.0409 | 22.9424 | 2.2747 | 20.6677 | anti_kl_profile_7_9 | 0.0323 | 22.9424 |
| llmjp | fixed_tail8 | 0.1500 | 5 | 0.0419 | 0.0316 | 0.0402 | 24.6030 | 3.8676 | 20.7353 | anti_kl_profile_7_9 | 0.0318 | 24.0318 |
| llmjp | fixed_tail8 | 0.2500 | 8 | 0.0419 | 0.0285 | 0.0390 | 31.8942 | 6.7853 | 25.1089 | anti_kl_profile_7_9 | 0.0285 | 31.8942 |
| llmjp | fixed_tail8 | 0.3500 | 11 | 0.0419 | 0.0261 | 0.0379 | 37.7595 | 9.4925 | 28.2670 | anti_kl_profile_7_9 | 0.0261 | 37.7595 |
| llmjp | fixed_tail8 | 0.5000 | 16 | 0.0419 | 0.0193 | 0.0355 | 53.8726 | 15.2421 | 38.6305 | anti_kl_profile_7_9 | 0.0193 | 53.8726 |

## Part B: is document KL-riskiness transferable across degradation mechanisms (the causally-valid signal a real credit system would need)?

| model | base_strategy | candidate_signal_strategy | n_docs | spearman_rho | p_value |
|---|---|---|---|---|---|
| olmoe | fixed_tail4 | kl_profile_3_5 | 32 | 0.7500 | 0.0000 |
| olmoe | fixed_tail4 | p95_profile_3_5 | 32 | 0.7529 | 0.0000 |
| olmoe | fixed_tail4 | gate_mass_profile_3_5 | 32 | 0.8526 | 0.0000 |
| olmoe | fixed_tail4 | anti_kl_profile_3_5 | 32 | 0.8526 | 0.0000 |
| olmoe | fixed_tail4 | kl_profile_2_4_6 | 32 | 0.8101 | 0.0000 |
| llmjp | fixed_tail8 | kl_profile_7_9 | 32 | 0.9765 | 0.0000 |
| llmjp | fixed_tail8 | p95_profile_7_9 | 32 | 0.9795 | 0.0000 |
| llmjp | fixed_tail8 | gate_mass_profile_7_9 | 32 | 0.9791 | 0.0000 |
| llmjp | fixed_tail8 | anti_kl_profile_7_9 | 32 | 0.9850 | 0.0000 |
| llmjp | fixed_tail8 | kl_profile_6_8_10 | 32 | 0.9842 | 0.0000 |