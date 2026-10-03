# Fixed KV pool: full versus chunk reservation

MEASUREMENT_ONLY

| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Maximum ITL s | Request max-ITL p99 s | Pooled ITL p99 s | Preemptions / repeated before output / recomputed positions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| repeat0-full | COMPLETE | 32/32 | 23.436882 | 1.365369 | 21.039541 | 4.589342 | 3.792215 | 0.028940 | 2 / 0 / 7685 |
| repeat0-chunk | COMPLETE | 32/32 | 25.170791 | 1.271315 | 22.686492 | 5.973160 | 4.959903 | 0.033027 | 70 / 68 / 123927 |
| repeat1-chunk | COMPLETE | 32/32 | 24.557693 | 1.303054 | 22.153692 | 5.866459 | 4.876939 | 0.030850 | 70 / 68 / 123927 |
| repeat1-full | COMPLETE | 32/32 | 23.393804 | 1.367884 | 20.962592 | 4.572049 | 3.777411 | 0.027786 | 2 / 0 / 7685 |

Each request and all partial cells remain in analysis.json. Only complete matched pairs are compared.

- {"repeat": 0, "baseline": "repeat0-full", "action": "repeat0-chunk", "status": "DESCRIPTIVE_MATCHED_PAIR", "engine_args_equal_except_reservation": true, "throughput_relative_change": -0.06888576473472163, "wall_relative_change": 0.07398207666226453, "mean_completion_relative_change": 0.07827882071929126, "max_itl_change_s": 1.3838175404816866, "request_max_itl_p99_change_s": 1.1676877429615713, "preemptions_change": 68, "recomputed_positions_change": 116242, "output_token_sequences_equal": false}
- {"repeat": 1, "baseline": "repeat1-full", "action": "repeat1-chunk", "status": "DESCRIPTIVE_MATCHED_PAIR", "engine_args_equal_except_reservation": true, "throughput_relative_change": -0.04739407538323481, "wall_relative_change": 0.0497520266864826, "mean_completion_relative_change": 0.05682029016333767, "max_itl_change_s": 1.2944103553891182, "request_max_itl_p99_change_s": 1.0995284023042773, "preemptions_change": 68, "recomputed_positions_change": 116242, "output_token_sequences_equal": false}

Single-model native in-process fixed-pool observation. Request times include waiting, recomputation and instrumentation. Work counts are executed interval overlap, not pure GPU time. Repeated preemption identifies no new generated output between consecutive preemptions, not causal wasted-time savings. Small repeated document cohorts do not establish production tails, quality or policy generalization.
