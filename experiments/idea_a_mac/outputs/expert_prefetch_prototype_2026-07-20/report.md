# Expert-Prefetch System Prototype: Measured Latency Result

| model | cache_capacity | prefetch_budget | h2d_time_us | layer_compute_us_median | mean_reactive_latency_ms | mean_predictive_latency_ms | mean_latency_saving_pct | ci_low_pct | ci_high_pct | mean_reactive_paging_latency_ms | mean_predictive_paging_latency_ms | mean_paging_only_saving_pct | paging_ci_low_pct | paging_ci_high_pct | mean_reactive_miss_rate | mean_predictive_miss_rate | n_docs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe | 8 | 8 | 248.6240 | 10087.9762 | 200.5051 | 194.3346 | 3.0762 | 2.9175 | 3.2200 | 38.8164 | 32.6459 | 16.0095 | 15.1252 | 16.8527 | 0.7666 | 0.6431 | 480 |
| llmjp | 6 | 6 | 81.6320 | 5114.9679 | 95.5297 | 93.7272 | 1.8866 | 1.7862 | 1.9743 | 13.7157 | 11.9132 | 13.2588 | 12.4856 | 13.9576 | 0.8248 | 0.7149 | 480 |