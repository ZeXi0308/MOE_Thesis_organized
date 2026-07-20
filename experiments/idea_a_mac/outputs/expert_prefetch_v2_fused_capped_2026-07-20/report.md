# Expert-Prefetch System Prototype v2 (fused/batched compute)

| model | cache_capacity | prefetch_budget | runtime_capped_budget | h2d_time_us | fused_layer_compute_us_median | mean_reactive_latency_ms | mean_predictive_latency_ms | mean_latency_saving_pct | ci_low_pct | ci_high_pct | mean_latency_saving_capped_pct | ci_low_capped_pct | ci_high_capped_pct | mean_reactive_miss_rate | mean_predictive_miss_rate | mean_predictive_capped_miss_rate | n_docs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe | 8 | 8 | 2 | 248.9280 | 543.5040 | 47.5623 | 63.1006 | -32.9665 | -33.7197 | -32.2712 | 5.3164 | 5.0001 | 5.6469 | 0.7666 | 0.6431 | 0.7163 | 480 |
| llmjp | 6 | 6 | 1 | 81.5680 | 101.7680 | 15.3334 | 19.3470 | -26.4514 | -27.1053 | -25.8224 | 3.0668 | 2.8743 | 3.2649 | 0.8248 | 0.7149 | 0.7963 | 480 |