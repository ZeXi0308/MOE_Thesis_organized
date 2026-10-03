# Interleaved Natural versus Synthetic Input A/B on RTX 5090

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- order: `AB/BA alternating by repeat`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

| phase | batch_size | independent_sequence_pairs | synthetic_latency_median_ms | natural_latency_median_ms | natural_over_synthetic_median | natural_over_synthetic_min | natural_over_synthetic_max | natural_delta_pct_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 5 | 43.6577 | 43.6166 | 0.999483 | 0.998961 | 1.00647 | -0.0516996 |
| decode | 4 | 5 | 73.6468 | 73.6893 | 1.00058 | 0.967516 | 1.03475 | 0.057785 |
| decode | 8 | 5 | 79.3293 | 80.4873 | 1.0113 | 1.0003 | 1.01903 | 1.13011 |
| prefill | 1 | 5 | 83.3015 | 83.8398 | 1.00214 | 0.995666 | 1.08017 | 0.214072 |
| prefill | 4 | 5 | 86.4577 | 86.8437 | 1.00309 | 0.96099 | 1.05601 | 0.3086 |
| prefill | 8 | 5 | 87.045 | 88.2227 | 1.00087 | 0.904011 | 1.01556 | 0.0872876 |

Ratios use the median forward latency within each independent sequence, then the median across
sequence pairs. This controls run-order drift better than comparing two standalone runs, but the
sample size remains descriptive. It measures local input sensitivity, not receiver congestion.

## Untimed route census

| workload | phase | batch_size | route_events | max_to_mean_mean | load_cv_mean | active_expert_fraction_mean | normalized_entropy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| natural | decode | 1 | 128 | 2 | 1 | 0.5 | 0.8 |
| natural | decode | 4 | 128 | 1.99219 | 0.56253 | 0.894531 | 0.941053 |
| natural | decode | 8 | 128 | 1.96094 | 0.523449 | 0.961426 | 0.953264 |
| natural | prefill | 1 | 16 | 1.93066 | 0.474337 | 0.998047 | 0.96548 |
| natural | prefill | 4 | 16 | 1.8208 | 0.387216 | 1 | 0.977603 |
| natural | prefill | 8 | 16 | 1.83997 | 0.357622 | 1 | 0.980998 |
| synthetic | decode | 1 | 128 | 2 | 1 | 0.5 | 0.8 |
| synthetic | decode | 4 | 128 | 1.98047 | 0.571131 | 0.899414 | 0.940175 |
| synthetic | decode | 8 | 128 | 1.93359 | 0.531608 | 0.957031 | 0.950771 |
| synthetic | prefill | 1 | 16 | 1.87207 | 0.461521 | 1 | 0.966109 |
| synthetic | prefill | 4 | 16 | 1.74023 | 0.389048 | 1 | 0.97659 |
| synthetic | prefill | 8 | 16 | 1.80493 | 0.420258 | 1 | 0.972363 |
