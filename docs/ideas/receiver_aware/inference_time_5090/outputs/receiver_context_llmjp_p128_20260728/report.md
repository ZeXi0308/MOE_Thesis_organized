# Receiver Multi-MoE Inference-Time 5090 Result

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- model revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- discovered MoE blocks: `16`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 48 | 42.9307 | 45.0882 | 23.2934 | 44.9889 | 1.04794 | 37.217 | 43.6904 | 0.827454 | 0.83437 | 16 |
| decode | 4 | 48 | 73.7503 | 77.4791 | 54.2373 | 75.5725 | 1.02471 | 67.3176 | 70.5625 | 0.89 | 0.894451 | 16 |
| decode | 8 | 48 | 80.2624 | 86.2945 | 99.673 | 82.2759 | 1.02509 | 73.968 | 80.7469 | 0.898553 | 0.900411 | 16 |
| prefill | 1 | 3 | 83.1806 | 83.8075 | 1538.82 | 85.5181 | 1.0281 | 76.8311 | 77.0374 | 0.898419 | 0.898851 | 16 |
| prefill | 4 | 3 | 87.3866 | 88.1931 | 5859.02 | 91.1396 | 1.04295 | 81.9895 | 84.2515 | 0.899604 | 0.900506 | 16 |
| prefill | 8 | 3 | 88.3577 | 89.865 | 11589.3 | 90.0095 | 1.0187 | 80.7156 | 83.3596 | 0.896744 | 0.897981 | 16 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 16 | 0.0641658 | 0.252647 | 0.00791282 |
| decode | 4 | 16 | 0.0651713 | 0.259724 | 0.0296446 |
| decode | 8 | 16 | 0.064619 | 0.257185 | 0.0245109 |
| prefill | 1 | 16 | 0.0643867 | 0.252514 | 0.00868749 |
| prefill | 4 | 16 | 0.0640628 | 0.254348 | 0.0116461 |
| prefill | 8 | 16 | 0.0647493 | 0.254016 | 0.0121903 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | nan | nan | nan | nan | 2 | 2 | 1 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 8 | -0.3849 | 0.346422 | -0.738095 | 0.0365528 | 2 | 1.98047 | 0.571131 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 8 | 0.0514344 | 0.90373 | -0.0714286 | 0.866526 | 2 | 1.93359 | 0.531608 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
