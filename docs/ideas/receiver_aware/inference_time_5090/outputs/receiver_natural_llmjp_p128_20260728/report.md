# Receiver Multi-MoE Inference-Time 5090 Result

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- model revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- discovered MoE blocks: `16`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 48 | 45.8263 | 46.7211 | 21.8215 | 47.827 | 1.04366 | 39.6034 | 40.1436 | 0.828611 | 0.831193 | 16 |
| decode | 4 | 48 | 78.8831 | 82.7493 | 50.708 | 80.805 | 1.02436 | 72.122 | 75.9995 | 0.891432 | 0.897148 | 16 |
| decode | 8 | 48 | 85.1079 | 87.7694 | 93.9983 | 87.5885 | 1.02915 | 78.7157 | 83.8672 | 0.899778 | 0.905704 | 16 |
| prefill | 1 | 3 | 87.9831 | 88.0366 | 1454.82 | 90.7636 | 1.0316 | 81.6586 | 82.1514 | 0.899684 | 0.90002 | 16 |
| prefill | 4 | 3 | 91.9117 | 92.6645 | 5570.56 | 94.01 | 1.02283 | 84.625 | 86.5149 | 0.90017 | 0.901607 | 16 |
| prefill | 8 | 3 | 93.3629 | 97.0429 | 10968 | 95.9268 | 1.02746 | 86.2414 | 86.8729 | 0.899034 | 0.900839 | 16 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 16 | 0.0643188 | 0.252338 | 0.00813381 |
| decode | 4 | 16 | 0.0646698 | 0.256315 | 0.021336 |
| decode | 8 | 16 | 0.0638222 | 0.254395 | 0.0135452 |
| prefill | 1 | 16 | 0.0637782 | 0.253679 | 0.00982298 |
| prefill | 4 | 16 | 0.064831 | 0.255174 | 0.0145985 |
| prefill | 8 | 16 | 0.0638873 | 0.254047 | 0.0111641 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | nan | nan | nan | nan | 2 | 2 | 1 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 8 | -0.57735 | 0.133975 | -0.904762 | 0.00200828 | 2 | 1.99219 | 0.56253 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 8 | -0.545397 | 0.162075 | -0.833333 | 0.0101755 | 2 | 1.96094 | 0.523449 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
