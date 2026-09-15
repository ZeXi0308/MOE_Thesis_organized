# Receiver Multi-MoE Inference-Time 5090 Result

- model: `jamesdborin/tiny-mixtral`
- model revision: `fa41eeceac446bc43fccc3e868a8fad85c7bd101`
- discovered MoE blocks: `2`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 160 | 2.26541 | 2.30957 | 441.422 | 2.43598 | 1.0753 | 1.05317 | 1.08068 | 0.432152 | 0.437636 | 2 |
| decode | 4 | 160 | 3.08678 | 3.40778 | 1295.85 | 3.29166 | 1.06637 | 1.85819 | 2.2414 | 0.563594 | 0.600445 | 2 |
| decode | 8 | 160 | 3.50176 | 3.95174 | 2284.57 | 3.71379 | 1.06055 | 2.26723 | 2.74065 | 0.601224 | 0.642699 | 2 |
| decode | 16 | 160 | 3.89459 | 4.25945 | 4108.26 | 4.0952 | 1.05151 | 2.62576 | 2.88922 | 0.641719 | 0.663658 | 2 |
| prefill | 1 | 5 | 4.24141 | 4.60386 | 30178.7 | 4.49795 | 1.06049 | 2.84419 | 27.1097 | 0.63233 | 0.888232 | 2 |
| prefill | 4 | 5 | 4.60003 | 4.8841 | 111304 | 4.8248 | 1.04886 | 3.0743 | 3.47901 | 0.637188 | 0.659687 | 2 |
| prefill | 8 | 5 | 4.95069 | 5.51408 | 206840 | 5.19802 | 1.04996 | 3.25258 | 3.7673 | 0.626867 | 0.656869 | 2 |
| prefill | 16 | 5 | 5.85818 | 6.22207 | 349597 | 6.04582 | 1.03203 | 3.81523 | 4.6729 | 0.631052 | 0.672796 | 2 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 2 | 0.514508 | 1 | 0.0290161 |
| decode | 4 | 2 | 0.527578 | 1 | 0.0551561 |
| decode | 8 | 2 | 0.508138 | 1 | 0.0162751 |
| decode | 16 | 2 | 0.504908 | 1 | 0.00981655 |
| prefill | 1 | 2 | 0.517072 | 1 | 0.0341445 |
| prefill | 4 | 2 | 0.5136 | 1 | 0.0271992 |
| prefill | 8 | 2 | 0.513711 | 1 | 0.0274225 |
| prefill | 16 | 2 | 0.513898 | 1 | 0.0277959 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | nan | nan | nan | nan | 4 | 4 | 1.73205 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 16 | -0.654909 | 0.00589906 | -0.894321 | 2.97273e-06 | 3 | 2.71875 | 0.991912 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 16 | -0.19234 | 0.475439 | -0.570588 | 0.0209907 | 3.5 | 3.20312 | 1.03922 | single-GPU route imbalance correlation; not receiver congestion |
| 16 | 16 | 0.0582154 | 0.830428 | -0.158824 | 0.556861 | 3 | 2.75 | 0.849139 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
