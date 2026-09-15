# Receiver Multi-MoE Inference-Time 5090 Result

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- model revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- discovered MoE blocks: `16`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 48 | 43.1351 | 44.7524 | 23.183 | 45.0927 | 1.04538 | 37.3803 | 37.6656 | 0.829142 | 0.830295 | 16 |
| decode | 4 | 48 | 72.5066 | 76.5125 | 55.1674 | 74.5231 | 1.02781 | 66.2112 | 69.9937 | 0.888816 | 0.893944 | 16 |
| decode | 8 | 48 | 78.7583 | 87.1932 | 101.577 | 80.8165 | 1.02613 | 72.5508 | 81.5392 | 0.897803 | 0.906936 | 16 |
| prefill | 1 | 3 | 86.9445 | 87.3915 | 5888.82 | 88.7535 | 1.02081 | 79.7818 | 81.6 | 0.899611 | 0.900353 | 16 |
| prefill | 4 | 3 | 90.4902 | 94.8044 | 22632.3 | 93.1695 | 1.02961 | 82.9995 | 87.7068 | 0.890844 | 0.895664 | 16 |
| prefill | 8 | 3 | 106.383 | 113.195 | 38502.3 | 108.017 | 1.01536 | 94.622 | 97.5246 | 0.894031 | 0.898928 | 16 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 16 | 0.0642145 | 0.252648 | 0.00801428 |
| decode | 4 | 16 | 0.0662186 | 0.262042 | 0.0368353 |
| decode | 8 | 16 | 0.0654729 | 0.259532 | 0.0295295 |
| prefill | 1 | 16 | 0.0648668 | 0.254697 | 0.0140112 |
| prefill | 4 | 16 | 0.0671484 | 0.256815 | 0.021858 |
| prefill | 8 | 16 | 0.0715743 | 0.263206 | 0.0431712 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | nan | nan | nan | nan | 2 | 2 | 1 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 8 | -0.755929 | 0.0300197 | -0.928571 | 0.000862968 | 2 | 1.99219 | 0.605927 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 8 | -0.206095 | 0.624375 | -0.738095 | 0.0365528 | 2 | 1.9707 | 0.55901 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
