# Receiver Multi-MoE Inference-Time 5090 Result

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- model revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- discovered MoE blocks: `16`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 48 | 43.0294 | 47.8699 | 23.2399 | 45.0191 | 1.04624 | 37.2847 | 37.8273 | 0.828523 | 0.830451 | 16 |
| decode | 4 | 48 | 73.0359 | 77.946 | 54.7676 | 75.3172 | 1.03124 | 66.9832 | 72.987 | 0.889526 | 0.894327 | 16 |
| decode | 8 | 48 | 79.145 | 80.8112 | 101.081 | 81.6436 | 1.03157 | 73.2948 | 78.8472 | 0.897446 | 0.901065 | 16 |
| prefill | 1 | 3 | 90.2591 | 96.566 | 22690.2 | 92.1015 | 1.02041 | 82.0059 | 90.1236 | 0.890515 | 0.897575 | 16 |
| prefill | 4 | 3 | 134.887 | 140.49 | 60732.2 | 135.155 | 1.00199 | 120.447 | 127.109 | 0.89117 | 0.893559 | 16 |
| prefill | 8 | 3 | 191.538 | 193.079 | 85539.2 | 185.128 | 0.966535 | 162.15 | 167.336 | 0.876018 | 0.878878 | 16 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 16 | 0.0643198 | 0.252643 | 0.00845269 |
| decode | 4 | 16 | 0.0659091 | 0.261074 | 0.0360642 |
| decode | 8 | 16 | 0.0653133 | 0.258562 | 0.0317443 |
| prefill | 1 | 16 | 0.0645866 | 0.25334 | 0.0102289 |
| prefill | 4 | 16 | 0.0633587 | 0.253129 | 0.0111674 |
| prefill | 8 | 16 | 0.0638291 | 0.253922 | 0.0135757 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | nan | nan | nan | nan | 2 | 2 | 1 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 8 | 0.503953 | 0.202884 | -0.357143 | 0.385121 | 2 | 1.99219 | 0.607493 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 8 | -0.763763 | 0.027396 | -1 | 0 | 2 | 1.94141 | 0.537879 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
