# Receiver Multi-MoE Inference-Time 5090 Result

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- model revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- discovered MoE blocks: `16`
- status: `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Timing summary

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | unprofiled_goodput_median_tokens_per_s | profiled_latency_median_ms | instrumentation_ratio_median | moe_sum_median_ms | moe_sum_p95_ms | moe_fraction_median | moe_fraction_p95 | moe_calls_per_forward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 160 | 43.0091 | 45.5139 | 23.2509 | 44.7468 | 1.04041 | 37.0536 | 40.2143 | 0.828373 | 0.829779 | 16 |
| decode | 4 | 160 | 72.671 | 79.0217 | 55.0426 | 74.371 | 1.02339 | 66.0307 | 70.5179 | 0.888345 | 0.893638 | 16 |
| decode | 8 | 160 | 79.0144 | 87.4374 | 101.247 | 80.9558 | 1.02457 | 72.6989 | 75.6283 | 0.898232 | 0.900034 | 16 |
| decode | 16 | 160 | 82.162 | 89.7087 | 194.737 | 83.9254 | 1.02146 | 75.6551 | 82.3474 | 0.901804 | 0.904019 | 16 |
| decode | 32 | 160 | 83.0975 | 92.7581 | 385.09 | 85.077 | 1.02382 | 76.7329 | 85.3019 | 0.902149 | 0.910416 | 16 |
| prefill | 1 | 5 | 83.3517 | 84.2406 | 1535.66 | 86.0235 | 1.03205 | 77.2285 | 77.4462 | 0.898368 | 0.899288 | 16 |
| prefill | 4 | 5 | 86.8085 | 87.9187 | 5898.04 | 89.6654 | 1.03291 | 80.691 | 83.171 | 0.898076 | 0.89974 | 16 |
| prefill | 8 | 5 | 88.407 | 89.7151 | 11582.8 | 91.1684 | 1.03124 | 81.7761 | 83.2454 | 0.896978 | 0.898139 | 16 |
| prefill | 16 | 5 | 94.8657 | 102.15 | 21588.4 | 95.0406 | 1.00184 | 84.6786 | 87.2684 | 0.891835 | 0.895711 | 16 |
| prefill | 32 | 5 | 99.1609 | 105.472 | 41306.6 | 100.685 | 1.01537 | 90.1756 | 96.8988 | 0.895622 | 0.898213 | 16 |

## Layer-time concentration

| phase | batch_size | num_moe_layers | max_single_layer_share | top4_layer_share | layer_latency_cv |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 16 | 0.064306 | 0.252739 | 0.00833614 |
| decode | 4 | 16 | 0.0657692 | 0.261062 | 0.036454 |
| decode | 8 | 16 | 0.064485 | 0.257358 | 0.0244259 |
| decode | 16 | 16 | 0.0636218 | 0.252344 | 0.00903841 |
| decode | 32 | 16 | 0.0636843 | 0.252023 | 0.00587989 |
| prefill | 1 | 16 | 0.0645559 | 0.25317 | 0.0102523 |
| prefill | 4 | 16 | 0.0640134 | 0.252765 | 0.00807822 |
| prefill | 8 | 16 | 0.0647675 | 0.254387 | 0.012189 |
| prefill | 16 | 16 | 0.0644127 | 0.256065 | 0.0154049 |
| prefill | 32 | 16 | 0.0644126 | 0.255864 | 0.0194448 |

## Route imbalance correlation (descriptive only)

| batch_size | n_decode_steps | spearman_latency_vs_route_max_to_mean_mean | pvalue_max_to_mean_descriptive_only | spearman_latency_vs_route_load_cv_mean | pvalue_load_cv_descriptive_only | route_max_to_mean_max_median | route_max_to_mean_mean | route_load_cv_mean | evidence_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | nan | nan | nan | nan | 2 | 2 | 1 | single-GPU route imbalance correlation; not receiver congestion |
| 4 | 16 | 0.0840168 | 0.757053 | -0.973529 | 2.2769e-10 | 2 | 1.99805 | 0.63236 | single-GPU route imbalance correlation; not receiver congestion |
| 8 | 16 | -0.712434 | 0.0019559 | -0.767647 | 0.000516884 | 2 | 1.94531 | 0.552317 | single-GPU route imbalance correlation; not receiver congestion |
| 16 | 16 | -0.514802 | 0.0413014 | -0.458824 | 0.0738355 | 2 | 1.90088 | 0.489877 | single-GPU route imbalance correlation; not receiver congestion |
| 32 | 16 | 0.0739108 | 0.785597 | -0.558824 | 0.0244365 | 2 | 1.86816 | 0.46973 | single-GPU route imbalance correlation; not receiver congestion |

## Interpretation

The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;
`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator
and is a local cumulative MoE-block characterization, not an EP return-path fraction.

This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural
continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be
called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.
