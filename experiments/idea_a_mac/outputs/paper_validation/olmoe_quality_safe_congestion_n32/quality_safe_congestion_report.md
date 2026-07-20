# Quality-Safe Critical-Flow Congestion Frontier

## Boundary

This is bandwidth-only trace replay with correct combine direction
`expert_owner_rank -> token_origin_rank`.  `selector_full_mean_token_kl` is
copied from the corresponding full-selector fake-quant experiment; subset
spending policies were not re-evaluated for quality and must not inherit that
number as a measured KL.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`; EP=`8`; GPUs/node=`4`
- concurrent jobs: `[1, 2, 4, 8, 16]`; origin modes: `['balanced', 'hotspot']`
- layer selector: `kl_profile_2_4_6` = `[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 6, 2]`
- calibrated gate threshold: `0.045898`
- calibrated cumulative tail-mass budget: `0.159424`

## Largest-concurrency view

| origin_mode | selector | spending | safe_fraction_of_pairs | int4_fraction_of_pairs | payload_saving_vs_bf16 | remote_wire_saving_vs_bf16 | bottleneck_saving_vs_uniform_fp8 | selector_full_mean_token_kl |
|---|---|---|---|---|---|---|---|---|
| balanced | none | uniform_fp8 | 0.000000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| balanced | fixed_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.624431 | 0.230190 | 0.030320 |
| balanced | fixed_rank | remote_only | 0.500000 | 0.250006 | 0.562502 | 0.624431 | 0.230190 | 0.030320 |
| balanced | fixed_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.562712 | 0.116694 | 0.030320 |
| balanced | fixed_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.623020 | 0.230190 | 0.030320 |
| balanced | layer_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.624554 | 0.235222 | 0.019569 |
| balanced | layer_rank | remote_only | 0.500000 | 0.250253 | 0.562563 | 0.624554 | 0.235222 | 0.019569 |
| balanced | layer_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.562315 | 0.119294 | 0.019569 |
| balanced | layer_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.623020 | 0.235222 | 0.019569 |
| balanced | gate_threshold | all_safe | 0.530997 | 0.530997 | 0.632749 | 0.632211 | 0.238221 | 0.018361 |
| balanced | gate_threshold | remote_only | 0.530997 | 0.265638 | 0.566409 | 0.632211 | 0.238221 | 0.018361 |
| balanced | gate_threshold | random_budget | 0.530997 | 0.265492 | 0.566373 | 0.566251 | 0.119760 | 0.018361 |
| balanced | gate_threshold | critical_budget | 0.530997 | 0.265492 | 0.566373 | 0.630645 | 0.238221 | 0.018361 |
| balanced | gate_tailmass | all_safe | 0.510047 | 0.510047 | 0.627512 | 0.626870 | 0.232556 | 0.021300 |
| balanced | gate_tailmass | remote_only | 0.510047 | 0.254907 | 0.563727 | 0.626870 | 0.232556 | 0.021300 |
| balanced | gate_tailmass | random_budget | 0.510047 | 0.255021 | 0.563755 | 0.563415 | 0.118294 | 0.021300 |
| balanced | gate_tailmass | critical_budget | 0.510047 | 0.255021 | 0.563755 | 0.625402 | 0.232556 | 0.021300 |
| hotspot | none | uniform_fp8 | 0.000000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| hotspot | fixed_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.624799 | 0.242007 | 0.030320 |
| hotspot | fixed_rank | remote_only | 0.500000 | 0.253020 | 0.563255 | 0.624799 | 0.242007 | 0.030320 |
| hotspot | fixed_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.562960 | 0.122962 | 0.030320 |
| hotspot | fixed_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.622085 | 0.242007 | 0.030320 |
| hotspot | layer_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.625295 | 0.248295 | 0.019569 |
| hotspot | layer_rank | remote_only | 0.500000 | 0.254027 | 0.563507 | 0.625295 | 0.248295 | 0.019569 |
| hotspot | layer_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.562619 | 0.124214 | 0.019569 |
| hotspot | layer_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.622294 | 0.248295 | 0.019569 |
| hotspot | gate_threshold | all_safe | 0.530997 | 0.530997 | 0.632749 | 0.632577 | 0.255542 | 0.018361 |
| hotspot | gate_threshold | remote_only | 0.530997 | 0.268791 | 0.567198 | 0.632577 | 0.255542 | 0.018361 |
| hotspot | gate_threshold | random_budget | 0.530997 | 0.265492 | 0.566373 | 0.566214 | 0.125919 | 0.018361 |
| hotspot | gate_threshold | critical_budget | 0.530997 | 0.265492 | 0.566373 | 0.629601 | 0.255542 | 0.018361 |
| hotspot | gate_tailmass | all_safe | 0.510047 | 0.510047 | 0.627512 | 0.627334 | 0.244751 | 0.021300 |
| hotspot | gate_tailmass | remote_only | 0.510047 | 0.258161 | 0.564540 | 0.627334 | 0.244751 | 0.021300 |
| hotspot | gate_tailmass | random_budget | 0.510047 | 0.255021 | 0.563755 | 0.563834 | 0.123654 | 0.021300 |
| hotspot | gate_tailmass | critical_budget | 0.510047 | 0.255021 | 0.563755 | 0.624530 | 0.244751 | 0.021300 |

## Interpretation

- `all_safe` applies INT4 to every pair admitted by the quality selector.
- `remote_only` applies INT4 only to admitted inter-node pairs.
- `random_budget` and `critical_budget` use the same admitted-pair count per layer.
- `critical_budget` is an online trace-level upper bound, not a deployable scheduler yet.
