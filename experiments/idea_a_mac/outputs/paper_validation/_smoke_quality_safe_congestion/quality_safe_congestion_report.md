# Quality-Safe Critical-Flow Congestion Frontier

## Boundary

This is bandwidth-only trace replay with correct combine direction
`expert_owner_rank -> token_origin_rank`.  `selector_full_mean_token_kl` is
copied from the corresponding full-selector fake-quant experiment; subset
spending policies were not re-evaluated for quality and must not inherit that
number as a measured KL.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`; EP=`8`; GPUs/node=`4`
- concurrent jobs: `[1]`; origin modes: `['balanced']`
- layer selector: `kl_profile_2_4_6` = `[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 6, 2]`
- calibrated gate threshold: `0.045898`
- calibrated cumulative tail-mass budget: `0.159424`

## Largest-concurrency view

| origin_mode | selector | spending | safe_fraction_of_pairs | int4_fraction_of_pairs | payload_saving_vs_bf16 | remote_wire_saving_vs_bf16 | bottleneck_saving_vs_uniform_fp8 | selector_full_mean_token_kl |
|---|---|---|---|---|---|---|---|---|
| balanced | none | uniform_fp8 | 0.000000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| balanced | fixed_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.624234 | 0.248469 | 0.030320 |
| balanced | fixed_rank | remote_only | 0.500000 | 0.253571 | 0.563393 | 0.624234 | 0.248469 | 0.030320 |
| balanced | fixed_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.560586 | 0.121172 | 0.030320 |
| balanced | fixed_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.612752 | 0.225503 | 0.030320 |
| balanced | layer_rank | all_safe | 0.500000 | 0.500000 | 0.625000 | 0.627625 | 0.255249 | 0.019569 |
| balanced | layer_rank | remote_only | 0.500000 | 0.260491 | 0.565123 | 0.627625 | 0.255249 | 0.019569 |
| balanced | layer_rank | random_budget | 0.500000 | 0.250000 | 0.562500 | 0.564633 | 0.129265 | 0.019569 |
| balanced | layer_rank | critical_budget | 0.500000 | 0.250000 | 0.562500 | 0.615048 | 0.230096 | 0.019569 |
| balanced | gate_threshold | all_safe | 0.540848 | 0.540848 | 0.635212 | 0.633640 | 0.267279 | 0.018361 |
| balanced | gate_threshold | remote_only | 0.540848 | 0.272768 | 0.568192 | 0.633640 | 0.267279 | 0.018361 |
| balanced | gate_threshold | random_budget | 0.540848 | 0.270313 | 0.567578 | 0.566273 | 0.132546 | 0.018361 |
| balanced | gate_threshold | critical_budget | 0.540848 | 0.270313 | 0.567578 | 0.622485 | 0.244969 | 0.018361 |
| balanced | gate_tailmass | all_safe | 0.506027 | 0.506027 | 0.626507 | 0.627297 | 0.254593 | 0.021300 |
| balanced | gate_tailmass | remote_only | 0.506027 | 0.259821 | 0.564955 | 0.627297 | 0.254593 | 0.021300 |
| balanced | gate_tailmass | random_budget | 0.506027 | 0.252902 | 0.563225 | 0.563211 | 0.126422 | 0.021300 |
| balanced | gate_tailmass | critical_budget | 0.506027 | 0.252902 | 0.563225 | 0.614829 | 0.229659 | 0.021300 |

## Interpretation

- `all_safe` applies INT4 to every pair admitted by the quality selector.
- `remote_only` applies INT4 only to admitted inter-node pairs.
- `random_budget` and `critical_budget` use the same admitted-pair count per layer.
- `critical_budget` is an online trace-level upper bound, not a deployable scheduler yet.
