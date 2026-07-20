# Selector Causal Audit

- model: `allenai/OLMoE-1B-7B-0924`
- independent documents: 8 (`validation` offset 32)
- sequence length: 128
- routed pairs: 131072
- owner groups / fixed-rate tile: 8 / 64 pairs
- patched-full exactness: max/mean logit difference 0.0/0.0

## Pooled score correlation with one-pair local intervention

| scope | layer | score | pairs | spearman_vs_causal_gain |
|---|---|---|---|---|
| pooled | -1 | rank_score | 131072 | 0.395902 |
| pooled | -1 | gate | 131072 | 0.715198 |
| pooled | -1 | contribution | 131072 | 0.967101 |
| pooled | -1 | qerr | 131072 | 0.96694 |
| pooled | -1 | qbenefit | 131072 | 0.966897 |

## Across-layer correlation stability

| score | mean | median | min | max |
|---|---|---|---|---|
| contribution | 0.968121 | 0.975533 | 0.908549 | 0.981347 |
| gate | 0.856538 | 0.876022 | 0.732996 | 0.894354 |
| qbenefit | 0.970705 | 0.976873 | 0.924752 | 0.982628 |
| qerr | 0.970699 | 0.976848 | 0.924601 | 0.98263 |
| rank_score | 0.759574 | 0.768618 | 0.663678 | 0.811634 |

## Fixed-quota one-pair additive oracle recovery

| score | tiles | summed_one_pair_causal_gain | fraction_of_additive_oracle | rank_to_oracle_recovery |
|---|---|---|---|---|
| rank_score | 2554 | 625.39 | 0.918391 | 0 |
| gate | 2554 | 644.703 | 0.946753 | 0.347535 |
| contribution | 2554 | 671.284 | 0.985787 | 0.82584 |
| qerr | 2554 | 672.974 | 0.988268 | 0.856248 |
| qbenefit | 2554 | 673.004 | 0.988313 | 0.856789 |

## Cross-term and relative-error diagnostics

| tokens | median_cross_to_diagonal | median_abs_cross_to_diagonal | p95_abs_cross_to_diagonal | fraction_abs_cross_gt_0p3 | fraction_negative_cross | pair_upgrade_positive_gain_fraction | low_relative_error_coefficient_cv |
|---|---|---|---|---|---|---|---|
| 16384 | 0.00190989 | 0.0190143 | 0.0655608 | 0.000915527 | 0.469971 | 0.983208 | 0.155358 |

## Interpretation boundary

`causal_local_gain` upgrades exactly one pair from fake MXFP4 to fake FP8 while
all other routed outputs for that token remain MXFP4.  It includes pair-vs-rest
error cancellation for that one intervention.  The tile table then sums these
one-pair gains, so its oracle is only an additive interventional oracle; it is
not the exact joint optimum after many simultaneous upgrades.  Cross terms,
BF16 accumulation order, downstream nonlinear propagation, routing drift,
native codec behavior, selector cost, network traffic, and latency remain
separate validation obligations.
