# Selector Causal Audit

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- independent documents: 8 (`validation` offset 16)
- sequence length: 128
- routed pairs: 262144
- owner groups / fixed-rate tile: 8 / 64 pairs
- patched-full exactness: max/mean logit difference 0.0/0.0

## Pooled score correlation with one-pair local intervention

| scope | layer | score | pairs | spearman_vs_causal_gain |
|---|---|---|---|---|
| pooled | -1 | rank_score | 262144 | 0.562373 |
| pooled | -1 | gate | 262144 | 0.566389 |
| pooled | -1 | contribution | 262144 | 0.823085 |
| pooled | -1 | qerr | 262144 | 0.823622 |
| pooled | -1 | qbenefit | 262144 | 0.823595 |

## Across-layer correlation stability

| score | mean | median | min | max |
|---|---|---|---|---|
| contribution | 0.828627 | 0.839491 | 0.724189 | 0.871127 |
| gate | 0.78247 | 0.794571 | 0.632279 | 0.832082 |
| qbenefit | 0.831996 | 0.841194 | 0.744533 | 0.872314 |
| qerr | 0.831976 | 0.841162 | 0.744595 | 0.8723 |
| rank_score | 0.747938 | 0.766977 | 0.595188 | 0.799005 |

## Fixed-quota one-pair additive oracle recovery

| score | tiles | summed_one_pair_causal_gain | fraction_of_additive_oracle | rank_to_oracle_recovery |
|---|---|---|---|---|
| rank_score | 4599 | 36117.5 | 0.995674 | 0 |
| gate | 4599 | 35925.8 | 0.990389 | -1.22163 |
| contribution | 4599 | 36163.8 | 0.99695 | 0.295015 |
| qerr | 4599 | 36170.8 | 0.997144 | 0.339729 |
| qbenefit | 4599 | 36171.6 | 0.997164 | 0.344464 |

## Cross-term and relative-error diagnostics

| tokens | median_cross_to_diagonal | median_abs_cross_to_diagonal | p95_abs_cross_to_diagonal | fraction_abs_cross_gt_0p3 | fraction_negative_cross | pair_upgrade_positive_gain_fraction | low_relative_error_coefficient_cv |
|---|---|---|---|---|---|---|---|
| 16384 | 0.00606547 | 0.0363071 | 0.123239 | 0.00299072 | 0.453796 | 0.886543 | 0.163852 |

## Interpretation boundary

`causal_local_gain` upgrades exactly one pair from fake MXFP4 to fake FP8 while
all other routed outputs for that token remain MXFP4.  It includes pair-vs-rest
error cancellation for that one intervention.  The tile table then sums these
one-pair gains, so its oracle is only an additive interventional oracle; it is
not the exact joint optimum after many simultaneous upgrades.  Cross terms,
BF16 accumulation order, downstream nonlinear propagation, routing drift,
native codec behavior, selector cost, network traffic, and latency remain
separate validation obligations.
