# Pool sampling scope correction

The original `pool_ranges`, `max_used_fraction`, and `minimum_free_blocks`
describe only scheduler before/after snapshots. They remain unchanged and valid
for that scope. Native preemption freed blocks between those snapshots, so their
minimum of two free blocks is not the minimum across all recorded observations.

Both native32 cells record zero free blocks at allocation-failure and preemption
points. The corrected analysis retains the original fields with an explicit
scope and separately adds `all_observed_pool_ranges` and `observed_zero_free`.
See [corrected report](../analysis-pool-corrected/report.md) and
[corrected JSON](../analysis-pool-corrected/analysis.json).

No raw data, request metric, execution, comparison, preemption count or
recomputed-token total changed. This corrects the pool sampling interpretation,
not the experiment outcome. The initial analysis remains retained here.
