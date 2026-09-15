# Bound on what absence rotation could buy

Arithmetic on sealed dynamics. Not a policy result: no
counterfactual execution is claimed, and the effect of holding
blocks for a victim on the survivors' TPOT is NOT captured.

## `repeat0-native32`

- steps 1345, pure-decode baseline **19.493 ms** (928 samples)
- **measured marginal recompute cost 31.1 ms** (2 victims). This is the SLOWDOWN of the steps that carry a recompute chunk, not their whole duration: the other ~30 requests keep decoding through it.

| victim | preempt | resume | absence steps | marginal recompute ms |
|---|---:|---:|---:|---:|
| -0003640 | 809 | 1030 | 221 | 30.0 |
| -0003571 | 931 | 1026 | 95 | 32.1 |

- total absence **316 request-steps** = 0.96% of all request-steps (this is the volume rotation CANNOT reduce)
- binding window 221 steps

| cooldown | swaps | rotated max absence | vs concentrated | recompute cost | throughput cost |
|---:|---:|---:|---:|---:|---:|
| 10 | 22 | 0.19 s | **22.1x** lower than 4.31 s | 1.37 s | **5.9%** |
| 20 | 11 | 0.39 s | **11.1x** lower than 4.31 s | 0.68 s | **3.0%** |
| 30 | 7 | 0.58 s | **7.4x** lower than 4.31 s | 0.44 s | **1.9%** |
| 60 | 3 | 1.17 s | **3.7x** lower than 4.31 s | 0.19 s | **0.8%** |
| 120 | 1 | 2.34 s | **1.8x** lower than 4.31 s | 0.06 s | **0.3%** |

Reference: the sealed comparison measured `safe29` (zero
preemption) at **-14.5% throughput** with TTFT p99 **18.0 s**.
A rotation point is only interesting if it beats that on both axes.

## `repeat1-native32`

- steps 1345, pure-decode baseline **19.549 ms** (928 samples)
- **measured marginal recompute cost 30.9 ms** (2 victims). This is the SLOWDOWN of the steps that carry a recompute chunk, not their whole duration: the other ~30 requests keep decoding through it.

| victim | preempt | resume | absence steps | marginal recompute ms |
|---|---:|---:|---:|---:|
| -0003640 | 809 | 1030 | 221 | 30.0 |
| -0003571 | 931 | 1026 | 95 | 31.8 |

- total absence **316 request-steps** = 0.96% of all request-steps (this is the volume rotation CANNOT reduce)
- binding window 221 steps

| cooldown | swaps | rotated max absence | vs concentrated | recompute cost | throughput cost |
|---:|---:|---:|---:|---:|---:|
| 10 | 22 | 0.20 s | **22.1x** lower than 4.32 s | 1.36 s | **5.8%** |
| 20 | 11 | 0.39 s | **11.1x** lower than 4.32 s | 0.68 s | **2.9%** |
| 30 | 7 | 0.59 s | **7.4x** lower than 4.32 s | 0.43 s | **1.9%** |
| 60 | 3 | 1.17 s | **3.7x** lower than 4.32 s | 0.19 s | **0.8%** |
| 120 | 1 | 2.35 s | **1.8x** lower than 4.32 s | 0.06 s | **0.3%** |

Reference: the sealed comparison measured `safe29` (zero
preemption) at **-14.5% throughput** with TTFT p99 **18.0 s**.
A rotation point is only interesting if it beats that on both axes.

