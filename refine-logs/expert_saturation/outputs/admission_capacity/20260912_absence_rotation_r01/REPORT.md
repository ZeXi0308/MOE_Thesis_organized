# Bound on what absence rotation could buy

Arithmetic on sealed dynamics. Not a policy result: no
counterfactual execution is claimed, and the effect of holding
blocks for a victim on the survivors' TPOT is NOT captured.

## `repeat0-native32`

- steps 1345, step period **19.229 ms** (1242 samples)

| victim | preempt | resume | absence steps |
|---|---:|---:|---:|
| -0003640 | 809 | 1030 | 221 |
| -0003571 | 931 | 1026 | 95 |

- total absence **316 request-steps** = 0.96% of all request-steps (this is the volume rotation CANNOT reduce)
- binding window 221 steps

| cooldown | swaps | rotated max absence | vs concentrated | recompute cost | throughput cost |
|---:|---:|---:|---:|---:|---:|
| 10 | 22 | 0.19 s | **22.1x** lower than 4.25 s | 5.10 s | **19.7%** |
| 20 | 11 | 0.38 s | **11.1x** lower than 4.25 s | 2.55 s | **9.9%** |
| 30 | 7 | 0.58 s | **7.4x** lower than 4.25 s | 1.62 s | **6.3%** |
| 60 | 3 | 1.15 s | **3.7x** lower than 4.25 s | 0.70 s | **2.7%** |
| 120 | 1 | 2.31 s | **1.8x** lower than 4.25 s | 0.23 s | **0.9%** |

Reference: the sealed comparison measured `safe29` (zero
preemption) at **-14.5% throughput** with TTFT p99 **18.0 s**.
A rotation point is only interesting if it beats that on both axes.

## `repeat1-native32`

- steps 1345, step period **19.271 ms** (1242 samples)

| victim | preempt | resume | absence steps |
|---|---:|---:|---:|
| -0003640 | 809 | 1030 | 221 |
| -0003571 | 931 | 1026 | 95 |

- total absence **316 request-steps** = 0.96% of all request-steps (this is the volume rotation CANNOT reduce)
- binding window 221 steps

| cooldown | swaps | rotated max absence | vs concentrated | recompute cost | throughput cost |
|---:|---:|---:|---:|---:|---:|
| 10 | 22 | 0.19 s | **22.1x** lower than 4.26 s | 5.10 s | **19.7%** |
| 20 | 11 | 0.39 s | **11.1x** lower than 4.26 s | 2.55 s | **9.8%** |
| 30 | 7 | 0.58 s | **7.4x** lower than 4.26 s | 1.62 s | **6.3%** |
| 60 | 3 | 1.16 s | **3.7x** lower than 4.26 s | 0.70 s | **2.7%** |
| 120 | 1 | 2.31 s | **1.8x** lower than 4.26 s | 0.23 s | **0.9%** |

Reference: the sealed comparison measured `safe29` (zero
preemption) at **-14.5% throughput** with TTFT p99 **18.0 s**.
A rotation point is only interesting if it beats that on both axes.

