# Expert-union measurement: adjudication of the frozen predictions

Thresholds were fixed before any route data existed. This script fits
nothing; it only reads what the collector wrote.

## `union`

- layers: 16, experts 64, top-k 8, median batch width 32.0
- layers at or above width 16: 16
- collector verdict: **CANDIDATE_RESIDENCY_HEADROOM**

- pure-decode steps only: True; skipped {'mixed_or_queued': 4, 'no_decode': 1, 'missing_route': 0, 'dropped_request_index_past_end': 0, 'step_lost_all_requests': 0}; hook mismatches None

| prediction | holds | key number |
|---|---|---|
| E1 router concentrates | **True** | 16/16 layers above null |
| E2 idle < 0.1 | **False** | max layer median idle 0.1875 |
| E3 saturates <= 4 steps | **False** | windows [2, 2, 4, 4, 32, 16, None, None, None, None, 16, 32, 8, 8, 8, 8] |

Scope: adjudicated on layers at or above the width threshold

| layer | steps | width | union p50 | idle p50 | uniform null | above null | sat99 | sat95 | skew C | never selected |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 255 | 32 | 0.9688 | 0.0312 | 0.0139 | yes | 2 | 1 | 2.660 | 0 |
| 1 | 255 | 32 | 0.9688 | 0.0312 | 0.0139 | yes | 2 | 1 | 2.171 | 0 |
| 2 | 255 | 32 | 0.9375 | 0.0625 | 0.0139 | yes | 4 | 2 | 2.392 | 0 |
| 3 | 255 | 32 | 0.9375 | 0.0625 | 0.0139 | yes | 4 | 2 | 4.000 | 0 |
| 4 | 255 | 32 | 0.8594 | 0.1406 | 0.0139 | yes | 32 | 4 | 4.056 | 0 |
| 5 | 255 | 32 | 0.8750 | 0.1250 | 0.0139 | yes | 16 | 4 | 3.999 | 0 |
| 6 | 255 | 32 | 0.8125 | 0.1875 | 0.0139 | yes | None | 32 | 3.414 | 0 |
| 7 | 255 | 32 | 0.8438 | 0.1562 | 0.0139 | yes | None | 16 | 2.873 | 1 |
| 8 | 255 | 32 | 0.8438 | 0.1562 | 0.0139 | yes | None | 16 | 4.098 | 0 |
| 9 | 255 | 32 | 0.8906 | 0.1094 | 0.0139 | yes | None | 4 | 3.759 | 0 |
| 10 | 255 | 32 | 0.9219 | 0.0781 | 0.0139 | yes | 16 | 2 | 3.194 | 0 |
| 11 | 255 | 32 | 0.9062 | 0.0938 | 0.0139 | yes | 32 | 2 | 3.469 | 0 |
| 12 | 255 | 32 | 0.9219 | 0.0781 | 0.0139 | yes | 8 | 2 | 4.879 | 0 |
| 13 | 255 | 32 | 0.9219 | 0.0781 | 0.0139 | yes | 8 | 2 | 3.318 | 0 |
| 14 | 255 | 32 | 0.9062 | 0.0938 | 0.0139 | yes | 8 | 2 | 3.081 | 0 |
| 15 | 255 | 32 | 0.9062 | 0.0938 | 0.0139 | yes | 8 | 2 | 2.926 | 0 |

U is a structural signal: not measured HBM traffic, not what the
fused backend loads, not proof that idle bytes are reclaimable.

