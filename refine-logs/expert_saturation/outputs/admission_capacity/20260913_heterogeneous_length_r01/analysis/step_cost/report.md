# Three-term decode step cost model

Width variable: `decode_requests`. Calibrated on `hom_b0` only.

| term | value | reading |
|---|---:|---|
| gamma (per step) | 9.4469 ms | launch, graph replay, host bookkeeping |
| alpha (per request) | 3.701 us | per-token dense work incl. routed experts |
| beta (per context token) | 102.5683 ns | attention scan over resident KV |

| cell | steps | decode s actual | predicted | total err | median err | p90 err |
|---|---:|---:|---:|---:|---:|---:|
| hom_b0 (calibration) | 2189 | 38.814 | 38.814 | -0.00% | 4.54% | 11.88% |
| hom_b1 | 2189 | 38.556 | 38.814 | +0.67% | 4.29% | 11.43% |
| het_b0 | 3156 | 48.342 | 50.245 | +3.94% | 10.09% | 13.85% |
| origdom_native | 1241 | 20.265 | 23.289 | +14.92% | 9.31% | 115.68% |

## Cost share at each cell's median step

| cell | median w | median context tok | fixed | width | context |
|---|---:|---:|---:|---:|---:|
| hom_b0 | 42 | 82023 | 52.4% | 0.9% | 46.7% |
| hom_b1 | 42 | 82034 | 52.4% | 0.9% | 46.7% |
| het_b0 | 21 | 63448 | 58.9% | 0.5% | 40.6% |
| origdom_native | 32 | 109803 | 45.4% | 0.6% | 54.1% |

## Decode-time gap explained by step count alone

| pair | d steps | d decode s | gamma x d steps | explained |
|---|---:|---:|---:|---:|
| hom_b0:het_b0 | +967 | +9.527 | +9.135 | 95.9% |
