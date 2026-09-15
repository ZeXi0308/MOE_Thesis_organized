# Closed-form admission-feasibility envelope

Primitives fitted out-of-sample: predictions for each campaign use
only the other campaign's episodes. 64 episodes validated.

## Measured primitives (pooled, for the envelope only)

- prefill tax: `5.2192 + 0.007780 x prefill_tokens` ms (1090 matched pairs)
- tax at prompt=128: **6.215 ms per admission**
- model pole: lambda = 1/tax = **160.9 req/s**

| capture bucket | c(w) ms | lambda_max from TPOT | lambda_max from throughput | feasible lambda | binding constraint |
|---:|---:|---:|---:|---:|---|
| 1 | 2.843 | 110.07 | 2.70 | **2.70** | throughput |
| 2 | 4.070 | 88.13 | 3.75 | **3.75** | throughput |
| 4 | 5.247 | 67.10 | 5.74 | **5.74** | throughput |
| 8 | 6.759 | 40.07 | 8.74 | **8.74** | throughput |
| 16 | 8.432 | 10.15 | 13.57 | **10.15** | TPOT |
| 24 | 9.319 | 0 (c>=SLO) | n/a | **0.00** | TPOT_at_zero_load |
| 32 | 9.933 | 0 (c>=SLO) | n/a | **0.00** | TPOT_at_zero_load |

## Critical operating point

- widest feasible bucket: **16** at c = 8.432 ms
- maximum joint-SLO-feasible arrival rate: **10.15 req/s** (inter-arrival **98.5 ms**)
- binding constraint there: TPOT

## Out-of-sample validation of the tau formula

- median relative error: **3.25%**
- p90 relative error: 6.09%
- max relative error: 16.38%
- TPOT pass/fail verdict agreement: **60/64**

| regime | cap | median width | bucket | realised lambda | predicted TPOT | measured TPOT | rel err | verdict match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bursty | 8 | 8 | 8 | 1.11 | 6.855 | 6.807 | 0.71% | yes |
| bursty | 8 | 8 | 8 | 1.13 | 6.856 | 6.785 | 1.05% | yes |
| bursty | 8 | 8 | 8 | 1.12 | 6.865 | 6.817 | 0.70% | yes |
| bursty | 8 | 8 | 8 | 1.13 | 6.866 | 6.790 | 1.12% | yes |
| bursty | 12 | 12 | 16 | 2.19 | 8.618 | 8.347 | 3.25% | yes |
| bursty | 12 | 12 | 16 | 2.20 | 8.618 | 8.277 | 4.13% | yes |
| bursty | 12 | 12 | 16 | 2.23 | 8.619 | 8.213 | 4.95% | yes |
| bursty | 12 | 12 | 16 | 2.20 | 8.617 | 8.315 | 3.63% | yes |
| bursty | 16 | 16 | 16 | 2.65 | 8.657 | 8.375 | 3.37% | yes |
| bursty | 16 | 16 | 16 | 2.69 | 8.661 | 8.201 | 5.60% | yes |
| bursty | 16 | 16 | 16 | 2.68 | 8.659 | 8.206 | 5.52% | yes |
| bursty | 16 | 16 | 16 | 2.68 | 8.658 | 8.222 | 5.31% | yes |
| bursty | 24 | 16 | 16 | 2.75 | 8.666 | 8.179 | 5.95% | yes |
| bursty | 24 | 16 | 16 | 2.76 | 8.667 | 8.199 | 5.71% | yes |
| bursty | 24 | 16 | 16 | 2.73 | 8.663 | 8.398 | 3.15% | yes |
| bursty | 24 | 16 | 16 | 2.74 | 8.664 | 8.197 | 5.69% | yes |
| bursty | 32 | 16 | 16 | 2.77 | 8.667 | 8.280 | 4.67% | yes |
| bursty | 32 | 16 | 16 | 2.74 | 8.665 | 8.392 | 3.24% | yes |
| bursty | 32 | 12 | 16 | 2.60 | 8.640 | 8.573 | 0.78% | yes |
| bursty | 32 | 16 | 16 | 2.64 | 8.656 | 8.276 | 4.59% | yes |
| bursty | 32 | 16 | 16 | 2.76 | 8.666 | 8.161 | 6.19% | yes |
| bursty | 32 | 12 | 16 | 2.28 | 8.625 | 8.553 | 0.85% | yes |
| bursty | 32 | 16 | 16 | 2.74 | 8.665 | 8.333 | 3.99% | yes |
| bursty | 32 | 16 | 16 | 2.75 | 8.666 | 8.195 | 5.74% | yes |
