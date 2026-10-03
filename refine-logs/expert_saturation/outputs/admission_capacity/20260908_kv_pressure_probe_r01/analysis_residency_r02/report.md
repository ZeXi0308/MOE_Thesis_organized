# MoE expert-residency tax on KV capacity and admission ceiling

Structural accounting over measured memory telemetry. No new execution.
The released-expert column is a **capacity bound, not a speedup**: paging cost
is deliberately not netted out.

## Measured memory split

| cell | params GiB | **experts GiB** | expert share | non-expert GiB | KV region GiB | expert GiB per KV GiB |
|---|---:|---:|---:|---:|---:|---:|
| repeat0-short-cap16 | 12.89 | **12.00** | 0.931 | 0.89 | 14.90 | 0.805 |
| repeat0-long-cap16 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat0-short-cap32 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat0-long-cap32 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat1-short-cap16 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat1-long-cap16 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat1-short-cap32 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |
| repeat1-long-cap32 | 12.89 | **12.00** | 0.931 | 0.89 | 15.00 | 0.800 |

## Residency tax and admission ceiling

top-k = 8/64 (12.5% of experts active per token)

| cell | context | idle expert GiB | idle/KV | concurrency now | if released | x | ceiling now | ceiling if released | bucket now -> if released |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| repeat0-short-cap16 | 4096 | 10.50 | 0.705 | 29.8 | 50.8 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat0-long-cap16 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat0-short-cap32 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat0-long-cap32 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat1-short-cap16 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat1-long-cap16 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat1-short-cap32 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |
| repeat1-long-cap32 | 4096 | 10.50 | 0.700 | 30.0 | 51.0 | 1.70 | 29 | 32 | 32 -> 32 |

## Where the ceiling lands on the measured cost staircase

| cell | step ms at ceiling now | if released | per-request ms now | if released | ratio |
|---|---:|---:|---:|---:|---:|
| repeat0-short-cap16 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat0-long-cap16 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat0-short-cap32 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat0-long-cap32 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat1-short-cap16 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat1-long-cap16 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat1-short-cap32 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |
| repeat1-long-cap32 | 9.964 | 9.964 | 0.3436 | 0.3114 | 0.9062 |

## Measured request-level outcomes in the same cells

| cell | completed | SLO attainment | goodput rps | throughput rps | TTFT p50 ms | TPOT p50 ms | ITL p50 ms | duration s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repeat0-short-cap16 | 32 | 0.500 | 0.7663 | 1.5327 | 5561.6 | 9.27 | 9.08 | 20.88 |
| repeat0-long-cap16 | 32 | 0.500 | 0.5582 | 1.1164 | 6881.5 | 13.68 | 12.98 | 28.66 |
| repeat0-short-cap32 | 32 | 1.000 | 2.4364 | 2.4364 | 18.2 | 11.47 | 11.30 | 13.13 |
| repeat0-long-cap32 | 0 | 0.000 | 0.0000 | 0.0000 | 351.1 | 20.12 | 19.72 | 16.40 |
| repeat1-short-cap16 | 32 | 0.500 | 0.8112 | 1.6225 | 4327.2 | 9.32 | 9.11 | 19.72 |
| repeat1-long-cap16 | 32 | 0.500 | 0.5634 | 1.1268 | 6794.6 | 13.57 | 12.96 | 28.40 |
| repeat1-short-cap32 | 32 | 1.000 | 2.4234 | 2.4234 | 17.5 | 11.53 | 11.38 | 13.20 |
| repeat1-long-cap32 | 0 | 0.000 | 0.0000 | 0.0000 | 373.6 | 20.26 | 19.75 | 16.52 |
