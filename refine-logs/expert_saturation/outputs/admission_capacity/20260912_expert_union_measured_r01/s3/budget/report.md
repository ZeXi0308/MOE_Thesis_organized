# Can reclaimed expert residency supply the KV deficit?

Measured inputs only: per-layer unions from the collector, and the
engine's own KV pool and weight footprint. Byte totals assume every
idle expert is fully reclaimable, so they are optimistic ceilings.

## The deficit

- KV pool: **122,752 tokens** (14.98 GiB), 128 KiB per token
- demand at max_num_seqs=32, context=4096: 131,072 tokens
- **deficit: 8,320 tokens = 1.0153 GiB = 2.03 requests**

## The two idle budgets

| budget | layer-equivalents | mean fraction | GiB | covers deficit | KV tokens bought | requests bought |
|---|---:|---:|---:|---:|---:|---:|
| instantaneous (W=1) | 1.5781 | 9.86% | **1.1836** | **116.6%** | 9,699 | 2.368 |
| stable (W=32) | 0.0938 | 0.59% | **0.0703** | **6.9%** | 576 | 0.141 |

Requiring the idle set to persist for 32 steps shrinks it by **16.8x**.

## Cost of harvesting each coherence window

Measured H2D bandwidth 56.44 GB/s; measured step time 9.93 ms.
An idle set that stays idle for only W steps must be re-staged every
W steps, so the amortised cost per step is bytes(W)/bandwidth/W.

| window W | idle GiB | covers deficit | transfer ms | amortised ms/step | vs step time |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1836 | 116.6% | 22.5 | 22.52 | **2.27x** |
| 2 | 0.6445 | 63.5% | 12.3 | 6.13 | **0.62x** |
| 4 | 0.3984 | 39.2% | 7.6 | 1.90 | **0.19x** |
| 8 | 0.2227 | 21.9% | 4.2 | 0.53 | **0.05x** |
| 16 | 0.1406 | 13.9% | 2.7 | 0.17 | **0.02x** |
| 32 | 0.0703 | 6.9% | 1.3 | 0.04 | **0.00x** |

No coherence window both covers the deficit and costs less than one step time.

## Per layer

| layer | width | idle W=1 | idle W=32 | sat99 | idle bytes W=1 (MiB) | idle bytes stable (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.0312 | 0.0000 | 2 | 24.0 | 0.0 |
| 1 | 32 | 0.0312 | 0.0000 | 2 | 24.0 | 0.0 |
| 2 | 32 | 0.0625 | 0.0000 | 4 | 48.0 | 0.0 |
| 3 | 32 | 0.0625 | 0.0000 | 4 | 48.0 | 0.0 |
| 4 | 32 | 0.1406 | 0.0000 | 32 | 108.0 | 0.0 |
| 5 | 32 | 0.1250 | 0.0000 | 16 | 96.0 | 0.0 |
| 6 | 32 | 0.1875 | 0.0312 | None | 144.0 | 24.0 |
| 7 | 32 | 0.1562 | 0.0312 | None | 120.0 | 24.0 |
| 8 | 32 | 0.1562 | 0.0156 | None | 120.0 | 12.0 |
| 9 | 32 | 0.1094 | 0.0156 | None | 84.0 | 12.0 |
| 10 | 32 | 0.0781 | 0.0000 | 16 | 60.0 | 0.0 |
| 11 | 32 | 0.0938 | 0.0000 | 32 | 72.0 | 0.0 |
| 12 | 32 | 0.0781 | 0.0000 | 8 | 60.0 | 0.0 |
| 13 | 32 | 0.0781 | 0.0000 | 8 | 60.0 | 0.0 |
| 14 | 32 | 0.0938 | 0.0000 | 8 | 72.0 | 0.0 |
| 15 | 32 | 0.0938 | 0.0000 | 8 | 72.0 | 0.0 |
