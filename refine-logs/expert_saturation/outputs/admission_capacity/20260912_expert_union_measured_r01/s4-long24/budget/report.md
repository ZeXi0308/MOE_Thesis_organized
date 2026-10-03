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
| instantaneous (W=1) | 2.3594 | 14.75% | **1.7695** | **174.3%** | 14,500 | 3.540 |
| stable (W=32) | 0.1250 | 0.78% | **0.0938** | **9.2%** | 768 | 0.188 |

Requiring the idle set to persist for 32 steps shrinks it by **18.9x**.

## Cost of harvesting each coherence window

Measured H2D bandwidth 56.44 GB/s; measured step time 9.93 ms.
An idle set that stays idle for only W steps must be re-staged every
W steps, so the amortised cost per step is bytes(W)/bandwidth/W.

| window W | idle GiB | covers deficit | transfer ms | amortised ms/step | vs step time |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.7695 | 174.3% | 33.7 | 33.66 | **3.39x** |
| 2 | 0.9258 | 91.2% | 17.6 | 8.81 | **0.89x** |
| 4 | 0.5156 | 50.8% | 9.8 | 2.45 | **0.25x** |
| 8 | 0.2930 | 28.9% | 5.6 | 0.70 | **0.07x** |
| 16 | 0.1758 | 17.3% | 3.3 | 0.21 | **0.02x** |
| 32 | 0.0938 | 9.2% | 1.8 | 0.06 | **0.01x** |

No coherence window both covers the deficit and costs less than one step time.

## Per layer

| layer | width | idle W=1 | idle W=32 | sat99 | idle bytes W=1 (MiB) | idle bytes stable (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 0.1094 | 0.0000 | 8 | 84.0 | 0.0 |
| 1 | 24 | 0.0938 | 0.0000 | 8 | 72.0 | 0.0 |
| 2 | 24 | 0.1250 | 0.0000 | 16 | 96.0 | 0.0 |
| 3 | 24 | 0.1250 | 0.0000 | 16 | 96.0 | 0.0 |
| 4 | 24 | 0.1719 | 0.0000 | 32 | 132.0 | 0.0 |
| 5 | 24 | 0.1562 | 0.0000 | 32 | 120.0 | 0.0 |
| 6 | 24 | 0.2031 | 0.0469 | None | 156.0 | 36.0 |
| 7 | 24 | 0.1875 | 0.0312 | None | 144.0 | 24.0 |
| 8 | 24 | 0.1875 | 0.0156 | None | 144.0 | 12.0 |
| 9 | 24 | 0.1406 | 0.0000 | 32 | 108.0 | 0.0 |
| 10 | 24 | 0.1094 | 0.0000 | 8 | 84.0 | 0.0 |
| 11 | 24 | 0.1562 | 0.0312 | None | 120.0 | 24.0 |
| 12 | 24 | 0.1406 | 0.0000 | 16 | 108.0 | 0.0 |
| 13 | 24 | 0.1406 | 0.0000 | 8 | 108.0 | 0.0 |
| 14 | 24 | 0.1562 | 0.0000 | 32 | 120.0 | 0.0 |
| 15 | 24 | 0.1562 | 0.0000 | 16 | 120.0 | 0.0 |
