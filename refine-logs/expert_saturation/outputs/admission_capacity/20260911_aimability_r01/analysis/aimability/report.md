# Aimability: information at decision time about effect time

108 sealed episodes. Nothing executed; no policy claim.
`frac resolved` is mutual information in excess of the per-episode
shuffle floor, divided by the effect-time bucket entropy. It upper-bounds
*every* predictor, so a value at zero means no controller can aim the action.

## Aimability decay

| lag (steps) | episodes | frac resolved (bucket only) | frac resolved (rich state) | MI rich (bits) | shuffle floor (bits) | H(target) | persistence | chance | episodes above floor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107 | 0.8537 | 0.8319 | 1.1384 | 0.0447 | 1.1815 | 0.9760 | 0.4815 | 97/107 |
| 2 | 107 | 0.7808 | 0.7633 | 1.0348 | 0.0470 | 1.1811 | 0.9617 | 0.4825 | 97/107 |
| 4 | 107 | 0.6810 | 0.6820 | 0.8992 | 0.0449 | 1.1803 | 0.9286 | 0.4857 | 97/107 |
| 8 | 107 | 0.5464 | 0.5645 | 0.7166 | 0.0457 | 1.1730 | 0.8754 | 0.4920 | 96/107 |
| 16 | 107 | 0.3962 | 0.4602 | 0.5120 | 0.0461 | 1.1625 | 0.8092 | 0.5075 | 94/107 |
| 32 | 107 | 0.1884 | 0.3201 | 0.3445 | 0.0484 | 1.1373 | 0.7162 | 0.5327 | 94/107 |
| 64 | 107 | 0.0687 | 0.2957 | 0.3762 | 0.0421 | 1.0084 | 0.5094 | 0.5728 | 88/107 |
| 128 | 105 | 0.2237 | 0.4064 | 0.4554 | 0.0449 | 1.0776 | 0.3236 | 0.5316 | 104/105 |

## Lag at which an admission cap actually binds

- dynamic episodes with at least one binding cap write: 20
- median binding lag: 0.0 steps (range 0-117)

A cap write that never binds is not an action and is excluded.

