# Which channel does the admission cap act through?

64 sealed episodes, grouped by engine so nothing is pooled.
The alignment ceiling assumes every padded graph slot is reclaimable at
zero cost, so it over-states what the capture mechanism could ever win.

## Static cap sweep per engine

| campaign | engine | regime | n | goodput spread | alignment ceiling (max) | admission cost (max) | rho gp~alignment | rho gp~admission | rho gp~padding |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20260908_capture_ladde | forward | bursty | 5 | 469.2% | 16.82% | 5.5% | +0.300 | +0.300 | +0.400 |
| 20260908_capture_ladde | forward | steady | 5 | 39.5% | 17.78% | 5.3% | -0.300 | +0.600 | -0.300 |
| 20260908_capture_ladde | reverse | bursty | 5 | 460.3% | 16.82% | 4.1% | +0.300 | -0.300 | +0.300 |
| 20260908_capture_ladde | reverse | steady | 5 | 54.9% | 17.93% | 5.2% | -0.300 | +0.900 | -0.300 |
| 20260908_capture_ladde | forward | bursty | 5 | 462.6% | 16.95% | 5.8% | +0.400 | +1.000 | +0.300 |
| 20260908_capture_ladde | forward | steady | 5 | 159.0% | 17.78% | 5.4% | -0.100 | -0.600 | -0.100 |
| 20260908_capture_ladde | reverse | bursty | 5 | 458.1% | 16.97% | 6.0% | +0.300 | +0.300 | +0.400 |
| 20260908_capture_ladde | reverse | steady | 5 | 65.6% | 17.96% | 5.0% | +0.700 | +0.200 | +0.700 |

## Magnitude bound

- median goodput spread across static caps: **308.6%**
- median alignment ceiling: **17.37%** of episode time
- ratio: the lever moves goodput about **18x** more than the entire
  alignment channel could account for, even granting free reclamation.

