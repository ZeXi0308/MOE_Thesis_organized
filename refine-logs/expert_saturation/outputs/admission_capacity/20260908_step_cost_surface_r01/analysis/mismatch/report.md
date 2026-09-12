# Target cap versus realised decode width

`share_target_aligned_but_width_not` is the load-bearing column: the fraction of
steps where the admission target *was* a capture point while the executed width
was not. If placing rungs on capture points worked, it would be near zero.

## aligned

| arm | episodes | median target | median realised width | median gap | width on capture point | target on capture point | **target aligned but width not** | distinct widths | wasted graph slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty_feedback | 2 | 12 | 16 | 4 | 0.994 | 1.000 | **0.006** | 4 | 0.003 |
| bursty_shadow | 2 | 32 | 16 | 16 | 0.994 | 1.000 | **0.006** | 6 | 0.004 |
| bursty_static | 8 | 20 | 16 | 4 | 0.996 | 1.000 | **0.004** | 5 | 0.002 |
| steady_feedback | 2 | 8 | 9 | 1 | 0.155 | 1.000 | **0.845** | 16 | 0.256 |
| steady_shadow | 2 | 32 | 14 | 18 | 0.247 | 1.000 | **0.753** | 26 | 0.170 |
| steady_static | 8 | 20 | 14 | 5 | 0.537 | 1.000 | **0.463** | 20 | 0.108 |

## sealed

| arm | episodes | median target | median realised width | median gap | width on capture point | target on capture point | **target aligned but width not** | distinct widths | wasted graph slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty_feedback | 2 | 14 | 12 | 0 | 0.922 | 0.715 | **0.093** | 5 | 0.029 |
| bursty_shadow | 2 | 32 | 16 | 16 | 0.994 | 1.000 | **0.006** | 6 | 0.004 |
| bursty_static | 8 | 14 | 14 | 0 | 0.832 | 0.750 | **0.003** | 4 | 0.060 |
| steady_feedback | 2 | 12 | 10 | 0 | 0.245 | 0.554 | **0.750** | 16 | 0.257 |
| steady_shadow | 2 | 32 | 14 | 18 | 0.268 | 1.000 | **0.732** | 26 | 0.167 |
| steady_static | 8 | 14 | 13 | 1 | 0.505 | 0.750 | **0.414** | 14 | 0.135 |

## aligned: feedback episodes in detail

- `forward/cell-010.json` steady: applied targets [24, 16, 8, 16, 8], median width 9, waste 0.2066, widths visited 16
- `forward/cell-011.json` bursty: applied targets [24, 16, 8, 16], median width 16, waste 0.0027, widths visited 5
- `reverse/cell-000.json` bursty: applied targets [24, 16, 8, 16, 8], median width 16, waste 0.0027, widths visited 4
- `reverse/cell-001.json` steady: applied targets [24, 16, 8, 16, 8], median width 9, waste 0.2516, widths visited 17

## sealed: feedback episodes in detail

- `forward/cell-010.json` steady: applied targets [16, 12, 8, 12, 8, 12, 8, 12], median width 10, waste 0.3014, widths visited 17
- `forward/cell-011.json` bursty: applied targets [16, 12], median width 16, waste 0.0014, widths visited 4
- `reverse/cell-000.json` bursty: applied targets [16, 12, 8], median width 8, waste 0.0387, widths visited 6
- `reverse/cell-001.json` steady: applied targets [16, 12, 8, 12], median width 11, waste 0.1570, widths visited 16

