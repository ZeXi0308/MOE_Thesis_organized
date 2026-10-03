# Native MoE step cost surface (observational reconstruction)

cells used 44/44; excluded 0
paired mixed steps 378

## Pure-decode step cost by decode width

| width | n | p50 ms | p10 | p90 | ms/token | decode tok/s | cells | cell p50 min | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16368 | 2.806 | 2.779 | 2.861 | 2.8060 | 356 | 21 | 2.789 | 3.655 |
| 2 | 250 | 4.016 | 3.749 | 4.536 | 2.0080 | 498 | 23 | 3.770 | 4.340 |
| 3 | 222 | 5.306 | 4.905 | 5.601 | 1.7686 | 565 | 20 | 5.212 | 5.522 |
| 4 | 4342 | 5.197 | 4.851 | 5.505 | 1.2994 | 770 | 23 | 5.136 | 5.346 |
| 5 | 285 | 7.032 | 6.653 | 7.388 | 1.4063 | 711 | 18 | 6.716 | 7.172 |
| 6 | 187 | 6.916 | 6.563 | 7.235 | 1.1527 | 868 | 18 | 6.700 | 7.111 |
| 7 | 166 | 7.013 | 6.544 | 7.353 | 1.0019 | 998 | 18 | 6.704 | 7.192 |
| 8 | 5913 | 6.763 | 6.315 | 7.193 | 0.8454 | 1183 | 36 | 6.553 | 7.152 |
| 9 | 156 | 8.421 | 7.977 | 8.836 | 0.9356 | 1069 | 14 | 8.148 | 8.820 |
| 10 | 257 | 8.499 | 8.141 | 8.752 | 0.8499 | 1177 | 14 | 8.231 | 8.606 |
| 11 | 167 | 8.615 | 8.136 | 8.854 | 0.7832 | 1277 | 14 | 8.314 | 8.718 |
| 12 | 1226 | 8.544 | 8.204 | 8.826 | 0.7120 | 1405 | 17 | 8.320 | 8.673 |
| 13 | 120 | 8.667 | 8.392 | 8.907 | 0.6667 | 1500 | 12 | 8.577 | 8.846 |
| 14 | 117 | 8.672 | 8.340 | 8.952 | 0.6194 | 1614 | 12 | 8.506 | 8.927 |
| 15 | 120 | 8.476 | 8.087 | 8.841 | 0.5651 | 1770 | 11 | 8.308 | 8.587 |
| 16 | 2760 | 8.410 | 8.117 | 8.698 | 0.5257 | 1902 | 24 | 8.273 | 8.702 |
| 17 | 100 | 9.375 | 9.142 | 9.705 | 0.5514 | 1813 | 7 | 9.225 | 9.526 |
| 18 | 54 | 9.300 | 9.006 | 9.539 | 0.5167 | 1935 | 6 | 9.242 | 9.434 |
| 19 | 52 | 9.353 | 9.046 | 9.701 | 0.4922 | 2032 | 6 | 9.138 | 9.667 |
| 20 | 57 | 9.328 | 8.973 | 9.523 | 0.4664 | 2144 | 6 | 9.073 | 9.460 |
| 21 | 55 | 9.421 | 9.189 | 9.678 | 0.4486 | 2229 | 6 | 9.299 | 9.548 |
| 22 | 59 | 9.425 | 9.134 | 9.610 | 0.4284 | 2334 | 6 | 9.328 | 9.521 |
| 23 | 75 | 9.508 | 9.239 | 9.753 | 0.4134 | 2419 | 6 | 9.318 | 9.672 |
| 24 | 144 | 9.450 | 9.227 | 9.641 | 0.3938 | 2540 | 12 | 9.339 | 9.537 |
| 25 | 106 | 9.943 | 9.733 | 10.153 | 0.3977 | 2514 | 6 | 9.885 | 10.061 |
| 26 | 31 | 9.982 | 9.915 | 10.119 | 0.3839 | 2605 | 3 | 9.955 | 10.016 |

## Prefill tax by decode width

| bucket | n | median pure ms | median mixed ms | median tax ms | tax/prefill-token ms |
|---|---:|---:|---:|---:|---:|
| width_1_2 | 36 | 4.032 | 19.659 | 15.247 | 0.11912 |
| width_3_6 | 74 | 5.771 | 13.504 | 7.866 | 0.06145 |
| width_7_10 | 81 | 7.098 | 13.536 | 6.414 | 0.03973 |
| width_11_14 | 62 | 8.701 | 13.459 | 4.742 | 0.03704 |
| width_15_18 | 47 | 8.428 | 13.749 | 5.192 | 0.03924 |
| width_19_22 | 29 | 9.402 | 13.762 | 4.408 | 0.03444 |
| width_23_28 | 49 | 9.499 | 13.620 | 4.109 | 0.03210 |

## Prefill tax by chunk size

| bucket | n | median pure ms | median mixed ms | median tax ms | tax/prefill-token ms |
|---|---:|---:|---:|---:|---:|
| prefill_65_200 | 349 | 8.503 | 13.593 | 5.261 | 0.04110 |
| prefill_201_600 | 6 | 6.800 | 15.083 | 8.283 | 0.01653 |
| prefill_601_1100 | 23 | 6.906 | 20.085 | 13.017 | 0.01281 |

## Prefill tax by arrival regime

| bucket | n | median pure ms | median mixed ms | median tax ms | tax/prefill-token ms |
|---|---:|---:|---:|---:|---:|
| steady | 353 | 8.501 | 13.598 | 5.270 | 0.04076 |
| bursty | 25 | 6.903 | 20.063 | 12.956 | 0.01281 |

## Per-episode mixed-step exposure

| cell | regime | cap | policy | pure steps | mixed steps | median pure ms | median mixed ms | mixed share of decode time | request-weighted mean ITL ms |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| forward/cell-000.json | steady | 4 | static | 1018 | 31 | 5.185 | 13.952 | 0.0781 | 5.357 |
| forward/cell-001.json | bursty | 4 | static | 1016 | 0 | 5.206 | - | 0.0000 | 5.209 |
| forward/cell-002.json | steady | 8 | static | 517 | 31 | 6.748 | 13.915 | 0.1193 | 7.120 |
| forward/cell-003.json | bursty | 8 | static | 508 | 0 | 6.764 | - | 0.0000 | 6.761 |
| forward/cell-004.json | steady | 16 | static | 303 | 31 | 8.313 | 14.021 | 0.1637 | 8.773 |
| forward/cell-005.json | bursty | 16 | static | 316 | 6 | 8.210 | 16.733 | 0.0364 | 8.150 |
| forward/cell-006.json | steady | 32 | static | 247 | 31 | 8.694 | 14.368 | 0.1910 | 9.771 |
| forward/cell-007.json | bursty | 32 | static | 308 | 6 | 8.226 | 15.140 | 0.0365 | 8.249 |
| forward/cell-008.json | steady | 4 | static | 4041 | 5 | 2.817 | 14.738 | 0.0074 | 2.921 |
| forward/cell-009.json | steady | 32 | static | 4051 | 1 | 2.822 | 14.829 | 0.0013 | 2.882 |
| reverse/cell-000.json | bursty | 32 | static | 314 | 6 | 8.120 | 14.934 | 0.0359 | 8.049 |
| reverse/cell-001.json | steady | 32 | static | 267 | 31 | 8.597 | 13.530 | 0.1692 | 9.408 |
| reverse/cell-002.json | bursty | 16 | static | 319 | 6 | 8.107 | 16.337 | 0.0361 | 8.041 |
| reverse/cell-003.json | steady | 16 | static | 306 | 31 | 8.198 | 13.371 | 0.1572 | 8.638 |
| reverse/cell-004.json | bursty | 8 | static | 508 | 0 | 6.731 | - | 0.0000 | 6.723 |
| reverse/cell-005.json | steady | 8 | static | 519 | 31 | 6.704 | 13.300 | 0.1156 | 7.004 |
| reverse/cell-006.json | bursty | 4 | static | 1016 | 0 | 5.193 | - | 0.0000 | 5.212 |
| reverse/cell-007.json | steady | 4 | static | 1016 | 31 | 5.165 | 13.586 | 0.0791 | 5.378 |
| reverse/cell-008.json | steady | 32 | static | 4064 | 0 | 2.794 | - | 0.0000 | 2.825 |
| reverse/cell-009.json | steady | 4 | static | 4053 | 1 | 2.789 | 14.189 | 0.0012 | 2.822 |
| forward/cell-000.json | steady | 8 | static | 526 | 31 | 6.691 | 13.766 | 0.1132 | 7.009 |
| forward/cell-001.json | bursty | 8 | static | 508 | 0 | 6.747 | - | 0.0000 | 6.751 |
| forward/cell-002.json | steady | 12 | static | 393 | 31 | 8.335 | 13.428 | 0.1287 | 8.443 |
| forward/cell-003.json | bursty | 12 | static | 378 | 6 | 8.378 | 15.022 | 0.0293 | 8.204 |
| forward/cell-004.json | steady | 16 | static | 326 | 31 | 8.288 | 13.467 | 0.1502 | 8.707 |
| forward/cell-005.json | bursty | 16 | static | 316 | 6 | 8.183 | 16.426 | 0.0362 | 8.121 |
| forward/cell-006.json | steady | 32 | static | 262 | 31 | 8.607 | 13.429 | 0.1711 | 9.432 |
| forward/cell-007.json | bursty | 32 | static | 311 | 6 | 8.118 | 14.974 | 0.0359 | 8.148 |
| forward/cell-008.json | steady | 32 | shadow | 254 | 31 | 8.618 | 13.504 | 0.1779 | 9.469 |
| forward/cell-009.json | bursty | 32 | shadow | 313 | 6 | 8.141 | 15.617 | 0.0366 | 8.170 |
| forward/cell-010.json | steady | 32 | feedback | 413 | 23 | 8.324 | 13.561 | 0.0941 | 8.573 |
| forward/cell-011.json | bursty | 32 | feedback | 322 | 6 | 8.138 | 16.296 | 0.0357 | 8.083 |
| reverse/cell-000.json | bursty | 32 | feedback | 378 | 8 | 7.357 | 14.042 | 0.0401 | 7.944 |
| reverse/cell-001.json | steady | 32 | feedback | 400 | 28 | 8.309 | 13.549 | 0.1192 | 8.427 |
| reverse/cell-002.json | bursty | 32 | shadow | 306 | 6 | 8.272 | 15.455 | 0.0367 | 8.330 |
| reverse/cell-003.json | steady | 32 | shadow | 271 | 31 | 8.722 | 13.669 | 0.1648 | 9.671 |
| reverse/cell-004.json | bursty | 32 | static | 309 | 6 | 8.181 | 15.183 | 0.0362 | 8.224 |
| reverse/cell-005.json | steady | 32 | static | 252 | 31 | 8.725 | 13.650 | 0.1790 | 9.599 |
| reverse/cell-006.json | bursty | 16 | static | 316 | 6 | 8.258 | 16.609 | 0.0364 | 8.168 |
| reverse/cell-007.json | steady | 16 | static | 321 | 31 | 8.317 | 13.481 | 0.1494 | 8.740 |
| reverse/cell-008.json | bursty | 12 | static | 378 | 6 | 8.411 | 15.161 | 0.0295 | 8.245 |
| reverse/cell-009.json | steady | 12 | static | 394 | 31 | 8.352 | 13.576 | 0.1296 | 8.461 |
| reverse/cell-010.json | bursty | 8 | static | 508 | 0 | 6.745 | - | 0.0000 | 6.742 |
| reverse/cell-011.json | steady | 8 | static | 528 | 31 | 6.757 | 13.698 | 0.1209 | 7.163 |
