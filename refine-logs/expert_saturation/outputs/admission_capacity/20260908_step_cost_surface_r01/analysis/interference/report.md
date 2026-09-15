# Decode-width buckets and admission interference (reconstruction)

cells used 44; excluded 0
matched mixed/pure pairs 378

## Capture-size verification (engine log vs assumed bucket edges)

- logs scanned: 4
- engine-logged `cudagraph_capture_sizes`: [[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]]
- decode FULL graphs captured: [7]
- assumed edges [1, 2, 4, 8, 16, 24, 32] match logged subset at or below 32: **True**
- decode graph count equals number of assumed edges: **True**

## Pure-decode cost is a staircase over CUDA-graph capture buckets

| capture bucket | widths present | median ms min | max | within-bucket spread % |
|---:|---|---:|---:|---:|
| 1 | [1] | 2.806 | 2.806 | 0.00 |
| 2 | [2] | 4.016 | 4.016 | 0.00 |
| 4 | [3, 4] | 5.197 | 5.306 | 2.08 |
| 8 | [5, 6, 7, 8] | 6.763 | 7.032 | 3.97 |
| 16 | [9, 10, 11, 12, 13, 14, 15, 16] | 8.410 | 8.672 | 3.11 |
| 24 | [17, 18, 19, 20, 21, 22, 23, 24] | 9.300 | 9.508 | 2.23 |
| 32 | [25, 26] | 9.943 | 9.982 | 0.39 |

| capture point transition | delta ms | jump % | ms per added request |
|---|---:|---:|---:|
| 1 -> 16 | 5.605 | 199.74 | 0.3736 |
| 16 -> 2 | -4.394 | -52.25 | 0.3139 |
| 2 -> 24 | 5.434 | 135.31 | 0.2470 |
| 4 -> 8 | 1.566 | 30.13 | 0.3914 |

## Per-request decode cost by width

| width | step ms | ms per request | bucket | padded waste % |
|---:|---:|---:|---:|---:|
| 1 | 2.806 | 2.8060 | 1 | 0.0 |
| 2 | 4.016 | 2.0080 | 2 | 0.0 |
| 3 | 5.306 | 1.7686 | 4 | 25.0 |
| 4 | 5.197 | 1.2994 | 4 | 0.0 |
| 5 | 7.032 | 1.4063 | 8 | 37.5 |
| 6 | 6.916 | 1.1527 | 8 | 25.0 |
| 7 | 7.013 | 1.0019 | 8 | 12.5 |
| 8 | 6.763 | 0.8454 | 8 | 0.0 |
| 9 | 8.421 | 0.9356 | 16 | 43.8 |
| 10 | 8.499 | 0.8499 | 16 | 37.5 |
| 11 | 8.615 | 0.7832 | 16 | 31.2 |
| 12 | 8.544 | 0.7120 | 16 | 25.0 |
| 13 | 8.667 | 0.6667 | 16 | 18.8 |
| 14 | 8.672 | 0.6194 | 16 | 12.5 |
| 15 | 8.476 | 0.5651 | 16 | 6.2 |
| 16 | 8.410 | 0.5257 | 16 | 0.0 |
| 17 | 9.375 | 0.5514 | 24 | 29.2 |
| 18 | 9.300 | 0.5167 | 24 | 25.0 |
| 19 | 9.353 | 0.4922 | 24 | 20.8 |
| 20 | 9.328 | 0.4664 | 24 | 16.7 |
| 21 | 9.421 | 0.4486 | 24 | 12.5 |
| 22 | 9.425 | 0.4284 | 24 | 8.3 |
| 23 | 9.508 | 0.4134 | 24 | 4.2 |
| 24 | 9.450 | 0.3938 | 24 | 0.0 |
| 25 | 9.943 | 0.3977 | 32 | 21.9 |
| 26 | 9.982 | 0.3839 | 32 | 18.8 |

## Where each episode ran inside the staircase

| cell | regime | cap | policy | admission events | max width | median width | mean padding waste | steps exactly at a capture point |
|---|---|---:|---|---:|---:|---:|---:|---:|
| forward/cell-000.json | steady | 4 | static | 32 | 4 | 4 | 0.004 | 0.983 |
| forward/cell-001.json | bursty | 4 | static | 8 | 4 | 4 | 0.000 | 1.000 |
| forward/cell-002.json | steady | 8 | static | 32 | 8 | 8 | 0.018 | 0.930 |
| forward/cell-003.json | bursty | 8 | static | 4 | 8 | 8 | 0.000 | 1.000 |
| forward/cell-004.json | steady | 16 | static | 32 | 16 | 16 | 0.092 | 0.650 |
| forward/cell-005.json | bursty | 16 | static | 7 | 16 | 16 | 0.001 | 0.997 |
| forward/cell-006.json | steady | 32 | static | 32 | 27 | 15 | 0.170 | 0.211 |
| forward/cell-007.json | bursty | 32 | static | 7 | 24 | 16 | 0.002 | 0.994 |
| forward/cell-008.json | steady | 4 | static | 32 | 2 | 1 | 0.000 | 1.000 |
| forward/cell-009.json | steady | 32 | static | 32 | 2 | 1 | 0.000 | 1.000 |
| reverse/cell-000.json | bursty | 32 | static | 7 | 24 | 16 | 0.002 | 0.994 |
| reverse/cell-001.json | steady | 32 | static | 32 | 26 | 14 | 0.155 | 0.258 |
| reverse/cell-002.json | bursty | 16 | static | 7 | 16 | 16 | 0.001 | 0.997 |
| reverse/cell-003.json | steady | 16 | static | 32 | 16 | 16 | 0.093 | 0.641 |
| reverse/cell-004.json | bursty | 8 | static | 4 | 8 | 8 | 0.000 | 1.000 |
| reverse/cell-005.json | steady | 8 | static | 32 | 8 | 8 | 0.019 | 0.927 |
| reverse/cell-006.json | bursty | 4 | static | 8 | 4 | 4 | 0.000 | 1.000 |
| reverse/cell-007.json | steady | 4 | static | 32 | 4 | 4 | 0.004 | 0.984 |
| reverse/cell-008.json | steady | 32 | static | 32 | 1 | 1 | 0.000 | 1.000 |
| reverse/cell-009.json | steady | 4 | static | 32 | 2 | 1 | 0.000 | 1.000 |
| forward/cell-000.json | steady | 8 | static | 32 | 8 | 8 | 0.024 | 0.909 |
| forward/cell-001.json | bursty | 8 | static | 4 | 8 | 8 | 0.000 | 1.000 |
| forward/cell-002.json | steady | 12 | static | 32 | 12 | 12 | 0.194 | 0.265 |
| forward/cell-003.json | bursty | 12 | static | 7 | 12 | 12 | 0.165 | 0.339 |
| forward/cell-004.json | steady | 16 | static | 32 | 16 | 14 | 0.095 | 0.592 |
| forward/cell-005.json | bursty | 16 | static | 7 | 16 | 16 | 0.001 | 0.997 |
| forward/cell-006.json | steady | 32 | static | 32 | 26 | 14 | 0.156 | 0.256 |
| forward/cell-007.json | bursty | 32 | static | 7 | 24 | 16 | 0.002 | 0.994 |
| forward/cell-008.json | steady | 32 | shadow | 32 | 26 | 14 | 0.160 | 0.264 |
| forward/cell-009.json | bursty | 32 | shadow | 7 | 24 | 16 | 0.002 | 0.994 |
| forward/cell-010.json | steady | 32 | feedback | 24 | 17 | 10 | 0.301 | 0.085 |
| forward/cell-011.json | bursty | 32 | feedback | 7 | 16 | 16 | 0.001 | 0.997 |
| reverse/cell-000.json | bursty | 32 | feedback | 9 | 16 | 8 | 0.039 | 0.847 |
| reverse/cell-001.json | steady | 32 | feedback | 29 | 16 | 11 | 0.157 | 0.405 |
| reverse/cell-002.json | bursty | 32 | shadow | 7 | 24 | 16 | 0.002 | 0.993 |
| reverse/cell-003.json | steady | 32 | shadow | 32 | 26 | 13 | 0.154 | 0.273 |
| reverse/cell-004.json | bursty | 32 | static | 7 | 24 | 16 | 0.002 | 0.994 |
| reverse/cell-005.json | steady | 32 | static | 32 | 26 | 14 | 0.162 | 0.254 |
| reverse/cell-006.json | bursty | 16 | static | 7 | 16 | 16 | 0.001 | 0.997 |
| reverse/cell-007.json | steady | 16 | static | 32 | 16 | 14 | 0.096 | 0.601 |
| reverse/cell-008.json | bursty | 12 | static | 7 | 12 | 12 | 0.165 | 0.339 |
| reverse/cell-009.json | steady | 12 | static | 32 | 12 | 12 | 0.194 | 0.264 |
| reverse/cell-010.json | bursty | 8 | static | 4 | 8 | 8 | 0.000 | 1.000 |
| reverse/cell-011.json | steady | 8 | static | 32 | 8 | 8 | 0.024 | 0.903 |

## Admission interference tax

- `fit_tax_vs_prefill_tokens_all`: n=378, intercept=5.6434, slope=0.006972, R2=0.1733
- `fit_tax_vs_prefill_tokens_width_ge7`: n=268, intercept=3.8527, slope=0.008826, R2=0.8498
- `fit_tax_vs_decode_width_at_chunk_128`: n=349, intercept=10.0815, slope=-0.305284, R2=0.4429

| exact prefill chunk | n | median tax ms | tax per token ms |
|---:|---:|---:|---:|
| 128 | 349 | 5.261 | 0.04110 |
| 512 | 5 | 8.339 | 0.01629 |
| 1008 | 8 | 12.122 | 0.01203 |
| 1016 | 15 | 13.088 | 0.01288 |

## Per-episode request-level attribution

| cell | regime | cap | policy | median TPOT ms | median interference share | median TPOT w/o interference ms | joint pass | joint pass w/o interference | TPOT failures attributable to interference | max reconstruction residual ms |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| forward/cell-000.json | steady | 4 | static | 5.406 | 0.0381 | 5.171 | 4 | 4 | 0 | 0.0543 |
| forward/cell-001.json | bursty | 4 | static | 5.239 | 0.0000 | 5.211 | 4 | 4 | 0 | 0.0397 |
| forward/cell-002.json | steady | 8 | static | 7.281 | 0.0523 | 6.815 | 8 | 8 | 0 | 0.0975 |
| forward/cell-003.json | bursty | 8 | static | 6.790 | 0.0000 | 6.740 | 8 | 8 | 0 | 0.0562 |
| forward/cell-004.json | steady | 16 | static | 9.032 | 0.0691 | 8.432 | 4 | 16 | 17 | 0.1102 |
| forward/cell-005.json | bursty | 16 | static | 8.289 | 0.0149 | 8.092 | 32 | 32 | 0 | 0.0830 |
| forward/cell-006.json | steady | 32 | static | 10.052 | 0.0566 | 9.324 | 4 | 10 | 6 | 0.1296 |
| forward/cell-007.json | bursty | 32 | static | 8.344 | 0.0189 | 8.112 | 32 | 32 | 0 | 0.0889 |
| forward/cell-008.json | steady | 4 | static | 2.897 | 0.0000 | 2.879 | 32 | 32 | 0 | 0.0282 |
| forward/cell-009.json | steady | 32 | static | 2.885 | 0.0000 | 2.867 | 32 | 32 | 0 | 0.0223 |
| reverse/cell-000.json | bursty | 32 | static | 8.146 | 0.0136 | 7.976 | 32 | 32 | 0 | 0.0795 |
| reverse/cell-001.json | steady | 32 | static | 9.702 | 0.0526 | 9.108 | 7 | 14 | 7 | 0.1150 |
| reverse/cell-002.json | bursty | 16 | static | 8.180 | 0.0144 | 7.998 | 32 | 32 | 0 | 0.0783 |
| reverse/cell-003.json | steady | 16 | static | 8.867 | 0.0650 | 8.325 | 6 | 16 | 12 | 0.0970 |
| reverse/cell-004.json | bursty | 8 | static | 6.742 | 0.0000 | 6.693 | 8 | 8 | 0 | 0.0546 |
| reverse/cell-005.json | steady | 8 | static | 7.132 | 0.0509 | 6.661 | 8 | 8 | 0 | 0.0719 |
| reverse/cell-006.json | bursty | 4 | static | 5.254 | 0.0000 | 5.222 | 4 | 4 | 0 | 0.0360 |
| reverse/cell-007.json | steady | 4 | static | 5.435 | 0.0376 | 5.203 | 4 | 4 | 0 | 0.0537 |
| reverse/cell-008.json | steady | 32 | static | 2.833 | 0.0000 | 2.818 | 32 | 32 | 0 | 0.0169 |
| reverse/cell-009.json | steady | 4 | static | 2.823 | 0.0000 | 2.809 | 32 | 32 | 0 | 0.0208 |
| forward/cell-000.json | steady | 8 | static | 7.121 | 0.0540 | 6.648 | 8 | 8 | 0 | 0.0740 |
| forward/cell-001.json | bursty | 8 | static | 6.759 | 0.0000 | 6.710 | 8 | 8 | 0 | 0.0555 |
| forward/cell-002.json | steady | 12 | static | 8.911 | 0.0473 | 8.469 | 7 | 12 | 5 | 0.0918 |
| forward/cell-003.json | bursty | 12 | static | 8.276 | 0.0082 | 8.141 | 16 | 16 | 0 | 0.0669 |
| forward/cell-004.json | steady | 16 | static | 8.953 | 0.0743 | 8.389 | 6 | 17 | 15 | 0.1035 |
| forward/cell-005.json | bursty | 16 | static | 8.275 | 0.0145 | 8.080 | 32 | 32 | 0 | 0.0874 |
| forward/cell-006.json | steady | 32 | static | 9.724 | 0.0511 | 9.117 | 7 | 14 | 7 | 0.1129 |
| forward/cell-007.json | bursty | 32 | static | 8.226 | 0.0137 | 8.051 | 32 | 32 | 0 | 0.0823 |
| forward/cell-008.json | steady | 32 | shadow | 9.732 | 0.0520 | 9.142 | 4 | 14 | 10 | 0.1290 |
| forward/cell-009.json | bursty | 32 | shadow | 8.225 | 0.0148 | 8.074 | 32 | 32 | 0 | 0.0988 |
| forward/cell-010.json | steady | 32 | feedback | 8.900 | 0.0248 | 8.510 | 7 | 17 | 10 | 0.1107 |
| forward/cell-011.json | bursty | 32 | feedback | 8.192 | 0.0145 | 7.994 | 32 | 32 | 0 | 0.0946 |
| reverse/cell-000.json | bursty | 32 | feedback | 8.262 | 0.0148 | 8.054 | 28 | 28 | 0 | 0.1052 |
| reverse/cell-001.json | steady | 32 | feedback | 8.876 | 0.0500 | 8.460 | 7 | 16 | 12 | 0.1188 |
| reverse/cell-002.json | bursty | 32 | shadow | 8.441 | 0.0182 | 8.187 | 32 | 32 | 0 | 0.1051 |
| reverse/cell-003.json | steady | 32 | shadow | 10.021 | 0.0504 | 9.387 | 5 | 10 | 5 | 0.1534 |
| reverse/cell-004.json | bursty | 32 | static | 8.288 | 0.0139 | 8.106 | 32 | 32 | 0 | 0.0877 |
| reverse/cell-005.json | steady | 32 | static | 9.889 | 0.0529 | 9.241 | 4 | 12 | 8 | 0.1220 |
| reverse/cell-006.json | bursty | 16 | static | 8.318 | 0.0146 | 8.120 | 32 | 32 | 0 | 0.0875 |
| reverse/cell-007.json | steady | 16 | static | 9.015 | 0.0646 | 8.436 | 6 | 16 | 16 | 0.1107 |
| reverse/cell-008.json | bursty | 12 | static | 8.383 | 0.0083 | 8.238 | 16 | 16 | 0 | 0.0794 |
| reverse/cell-009.json | steady | 12 | static | 8.968 | 0.0479 | 8.513 | 7 | 12 | 14 | 0.0984 |
| reverse/cell-010.json | bursty | 8 | static | 6.772 | 0.0000 | 6.721 | 8 | 8 | 0 | 0.0585 |
| reverse/cell-011.json | steady | 8 | static | 7.209 | 0.0544 | 6.723 | 8 | 8 | 0 | 0.0751 |

## Bounded coalescing projection (arithmetic only, not a policy result)

| cell | regime | cap | mixed steps | modelled interference ms | group2 | group4 | group8 |
|---|---|---:|---:|---:|---:|---:|---:|
| forward/cell-000.json | steady | 4 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-002.json | steady | 8 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-004.json | steady | 16 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-005.json | bursty | 16 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-006.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-007.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-008.json | steady | 4 | 5 | 24.9 | 17.2 | 13.4 | 9.5 |
| reverse/cell-000.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-001.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-002.json | bursty | 16 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-003.json | steady | 16 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-005.json | steady | 8 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-007.json | steady | 4 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-000.json | steady | 8 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-002.json | steady | 12 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-003.json | bursty | 12 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-004.json | steady | 16 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-005.json | bursty | 16 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-006.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-007.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-008.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| forward/cell-009.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| forward/cell-010.json | steady | 32 | 23 | 123.6 | 81.3 | 58.1 | 46.6 |
| forward/cell-011.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-000.json | bursty | 32 | 8 | 57.9 | 42.5 | 34.8 | 31.0 |
| reverse/cell-001.json | steady | 32 | 28 | 142.9 | 89.0 | 62.0 | 50.4 |
| reverse/cell-002.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-003.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-004.json | bursty | 32 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-005.json | steady | 32 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-006.json | bursty | 16 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-007.json | steady | 16 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-008.json | bursty | 12 | 6 | 50.2 | 38.7 | 34.8 | 31.0 |
| reverse/cell-009.json | steady | 12 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
| reverse/cell-011.json | steady | 8 | 31 | 154.5 | 96.7 | 65.8 | 50.4 |
