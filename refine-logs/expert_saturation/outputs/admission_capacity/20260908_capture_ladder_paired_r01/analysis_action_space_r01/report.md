# Step-level width-nudge action space: does it exist?

cells 100; max nudge +/-2 requests; prompt 128 tokens

Withholding a decoding request is illegal under the frozen non-preemption
invariant, so only *filling* upward to a capture point is a candidate action.

| regime | episodes | pure steps | already aligned | **legal fill** | blocked: no queue | illegal withhold candidates | median net share | max net share | episodes net>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 48 | 18446 | 0.9194 | **0.0000** | 0.0000 | 0.0030 | 0.000000 | 0.000000 | 0 |
| steady | 52 | 34262 | 0.7584 | **0.0039** | 0.0771 | 0.1154 | 0.000000 | 0.000000 | 0 |

## Mechanical verdict

- bursty: ACTION_SPACE_ABSENT (legal fill steps 0.0000 < 0.05)
- steady: ACTION_SPACE_ABSENT (legal fill steps 0.0039 < 0.05)

## Measured pure-decode cost used for the saving term

| width | median ms |
|---:|---:|
| 1 | 2.806 |
| 2 | 4.027 |
| 3 | 5.305 |
| 4 | 5.200 |
| 5 | 6.986 |
| 6 | 6.861 |
| 7 | 6.873 |
| 8 | 6.751 |
| 9 | 8.404 |
| 10 | 8.459 |
| 11 | 8.579 |
| 12 | 8.513 |
| 13 | 8.631 |
| 14 | 8.628 |
| 15 | 8.473 |
| 16 | 8.388 |
| 17 | 9.308 |
| 18 | 9.267 |
| 19 | 9.307 |
| 20 | 9.252 |
| 21 | 9.416 |
| 22 | 9.368 |
| 23 | 9.427 |
| 24 | 9.374 |
| 25 | 9.954 |
| 26 | 9.962 |

## Per-episode detail

| cell | regime | cap | policy | ladder | pure steps | aligned | legal fill | no queue | gaps | gross ms | fill cost ms | net ms |
|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| forward/cell-000.json | steady | 8 | static | static | 525 | 478 | 0 | 34 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-001.json | bursty | 8 | static | static | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-002.json | steady | 12 | static | static | 391 | 106 | 0 | 28 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-003.json | bursty | 12 | static | static | 378 | 128 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-004.json | steady | 16 | static | static | 306 | 199 | 0 | 44 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-005.json | bursty | 16 | static | static | 316 | 315 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-006.json | steady | 24 | static | static | 268 | 101 | 0 | 67 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-007.json | bursty | 24 | static | static | 313 | 311 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-008.json | steady | 32 | static | static | 260 | 71 | 0 | 76 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-009.json | bursty | 32 | static | static | 308 | 306 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-010.json | steady | 32 | shadow | legacy | 252 | 65 | 0 | 65 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-011.json | bursty | 32 | shadow | legacy | 305 | 303 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-012.json | steady | 32 | feedback | legacy | 414 | 34 | 12 | 31 | {1: 5, 2: 7} | 12.1 | 67.7 | -55.6 |
| forward/cell-013.json | bursty | 32 | feedback | legacy | 376 | 189 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-014.json | steady | 32 | feedback | aligned | 312 | 71 | 14 | 126 | {1: 6, 2: 8} | 14.0 | 78.8 | -64.8 |
| forward/cell-015.json | bursty | 32 | feedback | aligned | 321 | 319 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-000.json | bursty | 32 | feedback | aligned | 315 | 313 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-001.json | steady | 32 | feedback | aligned | 320 | 199 | 17 | 21 | {1: 7, 2: 10} | 17.1 | 96.0 | -78.9 |
| reverse/cell-002.json | bursty | 32 | feedback | legacy | 380 | 191 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-003.json | steady | 32 | feedback | legacy | 413 | 34 | 11 | 28 | {1: 5, 2: 6} | 10.8 | 61.6 | -50.8 |
| reverse/cell-004.json | bursty | 32 | shadow | legacy | 308 | 306 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-005.json | steady | 32 | shadow | legacy | 256 | 70 | 0 | 67 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-006.json | bursty | 32 | static | static | 312 | 310 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-007.json | steady | 32 | static | static | 258 | 70 | 0 | 74 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-008.json | bursty | 24 | static | static | 313 | 311 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-009.json | steady | 24 | static | static | 277 | 91 | 0 | 81 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-010.json | bursty | 16 | static | static | 317 | 316 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-011.json | steady | 16 | static | static | 307 | 194 | 0 | 47 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-012.json | bursty | 12 | static | static | 378 | 128 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-013.json | steady | 12 | static | static | 392 | 103 | 0 | 29 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-014.json | bursty | 8 | static | static | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-015.json | steady | 8 | static | static | 519 | 479 | 0 | 28 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-000.json | steady | 8 | static | None | 522 | 479 | 0 | 31 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-001.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-002.json | steady | 16 | static | None | 305 | 197 | 0 | 45 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-003.json | bursty | 16 | static | None | 317 | 316 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-004.json | steady | 24 | static | None | 262 | 98 | 0 | 64 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-005.json | bursty | 24 | static | None | 307 | 305 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-006.json | steady | 32 | static | None | 250 | 58 | 0 | 65 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-007.json | bursty | 32 | static | None | 305 | 303 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-008.json | steady | 32 | shadow | None | 249 | 56 | 0 | 64 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-009.json | bursty | 32 | shadow | None | 307 | 305 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-010.json | steady | 32 | feedback | None | 415 | 71 | 26 | 151 | {1: 13, 2: 13} | 24.7 | 144.2 | -119.6 |
| forward/cell-011.json | bursty | 32 | feedback | None | 319 | 317 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-000.json | bursty | 32 | feedback | None | 324 | 322 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-001.json | steady | 32 | feedback | None | 431 | 60 | 29 | 152 | {1: 16, 2: 13} | 26.5 | 159.2 | -132.7 |
| reverse/cell-002.json | bursty | 32 | shadow | None | 311 | 309 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-003.json | steady | 32 | shadow | None | 256 | 69 | 0 | 70 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-004.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-005.json | steady | 8 | static | None | 538 | 479 | 0 | 47 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-006.json | bursty | 16 | static | None | 321 | 320 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-007.json | steady | 16 | static | None | 307 | 195 | 0 | 48 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-008.json | bursty | 24 | static | None | 308 | 306 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-009.json | steady | 24 | static | None | 263 | 94 | 0 | 69 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-010.json | bursty | 32 | static | None | 307 | 305 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-011.json | steady | 32 | static | None | 252 | 61 | 0 | 66 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-000.json | steady | 4 | static | None | 1018 | 1001 | 0 | 17 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-001.json | bursty | 4 | static | None | 1016 | 1016 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-002.json | steady | 8 | static | None | 517 | 481 | 0 | 26 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-003.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-004.json | steady | 16 | static | None | 303 | 197 | 0 | 43 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-005.json | bursty | 16 | static | None | 316 | 315 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-006.json | steady | 32 | static | None | 247 | 52 | 0 | 61 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-007.json | bursty | 32 | static | None | 308 | 306 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-008.json | steady | 4 | static | None | 4041 | 4041 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-009.json | steady | 32 | static | None | 4051 | 4051 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-000.json | bursty | 32 | static | None | 314 | 312 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-001.json | steady | 32 | static | None | 267 | 69 | 0 | 78 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-002.json | bursty | 16 | static | None | 319 | 318 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-003.json | steady | 16 | static | None | 306 | 196 | 0 | 46 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-004.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-005.json | steady | 8 | static | None | 519 | 481 | 0 | 27 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-006.json | bursty | 4 | static | None | 1016 | 1016 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-007.json | steady | 4 | static | None | 1016 | 1000 | 0 | 16 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-008.json | steady | 32 | static | None | 4064 | 4064 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-009.json | steady | 4 | static | None | 4053 | 4053 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-000.json | steady | 8 | static | None | 526 | 478 | 0 | 35 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-001.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-002.json | steady | 12 | static | None | 393 | 104 | 0 | 30 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-003.json | bursty | 12 | static | None | 378 | 128 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-004.json | steady | 16 | static | None | 326 | 193 | 0 | 67 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-005.json | bursty | 16 | static | None | 316 | 315 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-006.json | steady | 32 | static | None | 262 | 67 | 0 | 78 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-007.json | bursty | 32 | static | None | 311 | 309 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-008.json | steady | 32 | shadow | None | 254 | 67 | 0 | 69 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-009.json | bursty | 32 | shadow | None | 313 | 311 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| forward/cell-010.json | steady | 32 | feedback | None | 413 | 35 | 11 | 28 | {1: 5, 2: 6} | 10.8 | 61.6 | -50.8 |
| forward/cell-011.json | bursty | 32 | feedback | None | 322 | 321 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-000.json | bursty | 32 | feedback | None | 378 | 320 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-001.json | steady | 32 | feedback | None | 400 | 162 | 12 | 34 | {1: 6, 2: 6} | 11.4 | 66.6 | -55.2 |
| reverse/cell-002.json | bursty | 32 | shadow | None | 306 | 304 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-003.json | steady | 32 | shadow | None | 271 | 74 | 0 | 73 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-004.json | bursty | 32 | static | None | 309 | 307 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-005.json | steady | 32 | static | None | 252 | 64 | 0 | 64 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-006.json | bursty | 16 | static | None | 316 | 315 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-007.json | steady | 16 | static | None | 321 | 193 | 0 | 62 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-008.json | bursty | 12 | static | None | 378 | 128 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-009.json | steady | 12 | static | None | 394 | 104 | 0 | 31 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-010.json | bursty | 8 | static | None | 508 | 508 | 0 | 0 | {} | 0.0 | 0.0 | 0.0 |
| reverse/cell-011.json | steady | 8 | static | None | 528 | 477 | 0 | 38 | {} | 0.0 | 0.0 | 0.0 |
