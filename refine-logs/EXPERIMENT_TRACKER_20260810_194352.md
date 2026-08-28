# StableBatch C8 Action-Transfer Gate Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| C8T-M0 | M0 | freeze cohort | M1 ledger deterministic tie-break | 33 unique cells / 8 docs | cohort hash, 139 raw positives, M1 closure | MUST | DONE | 33 unique cells sealed before C8 outcomes; cohort file SHA-256 `369106423a66236c35efb4f1c1ad92d9c153fb732093abb77ae53b119fdfd3` |
| C8T-M1 | M1 | runner qualification | R/U/M1 + 8 one-C8-seven-M64 | synthetic fixtures | closure, exact fractions, per-doc/LODO | MUST | DONE | both unit-test groups 6/6; frozen lock SHA-256 `13cdda8f2a25a6913236ffedb6b46744534dc319d92e507276bca5e0cbac55e1` |
| C8T-M2 | M2 | decisive GPU Gate | fixed-C8 action transfer | frozen M1-positive cohort | recovered/harmed/net, transfer/gaps, final logits | MUST | DONE | RTX 5090 formal `run01` COMPLETE; 100.52 s; scientific result eligible |
| C8T-M3 | M3 | independent recompute | raw result ledger only | same run | summary mismatch count | MUST | DONE | local rerun equals sealed recompute byte-for-byte; `mismatch_count=0` |
| C8T-M4 | M4 | integrity review | fresh same-family reviewer | artifacts only | P0/P1, claim ceiling | MUST | DONE | PASS; P0=0, P1=0; Gate A / GO supported within retrospective single-stack claim ceiling |

**Frozen source correction**：`33` 是 best-per-cell 的 positive unique cells；全 rank surface 有 139 个 positive ranks。primary 始终按 33 cells 等权。
