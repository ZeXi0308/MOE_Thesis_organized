# StableBatch Selectability Decomposition Gate Tracker

| Run ID | Milestone | Purpose | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| SEL-S0 | S0 | freeze data/action/selector semantics | calibration old240 + fresh manifest first16 | overlap, hashes, B=33, feature and seed closure | MUST | IN_PROGRESS | fresh manifest SHA256 `2608ef5d93a9da36b816eddfb8e3bf495631f0f19934a703bc813757443695d7`; StableBatch text-hash overlap 0 |
| SEL-S1 | S1 | native-only feature scan and selector seal | fresh 16 docs × 15 cells | selector lock precedes outcomes; 4×33 unique actions | MUST | TODO | no M1/M64 side-call before seal |
| SEL-S2 | S2 | full held-out action surface | fresh 240 cells × 8 ranks | Oracle/static/online/shuffle reward and recovered gap | MUST | TODO | one RTX 5090, same action signature |
| SEL-S3 | S3 | independent aggregation | raw result ledger | mismatch count, LODO, final verdict | MUST | TODO | must not import runner classifier |
| SEL-S4 | S4 | experiment integrity review | sealed artifacts only | P0/P1, evidence ceiling | MUST | TODO | one audit only |
| C8T-M2 | PARKED | prior C8 transfer run | old positive cohort | none | CUT | PARKED_PRE_OUTCOME | frozen lock exists; no C8 outcome opened; superseded by user-selected Gate |

**Authority rule**：只有 `COMPLETE` + verified manifest + independent recompute 可以提供正式结论；pre-action scan/lock 本身不是科学结果。
