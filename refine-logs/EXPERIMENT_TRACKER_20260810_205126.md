# StableBatch Selectability Decomposition Gate Tracker

| Run ID | Milestone | Purpose | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| SEL-S0 | S0 | freeze data/action/selector semantics | calibration old240 + globally untouched eval-16 | overlap, hashes, B=33, feature and seed closure | MUST | DONE | 6/6 CPU tests PASS; exact 19-file frozen lock SHA256 `72e53c770f8c259ae74cde57b23d6f8dc22c0639f658c8e9ec815f6c64821a04`; manifest SHA256 `30e1d366f6e3578660931ec586ccf58d3b3d1886b4c80c8da73898c5e3dfa6f3`; ordered window digest `44321cf960c2903b2db143c9fe7a0530028591da8e3442dd6d919022fde81b67` |
| SEL-S1 | S1 | native-only feature scan and selector seal | fresh 16 docs × 15 cells | selector lock precedes outcomes; 4×33 unique actions | MUST | TODO | no M1/M64 side-call before seal |
| SEL-S2 | S2 | full held-out action surface | fresh 240 cells × 8 ranks | Oracle/static/online/shuffle reward and recovered gap | MUST | TODO | one RTX 5090, same action signature |
| SEL-S3 | S3 | independent aggregation | raw result ledger | mismatch count, LODO, final verdict | MUST | TODO | must not import runner classifier |
| SEL-S4 | S4 | experiment integrity review | sealed artifacts only | P0/P1, evidence ceiling | MUST | TODO | one audit only |
| C8T-M2 | PARKED | prior C8 transfer run | old positive cohort | none | CUT | PARKED_PRE_OUTCOME | frozen lock exists; no C8 outcome opened; superseded by user-selected Gate |

**Authority rule**：只有 `COMPLETE` + verified manifest + independent recompute 可以提供正式结论；pre-action scan/lock 本身不是科学结果。
