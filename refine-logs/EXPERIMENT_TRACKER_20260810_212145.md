# StableBatch Selectability Decomposition Gate Tracker

| Run ID | Milestone | Purpose | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| SEL-S0 | S0 | freeze data/action/selector semantics | calibration old240 + globally untouched eval-16 | overlap, hashes, B=33, feature and seed closure | MUST | DONE | 6/6 CPU tests PASS; exact 19-file frozen lock SHA256 `72e53c770f8c259ae74cde57b23d6f8dc22c0639f658c8e9ec815f6c64821a04`; manifest SHA256 `30e1d366f6e3578660931ec586ccf58d3b3d1886b4c80c8da73898c5e3dfa6f3`; ordered window digest `44321cf960c2903b2db143c9fe7a0530028591da8e3442dd6d919022fde81b67` |
| SEL-S1 | S1 | native-only feature scan and selector seal | fresh 16 docs × 15 cells | selector lock precedes outcomes; 4×33 unique actions | MUST | DONE | selector lock SHA256 `e2af3737df4e7e59edaa97afa10f71cd84063e25f56b1f949f08149a6993c6c6`; `outcome_rows_existed_at_seal=false`; `result_path_existed_at_seal=false` |
| SEL-S2 | S2 | full held-out action surface | fresh 240 cells × 8 ranks | Oracle/static/online/shuffle reward and recovered gap | MUST | DONE | run01 invalidated by foreign GPU contention and excluded; clean run02 COMPLETE: Oracle `57`, shuffle `-4`, static/online `-7`, ROG `-0.04918` |
| SEL-S3 | S3 | independent aggregation | raw result ledger | mismatch count, LODO, final verdict | MUST | DONE | aggregation recompute PASS with `mismatch_fields=[]`; raw-route verifier rederived 240 U + 1920 A arms with 0 mismatches; verdict `STOP_PREACTION_STABLEBATCH` |
| SEL-S4 | S4 | experiment integrity review | sealed artifacts only | P0/P1, evidence ceiling | MUST | DONE | fresh GPT-5.6-Sol ultra audit `WARN / P0=0 / P1=0`; WARN is stale bookkeeping/declarative config only, no verdict impact |
| C8T-M2 | SUPERSEDED | separate C8 transfer branch | old positive cohort | excluded from this Gate | CUT | HISTORICAL_NON_AUTHORITY | no C8 result was used to alter Selectability features, thresholds, B, seed or verdict |

**Authority rule**：run02 的 `COMPLETE` + verified manifest + independent aggregation + raw-route recompute 是正式裁决；最终 verdict 为 `STOP_PREACTION_STABLEBATCH`。pre-action scan/lock 本身不是科学结果。
