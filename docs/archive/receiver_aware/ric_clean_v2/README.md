# RIC clean v2 research program

本目录集中保存 RIC clean v2 及其后续 FJRC、PhaseMap、CRQM、RR-credit/多 rank 探索。它们共享同一套 route、LUT、oracle 与回放代码，因此作为一个研究程序归档，避免拆散依赖。

| 子线 | 当前结论 | 入口 |
|---|---|---|
| FJRC | `NO_GO_FJRC_JOIN_PHASE_INFORMATION` | [`FJRC_Corrected_Level1_Result_2026-07-23.md`](FJRC_Corrected_Level1_Result_2026-07-23.md) |
| PhaseMap | `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`；holdout 未打开 | [`PhaseMap_Phase5_Result_2026-07-23.md`](PhaseMap_Phase5_Result_2026-07-23.md) |
| Multi-rank RR / incast | route fan-in 与 smoke，非物理 EP 结论 | [`MultiRankTimedTrace_ExecutionGate_2026-07-23.md`](MultiRankTimedTrace_ExecutionGate_2026-07-23.md) |

- 配置：[`configs/`](configs/)
- 代码与测试：[`experiments/ric_clean_v2/`](experiments/ric_clean_v2/)
- 结果：[`outputs/`](outputs/)

归档移动不会重签历史 approval 或 provenance；输出中的旧路径作为 as-run 记录保留。
