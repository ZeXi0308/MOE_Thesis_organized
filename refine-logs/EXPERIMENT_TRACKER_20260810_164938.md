# CriticalSplit-MoE 实验跟踪器

> 更新时间：2026-08-10 16:49 +0800  
> 对应冻结计划：`refine-logs/EXPERIMENT_PLAN_20260810_164938.md`  
> 总状态：`FROZEN_BEFORE_IMPLEMENTATION / NOT_RUN`

## Run ledger

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| `CS-P0-BASE` | M0 | 历史回归与输入 hash | FrontierCredit original | frozen 8 cells | 13 tests、deterministic artifact | MUST | TODO | 不修改历史 runner/artifacts |
| `CS-P0-CONTRACT` | M1 | subset transition合同 | validated ready-subset launch | unit fixtures | conservation、ready age、service、busy executor、replay | MUST | TODO | 失败即 INVALID |
| `CS-P0-ACTUAL` | M2 | action-space资格 | actual-identity split exact Oracle | frozen 8 cells | eligible cells、whole capture、critical use、miss delta | MUST | TODO | max states=500,000 |
| `CS-P0-SHAM` | M3 | MoE identity必要性 | sham-identity split exact Oracle | same 8 cells | sham applicability、identity gap | MUST | TODO | 只变 revealed sibling map |
| `CS-P0-DECIDE` | M4 | 机械聚合与封存 | parent decision | all completed cells | SUPPORT/WEAKEN/INVALID、COMPLETE-last | MUST | TODO | paper_result=false |
| `CS-P1-ONLINE` | future | online signal | CriticalSplit policy | new frozen cells | unseen residual | CUT | BLOCKED_ON_P0 | P0正向后另行冻结 |

## Frozen gates

- actual eligible cells `>=2`；每个必须实际使用 `CRITICAL` proper subset。
- eligible-cell `median(whole_capture) <0.90`。
- sham-applicable eligible cells `>=2`，`median(identity_gap) >=0.10`。
- deadline miss delta `<=0`。
- 任一 integrity/replay/state-cap失败为 `INVALID`，不得扩大 cap 或改 cells救结果。

## Current authority

- FrontierCredit 历史 verdict 仍是 `FRONTIER_SIGNAL_NOT_SUPPORTED / simulation-only`。
- CriticalSplit 当前只有候选机制与冻结计划，没有实现或结果。
- 本轮不启动 GPU；P0 negative 后直接停止该 action formulation。

