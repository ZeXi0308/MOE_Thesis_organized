# 研究方向索引

当前总裁决见[当前研究状态](../current/README.md)。目录存在只表示材料被保留，不表示方向已经 GO。

## 下一研究问题

`PRIMARY_NEXT_CANDIDATE = Route-Conditioned Barrier Amplification Boundary`，状态为 `ORACLE_FIRST / UNVALIDATED / SAME_FAMILY_PROVISIONAL / BLOCKED_PROTOCOL_AMBIGUITY`。它不是可直接实现的机制；本轮 protocol-first preparation 发现 common natural regime、removable barrier/dependency semantics 与 resource capacity 未唯一冻结，双模型 identity-complete trace 和 complete measured surface 也均为 `NO`，因此 evaluator 未实现、formal Gate 未运行。边界见 [RCBA README](rcba/)；原 Top 3、评分和停止门见 [Next-Idea Jury](../../idea-stage/NEXT_IDEA_JURY.md)。

## 保留的历史 formulation / 实现资产

| Idea | 状态 | 文档与代码 |
|---|---|---|
| BCRD | `FORMULATION_ASSET / LOCAL_SIMULATOR_CORRECTED / REQUEST_DAG_OPEN / NOT_CURRENT_EXECUTION_LINE` | [目录](bcrd/) · [研究设计](bcrd/研究设计与三门验证协议.md) · [实验代码](bcrd/experiments/README.md) |
| DEPA-MoE | `DEVELOPMENT_ASSET / NOT_SCIENTIFIC / NOT_CURRENT_EXECUTION_LINE` | [说明与代码](depa_moe/README.md) |

BCRD 与 DEPA 不再是当前执行线。新 Primary 的 Oracle Gate 准备可以复用其 route、full-path breakdown、5090 service surface 和 frozen workload manifest，但不得运行旧 Gate、选择 action 或并行调参。

## 新登记但未授权升格的候选

| Idea | 状态 | 文档与边界 |
|---|---|---|
| RouteGuard-KV | `PROPOSED / KILL_PROBE_ONLY / NOT_CURRENT_MAINLINE` | [严格评审与收紧后 R0–R2 协议](routeguard_kv/README.md)；保留历史协议，不改变 Oracle-first Primary 的执行顺序 |
| RouteShape-SLO | `BLOCKED_RUNTIME_NOT_REPRESENTATIVE / P1_SMOKE_ONLY / NOT_CURRENT_MAINLINE` | [状态、代码与唯一下一实验](route_shape_slo/)；只登记探索，不改变 Oracle-first Primary。 |

## 保留的证据方向

| Idea | 当前结论 | 目录 |
|---|---|---|
| Rank-tail / FP8-first | 仅结构性 evidence；不是系统 GO | [A_rank_tail_fp8](A_rank_tail_fp8/) |
| Receiver-aware / CPR | fixed RankLane 冻结域停止；8×A100 existence 未测；FJRC/PhaseMap 已归档 | [receiver_aware](receiver_aware/) |
| ConfidenceGuard / Verify precision | sealed scientific result 为 NO-GO | [B_verify_precision](B_verify_precision/) |
| Energy-SLO | 单 GPU characterization；serving controller 未验证 | [energy_slo](energy_slo/) |
| Quality debt | NO-GO | [quality_debt](quality_debt/) |
| StableBatch | fresh Oracle opportunity 成立；static/online pre-action selector 双双失败，`STOP_PREACTION_STABLEBATCH` | [stablebatch](stablebatch/) |
| JoinStream | `FROZEN / WEAKEN_UPPER_BOUND_TOO_SMALL / NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`；三阶段证据保留，不得优化当前 formulation | [最终冻结](../current/JOINSTREAM_FINAL_FREEZE_2026-08-10.md) · [BCRD 记录](bcrd/) |

## 归档

- [已停止 ideas](../archive/killed_ideas/README.md)
- [Receiver-aware 历史 formulation](../archive/receiver_aware/README.md)
- [非活动但未形成正式科学结论的 ideas](../archive/inactive_ideas/README.md)

开发夹具、smoke、逻辑 bytes、单卡 LUT/H2D 都不能升级为多卡 EP、NCCL、TPOT 或 P99 结论。
