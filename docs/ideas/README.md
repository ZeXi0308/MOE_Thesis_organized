# 研究方向索引

当前总裁决见[当前研究状态](../current/README.md)。目录存在只表示材料被保留，不表示方向已经 GO。

## 当前候选

| Idea | 状态 | 文档与代码 |
|---|---|---|
| BCRD | `DESIGNED_AND_IMPLEMENTED / NOT_FORMALLY_RUN` | [目录](bcrd/) · [研究设计](bcrd/研究设计与三门验证协议.md) · [实验代码](bcrd/experiments/README.md) |
| DEPA-MoE | `DEVELOPMENT_ONLY_NOT_SCIENTIFIC` | [说明与代码](depa_moe/README.md) |

BCRD 与 DEPA 先复用同一套 route、full-path breakdown、5090 service surface 和 frozen workload manifest；不得并行调参竞争。

## 新登记但未授权升格的候选

| Idea | 状态 | 文档与边界 |
|---|---|---|
| RouteGuard-KV | `PROPOSED / KILL_PROBE_ONLY / NOT_CURRENT_MAINLINE` | [严格评审与收紧后 R0–R2 协议](routeguard_kv/README.md)；只允许廉价存在性判死，不改变共同 Gate 0/1 的执行顺序 |

## 保留的证据方向

| Idea | 当前结论 | 目录 |
|---|---|---|
| Rank-tail / FP8-first | 仅结构性 evidence；不是系统 GO | [A_rank_tail_fp8](A_rank_tail_fp8/) |
| Receiver-aware / CPR | fixed RankLane 冻结域停止；8×A100 existence 未测；FJRC/PhaseMap 已归档 | [receiver_aware](receiver_aware/) |
| ConfidenceGuard / Verify precision | sealed scientific result 为 NO-GO | [B_verify_precision](B_verify_precision/) |
| Energy-SLO | 单 GPU characterization；serving controller 未验证 | [energy_slo](energy_slo/) |
| Quality debt | NO-GO | [quality_debt](quality_debt/) |

## 归档

- [已停止 ideas](../archive/killed_ideas/README.md)
- [Receiver-aware 历史 formulation](../archive/receiver_aware/README.md)
- [非活动但未形成正式科学结论的 ideas](../archive/inactive_ideas/README.md)

开发夹具、smoke、逻辑 bytes、单卡 LUT/H2D 都不能升级为多卡 EP、NCCL、TPOT 或 P99 结论。
