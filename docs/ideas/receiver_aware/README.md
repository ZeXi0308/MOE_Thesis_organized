# Receiver-aware / EP return-path

本目录现在只保留仍需阅读的 CPR / RankLane 条件分支。此前的 receiver controller、DDRC、CJC、RIC、FJRC 和 PhaseMap 已按 formulation 移入 [`../../archive/receiver_aware/`](../../archive/receiver_aware/)。

## 当前分支

| 分支 | 状态 | 文档、代码与结果 |
|---|---|---|
| CPR / fixed RankLane | 冻结域内 `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20` | [`cpr_ranklane/`](cpr_ranklane/) |
| optimized EP return-path existence | `NOT_TESTED_REQUIRES_8XA100` | [`cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md`](cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md) |

全项目当前裁决见 [`../../current/README.md`](../../current/README.md)。单卡 codec/LUT 不能外推为 NCCL、RDMA、TPOT 或 P99 结果。
