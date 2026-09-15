# BCRD：Batch-Conscious Replica Dispatch

状态：`LOCAL_SIMULATOR_CORRECTED / GATE0_A_PARTIAL_IMPLEMENTED / GATE0_B_INPUTS_FROZEN / FORMAL_GATE0_OPEN / REQUEST_DAG_OPEN / NOT_FORMALLY_RUN`。

BCRD 研究 fixed replica set 内 contribution-to-replica assignment 与 bounded seal time，目标是判断 least-load 是否因打碎同一 expert 的 rows 而浪费足够多的批效率和 SLO 容量。

- [研究设计与三门验证协议](研究设计与三门验证协议.md)
- [实验代码、测试与运行边界](experiments/README.md)
- [当前总裁决](../../current/README.md)
- [Gate-0 A 审计账本](../../current/gate0_audit_2026-08-02.md)

2026-08-10 的 backup `CriticalSplit-MoE` 只在 FrontierCredit 冻结 8-cell full-DAG simulator 上资格化 proper-subset action space，裁决为 `WEAKEN_ACTION_SPACE`：expanded split Oracle 在 8/8 cells 与 whole-ready Oracle flow 相等，eligible cells 为 0。该结果不升级 BCRD formal Gate，也不触发 online/GPU 实现；权威记录见 [CriticalSplit tracker](../../../refine-logs/EXPERIMENT_TRACKER_20260810_173700.md)。

同日的 `JoinStream` 已完成三阶段证据链：CPU exact Oracle 只给出 `SUPPORT_ACTION_SPACE / CPU_EXPLORATORY_SIGNAL`；首个单 GPU synthetic pilot 裁决为 `WEAKEN_TAX_DOMINATES`；最终 realistic MoE-tail pilot 裁决为 `WEAKEN_UPPER_BOUND_TOO_SMALL / GATING_INSUFFICIENT / WEAKENS`，安全收益为 `0/4` cells。当前 formulation 已标记 `NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`，只允许保留 action legality、single-GPU schedulability、natural tail headroom 与 producer safety 证据，不得继续优化 gate/priority/polling/stream/notification，也不得外推 serving 或普遍不可行。最终封存见 [JOINSTREAM_FINAL_FREEZE](../../current/JOINSTREAM_FINAL_FREEZE_2026-08-10.md)。

现有 smoke 只检查会计与决策分支，不能作为正式科学结果。当前 formal Gate 2 因尚无跨 layer/step counterfactual request-DAG 而强制 `INVALID`。
