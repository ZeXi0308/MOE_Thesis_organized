# Receiver-aware / EP return-path

本目录现在只保留仍需阅读的 CPR / RankLane 条件分支。此前的 receiver controller、DDRC、CJC、RIC、FJRC 和 PhaseMap 已按 formulation 移入 [`../../archive/receiver_aware/`](../../archive/receiver_aware/)。

## 当前分支

| 分支 | 状态 | 文档、代码与结果 |
|---|---|---|
| CPR / fixed RankLane | 冻结域内 `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20` | [`cpr_ranklane/`](cpr_ranklane/) |
| optimized EP return-path existence | `NOT_TESTED_REQUIRES_8XA100` | [`cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md`](cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md) |
| 5090 multi-MoE inference-time | `SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE` / serving+OLMoE `BLOCKED_REMOTE_INSTANCE_CLOSED` | [`inference_time_5090/`](inference_time_5090/) |

全项目当前裁决见 [`../../current/README.md`](../../current/README.md)。单卡 codec/LUT 不能外推为 NCCL、RDMA、TPOT 或 P99 结果。
单卡多 MoE 层 inference time 只能补完整推理分母与累计本地 MoE 时间，不能证明 receiver congestion。

2026-07-27 的 5090 表征补齐了完整 KV-decode inference-time 口径：16 个本地
MoE block 的累计时间占 profiled decode 的中位比例约 82.8%–90.2%，且该成本在
16 层间近似均匀累积。这证明完整 inference denominator 不能忽略多 MoE 层，但其中
包含 router、expert compute 与本地 combine；实验没有 EP ranks 或 return all-to-all，所以不是
receiver congestion 证据，也不改变 fixed RankLane NO-GO。

同日扩展实验在 observer tax ≤6.55% 的 coarse 分解中看到 `expert_loop` 已占 profiled
decode 约 74.9%–85.7%；context 128/512/2048 和自然/合成配对 A/B 都未出现新的单卡拥塞
信号。vLLM serving 与 OLMoE 跨模型项因远端实例关闭未完成，不能写成 serving 或跨模型证据。
