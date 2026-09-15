# CPR / RankLane

文档、实验代码和结果已集中在本目录。

| 内容 | 入口 |
|---|---|
| fixed RankLane 冻结域裁决 | [`CPR_MoE_RankLane_5090快验协议与裁决_2026-07-25.md`](CPR_MoE_RankLane_5090快验协议与裁决_2026-07-25.md) |
| 单卡必要条件设计与 Code Review | [`CPR_MoE_5090必要条件验证与CodeReview.md`](CPR_MoE_5090必要条件验证与CodeReview.md) |
| 8×A100 return-path existence Gate | [`EP_Return_Path_8xA100存在性Gate.md`](EP_Return_Path_8xA100存在性Gate.md) |
| codec 硬门槛历史测量 | [`Receiver_Codec硬门槛测量结论_2026-07-21.md`](Receiver_Codec硬门槛测量结论_2026-07-21.md) |
| 代码 | [`experiments/`](experiments/) |
| 结果 | [`outputs/`](outputs/) |

当前边界：fixed RankLane 在已冻结 `p_return≤20%` 域内停止；真实优化后 EP return path 仍需 8×A100，不得用单卡结果替代。
