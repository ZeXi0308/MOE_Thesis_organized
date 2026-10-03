# StableBatch MaxGate-v1 selector 实验审计

**完整性结论**：`PASS`  
**置信度**：`0.98`  
**审查独立性**：`same-family provisional`  
**冻结科学判定**：`WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`  
**P0/P1**：无

## 核心结论

本轮结果可以按冻结边界使用。独立审计没有发现 selector outcome 泄漏、O/S 工作量不匹配、负 reward 过滤、阈值后改、patch/no-op 失效、重复不稳定或 artifact 损坏。

从 240 行 raw results 独立重算得到：

- 240 个唯一 victim-layer cells，来自 16 个文档窗口；
- 35 个 opportunity cells，覆盖 8 个文档，机会门槛通过；
- MaxGate-v1：`A_O=-3`；matched shuffle：`A_S=+3`；
- O 的 positive/tie/negative 为 `13/209/18`，S 为 `10/220/10`；
- 30 个 O/S 同 rank cells 的重复产物逐位一致；
- shuffle 每个 top-k rank 恰好 30 cells；
- 因 opportunity 通过且 `A_O<=A_S`，冻结 verdict 必然是 `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`。

## 证据边界

本轮只支持：在固定单 RTX 5090、16 个文档窗口、240 个离线 same-cell action-value cells、same-layer top-8，以及 synthetic all-M1 self-supervised reference 下，MaxGate-v1 总 signed reward 不优于一次冻结的 balanced matched shuffle。

它不支持 serving、动态控制器、EP/NCCL、多 GPU、自然发生率、跨数据泛化或“240 个统计独立样本”；R 也不是 ground truth。

## 研究决策

**MaxGate-v1：NO-GO。** 保留负结果，不删负 reward、不改阈值、不在这批已消费窗口上救援式调参。若继续 StableBatch，新的 selector 必须作为独立预注册假设，用新证据验证；本轮不授权进入 controller/serving 实验。

审计 trace： [.aris run05](/Users/leandrozhao/Desktop/毕设论文资料/.aris/traces/experiment-audit/2026-08-10_run05/001-observable-selector-integrity.response.md)
