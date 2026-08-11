# RouteSieve CCF-B Kill Verdict

- 日期：2026-07-29
- 候选：MoE route-summary sufficiency certificate / collision-regret witness
- 主查新：`4.8/10, KILL_CURRENT_CCFB`
- 敌对审稿：`3.2/10, KILL current formulation`
- 最终：`KILL / NO_GPU_PILOT`
- 评审独立性：`same-family / provisional`

## 结论

RouteSieve 不进入 RTX 5090 pilot。当前中心方法不是新的 CCF-B 级贡献：

1. Rice 的 algorithm-selection 框架已用 feature map 划分实例等价类，并以类内算法性能差异衡量特征质量；RouteSieve 的 summary collision class 与 robust regret 骨架与其高度同构。
2. DA-MoE 已使用 sorted/L2-normalized expert-count histogram 做 tactic matching；RaMP 已直接从 full runtime histogram 选 134–268 个 config，报告约 0.93% mean oracle regret，并做 2→4 参数 progressive refinement。
3. 对已测有限集 `D` 计算的 empirical regret `R_D` 是未观测全域 regret 的下界。`R_D <= epsilon` 不能签发 population/worst-case sufficiency certificate；添加字段直到每类 singleton 还会机械得到 0 regret。
4. `R_sigma(z) <= epsilon` 等价于“类内存在一个统一动作，对所有实例 regret 不超 epsilon”，主 theorem 基本是定义展开。

更多 workload、更多字段或 GPU 跑数不能修复上述方法性缺陷。

## 不进入 pilot 的理由

RouteSieve 只有在下述极窄情形下可以重新命名为 falsification audit：严格固定模型、精度、GPU、软件、cache protocol 和完整 action portfolio，仍能在同 sorted/full histogram 中找到 regret 置信下界超过 5% 的自然、可重放最优动作翻转，并用单因素干预定位可在线获取的遗漏字段。

这不再是“证明摘要充分”，而是“反例优先地否证现有 dispatcher key”。当前审稿判断该事件成功概率低；且单张 5090 缺多个独立 production backend。因此不用低成功率微基准消耗 GPU，继续第二轮其他候选。

## 证据边界

- 未运行 RouteSieve GPU pilot，无 route-summary collision 实测、无 backend ranking reversal、无任何 regret 跑数。
- 本结论 KILL 的是当前理论/方法 framing，不是实验证明“所有 route summary 都充分”。
- 现有 `grouped_owner_combine.py` 只是 backend-specific numerical counterfactual，不得当作通用 EP wire 或 latency 证据。

