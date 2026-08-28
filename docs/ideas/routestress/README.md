# RouteStress-WitnessPatch

> 状态：`SYSTEM_PIVOT_DEFINED / NEXT_GPU_ACTION_NOT_RUN / NOVELTY_NOT_VERIFIED`  
> 定位：StableBatch 的反例驱动系统化版本；不是又一个 static risk table

## 直接判断

2026-08-10 的正负结果不支持继续调 SemanticFence 或 ErrorToken 的静态阈值：

| 对象 | 当前证据 | 系统含义 |
|---|---|---|
| StableBatch 单贡献干预 | `12/32` 富集 targets、`8` victims 出现可重复下游 route membership 变化 | execution-shape delta 有可能传播；只保护局部 raw delta 不够 |
| SemanticFence exact contract | unrestricted `7,584/8,192` mismatch，但 `4,237` contract entries 放行 `0` | binary exact class 在该 stack 上退化为全 `M=1` |
| ErrorToken static transfer | keyed AUC `0.5389`，相对 `M-only` 仅 `+0.0076` | calibration exact fraction 不足以作为在线风险表 |
| ShapeShare | 与冻结 RouteShare coalition-cost/Shapley 对象重复 | 不实现，不用新名字复活 |

因此下一个机制不能再问“哪个 `(layer, expert, M)` 平均更安全”，而应当问：

> 能否从真实 heterogeneous co-batch 中生成、最小化并在 held-out 数据上确认“会传播的 execution-shape 反例”，再把它编译成一个局部 pack-surgery action？

## 完整机制

1. **Natural witness generation**  
   在 exploration split 中对相同 victim 构造真实 heterogeneous co-batch，记录 `pack shape → raw contribution → combine → downstream route/token` 的第一分歧。不再用重复同一 row 伪装自然 batch。

2. **Counterexample minimization**  
   对 positive pack 做 delta debugging：逐步移除 spectator rows，每次都要在同 stack 上重跑因果链，直到得到仍能重现传播的最小 witness。最小化结果是实验对象，不是安全证书。

3. **Held-out rule confirmation**  
   从 witness 提取只依赖 action-time 可观测量的 predicate，例如 stack digest、layer、expert、natural `M` 转换和 route-rank composition。规则必须在独立 confirmation split 上胜过 matched shuffle；失败就丢弃，禁止换 threshold/key 抢救。

4. **Minimal pack surgery**  
   runtime 只对命中已确认 predicate 的 contribution 执行 `M=1` protected call，其余 rows 重新压紧后保持 native batch。不改 router、top-k、expert identity、gate weight 或 dtype。动作必须以 `(request, step, layer, expert, row)` ledger 记录，不做未记账的隐式 fallback。

5. **Stack epoch and canary invalidation**  
   每个 model/backend/driver/kernel 组合对应一个 epoch。新版本不继承旧规则；小比例 canary 只用来检测规则失效并整个 invalidate，不在线学习或调阈值。

## 不重复的边界

- **RouteContract** 的目标是跨实现 semantic relation / correctness bug 与 capability certificate；本方向只处理合法同 backend 的 execution-shape 传播反例，不宣称 correctness proof。
- **SemanticFence** 生成静态 exact allowlist；本方向只使用已在 held-out 上确认传播的最小 witness，不用 raw exactness 代替传播 label。
- **MarginGate / LLM-42 邻近结构** 依赖 low-margin verify/rollback；本方向不读 future/next-layer margin，也不对所有 low-margin token 做通用 rollback。最终新意仍需专门查新。
- **RouteShare / ShapeShare** 使用 coalition cost、Shapley、virtual service unit 或 fairness attribution；本方向完全不计算这些对象。
- **FrontierCredit** 决定 ready queue 的 hold/flush 与 request-DAG 推进；本方向在 rows 已 ready 后只做 executor pack surgery，不改变 queue timing。

## 下一个唯一 GPU Pilot

**问题：** 用 exploration-only witness 编译的 predicate，在 fresh natural co-batch 上实际执行最小 `M=1` pack surgery，是否比同 action budget 的 shuffled surgery 阻止更多 downstream route divergence？

- 三臂：`native`、`witness-patch`、`budget-matched shuffled-patch`。
- 必要控制：三臂的 input、natural pack、protected-call 数、full-forward 数、重复数和顺序完全一致；只换 protected contribution identity。
- 主指标：`avoided reproducible downstream route divergences / protected contribution`，以 victim 为聚类单位。
- 次指标：protected action 的 GPU latency/launch overhead；不用 call-count proxy 充当 latency。
- 支持信号：witness-patch 效率至少为 shuffled 冻结均值的 `1.5×`，且增益覆盖至少 4 个 distinct victims。
- 削弱信号：与 shuffled 持平/更差，或 predicate 不能在 fresh split 触发。
- 无法判断仅限于：泄漏 future label，三臂工作不匹配，action 没有真正改变 contribution，same-arm 不稳定，或 artifact 损坏。
- 资源：复用 StableBatch capture/replacement/hash closure；预计单 RTX 5090 `30–90` 分钟。

## 当前 claim boundary

现在只能宣称“已定义一个不依赖 static exact/risk table 的反例驱动系统 Pilot”。尚未运行 natural heterogeneous co-batch 动作实验，不能宣称 RouteStress 有效、有新意、改善 serving，或适用于 EP/NCCL/RDMA/多 GPU/生产。

## 证据入口

- [StableBatch 实验审计](../../../artifacts/stablebatch_remote_20260810_run01/EXPERIMENT_AUDIT.md)
- [SemanticFence 实验审计](../../../artifacts/semanticfence_remote_20260810_run03/EXPERIMENT_AUDIT.md)
- [ErrorToken risk-transfer 汇总](../errortoken/experiments/outputs/risk_transfer_20260810_run01/summary.json)
- [ErrorToken risk-transfer 审计](../errortoken/experiments/outputs/risk_transfer_20260810_run01/EXPERIMENT_AUDIT.md)
- [RouteShare 冻结禁止换 Shapley 抢救](../../archive/killed_ideas/routeshare/RouteShare_Gate0_Phase2_冻结协议_2026-07-23.md)
