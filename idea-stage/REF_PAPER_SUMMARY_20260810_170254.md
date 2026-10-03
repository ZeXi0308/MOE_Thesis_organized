# 近期文献边界：MoE 数值稳定性 × 系统调度 × 运筹优化

**检索日期**：2026-08-10  
**用途**：为本轮 idea 扩展提供碰撞检查；只记录原始论文或官方文档支持的边界。  
**证据状态**：`LITERATURE_MAP / NO_EMPIRICAL_RESULT`

| 来源 | 已覆盖机制 | 对本轮候选的约束 |
|---|---|---|
| [RaMP](https://arxiv.org/abs/2604.26039) | 依据 batch size 与 expert histogram 选择 MoE kernel/configuration | “按路由分布选 kernel”不新；只能研究带数值风险硬约束的 fast/stable mode 联合决策。 |
| [MarginGate](https://arxiv.org/abs/2605.30218) | 低 margin token 的稀疏验证与确定性恢复 | 一般 margin-triggered verification 已碰撞；新动作需发生在 MoE expert 执行前，或证明新的资源分配结构。 |
| [LLM-42](https://arxiv.org/abs/2601.17768) | 非确定 fast path、固定形状 verifier、commit/rollback | 通用 verify/rollback 不应成为主贡献；优先 pre-execution admit/pack/stabilize。 |
| [Batch-conditioned refusal protocol](https://arxiv.org/abs/2605.27763) | 把 batch condition 作为处理变量，并用 batch-invariant kernel 做消融 | 离线 batched-vs-single flip 不能外推为持续 composition 效应；必须 exact-stack、continuous batching 和 batch-invariant ablation。 |
| [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | 官方全局 batch-invariant 模式，覆盖已测试的 MoE 模型，但可能付出性能代价 | “让 MoE batch invariant”不新；空缺是选择性稳定化相对全局稳定模式能否收回 goodput。 |
| [METRO](https://arxiv.org/abs/2512.09277) | memory-bound decode 中优化 activated-expert count | “考虑 expert pressure”本身不新；新模型不能改变 router 语义，风险证书需成为额外硬约束。 |
| [Gimbal](https://arxiv.org/abs/2606.15177) | frontend/backend 协同、expert pressure、queue 和 MINLP placement | 一般 cross-level scheduling/MINLP 已碰撞；需有不可退化的 semantic-risk 状态与约束。 |
| [ExpertPlex](https://arxiv.org/abs/2607.18002) | tile-level adaptive persistent expert kernels 与隔离 | persistent/tile kernel 不是单独新意；只能作为构造 row-invariant arithmetic 的系统载体。 |
| [AMoE/AEP](https://arxiv.org/abs/2505.08944) | layer queue、异步执行、adaptive rebatching 与 defragmentation | 动态 rebatching、time-expanded queue 本身不新；需要证书约束、deadline 和可证明的联合决策增益。 |

## 当前未被直接覆盖的交叉点

> 将 MoE expert 内部的数值脆弱性或不变性证书，与 queue、deadline、expert pressure 和执行模式共同组成受约束在线决策；动作发生在执行前的 pack、admit、stabilize 或 kernel-mode，而非输出后的通用验证。

这个交叉点仍只是研究空间，不自动等于论文贡献。方法必须具备共享且稀缺的动作预算、可校准风险状态、不能被简单阈值替代的联合决策，并分别对照 vLLM batch invariance、RaMP、MarginGate/LLM-42、Gimbal/AMoE。

## 强制区分实验

1. partner permutation：区分 row-local safety 与 pair compatibility；若前者成立，compatibility graph/hypergraph 必须降级。
2. global stable mode：与全局 batch-invariant kernel 比较 semantic violation、fallback、goodput。
3. simple policy：与 shape-only threshold、greedy、random、load-only 比较，防止把通用 OR 模板包装成创新。
4. exact-stack continuous batching：离线 expert side-call 只做机制资格，不支持 serving/EP/SLO 主张。
