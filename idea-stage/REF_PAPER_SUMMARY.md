# StableBatch 后续方向：文献与证据边界

**生成时间**：2026-08-10 17:59 +08:00  
**用途**：为 MaxGate-v1 负结果后的新一轮 idea jury 提供边界；不是 novelty 完成证明。  
**证据等级**：本地实验结论为 observed；论文映射为 source-grounded；未跑机制均为 hypothesis。

## 1. 当前本地证据

- `single_contribution_20260810_run01` 在富集目标上观察到：单个 expert contribution 的 M=1/M=64 raw-BF16 差异能够穿过 combine，并在部分目标中改变后续路由。它只证明一条窄因果传播链，不是自然发生率。
- `observable_selector_20260810_run01` 覆盖 16 个文档、240 个 victim-layer cells。MaxGate-v1 的总 signed reward 为 -3，冻结 balanced shuffle 为 +3；因此 **MaxGate-v1 NO-GO**，不能继续调阈值或升级为 controller。
- 本轮允许的问题是：换一个机制上不同的 action space，先用 Oracle 判断它是否有足够上界；不得把旧 selector 改名救活。

## 2. 最近工作的硬约束

1. [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) 已在 2026 年提供全局 batch-invariant 模式，并列出 Qwen3 MoE 等验证模型；官方说明其使用 deterministic kernels、禁用部分非确定优化并可能损失性能。因此“让 MoE batch invariant”本身不新。
2. [vLLM modular fused-MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/modular_kernel.py) 已把 `supports_batch_invariance` 纳入 kernel compatibility/dispatch。一般性的 deterministic-kernel selector 不足以成立。
3. [LLM-42](https://arxiv.org/abs/2601.17768) 已提出 non-deterministic fast path + fixed-shape verify/rollback，且只按需要付验证开销。一般性的 speculative verification 已被覆盖。
4. [MarginGate](https://arxiv.org/abs/2605.30218) 已用低 logit margin 稀疏触发 deterministic verifier/KV repair。一般性的 margin-triggered selective determinism 也已被覆盖。
5. [RaMP](https://arxiv.org/abs/2604.26039) 和 [DA-MoE](https://arxiv.org/abs/2607.23099) 都根据 live routing histogram / expert distribution 做 MoE kernel configuration dispatch；若新方向只是在性能目标上选 kernel，会直接碰撞。
6. 本地 Aurora、Lina、HOBBIT、AdapMoE 分别覆盖通信/部署调度、micro-op 与 popularity 调度、mixed-precision offload/cache、sensitivity-aware gating/cache。普通 expert scheduling、cache、precision allocation 不能作为新意。

## 3. 仍可能留下的空隙

- **MoE 内部粒度**：不是验证整 token，而是利用 expert-row / layer / execution-shape 结构控制第一次发生的数值分歧。
- **无回滚路径**：通过 fixed-shape capsule 或 certified execution plan 在执行前规避形状变化，而非先执行再重放。
- **局部成本模型**：只在能够证明有风险的 execution signature / stratum 上支付稳定执行成本，目标是在同一确定性标准下显著低于全局 batch invariance。
- **Oracle-first**：先量化最优单贡献、shape bucket 或 plan fallback 的收益上界；若 Oracle 也没有优势，立即停止相应 action space。

## 4. 查新后的禁区

- 不把 fixed reduction tree、fixed-shape GEMM 或 deterministic fused-MoE kernel 单独包装成主贡献。
- 不把 next-token margin、verify/rollback、KV repair 单独包装成主贡献。
- 不把 routing-histogram-aware kernel selection 单独包装成主贡献。
- 不把一般风险预算、bandit、DRO 或 queueing 公式当成 MoE 系统贡献；必须先有 MoE-specific action value。

