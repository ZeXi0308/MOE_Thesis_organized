# MoE 系统方向文献与证据地图

**生成时间**：2026-08-12 17:19:03 +08:00
**范围**：本地论文前三页基线 + 2025–2026 近期一手工作 + 当前工作区冻结结果
**用途**：给下一轮 idea 排名与查新提供输入，不构成方法 GO 或实验结果。

## 当前问题地图

现有结果反复暴露的不是“还缺一个更聪明的 scheduler”，而是三个更基础的问题：

1. 局部 expert-stage opportunity 是否真的传播到 full-request completion、TPOT、P99 或 goodput；
2. execution shape / kernel / reduction 引入的数值变化首先发生在哪里，以及是否改变后续 route、state、token 和工作量；
3. 若需要避免或修复差异，安全约束是否仍留下优于 canonical baseline 的性能空间。

当前工作区的系统/方法级 formal GO 仍为 `0`。RCBA 尚处于 `UNVALIDATED / BLOCKED_PROTOCOL_AMBIGUITY / FORMAL_GATE_NOT_RUN`；SemanticFence-v2 只保留 proxy action-space 与 shadow-verify 资格；StableBatch pre-action selector 已停止；JoinStream 已冻结；fixed-C8 correctness primitive 未通过 headline cost Gate。

## 本地论文基线

| 工作 | 已覆盖方向 | 对新 idea 的约束 |
|---|---|---|
| HOBBIT | mixed-precision expert offloading、cache、prefetch | 普通 offloading/cache/prefetch 组合不够新 |
| AdapMoE | adaptive expert gating/management、按输入变化的 expert 使用 | 动态 expert 数量或 gating 本身不是 residual |
| Aurora | MoE inference-time deployment/communication optimization | generic placement 与通信优化拥挤 |
| Lina | distributed MoE all-to-all priority、micro-op scheduling、expert popularity | 仅重排通信与执行优先级难形成独立贡献 |
| Zhenyu HPC Master Thesis | gate-aware partial combine 与 Energy-SLO 早期方案 | 近似 combine/能耗调度需要重新证明语义与 full-request 闭环 |

## 近期工作簇

### 1. Numerical conformance 与 deterministic execution

- [From Expert Reduction to Behavioral Divergence](https://arxiv.org/html/2607.28097) 已建立 reduction order 到 state、route、text divergence 的链条，并提出 numerical compatibility contract；其 residual 不在“发现数值差异”，而在真实 GPU inference incidence、来源定位及 performance-aware enforcement。
- [UniEP](https://arxiv.org/html/2604.19241) 已在 MoE training megakernel 中使用 deterministic global token mapping、完整 contribution-set 等待和 baseline-order reduction，并将数值一致性纳入自适应优化配置；它是所有“adaptive kernel + deterministic ordering”想法的最强碰撞。
- [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) 已提供 batch-size/order-independent inference，代价是限制部分优化，因此“批不变推理”本身不是空白。

### 2. Verification、rollback 与 partial repair

- [LLM-42](https://arxiv.org/html/2601.17768) 已覆盖 nondeterministic fast path、fixed-shape verification 与 rollback。
- [MarginGate](https://arxiv.org/html/2605.30218) 已覆盖 low-margin selective verification 和 KV repair。
- [Predict, Reuse, and Repair](https://arxiv.org/abs/2606.30389) 已说明 partial state repair 不是天然新颖；新的 residual 必须落在 MoE identity、精确依赖闭包和可审计 closure 上。

### 3. Routing-aware kernel 与 serving optimization

- [RaMP](https://arxiv.org/html/2604.26039) 与 [DA-MoE](https://arxiv.org/abs/2607.23099) 已做 routing-distribution-aware kernel/config dispatch。
- [Gimbal](https://arxiv.org/html/2606.15177)、[MoEless](https://arxiv.org/html/2603.06350)、[Mixture-of-Experts Serving](https://arxiv.org/html/2607.17880) 与 [FinDEP](https://arxiv.org/html/2512.21487) 使 generic cross-layer scheduling、elastic placement 与 fine-grained compute/communication scheduling 成为拥挤区。

### 4. Causal tracing 与 request critical path

- [COZ](https://www.usenix.org/conference/atc16/technical-sessions/presentation/curtsinger) 已用 performance experiments 建立 causal profiling。
- [CRISP](https://www.usenix.org/conference/atc22/presentation/zhang-zhizhou) 已做 large-scale request critical-path analysis。
- [TELLER](https://arxiv.org/abs/2608.01975) 已面向 LLM inference 重建 per-request call-chain 与 dependency-aware causal context，并用于 RCA。

因此，“收集 trace”“画 critical path”“做 causal slice”不能单独作为贡献；可能保留的 residual 是 MoE-specific identity-complete DAG 经真实 held-out intervention 盲验，并在失败时 fail closed 地阻止 Oracle/repair claim。

## 仍可能成立的结构性空白

最有价值的空白不是单个执行器，而是一条 inference-specific qualification chain：

1. 在 natural continuous batching 中定位 consequential divergence 的发生率、first source 与 full-request propagation；
2. 建立 stack-versioned conformance contract，区分 GEMM、combine、next-route、persistent state 与 token 边界；
3. 只在 source 和 residual 已证明后选择 conditional actuator：canonical kernel/config class、epilogue reducer 或 dependency-scoped repair；
4. 与 UniEP-like ordering、batch-invariant、single canonical 和 verify/rollback 做 charged full-request Pareto。

这个空白目前只足以支持 `PROCEED_TO_FROZEN_GATE`。若 UniEP-equivalent control 已消除 consequential divergence、安全配置类退化为 singleton、局部差异不传播到 full request，或资格化成本吞噬收益，则整个方法方向应停止，只保留测量基础设施。

## 证据边界

本轮未运行新的 CPU/GPU pilot；所有排序来自既有冻结结果、文献查新与 same-family provisional reviewer。任何新方向都不得用 local expert-stage projection 替代 full-request 结果。
