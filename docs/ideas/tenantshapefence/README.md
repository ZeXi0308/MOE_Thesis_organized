# TenantShapeFence

## 当前判断

**Plausible / PROCEED 到一次自然共批 Oracle；尚未形成成立的论文 Idea。**

候选论文题目：

> **TenantShapeFence: Security-Domain Isolation for Capacity-Free Numerical Interference in MoE Serving**

当前最小主张不是“全模型确定性”或“全模型 noninterference”，而是：

> 在没有 expert capacity、overflow、token dropping 或跨 token 数据依赖的 MoE 中，外域请求仍可能通过改变 grouped expert GEMM 的 `M/kernel regime`，影响 victim 的 expert 输出并传播到后续 route/token；按 `(expert, security_domain)` 隔离 expert 执行可以关闭这条特定通道。

这条主张只有在自然共批 Gate 成功后才成立。

## Observation → Insight → Mechanism → System

### Observation

已有单贡献实验已经证明：改变一个 expert contribution 的执行形状可以稳定改变 BF16 输出，并在部分 victim 上传播到后续 routing；至少一个固定 victim 出现了稳定 greedy-token flip。

该证据是受控 intervention，不等于真实共批攻击已经成立。

### Insight

标准 capacity-free MoE 虽然不会因其他请求占满 expert capacity 而丢弃 victim token，但 grouped expert kernel 仍把不同请求的 routed rows 合并到同一 GEMM。外域请求因而可以改变 victim 所在 expert call 的 `M` 与 kernel regime。

安全边界不必等同于“每个请求 singleton”。真正需要隔离的是跨安全域共同决定 expert execution shape 的那一层。

### Mechanism

1. 请求进入 runtime 时携带 security-domain tag。
2. dispatch 后按 `(expert_id, security_domain)` 形成 expert groups。
3. 同域请求继续共享 expert microbatch；跨域默认不共同决定 `M`。
4. 只有具备 batch-invariant certificate 的 expert kernel 才允许跨域 merge。
5. combine 保持原 token/expert 身份和权重，不引入 selector 或 outcome prediction。

### System

最小系统面包括：

- domain-aware router/dispatcher metadata；
- `(expert, domain)` grouped-GEMM planner；
- batch-invariant kernel capability registry；
- isolation–throughput policy 与监控。

如果第一 Gate 成功，系统论文的核心评价轴是跨域输出独立性、expert microbatch fragmentation、吞吐、p50/p99 latency 与相对 selective verified replay 的成本。

## 与最近工作的边界

- [Buffer Overflow in Mixture of Experts](https://arxiv.org/abs/2402.05526) 依赖 capacity-limited、跨 batch 的 routing 竞争；其默认无容量限制的 Mixtral 路径未显示该攻击。本方向研究的是 capacity-free 数值执行形状通道。
- [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)、SGLang deterministic inference、DeepSeek-V4 和 TBIK 修改更广的 kernel/reduction surface。本方向只隔离一个 MoE-specific 跨域通道，必须用更低成本证明价值。
- [LLM-42](https://arxiv.org/abs/2601.17768) 已提供更一般的 selective deterministic verification/replay。若 TenantShapeFence 不能显著降低保护成本，它不足以成为独立系统贡献。

新颖性审查当前为同族 provisional：`7.4/10, PROCEED with narrowed claim`。

## 冻结的第一 Gate

实验脚本：[`experiments/run_capacity_free_cobatch_oracle_5090.py`](experiments/run_capacity_free_cobatch_oracle_5090.py)

固定项：

- OLMoE、RTX 5090 软件栈、victim token window；
- batch size、sequence length、victim slot；
- attention、dense layers、normalization、LM head 与所有非 expert batching；
- 无 capacity、drop 或 overflow 路径。

变量：

- 外域 co-batch token 内容；
- unprotected native expert grouping 对比仅按 domain 拆分的 expert execution。

搜索使用 32 个既有 request-disjoint 16-token windows，在 `B=8,16` 下做粗粒度 repeated-request attacker sweep。排名后候选重复两次验证。

### GO

至少一个候选同时满足：

1. 相同 batch shape 下，替换外域请求稳定改变 victim 的 route membership 或 greedy token；
2. victim 使用的至少一个 expert 的总 `M` 随外域 workload 改变；
3. 每个 routed contribution 都被执行，无 capacity/drop；
4. 只做 `(expert, domain)` split 后，两个外域 workload 下 victim 的 router logits、routes 与 final logits 完全相同；
5. 四个 arm 都 repeat-stable。

GO 只允许声称“自然 capacity-free expert-GEMM shape channel 在该模型/GPU/软件栈存在，expert-only domain split 能关闭该实例”。

### Weak

若只有 final-logit 数值变化而没有 route-membership/token 变化，则记录为 `WEAK_NUMERIC_CHANNEL_ONLY`，不足以进入系统实现。

### NO-GO

若自然 attacker sweep 无稳定 route/token signal，停止 TenantShapeFence 作为 headline mechanism；不立即加入攻击优化器来救结果。

若 unprotected 有信号但 expert-only split 无法消除，则当前归因不成立，结果标为 residual/outside-expert inconclusive，而不是宣称防御失败或全方向失败。

## 当前证据边界

- 已实现 runner，完成语法检查和一个 shape-sensitive toy expert 的因果单测。
- 真实 GPU Gate **尚未运行**；远端入口当前在 SSH 握手前关闭。
- 未验证自然共批通道、跨模型普遍性、攻击可控性、真实 serving 开销和质量影响。
- 不复活 MaxGate，不建立第三个 pre-action row selector。
