# SpectatorRoute（N05）

> 状态：`EXPLORATORY / PHASE0_ONLY / NOT_CURRENT_MAINLINE`  
> 证据等级：文献与代码审计、synthetic logger preflight，以及一次真实 5090 的**审计前诊断性强阳性**；**尚无审计接受的 pretrained-MoE Phase-0 结果，尚无攻击证据，尚无 CCF-B method GO**。

## 2026-07-29 run01 权威状态

`experiments/outputs/phase0a_5090_20260729_run01` 的 raw artifacts 内部一致，独立复算得到 `64/64` positive victims、`8192/8192` joint-positive cells、`0` unstable cells；但它在完整性审计通过前执行，没有 GPU UUID/外来 PID 连续监控、raw BF16 bit 计数、whole-parent watchdog、mandatory real-GPU acceptance 和最终 `COMPLETE.json`。因此该 run 仅为 `INVALIDATED_PRE_AUDIT_DIAGNOSTIC_ONLY`，其 `phase0b_authorized=true` 已撤销。原始文件保留，不删改，不作为正式证据。

修复版必须保持 64 victims、16×top-8 cells、原 M grid、10 repeats 和 `>=8` gate 完全不变，创建新 lock、新 acceptance artifact 与新输出目录后重跑。只有最后存在且校验通过的 `COMPLETE.json` 才能授权 Phase-0B。

## 当前唯一获准检验的主张

独立 fresh jury 在 16 个候选中给出 `15 KILL + 1 PHASE0_ONLY`。唯一存活的是下面这个可证伪的现象主张，而不是防御方法：

> co-batched spectator 的 prompt-induced MoE route shape，能否改变 victim 所在 expert 的实际 kernel/reduction regime，并以显著高于 matched-random spectator 的概率造成 victim expert output、后续 route 或最终 token 改变？

`stable-shape/canonical fallback` 不能写成新方法：它与 [MarginGate](https://arxiv.org/abs/2605.30218) 和 [LLM-42](https://arxiv.org/abs/2601.17768) 的选择性验证/回退结构直接相邻。当前只能先证明或杀死上面的物理因果链。

## 为什么这个问题不是凭空造的

- [RaMP](https://arxiv.org/abs/2604.26039) 和 [DA-MoE](https://arxiv.org/abs/2607.23099) 已证明 live expert histogram 会改变最佳 tile/kernel tactic；RaMP 的 kernel 还暴露 split-K 与 scatter reduction 等多种 configuration dimension。
- [batch-conditioned refusal](https://arxiv.org/abs/2605.27763) 已观察到低频 batch-conditioned flips，但其 continuous-composition 结果没有建立大规模 aggregate effect，batch-invariant kernel 消融又消除了复现到的 flips。它没有直接建立“prompt route shape → expert kernel regime → victim semantics”的链。
- vLLM 的 [batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) 仍标为 beta；官方 tracking [#27433](https://github.com/vllm-project/vllm/issues/27433) 把 DP support 列为 out of scope，[#30321](https://github.com/vllm-project/vllm/issues/30321) 只有用户级 DP+EP 不一致报告，不能当成已确认根因或问题规模。

## 执行顺序

1. [冻结 Phase-0 协议](N05_PHASE0_FROZEN_PROTOCOL_20260729.md)。
2. `Phase-0A`：真实 pretrained OLMoE hidden rows 上做 batch-row arithmetic capability gate；它只能 KILL 或授权 Phase-0B。
3. 只有 Phase-0A PASS 才做 prompt-only、matched-control 的 `Phase-0B`。
4. Phase-0B PASS 仍只授权重新查新和 method refinement；正式 EP、continuous scheduler、多租户安全与 P99 结论仍需多 GPU。

## 禁止越界

- 不把 synthetic matmul preflight 当作 pretrained-MoE 结果。
- 不把单 5090 的 batched Transformers 路径写成 EP/NCCL/RDMA 或生产多租户证据。
- 不在看到结果后改 M 网格、victim denominator、spectator 生成器、成功阈值或模型。
- 不因 Phase-0A 有数值差异就宣称 prompt-only attacker 可利用。
- 不把 canonical padding、margin fallback 或 batch-invariant mode 包装成当前的新贡献。
