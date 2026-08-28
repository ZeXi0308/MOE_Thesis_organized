# Top-3 Novelty Check Report

**生成时间**：2026-08-12 17:19:03 +08:00
**评审边界**：same-family / provisional；文献检索与一手来源已纳入，但不是 cross-family acceptance
**总裁决**：`C10 STOP_STANDALONE / C07 CAUTION / C02 PROCEED_TO_FROZEN_GATE_INFRASTRUCTURE_ONLY / METHOD_GO=0`

## Proposed Methods

- **C10 Canonical Segmented Epilogues**：保留 dynamic packed expert GEMM，在 epilogue/combine 按稳定 identity 恢复 canonical accumulation order。
- **C07 Causal Repair Cones**：对已验证失败的 MoE row 构建精确 downstream dependency cone，只撤销并重放该闭包，无法证明时 full rollback。
- **C02 Intervention-Calibrated Request-DAG Trace**：用 held-out 真实小干预盲验 identity-complete MoE request DAG，使其只在因果预测通过后支撑 Oracle/repair。

## 裁决摘要

| 候选 | Novelty score | 风险 | 裁决 | 真正 residual |
|---|---:|---|---|---|
| C10 | 2/10 | HIGH | `STOP_STANDALONE` | 只有先证明 inference 中的 first divergence 位于 epilogue、UniEP contract 未覆盖且实现有 full-request headroom，才可能重开 |
| C07 | 5/10 | MEDIUM-HIGH | `CAUTION` | MoE identity 精确依赖锥、invalidation certificate、closure proof；不是 selective verify/rollback 本身 |
| C02 | 4/10 | MEDIUM-HIGH | `PROCEED_TO_FROZEN_GATE / INFRASTRUCTURE_ONLY` | held-out real-intervention blind validation、missing-edge certificate、fail-closed Oracle qualification |

## C10 — Canonical Segmented Epilogues

### 最接近工作与重叠

[UniEP](https://arxiv.org/html/2604.19241) 已覆盖 deterministic identity mapping、等待完整 top-k contribution set、按 baseline order reduction 与 numerical consistency；[vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) 已覆盖 batch-size/order-independent inference；[From Expert Reduction to Behavioral Divergence](https://arxiv.org/html/2607.28097) 已覆盖 reduction order 到 behavior divergence 与 compatibility contract。

### 关键本地否证

StableBatch 的冻结证据在 gate weight/combine 之前的 raw expert output 已观察到 physical-M divergence。对当前主要现象而言，epilogue 无法恢复已经变化的 GEMM output，因此 C10 的 mechanism assumption 不成立。

### 裁决

`STOP_STANDALONE`。不投入 5–8 周实现。只保留一个条件 actuator：若 source-localization 在新 natural inference cells 中证明主要 residual 首次发生于 epilogue、UniEP-like control 仍未覆盖、且 charged full-request cost 优于 canonical baseline，才以新卡重开。

## C07 — Causal Repair Cones

### 最接近工作与重叠

[LLM-42](https://arxiv.org/html/2601.17768) 已覆盖 fast-path verification/rollback；[MarginGate](https://arxiv.org/html/2605.30218) 已覆盖 selective verification/KV repair；[Predict, Reuse, and Repair](https://arxiv.org/abs/2606.30389) 与 partial KV recomputation 说明 partial repair/recomputation 本身不新。

### 可保留 residual

从 request/step/layer/token/route/expert/row identity 构造精确传递闭包，输出可审计 invalidation certificate；只重放严格小于完整 per-request suffix 的 cone，并以 all-canonical reference 验证 closure。无法证明时必须 fail closed。

### 裁决

`CAUTION`。只有 C02 的 dependency trace 先通过资格验证后才能进入 `C07-G0 CONE_TIGHTNESS_AND_CLOSURE`。任意 closure mismatch、cone 大多退化为完整 suffix、普通 per-request rollback 已等价，或完整成本吞噬 headroom，均停止。

## C02 — Intervention-Calibrated Request-DAG Trace

### 最接近工作与重叠

[COZ](https://www.usenix.org/conference/atc16/technical-sessions/presentation/curtsinger) 已覆盖干预式 causal profiling；[CRISP](https://www.usenix.org/conference/atc22/presentation/zhang-zhizhou) 已覆盖 request critical path；[TELLER](https://arxiv.org/abs/2608.01975) 已覆盖 LLM call-chain、causal-context slicing 与 RCA。

### 可保留 residual

面向 continuous-batching MoE 的 identity-complete request/resource DAG，在 calibration/evaluation 分离的 held-out 真实干预上盲预测 completion-delta sign、affected request set 和 completion order；预测失败时输出有限 missing-edge class，并阻止任何 full-DAG Oracle 或 repair claim。

### 裁决

`PROCEED_TO_FROZEN_GATE / INFRASTRUCTURE_ONLY`。它值得先做，因为当前多个方向都卡在 local proxy 到 full request 的证据断层；但 trace schema 本身不够发表。只有证明 generic trace 在 MoE dynamic batching 下系统性给出错误因果结论，而 intervention-qualified identity DAG 能纠正，才可能形成测量型论文 residual。

## 对主线的修订

把初始 jury 的 C10 standalone Primary 改为：

> **Inference-Specific MoE Execution Conformance Pipeline**
> source localization → stack-versioned contract → conditional actuator → full-request Pareto

这只是 `PRIMARY_RESEARCH_PROGRAM / UNVALIDATED / PROCEED_TO_FROZEN_GATE`，不是 method candidate 或 GO。最强 reviewer objection 是：四段分别接近 causal profiling、UniEP conformance、现有 dispatch/rollback 与必要评测，把它们串联并不自动形成 novelty。它必须先证明一个 UniEP contract 未覆盖、只在 natural inference 中出现、且能被更便宜 actuator 消除的 residual。

## 唯一立即冻结 Gate

### `IECP-G0 — UNIEP_RESIDUAL_SOURCE_LOCALIZATION`

冻结两个模型 revision、每模型两个 natural continuous-decode load cells、每 cell 至少 32 paired episodes，以及完全相同 requests/arrival trace/seed。比较 canonical sequential、native dynamic 和实现 deterministic identity mapping、full contribution-set wait、baseline-order reduction 的 UniEP-equivalent control；至少覆盖三种 materially different kernel/tactic/packing configurations。

必须记录：

`raw expert output → epilogue/combine → next-router → persistent state → token → request completion`

并通过交换冻结 buffer 分离“相同 raw output + 不同 epilogue”和“不同 raw output + 相同 canonical epilogue”。

唯一通过条件是以下条件同时成立：

1. 两模型 natural cells 均有可复现 performance–conformance tension；
2. UniEP-equivalent control 后仍有 consequential state/route/token divergence；
3. first divergence 稳定落入预注册 source class；
4. 至少一个安全 non-canonical configuration 在 charged full-request accounting 下优于 canonical baseline；
5. 正结果落在 completion/TPOT/P99/goodput，而非 expert-stage projection。

任一条件不满足即停止 method program，只保留 C02 evaluator；不得用新 selector、fixed-C8、epilogue 调参或扩大 repair scope 抢救。

## Pilot 状态

本轮未运行新 pilot。原因是当前任务是候选查新与 Gate 排序，且 `IECP-G0` 的协议、trace 与 control 尚未冻结；`SKIPPED / NOT_RUN` 不计为实验结果。
