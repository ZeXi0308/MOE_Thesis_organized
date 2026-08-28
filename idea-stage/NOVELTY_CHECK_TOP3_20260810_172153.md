# Novelty Check Report：C09 / C02 / C04

**日期**：2026-08-10  
**审查路线**：多组定向检索 + fresh GPT-5.6-Sol xhigh 复核  
**独立性**：`same-family`  
**接受状态**：`provisional`  
**总裁决**：`PROCEED_WITH_CAUTION — 只保留 C09 短周期证伪`

## C09 — Pre-execution MoE row/shape safety certificate

### Proposed method

在固定 model/stack/layer/expert/M 下，从 routed hidden row 的输入数值与执行形状提取特征；只有冻结规则判断 native M>1 会与 isolated M1 raw-BF16 相等时才走 fast path，其余 fail closed 到 M1。先用 partner/slot permutation 验证“安全性是否真是 row-local”。

### Core claims

1. row safety 对 co-row identity 与 slot/order 足够稳定。
2. pre-execution rule 能以极低开销预测 native M>1 与 canonical M1 的 raw-bit equality。
3. selective fast path 能在零误放行或严格声明风险界下，优于全局 batch-invariant execution。

### Closest prior work

| Work | Overlap | Remaining delta |
|---|---|---|
| [Bit-Exact AI Inference Verification](https://arxiv.org/abs/2606.00279) | 用 operands 与 execution details 重建 bit-exact GPU inference | 使用 expensive emulation，而不是 cheap MoE row-level admission rule。 |
| [MMA-Sim](https://arxiv.org/abs/2511.10909) | bit-accurate matrix-accelerator arithmetic model | full simulator，不是低开销选择性 runtime certificate。 |
| [Floating-point robustness certification](https://arxiv.org/abs/2603.13334) | sound pre-execution floating-point bounds | 保护网络 robustness，不是 native M>1 vs M1 bit equality。 |
| [RaMP](https://arxiv.org/abs/2604.26039) | 基于 runtime MoE routing state 选 kernel/config | 性能选择，无 numerical-equivalence hard constraint。 |
| [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | 全局 batch-invariant execution，覆盖测试过的 MoE | C09 只能以选择性稳定化、goodput 回收形成差异。 |

### Assessment

- **Novelty**：5.0/10。
- **Recommendation**：`PROCEED_WITH_CAUTION`。
- **Genuinely new delta**：尚未找到“cheap、pre-execution、MoE expert-row 级、预测 native batched BF16 是否与 isolated M1 bitwise equal”的直接同构方法。
- **Strongest collision**：bit-exact emulation、generic floating-point certification 与 RaMP input-dependent dispatch 已分别覆盖三块基础；把 classifier/error bound 应用到 MoE 不是自动创新。
- **Required positioning**：没有 soundness proof 时只能叫 fail-closed empirical predictor，不能叫 certificate。论文级主张必须同时具备 MoE-specific mechanism、保守风险边界和相对全局 batch-invariant kernel 的实际 goodput 优势。

## C02 — Asymmetric exact-row commit + selective rescue

### Proposed method

对 M2 两行使用 pre-execution row rule；执行 M2 后只保留事先认证的行，未认证 partner 用 M1 补算，再按 sealed row identity 拼接。

### Closest prior work

| Work | Overlap | Remaining delta |
|---|---|---|
| [LLM-42](https://arxiv.org/abs/2601.17768) | nondeterministic fast path、fixed-shape verify、commit/rollback | token/state 级 post-verification，不是 pre-certified expert-row partial commit。 |
| [MarginGate](https://arxiv.org/abs/2605.30218) | sparse deterministic verification 与局部 K/V repair | logit-margin/token repair，不是 raw expert-row equality。 |
| [Bit-Exact AI Inference Verification](https://arxiv.org/abs/2606.00279) | exact recomputation/comparison | 未提出 one-row commit + one-row M1 rescue。 |

### Assessment

- **Novelty**：3.5/10。
- **Recommendation**：`ABANDON AS STANDALONE`。
- **Genuinely new delta**：只有“执行前已知安全、expert-row 粒度 partial commit”这一窄执行技巧。
- **Strongest collision**：结构上仍是 verified speculation + selective repair；从 token/KV 换成 expert row 更像专用化。
- **Disposition**：若 C09 成立，把 C02 作为 downstream executor/ablation。必须完整计入 certificate、M2、M1 rescue、stitching 成本，并对 two-M1、safe-only packing、global batch-invariant 与普通 verify/rollback。

## C04 — Grouped execution preserving serial-M1 arithmetic

### Proposed method

把多条逻辑独立 M1 expert GEMM 放入一次 grouped/persistent launch；共享 dispatch 与 weight residency，但逐行保持 canonical M1 的 K traversal、accumulator、activation/cast 和 output slot。

### Closest prior work

| Work | Overlap | Remaining delta |
|---|---|---|
| [CUTLASS Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) | 一次 persistent kernel 调度多个独立 GEMM problems | 不保证复现某个 legacy serial-M1 implementation 的 bits。 |
| [vLLM batch-invariant implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/batch_invariant/) | 固定 K accumulation order 以实现 batch invariance | general BMM，不是特定 MoE grouped-M1 executor。 |
| [TBIK](https://arxiv.org/abs/2511.17826) | 固定 reduction tree 保证 bitwise identity | 面向 TP-size invariance。 |
| [ExpertPlex](https://arxiv.org/abs/2607.18002) | adaptive persistent expert kernels | goodput 优化，不保护 legacy serial-M1 identity。 |
| [TMA-Adaptive FP8 Grouped GEMM](https://arxiv.org/abs/2508.16584) | variable grouped GEMM 与 valid-data numerical equivalence | 非 canonical BF16 M1 raw-bit contract。 |

### Assessment

- **Novelty**：1.5/10。
- **Recommendation**：`ABANDON AS THESIS NOVELTY`。
- **Genuinely new delta**：若能在特定 legacy-M1 exact contract 下得到明显正吞吐，可形成有价值的 implementation result。
- **Strongest collision**：CUTLASS 已有 grouped persistent execution，vLLM 已有 fixed-reduction batch-invariant kernels；C04 是两类既有机制加 qualification test 的交集。
- **Disposition**：作为 infrastructure/baseline 保留，不作 thesis headline。只有相对 CUTLASS、vLLM invariant kernels 与 serial M1 的非显然 kernel design + 强实测收益，才可能重新升格。

## Overall novelty assessment

- **Decision**：`PROCEED_WITH_CAUTION`。
- **Retain**：C09，仅做短周期、预注册证伪。
- **Conditional module**：C02，仅在 C09 存活后评估。
- **Infrastructure**：C04，不作论文新意。
- **Mandatory kill order**：
  1. partner permutation（固定 focal slot）；若 label 随 partner 变化，杀 row-local C09，转向 pair relation。
  2. slot permutation；若只随 slot 变化，模型对象必须扩为 row+slot+shape。
  3. document-disjoint frozen predictor；任一 false admission 杀 raw-exact positioning。
  4. 对 exact emulation、shape-only、global vLLM batch invariance。
  5. 全执行器 latency；若收益为零，只保留 characterization finding。

## Missing-search risks

- 未穷尽专利、cuBLASLt/TensorRT 私有实现与 vendor techniques；C04 风险最高。
- correctly rounded dot product、interval arithmetic、approximate-computing guards 的早期文献未穷尽，可能继续压低 generic C09 certificate。
- 2026 preprints 与 beta 文档可能变化。
- 没找到直接 C09 collision 不等于证明新颖。
- 当前没有 partner-invariant label、pre-execution predictor、sound certificate、kernel speedup 或 serving goodput 证据。
