Verdict: `PARTIALLY_NOVEL`
Confidence: `0.93`

Only one narrow residual survives: same global batch width and KV-length/padding vector, but different companion identities, first becoming observably different at the MoE output. Even that is not yet a defensible “expert microbatch grouping mechanism” claim because the artifacts do not record per-expert `M`, token order, kernel choice, per-expert target outputs, or pre/post-combine intermediates.

### Per-claim collision matrix

| Claim | Verdict | Strongest collision | Assessment |
|---|---|---|---|
| 1. Generic batch-dependent numerical/output variation | `NOT_NOVEL` | [Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/f80094a824ba5912d4a2de169c404a40-Paper-Conference.pdf), [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | These works already establish batch-size/kernel-shape-dependent floating-point variation, attention/GEMM reduction-order causes, output divergence, and production batch-invariance support, including tested MoE models. |
| 2. Heterogeneous KV/padding -> attention; companion identity -> MoE output | `PARTIALLY_NOVEL` for the observation; `UNCLEAR` for the proposed mechanism | [From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097), [Bit-Exact AI Inference Verification](https://arxiv.org/abs/2606.00279), [UniEP](https://arxiv.org/abs/2604.19241), [Buffer Overflow in MoE](https://openreview.net/forum?id=SKWidEjUgU) | The attention half is known. MoE numerical propagation, token ordering, reduction semantics, and cross-batch output interference are also known. The residual is specifically a capacity-free, same-global-shape companion-identity effect first observed at MoE output. However, the current evidence localizes the first observed difference only; it does not distinguish expert-GEMM `M`/kernel changes from sorting, combine order, `index_add`, or another MoE backend detail. Buffer Overflow is weaker overlap because its interference comes from capacity-limited routing. |
| 3. Near-tie expert top-k as secondary amplifier | `NOT_NOVEL` | [R3](https://arxiv.org/abs/2510.11370), [VSRAQ](https://arxiv.org/abs/2606.05688), [MarginGate](https://arxiv.org/abs/2605.30218) | R3 already identifies small perturbation -> discrete expert-route change -> amplified MoE output discrepancy. VSRAQ directly protects the kth-versus-(k+1) expert boundary. MarginGate establishes the analogous deployable near-margin trigger for output-token flips. The repository's 13/18 result is a scoped corroborating measurement, not a new mechanism. |
| 4. Execution-conformance measurement/contract direction | `NOT_NOVEL` as a broad direction | [From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097), [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/), [Bit-Exact AI Inference Verification](https://arxiv.org/abs/2606.00279) | The July paper explicitly proposes a sparse-MoE numerical compatibility contract and a hierarchy spanning token/logits, persistent state, layer state, routing, and operator intermediates. vLLM operationalizes batch invariance as a user-visible contract. The A/B/C/D source-localization protocol may be useful tooling, but it is not presently a standalone research contribution. |
| 5. Verify/repair method novelty | `NOT_NOVEL` and currently unsupported | [LLM-42](https://arxiv.org/abs/2601.17768), [MarginGate](https://arxiv.org/abs/2605.30218), [UniEP](https://arxiv.org/abs/2604.19241) | LLM-42 already provides fixed-shape verify/rollback with KV-state replacement. MarginGate already provides sparse margin-triggered verification and current-column KV repair. UniEP provides deterministic MoE token ordering. The canonical evidence tested no verifier, repair action, guarantee, overhead, or serving integration. |

### Strongest prior art

The strongest direct collision is [From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097), not the older generic batch-invariance literature. It already connects sparse-MoE numerical execution semantics to later route divergence, persistent state, delayed token divergence, and a hierarchical runtime-conformance contract.

Its limitation leaves the only credible residual: it manually varies cross-expert aggregation order and does not measure natural companion-conditioned expert microbatch shapes in GPU continuous batching. That is narrower than the current report's broad “execution-conformance” framing.

### What cannot be claimed

- First discovery of batch-dependent LLM nondeterminism.
- First observation that numerical perturbations flip MoE routing.
- Novelty of expert top-k near-boundary amplification.
- First execution-conformance or numerical compatibility contract.
- Any verify/rollback/selective-repair method.
- That companion identity acts specifically through expert microbatch grouping; current artifacts establish first observed divergence at MoE output, not the exact internal operator.
- Native-serving, cross-model, quality, token-outcome, latency, capacity, SLO, or security-isolation consequences. The current propagation had persistent logits/routes but `0/8` changed predicted tokens.

### Novelty residual

A defensible residual would be:

> In natural continuous MoE decoding, companion identity can alter a target request through hidden per-expert physical batch shape even when global width, KV-length/padding vector, target pre-MoE state, target router scores, and target selected experts are fixed.

This would be a useful MoE-specific measurement finding only if the per-expert execution mechanism is directly isolated in a native runtime. It is not yet a method contribution.

### One decisive next experiment

In one representative native serving runtime, reproduce two C/D events from the same target pre-MoE state and instrument the first divergent MoE layer:

1. Record companion routes, per-target-expert token batch `M`, token order, kernel/config ID, each target expert's output, and the weighted pre-combine result.
2. Run natural C/D.
3. Canonicalize only each target expert's `M` and token order while leaving cross-expert combine semantics unchanged.
4. Separately canonicalize only cross-expert combine/reduction order while allowing expert `M` to vary.

Decision:

- Difference disappears only after canonicalizing per-expert `M`/order: the narrow novelty residual survives.
- Difference disappears after canonicalizing combine order: it collapses into the July 2026 expert-reduction paper.
- Difference disappears in the native runtime: classify the current result as a custom-runtime artifact.
- No downstream route or token effect after mechanism isolation: retain it as low-level characterization, not a thesis headline.

Evidence type: `CUSTOM_CONTINUOUS_RUNTIME + PRIMARY-SOURCE NOVELTY REVIEW`
Oracle/headroom: `NOT_RUN`
Failure category: `MEASUREMENT_ONLY / NOVELTY_NARROWED`
Claim ceiling: one-runtime MoE source-localization hypothesis; no method GO.

No files were edited. The novelty-check trace was not written because the assigned review was explicitly read-only.
