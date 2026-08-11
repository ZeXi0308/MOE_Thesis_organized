## Verdict

Ranking for a CCF-C/B systems paper, conditional on a successful first oracle pilot:

1. **TenantShapeFence — 7.4/10 — PROCEED, but narrow the claim**
2. **RouteSnap / RouterCanon — 6.2/10 — CAUTION**
3. **IntervalTopK / RouteCert — 5.5/10 — CAUTION**

None should revive the failed pre-action selector or BF16 expert-output canonicalization directions.

The strongest paper candidate is **TenantShapeFence**, because the combination “capacity-free numerical co-batch integrity channel + MoE-specific isolation mechanism + measured throughput/security frontier” is not subsumed by the primary literature I found. But “numerical noninterference” is currently too broad: grouping expert rows by domain protects only the demonstrated expert-GEMM shape path, not every batch-sensitive operator in the model.

## Evidence boundary

- **Observed, user-provided:** execution shape changed OLMoE BF16 expert outputs, propagated to ordered top-k changes, and produced at least one stable token flip. I did not rerun these experiments.
- **Narrow negative:** the 240-cell pre-action selector gate and expert-output bitmask/dither canonicalization failed. This rules out rescuing these ideas through row prediction or lossless-looking expert-output rounding.
- **Literature-confirmed:** primary arXiv/OpenReview papers and official project documentation were searched through 2026-08-10.
- **Unverified:** cross-model prevalence, practical attacker control, production overhead, sound error envelopes, and CCF-level generality.
- An unsuccessful search is not proof that unpublished or unindexed work does not exist.

## Core claims

| Candidate | Claim assessment |
|---|---|
| RouteSnap | Runtime, versioned router-logit lattice as a separate control-plane ABI appears **new in mechanism**. Routing stability itself and margin preservation are crowded. |
| TenantShapeFence | Capacity-free numerical cross-tenant influence appears **materially distinct** from capacity-overflow attacks. `(expert, domain)` grouping appears new in this context. Whole-model “noninterference” is not established. |
| RouteCert | Runtime middle-layer certification plus escalation appears somewhat new. The top-k gap theorem itself is not novel, and selective escalation is crowded. |

## Closest prior work

| Prior work | Closest candidate | Collision | Surviving delta |
|---|---|---|---|
| Thinking Machines batch-invariant operators; vLLM/SGLang deterministic modes; DeepSeek-V4; LayerCast; TBIK | All | Make broad operator/kernel surfaces invariant across batch or TP shapes. [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [vLLM](https://docs.vllm.ai/en/stable/features/batch_invariance/), [SGLang](https://sgl-project.github.io/advanced_features/deterministic_inference.html), [LayerCast](https://arxiv.org/abs/2506.09501), [TBIK](https://arxiv.org/abs/2511.17826) | Router-only control-plane intervention; expert-domain isolation; intermediate certification rather than whole-stack invariant kernels. |
| LLM-42 | All, especially TenantShapeFence | Per-request selective determinism already exists; only protected requests pay verification. It reports throughput within 3% of nondeterministic mode when 1/11 requests asks for determinism. [LLM-42](https://arxiv.org/abs/2601.17768) | TenantShapeFence avoids replay by preventing one MoE-specific channel; RouteSnap modifies the internal decision plane; RouteCert proves route stability before output verification. |
| MarginGate | RouteSnap, RouteCert | Uses output-logit margins to selectively invoke deterministic verification and repairs K/V state. [MarginGate](https://arxiv.org/abs/2605.30218) | RouteCert must provide a sound middle-layer certificate, not a calibrated risk predictor; RouteSnap commits canonical routes directly. |
| EAQuant, EAC-MoE, ExpertQuant/RouteQuant, VSRAQ, DREAM-MoE | RouteSnap, RouteCert | Preserve router logits, rankings, boundary margins, and downstream routing under model quantization. [EAQuant](https://arxiv.org/abs/2506.13329), [EAC-MoE](https://arxiv.org/abs/2508.01625), [VSRAQ](https://arxiv.org/abs/2606.05688), [DREAM-MoE](https://openreview.net/pdf?id=Wyhqwjl51A), [RouteQuant](https://openreview.net/pdf?id=bPsPPI65hf) | Different perturbation source and lifecycle: runtime execution-shape drift rather than compression-time calibration. |
| RouteQuant’s top-k gap proposition | RouteCert | Explicitly gives a sufficient logit-gap condition for preserving ordered top-k under quantization. | Sound runtime envelopes, propagation, and selective execution are not provided. |
| Buffer Overflow in MoE | TenantShapeFence | Establishes malicious co-batch influence on victims through capacity-limited, batch-dependent routing. Default no-capacity Mixtral was not vulnerable. [Buffer Overflow](https://arxiv.org/abs/2402.05526) | Capacity-free numerical kernel-shape channel and domain-tagged expert execution isolation. |
| StableMoE | RouteSnap | Freezes a distilled router to stop training-time routing fluctuation. [StableMoE](https://arxiv.org/abs/2204.08396) | Does not address runtime execution-shape-induced hidden-state drift. |

## 1. RouteSnap / RouterCanon

**Score: 6.2/10 — CAUTION**

Exact honest differentiator:

> An inference-time, versioned router decision ABI that snaps only the discrete expert-selection plane under execution-shape drift, while leaving expert mixing values and most numerical kernels unchanged.

This is meaningfully different from quantization papers, which optimize a compressed model to reproduce a full-precision routing reference, and from deterministic-serving work, which fixes or verifies the broader forward pass.

Fatal collision/validity risks:

- A deterministic lattice is not itself batch-invariant. Two shape-dependent logits can land on opposite sides of a snap-cell boundary.
- Fixed expert-ID tie-breaking only resolves equal snapped values; it does not solve boundary crossing.
- An escape path can itself disagree across shapes unless its predicate is canonical or verified.
- Stable routes do not imply stable tokens because attention, dense layers, normalization, mixing weights, and later logits remain shape-sensitive.
- Reviewers may reduce the contribution to “quantize router logits at inference,” especially given VSRAQ, DREAM-MoE, and RouteQuant.
- The novelty therefore lives in the systems contract and measured route/quality/overhead frontier, not in rounding as an algorithm.

Single must-beat baseline: **router-only FP32 computation plus fixed expert-ID tie-breaking**. If that recovers the same consistency cheaply, the lattice is unnecessary.

## 2. TenantShapeFence

**Score: 7.4/10 — PROCEED**

Exact honest differentiator:

> Security-domain isolation of grouped MoE expert execution, preventing another domain’s rows from changing a victim domain’s expert GEMM `M` and kernel regime even when routing has no capacity limit, overflow, token dropping, or cross-token data dependency.

That is narrower and more defensible than “numerical noninterference domains.”

Fatal collision/validity risks:

- `(expert, domain)` grouping does **not** establish whole-model noninterference. Other batch-sensitive GEMMs, RMSNorm, attention, collectives, dispatch ordering, and scheduler choices remain possible paths.
- LLM-42 already provides selective per-request determinism and is much more general. TenantShapeFence must win materially on throughput/latency or expose a different security contract.
- The failed selectability gate weakens a targeted-attack story. The paper may need to frame the threat as an integrity boundary against arbitrary/untrusted co-tenancy unless a practical coarse attacker is demonstrated.
- Domain fragmentation can destroy expert microbatch efficiency; if trusted-domain batches are usually small, the claimed cost advantage may disappear.
- If the effect exists only in one OLMoE kernel/model/GPU combination, it is a bug report, not a systems paper.

Single must-beat baseline: **LLM-42’s per-request selective deterministic mode**.

Proceed only with the narrowed claim: **expert-GEMM shape noninterference**, with whole-model protection obtained only when all remaining cross-domain execution paths are separately batch-invariant or isolated.

## 3. IntervalTopK / RouteCert

**Score: 5.5/10 — CAUTION**

Exact honest differentiator:

> A sound runtime certificate for an internal MoE control-flow decision, derived from execution-error envelopes and router-weight propagation, with fallback only when the top-k route cannot be proved invariant.

Fatal collision/validity risks:

- The condition “kth lower bound exceeds every outsider’s upper bound” is standard interval-ranking logic; RouteQuant already publishes an explicit top-k gap proposition.
- If the error envelope is empirical, percentile-based, or calibrated, this is not a proof and collapses toward MarginGate.
- A sound floating-point envelope requires kernel/backend/shape-specific forward-error accounting. A generic hidden-state interval propagated through many transformer layers is likely to explode and certify almost nothing.
- If escalation means replaying a deterministic forward path, the system begins to look like a more complicated MarginGate/LLM-42.
- A proof that covers only the next router does not certify subsequent routes or the emitted token.

Single must-beat baseline: **MarginGate using one calibrated output-margin threshold**.

This becomes publishable only if the pilot shows both soundness and a high non-vacuous certification rate. A mathematically correct certificate that escalates almost every token is not a systems contribution.

## Final recommendation

**TenantShapeFence is the best conditional CCF-C/B systems candidate.** Its paper-shaped contribution is:

> A previously unrecognized, capacity-free cross-tenant integrity channel in MoE serving; a domain-aware expert-execution boundary that closes that channel; and a measured isolation/throughput frontier against selective verified replay.

For that claim to survive review, the first oracle must show that changing only foreign-domain expert rows changes the victim’s expert kernel regime and downstream route/token, and that domain partition removes this dependence without silently relying on whole-model singleton execution. If the residual path is outside the expert block, or LLM-42 is equally cheap, the paper case collapses.

RouteSnap is the stronger fallback engineering idea. RouteCert has the weakest standalone novelty and the highest risk of becoming either a textbook lemma or a vacuous verifier.
