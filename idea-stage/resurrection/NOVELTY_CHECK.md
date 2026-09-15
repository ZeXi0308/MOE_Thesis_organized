# Novelty Check — execution-conformance result

**Date:** 2026-08-13
**Reviewer:** fresh GPT-5.6-Sol xhigh, same-family/provisional
**Verdict:** `PARTIALLY_NOVEL`
**Confidence:** `0.93`
**Recommendation:** `PROCEED_WITH_CAUTION` as a narrow measurement question; do not present a method contribution.

## Proposed result

The canonical GPU experiment decomposes a target request's same-pre-state serial-versus-batched divergence into an attention-shape path and a same-global-shape companion-identity path whose first observed difference is at the MoE output. It also measures near-boundary association and two-step teacher-forced propagation; it does not establish causal margin amplification.

## Claim-level assessment

| Candidate claim | Novelty | Strongest collision | Allowed interpretation |
|---|---|---|---|
| Generic batch-dependent numerical/output variation | Low / not novel | [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [NeurIPS 2025 numerical nondeterminism](https://proceedings.neurips.cc/paper_files/paper/2025/file/f80094a824ba5912d4a2de169c404a40-Paper-Conference.pdf), [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | Background and replication only. |
| Heterogeneous KV/padding first diverges at attention | Low | Same generic batch-invariance/numerical-execution work | Scoped causal diagnostic, not novelty. |
| Fixed global width/KV-length vector, different companions, first observed difference at MoE output | Medium observation; mechanism unclear | [From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097), [Bit-Exact AI Inference Verification](https://arxiv.org/abs/2606.00279), [UniEP](https://arxiv.org/abs/2604.19241), [Buffer Overflow in Mixture of Experts](https://openreview.net/forum?id=SKWidEjUgU) | The only surviving residual. Current tensors do not identify expert GEMM `M`, sorting, kernel selection, combine order, or `index_add` as the exact cause. |
| Near-tie top-k association | Low / not novel | [R3](https://arxiv.org/abs/2510.11370), [VSRAQ](https://arxiv.org/abs/2606.05688), [MarginGate](https://arxiv.org/abs/2605.30218) | The 13/18 result is descriptive characterization, not independent causal evidence. |
| Broad execution-conformance contract | Low / not novel | [From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097), [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | The A/B/C/D protocol can be useful tooling, not a standalone contract contribution. |
| Verify/rollback/selective repair | Not novel and untested here | [LLM-42](https://arxiv.org/abs/2601.17768), [MarginGate](https://arxiv.org/abs/2605.30218), UniEP | No method claim is allowed. |

## Surviving novelty residual

The only potentially defensible new finding is:

> In natural continuous MoE decoding, companion identity may alter a target request through a hidden per-expert physical batch-shape or ordering path even when global width, the KV-length/padding vector, and the target row are fixed.

The current evidence supports only “first observed at MoE output.” It does **not** yet support “caused by expert microbatch grouping.” That stronger phrase requires direct per-expert mechanism capture.

## What cannot be claimed

- first batch-dependent LLM nondeterminism;
- first numerical perturbation causing MoE route flips;
- novel or causal top-k margin amplification;
- first execution-conformance contract;
- a verify/repair mechanism;
- exact expert-microbatch causality;
- native-serving, cross-model, quality, token-outcome, latency, capacity, SLO, or security impact.

## Decisive next experiment

In one representative native runtime, reproduce `steady:olmoe-dev-steady-002:5` and `bursty:olmoe-dev-bursty-002:0`, then instrument the first divergent MoE layer. Record companion routes, each target expert's physical `M`, token order, kernel/config ID, target expert outputs, and weighted pre-combine output. Compare:

1. natural original versus exact-length different-document companions;
2. per-expert `M` and token order canonicalized, combine order unchanged;
3. combine/reduction order canonicalized, expert `M` allowed to vary.

If only per-expert canonicalization removes the difference, the narrow residual survives. If combine-order canonicalization removes it, the result collides with the July 2026 expert-reduction work. If native execution removes it, the current result is a custom-runtime artifact.

Trace: `.aris/traces/novelty-check/2026-08-13_run01/`.
