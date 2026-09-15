# Resurrection Experiment Audit

**Date:** 2026-08-13
**Overall:** `WARN`, P0=0
**Acceptance:** same-family provisional

Two distinct surfaces were audited.

## 1. Corrected local three-arm runner

Final verdict: `PASS`, no remaining P0/P1. Before that pass, two targeted review rounds forced six corrections: global rather than local first-divergence ordering, paired-repeat qualification, removal of a physical-shape overclaim, a stronger near-tie criterion, cross-regime gating, and exact dtype/shape/float32-digest equality before a router-kernel classification. `py_compile` passes and all five focused CPU protocol tests pass.

This runner was **not** the producer of the canonical GPU evidence. Its remaining P2 engineering limits are non-retention of failed-run bundles, no canonical combine implementation, and no imported-producer source hash.

## 2. Canonical four-arm GPU evidence

Final integrity verdict: `WARN`, P0=0. Fresh outputs, claimed metrics, completion chronology, and both post-run seals pass. Five unique claim-level P1s were found across the bounded audits and corrected or explicitly bounded:

1. near-boundary association is not causal amplification;
2. expert-execution grouping is not measured;
3. exact all-six historical serial reproduction is not established;
4. raw `/tmp` capture provenance is not self-contained;
5. prevalence first material difference is attention output in 23/24, while first exact difference is there in 24/24.

The provenance warning cannot be repaired retroactively without the missing raw captures. It does not invalidate the retained fresh A/B/C/D measurements; it limits them to source-selected batched states whose source hashes were checked at execution time.

A final bounded correction recheck returned `PASS`, P0=0, P1=0 across the six requested claim boundaries. Overall status remains `WARN` solely because provenance is not self-contained.

## Final claim ceiling

Stable single-OLMoE/single-RTX5090/custom-runtime same-pre-state conformance differences, including a companion-conditioned effect first observed at MoE output and teacher-forced two-step logit/KV persistence. No exact internal mechanism, native transfer, quality, latency, capacity, controller, or SLO claim.

Evaluation type: `self_supervised_proxy_over_real_gpu_execution`.

The only next experiment is the two-event native-runtime operator disambiguation frozen in `RESURRECTION_AUDIT.md`.
