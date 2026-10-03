# Longrun A Experiment Audit

**Date:** 2026-08-13
**Review:** fresh GPT-5.6-Sol, same-family provisional
**Overall verdict:** `WARN`
**Integrity status:** `warn`
**Evidence type:** `self_supervised_proxy` over real GPU execution

No P0 integrity failure, fake ground truth, future leakage into the primary comparison, missing claimed output, or post-run seal mismatch was found. The valid evidence supports only `MEASUREMENT_ONLY / CUSTOM_CONTINUOUS_RUNTIME`. The remaining warning is structural: the source captures used to choose the six events were stored under `/tmp` and are absent locally, so original-event provenance cannot be independently replayed from the retained bundle. This does not invalidate the fresh same-pre-state A/B/C/D measurements, but it lowers their provenance ceiling.

## Checks

### A. Ground-truth provenance: WARN

The source ledger is model-generated execution metadata, explicitly not semantic ground truth. Arm C closes selected batched input, route, and token identity. Historical serial routes are diagnostic metadata rather than a validity condition, and one selected event did not reproduce its historical serial route. Source hashes were validated while the run executed, but the raw source directories are absent from the workspace.

### B. Score normalization: PASS

No metric is normalized by the model's own predicted score distribution. Exact equality, allclose, raw deltas, relative L2, cosine, route membership, and the preregistered `0.01` margin are retained. The 13/18 statistic supports association only; 18/18 lost/gained crossings follow from the route-flip definition and are not independent causal amplification evidence.

### C. Results and hashes: PASS

All reported main-run route counts, propagation counts/ranges, prevalence counts, and sampled isolation counts recompute from retained JSON. Both `POST_RUN_SEAL.json` files verify exactly: 35 main files and 79 prevalence files, with zero missing, extra, size, or SHA-256 mismatch. The prevalence first exact A/C difference is attention output in 24/24 cases; its first material difference is there in 23/24, not 24/24.

### D. Called paths and metric semantics: WARN

The reported capture and comparison paths are called. The runner does not record per-expert `M`, token order, kernel/config ID, individual expert output, dispatch/sort state, or combine intermediates, so expert-grouping causality is unmeasured. The machine source classifier is exact-bit sensitive, and its material-stage helper omits late final-logit/cache/mask stages; neither defect changes these reported comparisons because the scientific report uses explicit non-allclose stages and route changes.

### E. Scope: WARN

Scope is one OLMoE revision, one RTX 5090, one custom cached-decode runtime, six outcome-enriched selected states, repeat calls inside one process, four teacher-forced two-step propagation events, and eight unique prevalence targets reused across three widths. GPU isolation was sampled, not a proof of clock, thermal, host, or every sub-interval exclusivity. The result cannot support population incidence, arrival-pattern causality, free-running quality, native serving, latency, capacity, controller, second-model, or multi-GPU claims.

### F. Evaluation type

`self_supervised_proxy_over_real_gpu_execution`: controlled model-output conformance, not `real_gt`, human evaluation, or request-level serving evaluation.

## Findings and disposition

- P0: none.
- P1, near-tie amplification: corrected to descriptive association.
- P1, expert grouping cause: corrected to “first observed at MoE output; exact operator unresolved.”
- P1, historical serial reproduction: corrected to selected batched-state closure plus stable same-pre-state A/C divergence.
- P1, source provenance: explicitly marked non-self-contained because raw `/tmp` captures are absent.
- P1, prevalence localization: corrected from material 24/24 to exact 24/24 and material 23/24.
- P2, repeat independence: bounded to three alternating-order calls inside one model load.
- P2, prevalence dependence: bounded to eight targets reused across widths; no population inference.
- P2, exact-sensitive machine classifier: deferred; report-level claims use material stages.

The sealed raw artifact report was not rewritten. `REPORT_ADDENDUM.md` carries every narrative correction. The root Resurrection report and pointer summary use the corrected wording.

A final bounded same-family recheck inspected only these corrections and returned `PASS`, P0=0, P1=0. The overall evidence status remains `WARN` because missing raw source captures are an irreducible provenance limitation, not an uncorrected claim.

## Claim impact

- Source localization: `SUPPORTED_WITH_QUALIFIERS`.
- Companion-conditioned MoE-output externality: `SUPPORTED_WITH_QUALIFIERS`; internal cause unresolved.
- Near boundary: `ASSOCIATION_ONLY`.
- Propagation: `SUPPORTED_DIAGNOSTIC`; teacher-forced, no token/quality claim.
- Prevalence: `SUPPORTED_BOUNDED`; exact 24/24 versus material 23/24 explicitly separated.
- Latency/capacity/controller: `NOT_RUN`.
- Method/system GO: `0`.

## One next smallest experiment

Reproduce one frozen steady and one frozen bursty A/C/D state in a compatible native runtime and directly observe per-expert `M`/order, kernel choice, individual expert output, and combine/reduction order. This tests transfer and disambiguates the unresolved operator without implementing a controller.

This review is same-family and provisional. It is an integrity safety net, not independent model-family confirmation.
