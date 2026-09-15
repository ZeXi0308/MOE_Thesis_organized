# Resurrection Summary Artifact

This is the single non-canonical summary bundle allowed by the Resurrection Audit. It points to retained raw evidence; it does not copy, mutate, reseal, or replace it.

## Verdict

`PRIMARY_RESURRECTION_SIGNAL_FOUND`, with scientific status `MEASUREMENT_ONLY` and formal method/system GO count unchanged at zero.

In one OLMoE BF16 custom cached-decode runtime on one RTX 5090 with sampled process-isolation checks, six source-selected batched states were reconstructed from the same target pre-step state. All arms were stable across three alternating-order calls within one loaded model process. The result is `MIXED_SOURCE_IDENTIFIED`: heterogeneous KV/padding execution first changes the target at layer-0 attention output, while exact-shape companion replacement first becomes observably different at MoE output. The exact internal MoE operator remains unresolved. Near-tie margins are associated with many flips, but causal amplification was not shown.

Four events were advanced for two policy-specific cached-decode steps. Final logits remained different in 8/8 steps and route membership in 6/8; greedy next tokens changed in 0/8. A separate 24-case width-2/4/8 probe found route flips in the sampled bursty cases but not the sampled steady step-0 cases. This is a conditional execution-conformance phenomenon, not a prevalence estimate or arrival-pattern causal result.

## Evidence pointers

- Main canonical bundle: `artifacts/longrun_A_execution_conformance/20260812T204037Z/`
- Bounded prevalence bundle: `artifacts/longrun_A_execution_conformance/prevalence/20260812T205319Z/`
- Scientific report: `idea-stage/longrun_A_execution_conformance/SOURCE_LOCALIZATION_REPORT.md`
- Route Capacity implication: `idea-stage/longrun_A_execution_conformance/CAPACITY_IMPLICATION.md`

Both `POST_RUN_SEAL.json` manifests verified exactly (35 and 79 files, no missing, extra, size, or SHA-256 mismatch). All JSON files in both completed bundles parsed successfully. Invalid and superseded attempts remain retained. The raw `/tmp` captures used for event selection were hash-checked during execution but are absent locally, so original-event provenance is not independently replayable from this summary or the canonical bundles.

## Claim boundary

The evidence supports a repeat-stable, MoE-specific diagnostic over six source-selected batched states in this one custom runtime. It does not establish exact all-event historical serial reproduction, a specific expert-grouping cause, native-serving transfer, semantic quality impact, request latency, capacity, SLO-goodput, multi-GPU behavior, or a useful controller. Route Capacity remains observationally open and its causal action Oracle remains paused.

The next and only experiment is a two-event native-runtime MoE mechanism disambiguation, using the same OLMoE revision and target prefixes. It separates per-expert physical `M`/token order from combine/reduction order. Failure of runtime/model compatibility is `NATIVE_RUNTIME_TRANSFER_BLOCKED`, not permission to swap models post hoc.
