# Longrun A Source Localization Report

Date: 2026-08-13
Canonical bundle: `artifacts/longrun_A_execution_conformance/20260812T204037Z/`

## Verdict

**Main verdict: `MIXED_SOURCE_IDENTIFIED`.** In this OLMoE BF16 custom cached-decode runtime, all six selected batched source states closed and their same-pre-state A/C differences were stable within one model process; exact reproduction of every historical serial route was not established. The first non-allclose A/C difference is the layer-0 self-attention output in all six selected events. A second, independently controlled C/D difference first appears at the MoE output under changed companion identity while width and the full KV-length/padding vector are fixed; the exact internal MoE operator is not measured.

`NEAR_BOUNDARY_ASSOCIATION` is supported as a secondary description: 13 of 18 A/C route-flip layers fall within the frozen `0.01` selection-boundary margin. The 18/18 lost/gained-expert order crossings follow from how a route flip is defined and are not independent evidence of causal amplification.

**Paper-direction verdict: `PIVOT_TO_EXECUTION_CONFORMANCE_MEASUREMENT`.** No conformance-preserving method, native-runtime transfer, or action controller was tested.

## Evidence integrity

- Evidence tier: `CUSTOM_CONTINUOUS_RUNTIME` on one RTX 5090; not native serving.
- Model/runtime: OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16, Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6.
- Six source-selected events: three steady and three bursty. Their raw capture hashes were validated when the runs executed, but the referenced `/tmp` capture directories are no longer retained locally, so original-event provenance is not independently replayable from this bundle alone.
- Four arms per event and three measured repeats per arm; arm order alternated within one loaded model process. These are repeat calls, not independent process restarts.
- Same-arm target tensors, routes, final logits, output cache, and tokens were repeat-stable for every arm/event.
- Arm B clone rows were exact for route, router logits, final logits, and post-step cache.
- Arm C closed source input tokens, all request/layer routes, and predicted tokens against the original ledger.
- Historical serial-route identity was diagnostic rather than a validity condition. `steady:olmoe-dev-steady-000:1` did not match its recorded historical serial route in any repeat, so this report does not claim exact all-event historical serial reproduction.
- The GPU process monitor sampled 377 times with no isolation violation.
- Five D events replaced all 3/3 companions with exact-length different-document states. `steady-003:4` replaced only 2/3 because no distinct exact-length candidate existed; its C/D identity result is therefore partial.
- Failed attempt `201023Z` is retained as `INVALID_HARNESS_CONTROL`; partial attempt `201415Z` is retained as `INVALID_GPU_ISOLATION`. Neither contributes a measurement.
- Complete pre-attention run `203419Z` is retained as corroboration; `204037Z` adds the single preregistered attention-output localization point and is canonical.

## First-divergence evidence

All entries below are stable over three repeats. “First” is the first non-allclose tensor or route stage. Route counts are target-row layer-membership flips.

| Event | A/B first; flips | A/C first; flips | C/D first; flips | A/C near-boundary flips |
|---|---:|---:|---:|---:|
| steady-000 step 1 | attention output L0; 1 | attention output L0; 1 | MoE output L0; 0 | 1/1 |
| steady-002 step 5 | MoE output L0; 0 | attention output L0; 10 | MoE output L0; 10 | 7/10 |
| steady-003 step 4 | MoE output L0; 0 | attention output L0; 2 | MoE output L2; 1 | 2/2 |
| bursty-000 step 0 | attention output L0; 0 | attention output L0; 1 | MoE output L2; 0 | 1/1 |
| bursty-001 step 0 | attention output L0; 0 | attention output L0; 1 | MoE output L3; 0 | 1/1 |
| bursty-002 step 0 | attention output L0; 1 | attention output L0; 3 | MoE output L0; 0 | 1/3 |

The target residual entering decoder layer 0 remains equal in every A/C event. The first attention-output max-absolute deltas are BF16-scale (`3.05e-5` to `2.44e-4` for A/C), but the following normalization expands the observed pre-router delta up to `0.0117`. This explains why the historical first route flip may occur at layer 4 or later even though its causal numerical trajectory diverges at layer 0.

The controls separate three contributions:

1. **Width-only execution:** A/B causes two route-flip layers across two of six events and stable final-logit differences in all six. Width alone is real but insufficient to explain the 18 A/C flips.
2. **Heterogeneous KV/padding physical shape:** A/C and A/D have the same route-flip count in four events, while C/D has no route flip there. Their first A/C difference is attention output, establishing a companion-independent physical-execution contribution.
3. **Companion identity / MoE-output externality:** C/D attention outputs stay equal until a MoE output first differs. C/D changes target route membership in two events (11 layers total), dominated by steady-002 where D matches A but original companions C produce 10 flips. This establishes a stable companion-conditioned MoE-output effect in this implementation, not its internal grouping/kernel mechanism and not semantic cross-request attention.

## Top-k boundary

Across the 18 A/C route-flip layers:

- 13/18 are associated with the frozen `0.01` kth-versus-(k+1) margin;
- the minimum/median/maximum smaller-side margin is `0 / 0.00390625 / 0.0546875`;
- 18/18 contain a lost/gained expert logit-order crossing.

The evidence therefore supports frequent near-boundary association, but five flips lie outside the frozen threshold. Causal amplification was not independently tested, and a pure near-tie explanation is rejected.

## Lightweight prevalence

The separate bundle `artifacts/longrun_A_execution_conformance/prevalence/20260812T205319Z/` tests four natural heterogeneous step-0 targets per captured workload at widths 2, 4, and 8. It contains 24 A/C cases with three alternating-order repeats each. All same-arm fingerprints were stable; the sampled GPU monitor passed 450 samples with no violation. The historical width-4 source batches closed all-row input, route, and token identity in 24/24 repeats.

| Captured workload | width 2 | width 4 | width 8 |
|---|---:|---:|---:|
| steady | 0/4 targets | 0/4 targets | 0/4 targets |
| bursty | 4/4 targets | 3/4 targets | 3/4 targets |

The bursty cases contain 6, 5, and 5 route-flip layers respectively; none changes the predicted token. The first **exact** A/C difference is layer-0 attention output in 24/24 cases. The first **material** difference is layer-0 attention output in 23/24; `steady:w2:olmoe-dev-steady-003` first becomes non-allclose at the pre-router hidden state. Thus the phenomenon is not confined to one selected event or to width 4, but neither is it unconditional: the tested steady step-0 requests have numerical differences without route membership flips. Because request/document sets differ between the two captures, this is a bounded workload/regime comparison, not a causal arrival-pattern comparison or population incidence estimate.

## Downstream propagation

Four preregistered events continued from each arm's own target KV for two serial, companion-free, teacher-forced steps. Cache length, mask content, and cross-arm non-aliasing closed before and after every step; all propagation fingerprints were stable across repeats.

- Arm C versus A retained final-logit differences in 8/8 later steps.
- Max absolute final-logit delta ranged from `0.09375` to `0.625`; relative L2 ranged from `0.00575` to `0.02321`.
- Route membership still differed in 6/8 later steps, totaling nine layer flips.
- Predicted next token differed in 0/8 later steps.

Propagation verdict: **`PERSISTENT_LOGIT_EFFECT` with frequent persistent route effects; no `TOKEN_OUTCOME_EFFECT`.** The target-step perturbation enters arm-specific KV and remains observable after companions are removed. This does not by itself establish quality impact.

Instrumented whole-model-call times were recorded, but the hooked custom eager calls are not a benchmark and have no arrival/queue/request-completion denominator. **`REQUEST_LATENCY_EFFECT` is not established.**

## Claim ceiling and exclusions

Measured: same-pre-state causal execution conformance from six source-selected batched states, target-row route/logit/KV propagation, controlled A/B/C/D effects, and bounded width-2/4/8 step-0 prevalence for one model/runtime/GPU. Fresh outputs and post-run seals are self-contained; original capture provenance is hash-recorded but not locally replayable because those raw `/tmp` sources were not retained.

Not measured: `NATIVE_RUNTIME_TRANSFER_NOT_RUN`; a second model was not run; multi-GPU EP/NCCL, production TTFT/TPOT/P99, request-level latency, semantic quality, safe capacity, an action-conditioned running-set budget, and SLO-goodput were not measured.

## Required final fields

- Strongest baseline/control: same-pre-state Arm A, width-only clone Arm B, original Arm C, and exact-length different-document Arm D.
- Oracle/headroom: `NOT_RUN`; this was not a capacity Oracle.
- Failure category: `MEASUREMENT_ONLY`, not a method GO.
- Resurrection condition: reproduce the same attention/MoE conformance chain in a representative native serving runtime or show that it disappears under an equivalent native padding/cache path.
- One next smallest experiment: run one frozen steady and one frozen bursty A/C/D source check in one compatible native serving runtime, retaining the same target token/KV/position and exact-length companion controls and directly observing the internal MoE grouping/operator boundary. This two-event transfer decides whether the thesis result is generic execution conformance or a custom-runtime artifact.

## Mechanism and novelty addendum

The internal expert-execution grouping hypothesis is not a directly measured operator. The experiment establishes that, with global width and the full KV-length/padding vector fixed, the first **observed** C/D difference is at the MoE output. It did not record per-expert physical `M`, token order, kernel/config selection, individual target-expert outputs, or pre/post-combine accumulation. Sorting, expert GEMM shape/order, combine order, `index_add`, or another MoE implementation detail therefore remain unresolved.

An adjacent workspace process left a provisional novelty note at `idea-stage/resurrection/NOVELTY_CHECK.md`. It is not part of this Goal's scientific verdict and is not used to upgrade novelty. This report's only claim is the bounded measurement above; any novelty claim requires its own verified literature workflow and native-runtime mechanism isolation.
