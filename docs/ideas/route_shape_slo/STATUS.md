# RouteShape-SLO status and evidence ledger

**Date:** 2026-08-12
**Branch inspected:** `agent/publish-current-moe-code`
**HEAD inspected:** `d905e11`
**Verdict:** `BLOCKED_RUNTIME_NOT_REPRESENTATIVE`

## Repository fact inventory

| Object | Existing capability | Evidence type | Reusable? | Decisive gap |
|---|---|---|---|---|
| Continuous cached decode | `capture_native_routes.py` performs batch-one cached decode; `capture_continuous_decode.py` implements a mutable active set and batched KV decode. | Producer capability is `[Not measured]`; existing batch-one outputs are `[Observed isolated GPU primitive]`. | Producer code and identity tests: yes. | No completed representative `CAPTURE_COMPLETE + routes + decode_batches + request_ledger` bundle. |
| Route identity | `core.py` binds request/input event/decode step/layer/top-k/expert/gate weight; both capture paths conserve these identities. | Implemented schema; batch-one GPU route observations. | Yes. | Formal continuous multi-request artifact is absent. |
| Request/window ledger | Continuous producer records request lifecycle and per-model-call batch membership. The new builder derives stable window IDs. | Code capability only for current P0 target. | Yes. | No native serving window/queue artifact in the workspace. |
| Step/E2E latency | Continuous producer synchronizes CUDA around a whole model call; StableBatch has CUDA-synchronized whole-step and expert-stage timing. | `[Observed isolated GPU primitive]` for StableBatch. | Whole-step smoke: yes. | No serving queue, admission, TTFT/TPOT or request-E2E denominator; no paired hook-overhead A/B. |
| Queue/load telemetry | Continuous producer exposes active IDs, scheduled IDs, pending count and prior KV lengths. | `[Not measured]` on a representative runtime; derivable development telemetry only. | Partly. | `pending_request_count` includes not-yet-arrived work and is not queue depth; StableBatch has no observed queue/running counter. |
| Expert service curve | Isolated selected-expert CUDA microbenchmark and StableBatch expert-call ledger exist. | `[Observed isolated GPU primitive]`; analytic smoke also exists. | Mechanism sanity only. | No natural serving surface or EP/A2A/queue capacity result. |
| Natural workload | Frozen WikiText documents and BurstGPT arrival source are present for 128 requests/model. | `[Not measured]`; this is an input manifest, not runtime evidence. | Yes. | The two formal continuous cells were not run; deadlines are fixed arrival + 60 s, not an observed calibrated SLO. |
| SLO definition | Protocol text and modeled deadlines exist. | `[Analytic model]` only. | Only as a template. | No frozen baseline-derived serving SLO and no completed-token violation ledger. |
| Single-GPU evidence | OLMoE and LLM-jp batch-one routes; OLMoE StableBatch whole-step/expert timing on RTX 5090. | `[Observed isolated GPU primitive]`. | P0/P1 pipeline smoke only. | Not continuous serving or admission-control capacity. |
| Multi-GPU EP evidence | Simulators and analytic replays only. | `[Analytic model]` / `[Synthetic fixture]`. | No for claims. | No observed EP, dispatch/combine, NCCL/RDMA, receiver pressure or multi-rank DAG. |

`capture_native_routes.py` is an identity-safe route producer, not a
wall-clock serving-window producer. Its own metadata excludes natural
continuous batching, dispatch/expert/combine timing, latency, energy and an
end-to-end denominator. Git-AI lookup found indexed authorship but no usable
prompt transcript, so this boundary comes from source and current authority,
not inferred author intent.

## P0 measurement-surface result

| Required field | Current representative artifact? | Notes |
|---|---:|---|
| request ID / decode step / route layer / expert ID | Partial | Observed in batch-one and StableBatch artifacts. |
| window ID / timestamp | No | Derivable for non-serving smoke, not observed as a native control window. |
| running sequences / active tokens / queue depth | No | StableBatch fields are sentinels/surrogates; future pending arrivals are never treated as queue. |
| batch tokens / mean and max KV | Partial | Derivable from custom producer/roster, not verified serving counters. |
| step service time / tokens completed | Partial | Whole-step isolated GPU timing; executed decode rows are not request completions. |
| gate weight | No in StableBatch smoke | BCRD route-v3 supports it; selected StableBatch ledgers do not. |
| instrumentation overhead | No | No paired hooked/unhooked run. |
| calibrated SLO | No | Existing deadline is not a serving SLO measurement. |

Mechanical P0 status: `BLOCKED_RUNTIME_NOT_REPRESENTATIVE`.

## P0/P1 smoke actually run

Source: sealed OLMoE BF16 eager RTX 5090 StableBatch run, filtered to
`arm=native_variable_m`, `phase=measured`, `repeat=0`.

- 256 measured windows, 2,048 request-steps, 16 layers, 64 experts, top-k 8.
- 16 request-disjoint proxy split units from one arrival replay.
- 135/45/60 aligned train/validation/test window pairs.
- Request and document overlap across splits: zero.
- M0--M3 consume only state available by the end of window `t`; M4 alone uses
  route features from `t+1`.
- M1 workload-only P95 pinball loss: `0.97172356`.
- M3 workload-plus-route P95 pinball loss: `0.94481007`.
- Apparent relative change: `+2.7697%`; dangerous-underprediction rate is
  `1/60 = 1.6667%` for both.
- M4 future-route predictor pinball loss: `0.73516210`, but this is only a
  future-route latency-prediction diagnostic, not a capacity-action Oracle.

These numbers are **not P1 evidence**. They are one-model, one-replay,
teacher-forced, non-serving, development-split smoke numbers with no native
queue, no capacity intervention and no overhead A/B. Therefore neither the
nominal `<3%` stop rule nor any positive trend may be applied.

P1 status: `SMOKE_ONLY_NOT_SCIENTIFICALLY_ELIGIBLE`.
P2 guard: `BLOCKED_P1_NOT_ELIGIBLE`; no Oracle replay executed.
P3 guard: `BLOCKED_P2_NOT_PASSED`; no controller executed.

## Formulation collision

The frozen action is only **`active_token_budget[t+1]`**, the prompt default.
The current harness has no verified active-token actuator, so P2 is explicitly
implementation-blocked; this smoke executes no capacity action. The missing
actuator is not used to silently substitute a max-running fallback, and no
second action is explored in parallel.

- BCRD chooses an equivalent physical expert replica and bounded seal/hold for
  already-routed contributions. RouteShape-SLO does neither.
- DEPA already includes launch/defer/reject and admitted-set composition. A
  scalar service-level active-token bound is therefore a narrow DEPA action.
- If later route value is mediated by replica fragmentation, it is a BCRD
  service-estimation submodule.
- If route adds no value beyond queue/token/KV/per-expert queue summaries,
  choose a simple token controller or stop.
- If only future route helps, retain it as Oracle/diagnostic.

Current relationship decision: no positive fold is yet authorized. Conditional
default is **DEPA submodule** for admission value, **BCRD submodule** only for
fragmentation-mediated value. Independent status requires P1, P2 and P3.

## Measured / inferred boundary

- **`[Observed real runtime]`:** not measured for representative continuous serving.
- **`[Observed isolated GPU primitive]`:** RTX 5090 OLMoE isolated teacher-forced whole-step/expert timing;
  earlier OLMoE/LLM-jp batch-one route captures.
- **`[Offline replay]`:** current feature extraction and ridge smoke; DEPA/BCRD
  action replays remain separate assets and are not capacity evidence.
- **`[Analytic model]`:** prior EP congestion and service/Oracle tools only.
- **`[Synthetic fixture]`:** smoke fixtures and some route/service fixtures only.
- **`[Not measured]`:** incremental route signal on representative runtime, safe
  capacity, future-route action headroom, causal historical-route benefit,
  controller overhead, native queue, calibrated SLO, multi-GPU EP.

## Answer to the research question

**Unknown and currently unmeasurable from the checked artifacts.** The smoke
does not show that route information changes safely supportable capacity; it
also does not show that route is merely a complicated restatement of token or
queue state. One bounded representative runtime capture is required before
either conclusion is defensible.

## Next smallest experiment

Only one experiment is authorized next: the frozen OLMoE continuous-decode P0
capture described in `NEXT_EXPERIMENT.md`. It qualifies the existing producer
only; paired overhead/native-scheduler qualification remains later. Do not run
LLM-jp, P1/P2/P3, an 8xA100 job, or a second predictor family first.
