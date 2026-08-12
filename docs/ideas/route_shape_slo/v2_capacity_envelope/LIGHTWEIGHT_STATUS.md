# Route-Conditioned Capacity Envelope lightweight status

**Date:** 2026-08-13

**Base repository HEAD:** `agent/publish-current-moe-code@beb08ee4e25dbf93ea3e199db671080c4d3ecea5`

The v2 files are intentionally still untracked, so the base commit alone cannot
reconstruct this development run. The retained runner/config hashes are in the
canonical bundle; the recovered executed-source hashes are:

| Executed v2 source | SHA256 |
|---|---|
| `capture_dev_continuous_decode.py` | `17ec528711154fea3e4db228401fd8bbfb2d01289729a0c86903f3afe7a99f97` |
| `qualify_dev_capture.py` | `4788cfaf51cc81124254d3fe14b43be162fd51db5a169e02a4e67abeaaf57b68` |
| `build_capacity_windows.py` | `8b02fed533997e6d7dea79dc95b048c7a616a36c7cd0d7ad7938724123d19ee7` |
| `analyze_capacity_signal.py` | `d6422bbd7d32f502329ee0a5492cbc3df98daac2344964e3018fe786168e2ffb` |
| `run_dev_capture.sh` | `1b47bbfeefb55454d341358eaffe78aca99a47e4bd219202f4b6339f2cd2727d` |

Local qualifier/builder code now adds fail-closed capture-hash and fixed-batch
proxy-label checks; those post-run hardenings do not rewrite the canonical
bundle. The current wrapper hash is
`5cda4159c94662e07efc07ff02ba42df31f1c9c5268b37d6757ff378df490f86`;
it differs from the recovered executed bytes only by a post-run docstring
clarification. The canonical bundle records the executed path but not wrapper
bytes, so this source provenance is recovered evidence rather than bundle-closed
provenance. The checked-in analyzer is byte-identical to the recovered executed
analyzer, and the canonical `report.md` remains the original GPU-produced file.
Its timing/M0--M4 values must be interpreted with this companion status, which
records the non-isolation caveat without rewriting canonical evidence.

**Verdict:** `PIVOT_TO_EXECUTION_CONFORMANCE`

## Repository fact inventory

| Object | Current capability | Reusable now | Actual v2 gap |
|---|---|---|---|
| Continuous decode producer | Custom Transformers arrival-ordered prefill, mutable active set, and left-padded batched cached decode. | Yes, as a development measurement substrate. | It is not a native serving scheduler. |
| Route identity | Native router logits close request/document/decode/layer/top-k/expert/gate-weight identity. | Batched route rows remain identity-complete. | Serial and batched execution did not choose identical experts. |
| Queue/running telemetry | Batch rows expose all active IDs, scheduled IDs, pending count, width, and timestamps. | Only with custom names. | `pending_request_count` includes future arrivals; there is no vLLM queue. |
| KV/batch shape | Per-request logical KV lengths and left padding are recorded. | Logical and padded token extents are usable. | No backend-native allocator bytes. |
| Step/request latency | CUDA-synchronized whole model-call plus request arrival/prefill/completion timestamps. | Development diagnostic only. | No expert-stage timing or representative TTFT/TPOT denominator. |
| Capacity actuator | Fixed `max_batch_size` selects the scheduled pure-decode batch. | Static running-set proxy only. | No dynamic admission actuator or action-conditioned rerun. |
| SLO definition | Calibration-fold baseline P95 x 1.10 is available inside P1. | Exploratory diagnostic. | It is not a serving SLO. |
| Two-model evidence | OLMoE and LLM-jp revisions are frozen. | OLMoE was sufficient for this first pivot. | No second-model conformance result. |

## Action semantics

Each scheduled request contributes exactly one decode token, therefore in this
producer:

```text
scheduled_active_tokens[t] = scheduled_request_count[t] = batch_size[t]
```

The only eventual capacity candidate would be `running_set_budget`, a
DEPA-style admission/running-set action. It remains unauthorized and
unimplemented because execution conformance failed before any action Oracle.

## Completed development evidence

The retained canonical development bundle is
`artifacts/route_capacity_envelope/dev/20260812T170512Z/` and contains exactly
the six configured files. It covers two document-disjoint 16-request episodes,
eight decode steps per request, 64 windows total, OLMoE revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, and one RTX 5090.

The fixed-batch router-output OFF/ON proxy passed token, logit, completion, and
ON-route stability. Median model-call overhead was `+0.0511%` and loop-wall
overhead was `+0.9313%`, both below the frozen `2%` limit. It did not replay
manifest arrival timing (`same_arrival_trace=false`), so it is not a complete
A3 pass. The serial audit observed batch-dependent expert assignment in both
regimes despite exact token parity: layer-assignment match was `96.6797%` for
steady and `94.7266%` for bursty, while whole-step assignment match was
`81.25%` and `37.50%`, respectively. This is an execution-conformance result,
not fake ground truth and not an eligible capacity result.

The diagnostic M0--M4 run completed on 62 aligned next-window examples. In the
retained run M3 reduced P95 pinball loss from `1.26975268` (M2) to `1.14749054`
(`+9.6288%`), with fold improvements of `+1.7428%` and `+27.7089%`; dangerous
underprediction stayed unchanged. A superseded run using the same workload and
config but a different Python environment produced `-24.0388%`. The logs do not
prove that the two pilots were isolated repeats, so the sign flip is an
uncontrolled stability warning, not a second capacity result. Neither value is
promoted to route-conditioned capacity evidence.

The superseded `170328Z` bundle was removed from the canonical tree and was
copied at audit time to ephemeral local quarantine at
`/private/tmp/rce-superseded-170328Z/`. The
sole retained `170512Z` bundle was selected because it applies the required
conformance veto, not because its nominal M3 metric is favorable. Both runs
reproduced exactly the same serial-vs-batched conformance fractions and
difference examples. The uncontrolled repeat conditions block timing/M0--M4
interpretation but do not erase the execution-conformance finding.

The fresh same-family integrity audit is `WARN` (provisional), not `FAIL`: it
found no fake target, self-normalized score, phantom number, or executed action.
It flags the post-hoc canonical selection, uncontrolled repeat conditions, and
an in-audit `report.md` byte mutation. After the mutation was recorded, the
report was restored exactly to the original GPU-produced SHA-256
`8ca96e8096796da966879151d49c1b8e048eb36a2117ce99b975339f062f7c1b`;
the warning and claim ceiling remain. No standalone audit/trace artifact is
retained, honoring this pilot's lightweight no-`.aris` constraint.

## Evidence Table

| 问题 | 观察结果 | 证据类型 | 当前含义 |
|---|---|---|---|
| Route 是否有 M2 之外的增量 | Retained diagnostic was `+9.63%`, but a same-workload/config run in another Python environment was `-24.04%`; route assignment itself was batch-dependent. | Real RTX 5090 custom-runtime diagnostic | Incremental capacity signal is not identified. |
| 信号是否进入 whole-step/request | Only CUDA-synchronized whole model-call and request completion were recorded; no expert-stage survival ledger exists. | Observed whole-call/request fields | RCBA-Lite mechanism survival remains unmeasured. |
| 动作是否非退化且可执行 | Active-token budget degenerates to scheduled running-set width; current cap is static only. | Source/runtime semantics | `running_set_budget` is the sole candidate, not a dynamic actuator. |
| Oracle 是否有容量空间 | No action-conditioned rerun was authorized. | `NOT_RUN` | No headroom or controller claim. |
| 简单策略是否已覆盖收益 | M2 was present, but M3 direction was execution-unstable and dangerous underprediction did not improve. | M0--M4 diagnostic | Cannot distinguish a real route residual from unresolved batch-context dependence. |
| 应独立还是并入 BCRD/DEPA | Capacity interpretation is blocked before module placement. | Frozen decision tree | Pivot to execution conformance; do not claim independent, BCRD, or DEPA value yet. |

## Measured boundary

- **real GPU:** one RTX 5090, OLMoE BF16, two 16-request episodes, 64 decode windows.
- **custom continuous runtime:** real batched cached decode and raw route/request/KV/whole-call telemetry; not native serving.
- **offline replay:** none; `--offline` only disabled network access while the
  custom runtime executed fresh model calls.
- **analytic/proxy:** pure-decode action degeneracy, padded-KV token extent, and diagnostic M0--M4 only.
- **not measured:** safe capacity, dynamic action headroom, expert-stage survival, controller gain, native queue/TPOT, second model, or multi-GPU behavior.

## Best current story

OLMoE's generated tokens remained identical, yet serial and batched execution
selected different experts for a nontrivial fraction of audited layers. That
batch-dependent conformance and the uncontrolled cross-run sign flip jointly
prevent interpreting historical route features as a capacity variable; their
causal relationship is not identified. The evidence therefore supports an
execution-conformance measurement problem, not a route-conditioned capacity
controller. The old 2.77% result is not inherited.

## Next smallest experiment

**Unique uncertainty:** does the observed serial-vs-batch-4 route divergence
occur in pre-top-k router logits or only when top-k turns near-ties into expert
identities?

This is the target invocation for the checked-in diagnostic. It first verifies
the sealed capture hashes and an idle GPU, then reconstructs states from the
frozen workload/token trace; it does not assume serialized KV/cache states.

**Execution status: `NOT_RUN`.** The prepared tool SHA-256 is
`ab4cb2e1f3091d55f8f1952b00de4a45673f52465211d8d70a6c19e0a816cd1b`;
it binds captured software/tokenizer/prompt identities and polls GPU process
isolation every `0.20 s`. Its CPU regression suite and final read-only code
review pass, but no GPU result is claimed from it in this bundle.

```bash
cd /root/autodl-tmp/rce_capacity_envelope_beb08ee
/root/miniconda3/bin/python3 \
  docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/compare_serial_batched_router_logits.py \
  --capture-dir /tmp/bcrd-gate0-smoke-rce-steady-20260812T170512Z \
  --requests 4 --decode-steps 8 --batch-size 4 --repeats 3 \
  --offline \
  --output /tmp/rce-v2-router-logit-conformance-$(date -u +%Y%m%dT%H%M%SZ).json
```

- Model/revision: `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- Requests/arrival: reconstruct the first four audited requests from the frozen
  steady workload and captured token trace, then bind all 32 comparisons by
  request ID and decode step. Arrival timing is not replayed because batch width
  is the isolated execution condition; compare width one with width four, three
  repeats each.
- Action: none; batch width is an execution condition, not a capacity action.
- Expected memory/time: conservatively reserve 24 GiB on one RTX 5090; at most
  18 GPU minutes. Actual peak memory remains unmeasured.
- `PROMISING` for the conformance pivot: tokens remain identical, but matched
  pre-top-k logits or top-k margins change consistently with batch width and
  explain the selected-expert differences.
- `STOP`: tokens diverge, within-arm repeats are unstable, the GPU is not
  isolated, the sealed raw input is unavailable, or outputs cannot bind to the
  same reconstructed request/decode-step identity.
- Output: one JSON diagnostic under `/tmp`; do not create another Route Capacity
  canonical bundle. This is an execution-conformance diagnostic only; it does
  not authorize a capacity action.

Direct answer: **current evidence does not show that route shape changes safe
capacity, and it also does not establish that route is merely a restatement of
queue/token/KV/per-expert load. The dominant observed phenomenon is
batch-dependent execution conformance, so capacity interpretation must stop.**
