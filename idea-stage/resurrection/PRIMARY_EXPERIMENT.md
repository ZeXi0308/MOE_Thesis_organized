# Primary Experiment — real-event execution-conformance source localization

> Frozen: 2026-08-13
> Scope: development diagnostic only; this document does not update `docs/current/README.md` or any sealed verdict.

## Execution addendum

The Primary was executed through the stricter four-arm harness in `idea-stage/longrun_A_execution_conformance/`, not through the three-arm runner frozen below. The canonical run added a width-only clone control and an exact-length different-document companion control, captured attention output at the one preregistered source-localization point, retained every repeat, and propagated four events for two arm-specific KV steps.

Canonical evidence:

- `artifacts/longrun_A_execution_conformance/20260812T204037Z/`
- `artifacts/longrun_A_execution_conformance/prevalence/20260812T205319Z/`
- `idea-stage/longrun_A_execution_conformance/SOURCE_LOCALIZATION_REPORT.md`

Result: `MIXED_SOURCE_IDENTIFIED`, evidence tier `CUSTOM_CONTINUOUS_RUNTIME`, scientific status `MEASUREMENT_ONLY`. All six selected events were repeat-stable. A/C first differed at layer-0 attention output; C/D first differed at MoE output. The separate 24-case probe found route flips only in the sampled bursty cases. No token change, native-serving transfer, request latency, capacity action, or method GO was established.

The runner and rules below remain a code-audited reproducibility path for the original A/B/C formulation; they were not the source of the GPU result. Do not mechanically combine their arm labels with the canonical four-arm result.

## Repository reality snapshot

- Local branch: `agent/publish-current-moe-code`.
- Local HEAD: `203a100967f89c1e9e6cdfb2238240c4120eb314` (`Add route-conditioned capacity envelope pilot`).
- Remote `origin/agent/publish-current-moe-code`: `203a100967f89c1e9e6cdfb2238240c4120eb314`, refreshed read-only on 2026-08-13.
- Pre-audit dirty state: only the user-owned untracked `AGENTS.md`; this audit adds only `RESURRECTION_AUDIT.md`, `idea-stage/resurrection/`, and at most one `artifacts/resurrection/<timestamp>/` bundle.
- Authority read: `AGENTS.md`, `docs/current/README.md`, `docs/ideas/README.md`, the target Route Capacity README/status/result, the specified archive indexes/correction audit, and `docs/current/JOINSTREAM_FINAL_FREEZE_2026-08-10.md`.

Frozen facts inherited:

1. Formal system/method GO count remains zero.
2. The retained development capture observed serial-vs-batched expert-assignment differences with token parity, but it did not localize their source.
3. Batch-dependent route invalidates a fixed-route **causal action counterfactual**; it does not automatically invalidate the **observational** question inside the same batched runtime.
4. The retained observational M3-vs-M2 result has an uncontrolled `+9.6288% / -24.0388%` sign flip and has no capacity authority.
5. The old source diagnostic selected the first four steady requests and ran independent serial/batched trajectories. It did not share the target pre-step state, include bursty events, run Arm C, or capture the first-divergence chain.

## One question, weakest link, and claim ceiling

**One research question:** when a retained real difference event is reconstructed, does serial width 1 versus the original width-4 batch still cause a stable expert-assignment change when the target request starts from the exact same pre-step KV/token state; if so, where does the first stable difference appear?

**Weakest causal link:** the historical serial and batched trajectories had already evolved independently before later decode events. Their route difference therefore conflated immediate batch execution with earlier KV/hidden-state divergence. That link must be closed before interpreting route as either a stable observational signal or an action-conditioned variable.

**One experiment:** source-bound A/B/C, one step per retained event, three repeats per arm.

**Allowed claim ceiling:** `CUSTOM_CONTINUOUS_RUNTIME_DEVELOPMENT_DIAGNOSTIC`. A positive result may localize an execution-conformance effect. It cannot establish native-serving behavior, request latency, safe capacity, SLO-goodput, or a controller benefit.

Accounting boundary:

```text
request time
= queue + prefill + decode non-MoE + exposed MoE + sampling + barrier/idle
```

This experiment measures none of those mutually exclusive time buckets. It only compares model-state tensors and selected experts at one decode call.

## Frozen real-event selection

The complete selection and source row/hash bindings are in `experiments/events.json`. The producer retained at most 16 difference examples per episode, so “largest” means largest mismatch count **within recorded examples**, not a global maximum.

| Profile | Regime | Event | Recorded mismatch layers |
|---|---|---|---|
| smoke + pilot | steady | `steady-002-step-005` | 4, 9, 10, 11 |
| pilot | steady | `steady-000-step-001` | 3, 7, 8 |
| pilot | steady | `steady-003-step-006` | 1, 5, 6 |
| smoke + pilot | bursty | `bursty-002-step-000` | 4, 5, 9 |
| pilot | bursty | `bursty-001-step-003` | 11, 12 |
| pilot | bursty | `bursty-000-step-001` | 1, 13 |

Each event is bound to `serial_audit.json`, `decode_batches.jsonl`, `request_ledger.jsonl`, and `routes.csv`, plus the capture sentinel SHA-256 values.

## Arms and invariants

From the captured batch schedule, replay only batches strictly before the selected event. Every replayed call must reproduce request order, decode step, input token, predicted token, logical KV lengths, left padding, and ordered per-layer route. Then fingerprint and clone the target pre-step cache/mask/token for every arm and repeat.

- **A — `A_serial`:** target only, width 1.
- **B — `B_original`:** target and the original three companions in their captured order. B must reproduce the captured current-step token, padding, and routes or the event is invalid.
- **C — `C_shuffled`:** same target row and same width; reverse only the three original companion rows. This preserves the companion identity set, KV-length multiset, and padding multiset.

Arm C tests companion row-order / physical-layout sensitivity. It does **not** test replacement-companion identity externality; `COMPANION_IDENTITY_EXTERNALITY` therefore remains unmeasured. Arm D is forbidden unless A/B/C leave padding versus companion layout unresolved and a later experiment is separately frozen.

Every arm runs at least three times with alternating order and the same seed, model/revision, runtime, GPU, target pre-state fingerprint, and target input token. B/C additionally keep the companion identity set and width fixed; A is intentionally width 1. A sampled `nvidia-smi` process monitor fails closed on overlap.

All A/B and B/C claims are repeat-qualified: compare A-vs-B and B-vs-C within the same repeat identity, require all three pairwise signatures to agree, and only then emit an event-level localization. A repeat-0-only comparison is never a result.

## Minimal capture chain

For only the target token and one decode call, capture:

1. pre-router hidden state;
2. pre-top-k router logits;
3. top-k boundary margin;
4. selected experts;
5. combined expert output before the residual add;
6. next-token logits.

No attention tensors, KV contents beyond the pre-state fingerprint, per-expert intermediate activations, or request-time proxy are added.

Frozen result classes:

- `ROUTER_KERNEL_SHAPE_EFFECT`: the earliest stable A/B difference in layer/signal execution order is router logits, and that candidate layer's pre-router hidden has exactly equal dtype, shape, and float32-copy SHA-256 across A/B. `allclose` alone is insufficient.
- `UPSTREAM_BATCH_CONTEXT_EFFECT`: the earliest stable A/B difference in layer/signal execution order is pre-router hidden.
- `UNRESOLVED_INPUT_DELTA_VS_KERNEL`: router logits are the first difference under the frozen numerical tolerance, but the candidate layer's pre-router hidden digest is not exactly equal; do not attribute this to the router kernel.
- `NEAR_TIE_AMPLIFICATION`: at least 75% of repeat-qualified assignment-change layers have a top-k boundary crossing with **both** A and B absolute boundary margins `<= 1e-2`.
- `NONDETERMINISTIC_RUNTIME`: any arm is unstable across repeats.
- `NOT_REPRODUCED`: B reproduces the captured event, but the shared-target-prestate A/B difference disappears.
- `COMPANION_IDENTITY_EXTERNALITY`: explicitly not measurable with the frozen shuffled-rows C arm.

The first divergence is selected globally in execution order: for each ascending decoder layer, `pre-router hidden -> router logits -> top-k margin -> selected experts -> combined expert output`, followed by next-token logits after the decoder. If B/C differs, report the secondary `COMPANION_ROW_ORDER_OR_LAYOUT_EFFECT`. If B/C does not differ, report `WIDTH_VS_COMPANION_CONTEXT_UNRESOLVED`; shuffled-row C cannot identify a `PHYSICAL_SHAPE_EFFECT` or separate width from companion context.

## Why this runner exceeds the normal exploratory LOC budget

The historical runner cannot answer the frozen question: it is steady-only, chooses the first four requests, executes two independent full trajectories, has no Arm C, and records neither pre-router hidden nor expert output. A smaller A/B microbenchmark also cannot bind a real event or distinguish width/physical-shape sensitivity from companion row layout.

The new code is limited to three necessary blocks: hash/identity-bound capture loading, deterministic prefix replay plus isolated cache forks, and the six-signal/repeat comparison. Those blocks directly close the single uncertainty above; they do not implement a controller, general tracing framework, coverage expansion, or formal sealing pipeline. The five CPU tests are limited to event identity/alignment, no-future replay cutoff, arm isolation/order, classification semantics, and repeat/cross-signal/cross-regime qualification (including exact-equal versus allclose-only pre-router hidden).

## Progressive execution

Smoke (two events, both regimes):

```bash
PYTHONPYCACHEPREFIX=/tmp/resurrection-pycache ./.venv/bin/python \
  idea-stage/resurrection/experiments/run_source_localization.py \
  --events idea-stage/resurrection/experiments/events.json \
  --profile smoke \
  --steady-capture-dir /tmp/bcrd-gate0-smoke-rce-steady-20260812T170512Z \
  --bursty-capture-dir /tmp/bcrd-gate0-smoke-rce-bursty-20260812T170512Z \
  --repeats 3 --offline --output /tmp/resurrection-source-smoke.json
```

Pilot (all six events, both regimes):

```bash
PYTHONPYCACHEPREFIX=/tmp/resurrection-pycache ./.venv/bin/python \
  idea-stage/resurrection/experiments/run_source_localization.py \
  --events idea-stage/resurrection/experiments/events.json \
  --profile pilot \
  --steady-capture-dir /tmp/bcrd-gate0-smoke-rce-steady-20260812T170512Z \
  --bursty-capture-dir /tmp/bcrd-gate0-smoke-rce-bursty-20260812T170512Z \
  --repeats 3 --offline --output /tmp/resurrection-source-pilot.json
```

- Exact model: `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16, offline cache only.
- GPU: exactly one RTX 5090; fail below 24 GiB free. The retained host previously showed 30.86 GiB free, but that is not current evidence until login succeeds.
- Planning estimate: 24 GiB reservation; approximately 20–45 GPU minutes for smoke + pilot, unverified until the corrected runner executes.
- Output: one non-overwriting smoke JSON and one pilot JSON, then mechanically combined into the single canonical artifact `artifacts/resurrection/<timestamp>/results.json`; both raw run payloads remain embedded.

## Mechanical decision rules

`INVALID` immediately if any capture hash, event row, prompt/token identity, prefix token/route/padding, B current-step reproduction, target fingerprint, runtime identity, or GPU isolation check fails.

`STOP` the current interpretation if:

- within-arm instability → `NONDETERMINISTIC_RUNTIME`;
- B reproduces the source but A/B no longer differs → `NOT_REPRODUCED`;
- only one regime reproduces and the other does not → regime-conditional diagnostic, no general route-conformance claim;
- a stable tensor difference exists but does not change assignment → measurement only.

`CONTINUE` only if at least one real event in both steady and bursty regimes has repeat-qualified A/B assignment divergence with B source reproduction, and all qualifying events across both regimes share one primary first-divergence class (`UPSTREAM_BATCH_CONTEXT_EFFECT` or `ROUTER_KERNEL_SHAPE_EFFECT`). A mixed primary class is `STOP_OR_MIXED_EVENT_RESULTS`, not a generic localization. The next step would then be controlled observational M3-vs-M2 repeats; it would **not** authorize a capacity controller.

`REOPEN causal action` only after a later running-set-budget experiment independently regenerates admitted requests, batch/KV/padding, route, queue, and completion for every budget.

If the RTX 5090 remains inaccessible after local integrity closure, Primary is `BLOCKED_AUTHENTICATION`, not NO-GO. Conditional Backup is the true in-loop Verify Precision eligibility design; Wildcard is offload/HBM-pressure Prefetch only after a real miss regime exists.
