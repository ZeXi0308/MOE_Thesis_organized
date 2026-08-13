# Longrun A Exact-Event Experiment Design

## Frozen question

For a real historical serial/batched mismatch, reconstruct the original batched pre-step state and branch from that one target state. Determine the first stable causal difference and whether it persists after companions are removed.

The historical serial route is event-selection metadata, not ground truth: historical serial and batched trajectories had different prior state evolution. Arm A is the operational same-pre-state serial reference.

## Accounting and evidence

The mutually exclusive interpretation chain is:

```text
same token + position + logical target KV
-> residual input
-> self-attention output (single source-localization follow-up point)
-> post-attention normalization / pre-router hidden
-> native BF16 router logits
-> ordered top-k and membership
-> MoE sublayer output (before decoder residual add)
-> final next-token logits
-> arm-specific KV
-> later serial teacher-forced route/logits/token
```

Instrumented model-call time is reported only as a custom eager whole-call diagnostic. Hook copies are excluded from the timed call but the hook itself still perturbs execution; it is not serving latency.

## State reconstruction

1. Prefill each request in the original four-request batch.
2. For every step before the selected event, execute the original batch with captured input tokens.
3. Verify batch membership, decode step, logical KV lengths, input tokens, all request/layer routes, and predicted tokens against the hash-bound source ledger.
4. Immediately before the target call, deep-clone every layer's K/V tensor for every arm.
5. Assert pairwise `untyped_storage().data_ptr()` non-aliasing within rows and across arms.
6. Arm C must reproduce the target source route and predicted token. Failure is `BLOCKED_STATE_RECONSTRUCTION`, not a method NO-GO.

## Arms

- A — target serial: width 1 and the target's natural physical KV length.
- B — width-only: original width, all rows are independent clones of the target with the same token, logical length, position, and semantics.
- C — original companions: original row order, request identities, forced tokens, logical lengths, physical max extent, and padding.
- D — matched alternative companions: target remains at its original row; every available companion is replaced by a different captured request/document at an exact matching `prompt_tokens + decode_step` logical KV length. The full per-row length vector, physical max extent, width, and padding vector are identical to C. If no distinct exact-length candidate exists, that row is explicitly retained and the event cannot by itself receive a pure physical-shape verdict.

Arm E is not preregistered. It may be added only if A/B/C/D cannot distinguish heterogeneous physical shape from other causes.

## Repeats and thresholds

- Three measured repeats for every event and arm; arm order alternates forward/reverse.
- The RNG seed is reset before each reconstructed event repeat.
- Same-arm stability uses raw BF16 target tensor hashes, routes, final logits, and predicted tokens; elapsed time is deliberately excluded.
- Numeric comparisons report exact match, allclose (`atol=1e-6`, `rtol=1e-5`), max absolute delta, relative L2, cosine, kth-minus-(k+1) boundary margin, and route membership.
- `0.01` is inherited before this run from the checked-in router-logit diagnostic only as a labelled near-tie summary. Continuous margins remain the authority; the threshold cannot manufacture the primary verdict.

## Classification

- `RUNTIME_NONDETERMINISM`: identical same-arm inputs differ across repeats.
- `ROUTER_KERNEL_SHAPE_EFFECT`: pre-router hidden is stable/equal, router logits stably differ with width.
- `UPSTREAM_BATCH_CONTEXT_EFFECT`: pre-router hidden already stably differs.
- `PHYSICAL_SHAPE_EFFECT`: C and fully decorrelated D agree in target route under the same heterogeneous KV/padding vector while both differ from A; A/B separately identifies any width-only component.
- `COMPANION_IDENTITY_EXTERNALITY`: C/D target results stably differ under matched width and length/padding multisets.
- `NEAR_TIE_AMPLIFICATION`: secondary only when continuous boundary margins and expert crossings support it.
- `NOT_REPRODUCED`: strict same-pre-state A/C has no route-membership divergence; this does not erase the historical full-trajectory observation.

## Propagation

For four preregistered events, take each arm's own output cache, split and deep-clone the target row, remove all companions, then serially teacher-force the same next two captured tokens. Record later routes, router margins, final-logit distance, predicted tokens, and instrumented whole-call time. No arm may reuse another arm's KV.

## Capacity decision boundary

- Observational track asks whether historical batched route adds stable next-window risk signal beyond M2. Old sign-flipped runs require controlled repeats; mean pinball alone is insufficient if dangerous underprediction does not improve.
- Causal track asks whether a real `running_set_budget` changes SLO-goodput. It remains paused unless each budget regenerates its own queue, batch, KV, route, and completion trajectory.
