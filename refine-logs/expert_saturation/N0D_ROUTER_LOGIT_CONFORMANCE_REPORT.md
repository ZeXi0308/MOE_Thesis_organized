# N0d Router-Logit Conformance Report

Date: 2026-08-24

## Verdict

`POSITIVE_MEASUREMENT /
PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`

In the frozen four-request OLMoE steady cell, serial and batch-4 execution were
forked from the same per-step KV state. The first reconstructed Expert-
assignment frontier occurs at `olmoe-dev-steady-000 / decode step 1 / layer 3`.
Within that same request and decode step, a nonzero router-logit difference is
already visible at layer 0. The frontier is identical against both serial-A and
serial-B and across three fresh OS processes.

`westc-r01` stopped before model loading because its capture path violated the
producer isolation contract and produced no scientific trajectory.
`westc-r02` is the first and only scientific execution, not a favorable rerun.

## Evidence type

`CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_THREE_FRESH_PROCESS`

- Model: `allenai/OLMoE-1B-7B-0924` revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16, greedy decode.
- Runtime: custom Transformers cached decode on one RTX 5090.
- Cell: four fixed requests, eight decode steps, batch width four.
- Repeats: three fresh processes with counterbalanced arm order.
- Evidence tier: custom-runtime matched-prestate execution-conformance
  measurement.

All manifest-listed r02 campaign bytes revalidate by SHA-256. A post-result
review also found two unmanifested local `frozen/__pycache__/*.pyc` files with
mtimes after the completion sentinel. They do not alter any listed scientific
input or result, but they keep exact local-directory provenance at `WARN`.

## What was measured

At every decode step, one canonical state was cloned into independent
`serial-A / batch-4 / serial-B` branches; the batch branch was discarded after
the step. In each process, eight fork checks covered 3,072 KV tensors and
440,401,920 elements that were value-equal and storage-disjoint.

The three processes all reproduce these facts:

- serial-A and serial-B are exact for all 32 token records and all 512
  request-step-layer router records;
- all arms preserve 32/32 input and predicted-token parity with the sealed
  request-ledger reference;
- among 512 serial-versus-batch records, 22 Expert-multiset changes occur in
  nine request-step cells; multiset match is `95.703125%` and ordered match is
  `83.984375%`;
- every router vector contains at least one non-allclose scalar; across 32,768
  scalars the exact/allclose fraction is `44.250488%`, maximum absolute delta
  is `0.28125`, and mean absolute delta is about `0.00648`;
- at the first assignment frontier, serial Experts are
  `[19,23,37,39,40,51,56,61]`, batch Experts are
  `[19,23,37,39,51,56,61,62]`, the layer-3 maximum logit delta is `0.015625`,
  and the qualifying layer-0 maximum delta in the same request-step is
  `0.0078125`.

The append-only evaluator-v3 replay validates all 4,608 retained Expert sets
for top-k value consistency against their 64 router logits. It returns the same
verdict with `selected_experts_topk_value_consistent=true`, no structural
errors, exact serial controls, exact token parity, and stable three-process
signatures. There are 48 exact boundary-tie rows, so exact GPU tie-break
identity is not independently reconstructed.

## What was not measured

The experiment did not capture the layer-0 pre-router hidden state. It therefore
cannot distinguish an upstream Attention/KV/padding/residual/RMSNorm difference
from the router gate Linear's `M=1` versus `M=4` execution shape. It also has no
same-width shuffled-companion arm, so physical batch shape and companion
identity remain confounded.

The Expert IDs are retained producer outputs and are independently validated
for top-k value consistency; they are not captured at an internal dispatch
hook, and the exact selected identity in boundary-tie rows is not reconstructed.
Coverage is limited to one model, one steady cell, BF16, a custom Transformers
runtime, and one GPU. No second model, bursty cell, native serving runtime,
multi-GPU EP, batch-specific future-state trajectory, full next-token-logit
comparison, semantic-quality result, request latency, TPOT/P99, SLO-goodput, or
full-request critical-path consequence was measured.

## Strongest baseline

The strongest negative control is serial-A versus serial-B, independently run
from the same canonical prestate. They are exact across all three processes,
512 router records, Expert assignments, and 32 token records. The measured
batch contrast therefore cannot be attributed to same-arm repeat instability.

This is not a performance-method experiment, so there is no scheduler or
Controller baseline.

## Oracle/headroom status

`NOT_RUN / LOCKED`

No action-conditioned Oracle, performance action, or full-request denominator
was defined. `method_go_authorized`, `action_oracle_authorized`,
`capacity_claim_authorized`, and `controller_authorized` all remain `false`.

## Claim ceiling

`CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY`

Allowed claim: for this fixed custom-runtime cell, a repeat-stable
serial-versus-batch reconstructed assignment difference has a nonzero
pre-top-k router-logit difference earlier in the same request-step.

Not allowed: a specific kernel or subsystem root cause, propagation through
future batch-specific KV, final-output or quality impact, a serving-capacity
signal, latency/SLO gain, Controller benefit, native serving transfer, or EP
generality.

## Failure category

N0d has no scientific failure category. It is `MEASUREMENT_ONLY`, not method
GO. `westc-r01` is `INFRASTRUCTURE_ABORT_BEFORE_MODEL_LOAD`; mechanism,
action, Oracle, capacity, latency, and Controller evidence remain `UNRUN`, not
NO-GO.

## Resurrection condition

N0d already has a stable positive measurement and should not be rerun with more
seeds. Reopen it only if new evidence invalidates capture provenance,
matched-prestate isolation, or Expert reconstruction. Otherwise continue to
source localization.

## One next smallest experiment

Run N0e only at the preregistered
`olmoe-dev-steady-000 / decode step 1 / layer 0` frontier. Capture the input to
`model.model.layers[0].mlp.gate` for serial-A, batch-4, and serial-B from the
same canonical prestate.

- If batch hidden state differs from both exact serial controls, the source is
  already upstream of the gate and should next be divided within
  Attention/KV/padding/residual/RMSNorm.
- If hidden state is bit-exact but router logits differ, the visible source is
  narrowed to the bias-free gate Linear's batch-shape-dependent numerical path.

N0e remains source localization only; it adds neither a Controller nor a
performance claim.
