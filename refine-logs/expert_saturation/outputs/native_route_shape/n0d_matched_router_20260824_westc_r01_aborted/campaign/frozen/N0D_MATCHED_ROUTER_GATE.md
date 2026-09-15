# N0d Matched-Prestate Router Gate

Date: 2026-08-23  
State: `PREPARE / GPU NOT YET RUN`  
Repository HEAD: `b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a`

## One research question

For the frozen four-request OLMoE steady cell, does a repeatable
serial-versus-batch-4 reconstructed Expert-assignment difference have an
earlier or same-layer nonzero difference in native pre-top-k router-logit
values when both arms begin from the same KV state?

This is not an N0c retry. N0c tested whether two historical native-vLLM token
drifts recur under capture/export paths. N0d tests a separate custom-runtime
serial/batched conformance observation with a matched-state intervention.

## Frozen execution

- Model: `allenai/OLMoE-1B-7B-0924` revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16, greedy decode.
- Runtime: one RTX 5090, Transformers custom cached decode, one visible GPU.
- Requests: `olmoe-dev-steady-000..003`, prompt lengths `123/6/128/9`.
- Scale: four requests, eight decode steps, batch width four.
- Repeats: exactly three fresh OS processes; no retry or favorable selection.
- Source: exact HEAD plus nine runtime files bound by SHA-256.
- Input: fresh seven-file continuous-decode capture from the same environment.

At each step one canonical state is cloned into independent branches:

```text
canonical per-request KV / attention mask / next token
  -> serial-A
  -> batch-4
  -> serial-B
```

All cloned KV tensors must be value-equal and storage-disjoint. Only serial-A
advances the next canonical state; batch-4 state is discarded. Therefore later
steps remain matched-state interventions rather than propagation along a
batch-specific future trajectory.

The arm order is counterbalanced across processes:

```text
p0: serial-A, batch-4, serial-B
p1: batch-4, serial-B, serial-A
p2: serial-B, serial-A, batch-4
```

## Decision rule

The evaluator, not an individual process, owns the final verdict.

Required controls:

1. fresh capture closes request/token identity and itself observes the frozen
   batch-dependent assignment phenomenon;
2. serial-A equals serial-B exactly within every process;
3. all arms match the frozen reference tokens;
4. serial and batch traces are stable across the three fresh processes;
5. batch-4 has the same double-sided causal-frontier signature against both
   serial controls and across all processes;
6. the GPU remains process-isolated.

For each request and decode step, the causal frontier is ordered only by
`layer`. For an assignment difference at `(step=s, layer=l)`, the source
category is `PRE_TOPK_NUMERICAL_DIVERGENCE` only when the same request has a
nonzero router-logit value difference in the same decode step `s` at a layer
no later than `l`. A difference from an earlier step cannot qualify because
the batch branch is discarded and every later step forks again from the
serial canonical state. All requests sharing the first global assignment
`(decode_step, layer)` are retained as a frontier set. `allclose` and a
near-boundary margin are magnitude diagnostics only; they cannot turn a
nonzero pre-top-k difference into a top-k-only result.

Final outcomes:

- `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`: all controls and
  three-process frontier agreement pass;
- `NO_DIVERGENCE`: valid matched-state execution has no assignment difference;
- `SERIAL_CONTROL_UNSTABLE`, `TOKEN_PARITY_FAILED`,
  `CROSS_PROCESS_UNSTABLE`, or `INCONSISTENT_FIRST_DIVERGENCE`: stop and do not
  localize a source;
- `INCONCLUSIVE`: reconstructed top-k changes without a preceding visible
  logit-value difference;
- `INVALID`: source, identity, isolation, scale, or artifact contract failed.

## Claim ceiling

`CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY`.

Even a positive result means only that, for one model and this fixed
four-request combined execution context, the first observable reconstructed
assignment divergence is already present in pre-top-k router-logit values. It
does not isolate router GEMM from Attention/KV/padding/companion effects. The
selected experts are reconstructed from native router logits with
`softmax+topk`; they are not an internal dispatch hook. No capacity, latency,
action, Controller, native serving, or multi-GPU EP claim is authorized.

## Stop / continue

- Positive and repeat-stable: next smallest experiment captures pre-router
  hidden state at the first frontier, then separates padding/shape from
  companion identity.
- Valid no-divergence: close the historical custom-runtime formulation; do not
  rescue it with more seeds.
- Unstable/invalid: fix only the identified execution or evidence defect; do
  not reinterpret it as a scientific negative result.
