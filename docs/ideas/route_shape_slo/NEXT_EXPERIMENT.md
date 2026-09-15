# Next smallest experiment: one OLMoE continuous-decode P0 capture

## Unique uncertainty

Can the repository's already-implemented continuous-decode producer complete on
the frozen RTX 5090 environment and emit an identity-closed bundle in which
route rows, request/decode membership, KV state, and CUDA-synchronized whole
model-call time share the same window?

This is one **P0 producer-capture experiment**, not P1, an overhead
qualification, an Oracle, or a controller. A passing run advances only to a
paired hook-overhead/native-scheduler qualification; it does not authorize
predictor training.

## Exact runnable command

Run from a clean committed checkout with one idle RTX 5090, PyTorch `2.8.0`,
Transformers `4.57.6`, and the frozen model already cached:

```bash
python3 docs/ideas/bcrd/experiments/capture_continuous_decode.py \
  --workload-manifest docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json \
  --preregistration docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json \
  --output-dir artifacts/bcrd_gate0/formal/route-shape-p0r-20260812/olmoe \
  --offline
```

This command is supported by the current producer. Formal mode intentionally
fails if the checkout is dirty, the canonical bytes differ from `HEAD`, the
GPU/dependency contract differs, or any identity/integrity check fails.

## Frozen model and scale

- Model: `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- Precision/device: BF16, one RTX 5090.
- Data: 128 frozen WikiText documents with BurstGPT-derived arrival timestamps,
  up to 16 decode steps/request, max batch size 8.
- Sole future action is `active_token_budget[t+1]`, the prompt default. This
  qualification run executes no capacity action because the current harness
  has no verified dynamic active-token actuator; that missing interface remains
  an explicit P2 implementation blocker.

## Resource estimate

The closest sealed OLMoE RTX 5090 run observed about `15,986 MiB` resident GPU
memory at completion on a 32 GiB card. Reserve 24 GiB. That prior value is real
GPU evidence for a different three-arm harness, not a measured peak for this
command.

The closest broader run spanned approximately 13 minutes from environment
capture to completion. Budget 30 minutes for model load, continuous decode,
and the producer's full serial audit. This is a conservative planning estimate,
not a new runtime measurement.

## GO / STOP

GO to the **next P0 qualification step only** if:

- both `RUN_STATUS.json=COMPLETE` and `CAPTURE_COMPLETE.json` exist;
- all 128 requests terminate and token/route serial-audit match fractions are
  exactly `1.0`;
- request/window/layer/top-k contribution identities close exactly;
- observed decode batch size reaches at least 2 and the active set changes;
- prefill and decode remain separated and whole-call timing is CUDA-synchronized.

STOP this producer path if any condition fails. Even on GO, retain
`BLOCKED_RUNTIME_NOT_REPRESENTATIVE` because the custom Transformers harness has
no paired route-hook overhead measurement, native serving queue/admission
semantics, calibrated serving SLO, or fresh two-model holdout.

## Output

```text
artifacts/bcrd_gate0/formal/route-shape-p0r-20260812/olmoe/
```

Do not start LLM-jp, P1/P2/P3, or an 8xA100 job before this one result is
audited.
