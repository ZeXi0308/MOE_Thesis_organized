# N0b evaluator addendum

Date: 2026-08-23

The remote campaign and its original report are preserved without modification.
The sealed report `valid-window-gate.json` has SHA-256
`522861b3823c76ae378b4aa28909d41270676f1eaac1c60229a29cbd31d930ae` and was
produced by the frozen v2 evaluator with SHA-256
`505672a2835cf66e323ade4d0496420c300fb67c6d69c57843adfa34c7b46bb8`.
Its terminal `CAMPAIGN_COMPLETE.json` has SHA-256
`484c63bfabf3b8efc5de0b9244eda2709fd81f99d5e688c4016f699ae3a163cb`.

## Correction

The v2 campaign reducer selected the first non-qualified process repeat. It
therefore surfaced repeat 0's timing failure while masking the stronger output
correctness failure retained in repeat 1. Pair-level measurements were not
affected by this reducer bug.

Evaluator v3 gives token/output drift precedence over timing-only failures
across all retained repeats, ranks known invalid and route-contract outcomes
explicitly, and lists every repeat failure at the campaign level. The final
evaluator has SHA-256
`f263366b895c213ba95f42f4004a948e8697a3a90e63b99f4122404cd3040c80`.
Its local replay is `valid-window-gate.evaluator-v3-r2.addendum.json` with
SHA-256
`96d3c042ffad91220384b8ada4642fa73ad8fe52fd5b8eab479e5f8e6759ebb1`.
The earlier v3 replay (`5179bba0...019694`) is retained and superseded; no
artifact was overwritten.
The corrected campaign verdict is:

```text
status: VALID_WINDOW_NOT_TRANSPARENT
failure_category: TELEMETRY_TOKEN_DRIFT
```

## Evidence retained

| Repeat | Optimized OFF/ON token parity | Lossless route comparison | P95 absolute TPOT deviation | P95 absolute wall deviation | Interpretation |
|---|---:|---:|---:|---:|---|
| 0 | PASS | 36/36 comparable and exact | 28.26% | 43.77% | Timing is interpretable and exceeds the frozen 5% Gate. |
| 1 | FAIL at `[512,16,1,0]` | 34/36 comparable; 34/34 exact among comparable cells | 20.34% | 28.24% | Timing is diagnostic only because output parity failed. |

The stock control also drifted in repeat 1 at `[512,8,2,0]`. Stock-OFF versus
optimized-OFF output parity passed in both repeats, so the valid-window source
patch is inert when telemetry is disabled. All eight bundles passed integrity,
source-identity, exclusive-GPU, and repeat-retention checks.

The replay command used the current evaluator with the two local bundle paths
for every one of `stock_off`, `stock_on`, `optimized_off`, and `optimized_on`,
and wrote to the new `evaluator-v3-r2` path. It returned exit code 1, the
expected code for a valid but scientifically failed Gate.

## Claim boundary

This is a valid negative result for the tested lossless, host-visible route
telemetry formulation on single-GPU eager vLLM 0.26. It is not pressure-latency,
capacity, scheduler, online-serving, or Expert Parallel evidence. Decode-cap and
Controller experiments remain blocked. The raw timing values from repeat 1 must
not be used as a confirmatory overhead estimate.
