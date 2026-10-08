# Expert Saturation Evaluator Remediation

Date: 2026-08-23  
Reviewer: independent same-family sub-agent `/root/experiment_integrity_reaudit`  
Overall verdict: `WARN`  
E0 verdict: `PASS`  
Blocking findings: `P0=0, P1=0`

This is an append-only remediation record. It does not replace or rewrite
`EXPERIMENT_AUDIT.md`, whose `FAIL` verdict correctly describes the evaluator
state before remediation.

## Audited source snapshot

| Component | SHA-256 |
|---|---|
| route OFF/ON comparator | `c6d16a1f219438a0525055096bd77278bc1a83119e9079fe37775abcc3b7013d` |
| route pivot analyzer | `f863a9cbae02f595da729226da91401bdbce13541b326443ae782e72dc2a4afc` |
| stock/optimized implementation comparator | `505672a2835cf66e323ade4d0496420c300fb67c6d69c57843adfa34c7b46bb8` |
| route probe runner/helper | `9a83209363a0fb68568a3c85cc42dfb578c407f4ae8c4306d78f147ffe433e44` |
| decode-cap branch runner | `717725a3aa4f0da0df55e154b825e6147caa95089363c191b539313709df65aa` |
| decode-cap analyzer | `06d05747e21ad4507f75bf70f29d4360abc3817288560d9227095ed54ada62cd` |

Repository HEAD was `b141c1d5`; the target research directory remained
untracked. No GPU experiment was run during E0.

## Closed fail-open paths

- Timing decisions are recomputed from request-level JSONL evidence; route
  metrics are recomputed from sealed NPZ tensors. Forged summaries fail.
- The telemetry deviation Gate is frozen at an absolute two-sided 5%; NaN,
  Inf, negative, or relaxed thresholds are invalid.
- Every required repeat is retained. Token drift or excessive timing deviation
  in any repeat removes the full Route-OFF timing join.
- Standalone, pivot, implementation, and decode scientific CLIs use
  `0=positive`, `1=valid non-positive`, and `2=invalid`.
- Source-file integrity and source semantics are separate. Only the exact
  audited producer/helper hashes can qualify; a missing, historical, or merely
  self-consistent unknown producer remains measurement-only or invalid.
- The stock/optimized Gate requires two complete repeats, exact validator-
  approved vLLM source hashes, full nonzero route support, token/route parity,
  and GPU isolation in all eight arm/repeat bundles.
- Decode-cap bundles seal both local Python sources and bind exact vLLM 0.26
  telemetry, offline-entrypoint, scheduler, and request-queue source hashes.
- Decode-cap requires exclusive-GPU evidence, freezes the 5% telemetry and 3%
  headroom thresholds, and treats zero low-arm goodput as an undefined relative
  baseline with exit 1 rather than a positive result.
- Missing, traversing, corrupt, truncated, wrong-shape, wrong-dtype,
  out-of-range, and duplicate-top-k artifacts fail closed.

## Verification

- Top-level experiment tests: `87/87 PASS`.
- GPU pressure-sketch tests: `5/5 PASS`.
- Python compilation and `git diff --check`: `PASS`.
- Historical full-bundle replay remains:
  `WORKING_SET_MEASUREMENT_ONLY / TELEMETRY_TRANSPARENCY_FAILED`, with Route
  ON/OFF token drift in `1/36` and `6/36` cells and nonzero CLI exit.

## Remaining non-blocking boundaries

1. Historical producer SHA `fa20398f...` has no recoverable source; old bundles
   remain producer-source-unverified structural evidence only.
2. Decode-cap accepts one frozen six-branch sextet and is exploratory until a
   controlled repeat-level GPU experiment is added.
3. Valid-window telemetry and decode-cap GPU Gates are still `UNRUN`; E0 PASS
   is evaluator qualification, not runtime, latency, headroom, or method GO.

## E0 decision

`PASS / CLOSE_E0`

The next and only authorized scientific step is the two-repeat stock versus
valid-window telemetry qualification on an explicitly authorized, isolated
GPU. No Controller implementation is authorized before request-level action
headroom survives its later Gate.
