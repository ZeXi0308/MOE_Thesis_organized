# Independent remediation re-audit

## Verdict

- Overall: `WARN`
- E0 evaluator-qualification Gate: `PASS / can close`
- Findings: `P0=0`, `P1=0`, `P2=3`
- GPU execution: `UNRUN`

The remediated evaluators fail closed for every previously demonstrated
counterexample and the additional adversarial cases exercised in this review.
The remaining warnings limit scientific scope; they do not block E0 closure.

## Audited source snapshot

| Component | SHA-256 |
|---|---|
| route OFF/ON comparator | `c6d16a1f219438a0525055096bd77278bc1a83119e9079fe37775abcc3b7013d` |
| route pivot analyzer | `f863a9cbae02f595da729226da91401bdbce13541b326443ae782e72dc2a4afc` |
| stock/optimized implementation comparator | `505672a2835cf66e323ade4d0496420c300fb67c6d69c57843adfa34c7b46bb8` |
| route probe runner/helper | `9a83209363a0fb68568a3c85cc42dfb578c407f4ae8c4306d78f147ffe433e44` |
| decode-cap branch runner | `717725a3aa4f0da0df55e154b825e6147caa95089363c191b539313709df65aa` |
| decode-cap analyzer | `06d05747e21ad4507f75bf70f29d4360abc3817288560d9227095ed54ada62cd` |

Repository HEAD was `b141c1d5`; the research directory was untracked.

## A-F status

- A. Ground-truth and provenance: `WARN`. New bundles are exact-source-bound;
  the historical producer source is unavailable and therefore remains
  measurement-only.
- B. Score normalization and accounting: `PASS`.
- C. Result existence and claim matching: `PASS`.
- D. Dead code and fail-open validation: `PASS`.
- E. Scope: `WARN`. Evidence is single-model historical fixed batching; the
  valid-window and decode-cap GPU Gates have not run.
- F. Evaluation classification: `PASS`.

## Verified remediations

- Raw request JSONL and route NPZ evidence are recomputed; forged summaries do
  not control a verdict.
- The telemetry guard is a frozen, finite, absolute two-sided 5% threshold.
- All required repeats are retained; token drift or excessive deviation in any
  repeat invalidates the corresponding timing join.
- Producer integrity is separated from approved source semantics. Historical
  or arbitrary self-consistent source does not qualify a scientific Gate.
- Stock/optimized telemetry requires all eight arm/repeat bundles, exact vLLM
  sources, complete valid route support, token/route parity, and exclusive-GPU
  evidence.
- Decode bundles bind the runner, helper, vLLM telemetry and actuator sources;
  zero low-arm goodput cannot manufacture relative headroom.
- Scientific CLI exits are `0=positive`, `1=valid non-positive`, `2=invalid`.

Verification completed with `87/87` top-level tests, `5/5` pressure-sketch
tests, Python compilation, and diff checks passing.

## Remaining P2 boundaries

1. Historical producer SHA `fa20398f...` is unrecoverable; those artifacts are
   structural, producer-source-unverified evidence only.
2. Decode-cap currently specifies one frozen six-arm sextet, so any future
   result is exploratory until repeat-level stability is added.
3. No valid-window or decode-cap GPU result exists.

## Claim impact and next Gate

Historical route shape remains `WORKING_SET_MEASUREMENT_ONLY`; historical
Route-ON/OFF timing remains invalid because of token drift. Valid-window and
decode-cap are implemented and evaluator-qualified, not performance-validated.
There is no capacity, SLO-goodput, action-headroom, Controller, or method-GO
claim.

The only next scientific step is the explicitly authorized, isolated-GPU,
two-repeat stock-versus-valid-window telemetry qualification. Decode-cap and
Controller work remain conditional on that Gate.
