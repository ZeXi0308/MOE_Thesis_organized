# Experiment Audit Report

**Date:** 2026-08-13

**Auditor:** GPT-5.6-Sol ultra, fresh same-family read-only agent (provisional)

**Project:** Route-Conditioned Capacity Envelope v2

## Overall Verdict: WARN

## Integrity Status: warn

No fake ground truth, self-normalized score, phantom metric, or capacity action
was found. The claim ceiling is nevertheless narrow because the canonical
selection was post-hoc, the two timing/M0--M4 pilots were not controlled
repeats, and `report.md` changed bytes during the audit. The fresh primary
reviewer returned this verdict and its reason code; its detailed A--F response
was interrupted after a completed delegated selection-bias check, so this
same-family audit remains provisional.

## Checks

### A. Ground Truth Provenance: PASS

The prediction target is observed next-window whole-model-call latency from a
real RTX 5090 custom runtime, not a target synthesized from model predictions.
M4 is explicitly labeled as a future-route diagnostic, and the report states
that no capacity action or safe-capacity claim was measured
(`artifacts/route_capacity_envelope/dev/20260812T170512Z/report.md:13-19`).

### B. Score Normalization: PASS

The bundle reports raw P95 pinball loss, dangerous-underprediction rate, and
SLO-risk false negatives for M0--M4. The relative M3/M2 comparison uses the M2
loss as the declared baseline; it does not divide by a maximum or statistic of
M3's own predictions (`metrics.json:4-39,281-287`).

### C. Result Existence and Selection Integrity: WARN

The retained files exist and the displayed `+9.6288%`, conformance veto, and
final verdict match `metrics.json` (`metrics.json:287-305,361`). The unfavorable
`-24.0388%` run also exists in audit quarantine and is disclosed
(`/private/tmp/rce-superseded-170328Z/report.md:3-17`). However, replacement of
the first canonical bundle was post-hoc. Retaining the second run has a valid
non-metric reason--it propagates the conformance veto--but no preregistered,
result-blind retention rule independently proves that selection was unrelated
to its favorable M3 value (`LIGHTWEIGHT_STATUS.md:78-94`).

During review, canonical `report.md` changed from SHA-256 `8ca96e...` to
`3bf68e...` by gaining a post-run caveat. After the reviewer recorded the drift,
the added line was removed and the file was restored byte-for-byte to the
original GPU-produced SHA-256
`8ca96e8096796da966879151d49c1b8e048eb36a2117ce99b975339f062f7c1b`.
This remediation prevents the final raw bundle from containing a silent
post-run rewrite; it does not erase the audit warning.

### D. Dead Code Detection: PASS

The M0--M4 methods all appear in the materialized result and the batch-dependent
route flag reaches the top-level verdict (`metrics.json:4-39,287-305,361`). The
new pre-top-k diagnostic is explicitly the next experiment and has not been
reported as executed (`LIGHTWEIGHT_STATUS.md:126-160`).

### E. Scope Assessment: WARN

Evidence is one model, one RTX 5090, two 16-request episodes, eight decode steps
per request, 64 windows, and 62 aligned next-window examples. It is a custom
Transformers runtime, not native serving, and the hook check did not replay the
arrival trace (`LIGHTWEIGHT_STATUS.md:60-85,107-114`). The two M3/M2 results
change sign (`+9.6288%` versus `-24.0388%`), while logs do not establish
exclusive or isolated repeat conditions. Consequently neither result supports
a stable route-residual or safe-capacity claim.

### F. Evaluation Type: real_gt (development custom runtime) plus proxy checks

The primary latency target and route identities are freshly observed real-GPU
runtime values. The fixed-batch router-output overhead check, padded-KV extent,
and M0--M4 capacity interpretation are explicitly proxy/analytic evidence; no
offline replay, action Oracle, controller, second model, or production serving
evaluation was run (`LIGHTWEIGHT_STATUS.md:68-76,107-114`).

## Action Items

- Keep the six-file GPU bundle byte-stable and treat the original report hash as
  canonical.
- Do not interpret either M3/M2 sign as capacity evidence and do not run stages
  C, D, or E under this result.
- Run only the frozen serial-vs-batch pre-top-k logit diagnostic next; any later
  M0--M4 repeat must predeclare retention, preserve every run durably, and log
  GPU/process isolation.

## Claim Impact

- `PIVOT_TO_EXECUTION_CONFORMANCE`: **supported**, scoped to this single-model
  custom-runtime observation.
- Batch-dependent expert assignment despite token parity: **supported**; it was
  reproduced in both regimes and both pilot bundles.
- Historical route adds `+9.6288%` capacity information: **unsupported as a
  capacity claim**; retain only as an unstable diagnostic number.
- Route shape changes safe capacity: **unsupported / not measured**.
- Dynamic `running_set_budget`, Oracle headroom, or controller gain:
  **unsupported / not run**.
- Telemetry overhead below 2%: **supported only for the fixed-batch proxy**, not
  for a complete arrival-matched A3 or native serving runtime.
