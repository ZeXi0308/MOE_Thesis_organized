# Post-stabilization same-family integrity audit

**Overall verdict:** `WARN / SAME_FAMILY_PROVISIONAL`
**Implementation and evidence integrity:** `PASS`
**Scientific eligibility:** `FAIL / BLOCKED`
**Unresolved P0:** `0`
**Unresolved P1:** `0`

No exploitable integrity defect remains in the reviewed snapshot. The sole
action is `next_window_active_token_budget`; configuration drift and a second
action are rejected. Feature availability is bound to the end of window `t`,
the target to `t+1`, and only M4 can consume future route. Request/document
components are split without overlap, while arrival-episode independence is a
separate eligibility check and is not replaced by request-disjoint proxies.

The P1 contract binds the two frozen model revisions, steady and bursty
regimes, and the frozen 5%/15%/3% thresholds. Route ingestion checks ownership,
contiguous layer/top-k/slot identities, duplicate experts within a token-layer,
gate-weight validity, and source provenance. M4 remains a future-route latency
diagnostic rather than capacity ground truth. P2 and P3 require exact upstream
schemas, literal booleans, empty blockers, the frozen action, and an actually
executed passing upstream result.

The canonical command completed with `26/26` tests passing and byte-identical
`metrics.csv`, `summary.json`, `environment.json`, and `report.md` outputs.

Scientific eligibility remains blocked: the source is one model, one arrival
episode, one RTX 5090, teacher-forced, and non-serving. It lacks a second frozen
model, independent steady/bursty episodes, gate weights, native queue/admission
telemetry, hook-overhead A/B, a verified active-token actuator, and
action-conditioned route regeneration. P2 and P3 are not run. Classification
`D` and `BLOCKED_RUNTIME_NOT_REPRESENTATIVE` are therefore correct.
