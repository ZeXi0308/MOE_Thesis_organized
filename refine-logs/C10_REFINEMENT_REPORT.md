# C10 Refinement Report

> Status：`TERMINATED / KILL`  
> Candidate evolution：CohortFence → HorizonFence  
> Evidence：conceptual review only; no pilot result

| Round | Proposal | Score | Verdict | Decisive finding |
|---|---|---:|---|---|
| 0 → 1 | backlog conservation + arrival-cohort partial-ID certificate | 5.20/10 | RETHINK | attribution estimand did not determine fast/slow action utility; generic censoring/accounting; always-abstain risk |
| 1 → 2 | action-specific no-prediction slow-migration futility certificate | 4.60/10 | KILL | generic robust dominance; future/DAG benefit bound widens while asynchronous unavoidable-cost bound collapses to zero |

## What improved

- separated fast and slow actuators;
- stopped calling scheduler-induced executed load inherently false;
- froze one detector/action/horizon;
- made the certificate one-sided and fail-closed;
- placed analytic non-vacuity before implementation.

## Why refinement stopped

The remaining repair would require a new causal-sensitivity/full-DAG method, not a refinement of C10. The frozen `RETHINK-ONCE` rule therefore terminates the route. See [final kill verdict](C10_KILL_VERDICT.md) and [Round-2 review](round-2-review.md).

## Handoff

The valid reusable insight is that local event savings do not compose into request/SLO savings without downstream causal closure. This motivates a separate C01/C02 candidate with a new Problem Anchor and new review run; it does not reopen scheduler-induced popularity endogeneity.
