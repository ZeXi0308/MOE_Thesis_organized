# RouteShape-SLO

> Status: `BLOCKED_RUNTIME_NOT_REPRESENTATIVE / P0_INCOMPLETE / P1_SMOKE_ONLY`
> Date: 2026-08-12
> Current authority remains [`docs/current/README.md`](../../current/README.md).

## Verdict

`BLOCKED_RUNTIME_NOT_REPRESENTATIVE`

The repository does not yet contain a completed, representative continuous
serving trace that aligns route shape with native queue/load/KV state and a
calibrated SLO. One existing RTX 5090 StableBatch ledger was reused for an
offline P0/P1 pipeline smoke. It is an observed isolated GPU primitive with a
teacher-forced frozen roster, not a serving-capacity experiment.

The smoke's workload-only versus workload-plus-route numbers are deliberately
non-scientific and cannot answer whether route information changes safe
capacity. P2 and P3 are fail-closed until an eligible P1 result exists.

## Files

- [`STATUS.md`](STATUS.md): evidence inventory, measured/inferred boundary,
  formulation collision, and next experiment.
- [`research_question.md`](research_question.md): frozen hypothesis, feature,
  action, information-time, and sequential Gate contract.
- [`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md): provisional same-family
  integrity audit, fixed findings, and remaining evidence limits.
- [`experiments/inspect_existing_assets.py`](experiments/inspect_existing_assets.py):
  read-only schema/provenance inspection for existing StableBatch and BCRD
  bundles.
- [`experiments/build_route_windows.py`](experiments/build_route_windows.py):
  identity-closed, zero-inclusive, permutation-invariant route-window feature
  builder.
- [`experiments/analyze_incremental_signal.py`](experiments/analyze_incremental_signal.py):
  grouped P95 ridge comparison for M0--M4.
- `experiments/replay_capacity_oracle.py` and
  `experiments/run_causal_controller.py`: P2/P3 guards; no action replay or
  controller is implemented or executed.

## Reproduce the smoke

The canonical local bundle is
`artifacts/route_shape_slo/20260812T190238+0800/`. Run its
`commands.sh` from the repository root. Generated artifacts remain ignored by
Git under the repository artifact policy.

## Scope

This exploration does not alter the current RCBA Oracle-first Primary. If an
eligible result later shows value only through physical replica fragmentation,
fold it into BCRD. If the value changes service-level active-work admission
independently of assignment, DEPA is the default collision; an independent
paper would still require P2/P3 headroom and causal-policy evidence.
