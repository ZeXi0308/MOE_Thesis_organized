# `active_target` action-space fairness addendum

**Verdict: `CPU_ACTION_SPACE_DEFECT_CONFIRMED / GPU_UNRUN`.** This finding is limited to the current single-target LTR-style transplant. It does not establish a bug in official LTR, a performance benefit, or a requirement to dominate native service on every metric.

## Exact source boundary

The checked selector is `ltr_style_selected.py`, 134 lines, SHA-256 `371c6eb596b009d12cec7c99dc6eae83c719a4c6af8bc48d69f1125da20599fd`. Its `begin_step` sets `active_target` to the first boosted `PREEMPTED` request before checking backend ownership or single-victim feasibility, and clears it only when the request disappears or its priority is no longer `-1`.

The verified official source and the repository transplant have different action spaces:

- `../../20260914_service_window_r01/neighbors/sources/ltr_scheduler.py:984-996` promotes every request whose idle counter reaches the threshold and globally sorts waiting, running and swapped requests by priority and score. It has no single `active_target` latch.
- `../../20260914_service_window_r01/neighbors/sources/ltr_scheduler.py:1137-1218` admits multiple ordered requests up to the request/token budget, then `reserve_free_blocks` may evict multiple lower-priority requests (`:1396-1452`).
- `../../20260914_service_window_r01/neighbors/sources/ltr_scheduler.py:1359-1365` decrements `runs` only for requests actually scheduled and increments idle for every unscheduled request.

The older local `../preparation/pkg/ltr_recompute_native.py:40-73` planner is itself a custom recompute integration, not official LTR. It nevertheless supplies a useful action-space negative control: an infeasible earlier row has no eviction/budget side effect and the loop continues to later rows.

## Executed counterexample

The existing `LTRCounters` and current selector were executed with `threshold=1`, `quantum=2`, `free_blocks=0` and three requests:

| Queue order | State | Need / held | Result after both waiters boost |
|---|---|---:|---|
| `old_unfunded` | `PREEMPTED`, arrival 0 | need 10 | chosen as `active_target`, then `DEFER` |
| `later_fundable` | `PREEMPTED`, arrival 1 | need 2 | never examined |
| `peer` | `RUNNING`, normal priority | held 2 | can fund only `later_fundable` |

Across three further calls the selector repeatedly returned `DEFER: no legal single victim can fund full history` for `old_unfunded`. Both waiters remained `priority=-1, quantum_remaining=2`; because neither was scheduled, quantum never decreased, and the idle threshold rearmed it. Thus `later_fundable` can be excluded indefinitely by the latch. The older local full-batch component selected `later_fundable` and victim `peer` on the same abstract resource state.

This is independent of request-ID tie breaking: the fixture uses distinct arrivals. A non-active boosted request is correctly excluded from the victim set by the strict lower-priority test; the defect is that a feasible boosted waiter is also excluded from target consideration. Serialization is bounded when the active target is loading or actually consuming quantum, but it is unbounded when the latched target stays infeasible or otherwise `DEFER`red.

## Minimum fair handling

When no real action owns the backend, scan boosted `PREEMPTED` requests in the declared queue order and choose the first request with an executable single-target action:

1. If current free blocks fund its complete history, emit `PRIORITIZE_WAITING`.
2. Otherwise require one legal lower-priority running victim that individually funds the history; if none exists, continue to the next boosted waiter without latching or mutating action state.
3. Latch `active_target` only after the adapter accepts `PRIORITIZE_WAITING` or `PREPARE_SELECTED`. Retain it through that target's actual prepare/commit/load/running epoch, then release it on quantum exhaustion or terminal completion.

No round-robin policy, timeout, extra threshold or multi-victim fallback is needed. All live requests must still pass through the same `begin_schedule/after_schedule` counter update, including candidates skipped for present infeasibility. Record skipped boosted candidates and waiters deferred behind a real active epoch so the single-target action-space loss remains visible.

## Victim-policy fairness boundary

The intended comparison is between two complete recovery policies under the same physical GPU/host budget, selected native-save backend and one-target/one-victim capability. It is not a single-factor attribution of only the trigger. Therefore G's `min_residency_steps`, progress/near-EOS guard and `max_absences_per_request` remain G policy behavior; forcing them into the LTR-style arm would suppress counter-authorized actions and turn the arm into a G/LTR mixture.

The common victim legality floor is only: distinct target/victim; victim is lower priority, `RUNNING`, pure decode and nonterminal on the prepare step; current physical ownership is valid; and that one victim plus current free blocks funds the target's complete current history. The frozen `most_output` tie/order used by this component is an explicit transplant choice, not official LTR or a physical-safety invariant. Both policy differences and realized peer cost must be reported in the eventual same-budget comparison.

The common backend still owns prepare-to-next-boundary revalidation, selected-prefix store registration/fallback semantics, exact forced preemption, required flush, async load readiness, target block safety and actual-result accounting. This addendum changes no adapter, selector, G/H artifact, test matrix or execution identity.
