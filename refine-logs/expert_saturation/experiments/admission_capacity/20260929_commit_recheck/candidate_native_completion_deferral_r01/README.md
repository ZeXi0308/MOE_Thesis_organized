CPU-only, unrun native allocation-failure capacity-deferral candidate.

This is a private copy of `candidate_native_bidkv_full_running_r01`, used with
native-full ordinary-only execution, ordinary backfill, native tail victim
selection, and the full-running, current-guard, and continuation probes off.
Set `A_NATIVE_CAPACITY_DEFERRAL=off|prefix_work|prefix_finish`.

After a running request's native `allocate_slots` returns `None`, and before
preemption, `prefix_work` can defer that current request when an already
scheduled pure-decode prefix has one positive token planned in this call.
`prefix_finish` additionally requires a known positive hard-cap remainder
that fits entirely in that witness's privately owned KV slots. It selects the
smallest remainder, then arrival and request ID. Both modes skip to the next
running request while preserving the prefix plan. The next pressure decision
rechecks every condition; there is no persistent hold. Unknown or unsafe
state uses the original native preemption path. `off` leaves that path active.

The store records actual deferrals in `capacity_deferrals`, with the failed
request and witness, their predecision counts and capacity, and native output
plan verification. `capacity_deferral_fallback_counts` records rejected
opportunities by reason. Both config and store record
`capacity_deferral_mode`. The focused CPU closure is
`test_capacity_deferral_cpu.py`; the copied full-running closure also runs.
No GPU results are included in this candidate.
