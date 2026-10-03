CPU-only, unrun native victim-size comparison candidate.

This is a private copy of `candidate_native_completion_deferral_r01`. Use
native-full ordinary-only execution with ordinary backfill. The three rules
are `A_NATIVE_VICTIM_RULE=tail|min_held_other|max_held_other`. In all three
arms, full-running, current-guard, self-preempt continuation, and capacity
deferral modes must be off.

At a native running allocation failure, the two new rules compute the
single-token block deficit from the failed current request's computed count,
privately owned blocks, and free blocks. They consider only unscheduled,
noncurrent suffix requests in pure decode with known exclusive KV ownership.
The smallest or largest held-block candidate that covers the deficit wins;
ties go to the later running index. Unknown state, no positive deficit, no
sufficient alternative, or active protection/preparation retains the native
tail. The native scheduler still performs its original preempt-and-retry path.

`victim_decisions` records the predecision deficit, current computed/held
counts, free blocks, each suffix candidate's held count and qualification,
selected victim, and fallback reason. The package makes no claim about host
copy coverage or actual reload cost. The focused CPU check is
`test_victim_size_cpu.py`, including the pinned native FCFS preemption branch.
No GPU results are included.
