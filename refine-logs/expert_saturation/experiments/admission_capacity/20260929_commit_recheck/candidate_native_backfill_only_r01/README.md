# Private native full-save ordinary-backfill contrast (GPU unrun)

This candidate copies `candidate_waiter_backfill_r01` into a new immutable identity. The frozen 128-request input, warmups, model/runtime pins, native scheduler source pin, full-save implementation, and metric capture are unchanged. Only `pkg/run.sh`, `pkg/run_recovery_cadence.py`, and `pkg/staged_store_rotation.py` differ from that source package.

The two supported cells are:

- `pkg/run.sh native_full_native performance off ABSOLUTE_OUTPUT_DIR`: existing native full-save service, with no scheduling adapter installed.
- `pkg/run.sh native_full_ordinary_only performance ordinary ABSOLUTE_OUTPUT_DIR`: the same native full-save connector calculation, with the already qualified ordinary eligible-waiter selection hook. The copied adapter installs `store_scope='native_full'`, leaves `connector_scheduler._calc_num_offloadable_tokens` unmodified, and explicitly disables its entire forced-rotation proposal path. Native memory-pressure preemptions remain possible; `applied_rotations` and `status.forced_rotations` must remain zero.

The ordinary gate is unchanged: a PREEMPTED FCFS head whose complete history does not fit, another PREEMPTED waiter absent at least 30 schedule steps whose complete history does fit, a sequence slot, known zero native in-flight reservation, no skipped waiters/global connector jobs/pending push, pure-decode running requests, and valid clean block ownership. Unknown states retain the original native waiting path. It starts Q1 first-output protection for a chosen waiter and adds no victim preemption. Full native saving can itself reduce eligible windows; an action count of zero is a valid coverage result, not permission to weaken the gate.

The candidate is a strong simple backfilling control, not a new scheduling mechanism. The native baseline and hybrid share the full-save backend, but only the hybrid installs scheduler hooks, so one ordered pair is neither a same-state counterfactual nor an adapter-overhead isolation. The working prediction is fewer than the 44 ordinary actions observed under selected saving on this same frozen input, potentially zero: the earlier native full-save reference had 43 natural preemptions, while selected-saving ordinary had 496 total preemptions and many more waiting opportunities. This prediction is falsifiable by the archived choice, admission, gate-count, and complete-cohort records; no GPU result is claimed here.

`manifest.json` covers the exact 25 payload files. `verify_package.py` checks every payload hash without importing vLLM or CUDA. The focused CPU closure test is `../test_native_backfill_only_cpu_r01.py` and is intentionally outside the deployable payload.
