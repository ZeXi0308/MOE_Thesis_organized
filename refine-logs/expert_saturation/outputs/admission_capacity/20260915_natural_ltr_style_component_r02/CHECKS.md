# r02 minimal successor checks

Only runtime change from accepted r01 is deleting one aggregate prepare-growth prefilter line in `pkg/ltr_style_native.py`. All other runtime/input/controller/CPU-check source bytes match r01; the changed adapter compiles. No old suite, new fixture, SSH connection, GPU initialization or experiment was run for this preparation.

`cpu_checks.json` is the unchanged **reused r01 PASS_CPU_ONLY receipt**, not a new r02 execution result. It covers HOL/accept-only latch, positive-call Q accounting, pending-load/first-output behavior, slot/foreign-queue cases, actual native scheduling/preemption-loop growth, selected store/commit and failure handling. Its CPU allocator, outputs and transfers remain fixtures.

The only new semantic evidence is the already completed independent model-side counterexample, consumed without rerunning:

- Report: `refine-logs/expert_saturation/outputs/admission_capacity/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD.json`
- Source: `refine-logs/expert_saturation/outputs/admission_capacity/20260915_joint_growth_decision_r01/check_prepare_sum_guard.py`
- Native scheduler SHA256: `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`
- Report SHA256: `88449eadf64ba7d9881da6bfcf701d606bb59bc738aba51fc55501fc58bda84d`
- Counterexample source SHA256: `0c467d28607478b4929a9372710f5cebcef8ef3d7099f908f8078ab119d51919`

With free0 and aggregate peer growth1, both arms naturally preempt peer. Removing only the prefilter lets V decode once and register store7, then the shared contract returns READY, flushes store7 and schedules target17. Those tokens/outputs/transfer registrations are CPU fixtures. The counterexample establishes legal action availability; it does not remove peer cost, establish a complete-service advantage or qualify native asynchronous loading.

Runtime input/model/EOS/resource/measurement/threshold/quantum/budget are unchanged: G64, T30/Q10, 4096 usable GPU blocks plus null, 16GiB host, one180s diagnostic. Actual GPU initialization/type/source/pool checks, store/load completion, quantum evolution and all request costs remain UNRUN. Prior r01 prelaunch errors are preserved separately. Final payload and archive hashes pin this successor; root must explicitly accept r02 before a new prelaunch attempt.
