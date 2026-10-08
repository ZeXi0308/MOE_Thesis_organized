# Necessary timing preparation checks

PASS: D/E input and warmup bytes, EOS capture, selector, recovery contract and native schedule patch are unchanged. Actual adapter begin/hold/limiter/uninstall closures are identical to E. Changes only select the saving scope, use common lightweight capture, and disable detailed diagnostic observations in both arms. Syntax, shell and CLI checks passed.

Three existing request-measurement fixtures were reused against the frozen copy: naturalEOS/policy cost, sparse preempt passthrough/restoration, and failure/partial-result retention. All passed. No D/E GPU qualification or full oldEOS suite was rerun. New sparse analysis checks cover1–2delivered outputs before repreemption, terminal without a new output, invalid count alignment and non-scope configuration mismatch. No new SLO/latency threshold.

The reused measurement SHA256 is 1b322be02505381dedb0aaf47f58513dfef61d60bf98e8d9590e18e3012534e4. New/added executable code is 249lines (223new analyzer/check lines plus26added lines in inherited controller/runner/adapter/shell), below500; no new controller policy or predictor.

Reproduce from repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/check_preparation.py
```

GPU UNRUN. Next gate is root accepting this exact version in CURRENT and delegating one whole-group GPU window. Source/runtime hashes, actual4096usable+1null GPU/16GiBhost allocations and process isolation remain runtime checks. The fixed selected/full/full/selected budget is256measured requests; no added diagnostic or repeat.
