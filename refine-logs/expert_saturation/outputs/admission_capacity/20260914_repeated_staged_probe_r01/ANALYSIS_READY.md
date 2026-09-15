# Analysis entry prepared

analyze_repeated_staged_probe.py requires actual COMPLETE arms,32 complete1024-output requests, DRAINED adapter, at least2 matched prepare/commit/raw-preemption events and matching target-new-output releases. Missing/incomplete inputs raise; no synthetic result is emitted. Save-on must have positive completed transfer observations, save-off must have none.

Each engine call contributes once to prefill, mixed-recompute, decode or no-scheduled-token time. Full episode wall minus sum of engine-call durations is recorded separately. Device transfer statistics are not subtracted again from engine time. Full-request mean/wall/maxITL and actual action/cancellation counts are retained. Fixed off→on n=1 cannot establish a stable method effect.

Only source compilation has been checked for this new analyzer. It has not consumed a completed repeated GPU pair. Full model/execution correspondence, overhead decomposition and metric consistency will be checked against actual results.

Command after readback: python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_repeated_staged_probe.py <readback/results> <new-analysis.json>
