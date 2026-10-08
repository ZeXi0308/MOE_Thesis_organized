# Fresh targeted experiment integrity review

Auditor: /root/staged_event_audit, GPT-5.6-Sol ultra, fresh same-family provisional.
Overall WARN. No P0/P1 for SINGLE_EVENT_EXECUTION_QUALIFIED / MEASUREMENT_ONLY.

A PASS: model/workload identities match and no quality GT claim is made. Reviewer independently found identical32×1024 output sequences; this is not dataset quality or bitwise KV fidelity. Evidence: readback/results/save-{off,on}/config.json:2-37,49-56; environment.json:12-32; manifest.json:2-20.
B PASS: complete32-request denominators, off baseline percentage denominator, raw time/recompute metrics and phase counts independently matched. Evidence: analyze_staged_store_probe.py:17-19,48-55; readback/pkg/native_capture.py:64-100; metrics.py:62-155.
C WARN: principal metrics, archive19 hashes, store114 at329/flush330 and completed store/load match. Evidence: save-on/selective-store.json:225-248; offload-events.json:5331-5374; readback/group-status.json:2-25. Original action files retain initialization status INSTALLED_GPU_UNVALIDATED; that field is stale, not the execution verdict.
D PASS: run_probe installs and invokes the staged wrapper/observer; original rotation installer is not called. Evidence: readback/pkg/run_probe.py:171-202; staged_store_once.py:67-157.
E WARN: fixed off→on n=1 and pre-action1.140924s drift prohibit causal timing claims. preparation_prefix_equal checks scheduled work, not thermal, clock, complete physical state or early queue counts. Evidence: analyze_staged_store_probe.py:53-54; time_and_calls.json:40-44; REPORT.md:20-24.
F PASS: NATIVE_SERVING_INPROCESS_HOST_CAPTURE, one event/arm, one OLMoE workload on one5090. No full-rotation/Oracle/quality/multi-GPU claim. Evidence: analysis.json:63-69; REPORT.md:16-24.

Unchecked: no exhaustive line-by-line audit of the two~287MB raw files, no new GPU replay, no installed vLLM inspection beyond retained hashes, no separate structural-model audit. Review stopped at this bounded scope. No .aris generated per repository instructions.

Correction for downstream consumption: preserve immutable original action JSON; use readback/group-status.json and per-arm status.json plus the validated event/worker chain. Their COMPLETE status does not upgrade this to a method GO. New repeated adapter now emits DRAINED/ERROR on uninstall; that new code was not covered by this review.
