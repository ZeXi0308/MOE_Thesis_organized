# Two-stage save/preempt CPU qualification

Status: CPU_CONDITIONAL_CONTRACT_PASS. GPU integration UNRUN.

Question: can a one-step preparation preserve an eligible victim and reserve the original absent request's recovery space?

Reused frozen step 329 and native block0 steps 329/330: victim 0000001 advances 3398→3399 computed, 327→328 output, retains 213 allocated blocks; original absent 0003640 remains unchanged. Free blocks fall 148→145. The plan saves only 3392 already computed tokens (212 complete blocks). Ten targeted cases pass, including cancellation on changed state, missing store, changed ownership and lost funding; pending load cannot execute and its outstanding allocation is reserved.

The raw next-state records block counts, not physical IDs. Stable IDs and a registered store in the success test are explicitly fixtures. READY means eligible to invoke the native fenced preemption path, never permission for unfenced block reuse. Actual registration, completion fence, asynchronous queue integration and full-request outcomes remain unverified. This is neither a performance model nor a method GO.

Strongest baseline remains most-output; eventual comparison needs original most-output and identical staged actions with saving disabled to charge the preparation cost. No new Oracle bound. Current failure category is unverified integration, not scientific NO-GO.

Next smallest experiment: one native staged event, verify store coverage and completion before block reuse, original-target queue priority and load readiness. GPU execution waits for the existing Qwen group and queued funding-filter group to release resources. SSH currently closes before authentication, so remote state is UNKNOWN; no new job was started.

Command: `python3 refine-logs/expert_saturation/experiments/admission_capacity/check_staged_save_contract.py`
