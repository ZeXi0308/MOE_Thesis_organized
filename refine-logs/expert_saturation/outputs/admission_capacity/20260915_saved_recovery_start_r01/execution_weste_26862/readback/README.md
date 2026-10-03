# Saved recovery start: remove one global cooldown

Status: PREPARED / UNRUN. Root is the sole execution owner and must accept this exact archive before launch. This package creates no background job or queue entry. Existing shared-host groups must release both GPUs and the whole-host lock before startup; busy GPUs or failed occupancy queries abort. Results belong only to this version.

Question: with native KV saving enabled, does removing the global 20-step rotation cooldown reduce long generation pauses at an acceptable measured service-efficiency cost? The changed field is `RotationConfig.min_steps_between_swaps`: current=20, eager=0. Both values use the same online observations and selector. Zero removes only this global gate; it does not force immediate restoration or change eligibility, resource funding, target rank or token allocation.

Inherited invariants: minimum absence 30 steps, minimum victim residency 30 steps, most-output victim order, longest-observed-absence target order with unchanged deterministic tie breaking, maximum 8 absences, progress guard 0.90 against the declared output cap, staged preparation/commit, and target protection until its first actual new output. No actual EOS, outcome-selected step or learned request identity enters the action. Preparation still lets the victim decode one token, saves only its materialized full 16-token KV blocks, and excludes the unmaterialized decode position/trailing partial block. Native store completion/flush, cache validity/eviction, lookup/load and recomputation fallback are unchanged. Both arms use saving; current/eager never mean save-off/on.

Frozen order: **diagnostic-eager → block0-current → block0-eager → block1-eager → block1-current**. All cells run serially on GPU `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`, using `/root/autodl-tmp/moe-research-gpu.lock`; the second GPU remains free for process isolation. The existing current diagnostic from `20260915_repeated_kv_service_r01/execution_weste_26862/readback/results/diag-on` is reused for descriptive localization only. It is not a new-contract repeat and is excluded from the two new timing pairs.

The diagnostic must complete all 32 requests, apply the recorded eager configuration, execute native stores and loads, and contain a committed preparation less than 20 steps after the preceding committed preparation. This last condition demonstrates a realized action that the current global cooldown forbids in that state; it is not a counterfactual performance estimate. Invalid diagnostics or no new action retain their raw data and leave all timing cells UNRUN. Four primary timing cells proceed only after this action gate passes. No favorable-result threshold controls which timing repeats survive.

Resources and workload: the already verified cloned vLLM 0.26 / torch 2.11 / Transformers 5.15 environment and offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; 6656 usable 16-token GPU KV blocks / 13,960,740,864 bytes; native host KV capacity 16 GiB; one rank; 1024 scheduler token budget; 32 requests per cell, 3072 prompt tokens, declared/forced 1024 outputs, original 50 ms arrival trace. The value 1024 is the configured output cap used by this fixed-length experiment, not an observed natural EOS or a model of unknown natural completion. Measured budget is at most 160 requests / 163,840 generated tokens across five cells, with at most 120 seconds per measured episode. Each process retains the identical three inherited warmups (32 short at cap16, 32 short at cap32, 2 long at cap2; 16 outputs each), followed by drain and connector-cache reset. Warmups are outside the measured request budget and performance denominator. No additional model copy/download or repeat is authorized by this package.

Primary objective: long generation pauses, reported as the maximum engine-return new-output gap and its per-request distribution. Necessary costs: full-cohort makespan/output rate and mean request completion; TTFT is also retained. Exploration reports the tradeoff and both balanced pairs separately; no requirement that every metric improve, no post-hoc SLO threshold, no best-repeat selection. All native saving/loading, scheduling, recomputation and protection costs remain in the measured engine path. Initialization/warmups are excluded; post-request drain is reported separately and never added twice to overlapping transfers.

The eager diagnostic retains native stores/loads, executed recovery work, L/E/S/F and re-preemption/useful-output evidence. E is only observed resource funding, not complete policy readiness. S is host load submission or a recovery-bearing engine-call boundary, not DMA/kernel start. Engine calls can enclose preemption/E, so negative sub-call waiting is not a causal acceleration. Diagnostic timings never enter primary performance comparisons. Host allocated KV, cache-valid entries, process RSS/HWM and parent cgroup usage are overlapping views, not summable allocations. Full-service outcomes, less recomputation, earlier recovery start and earlier first new output are separate conclusions.

Interpretation is restricted to this saved-KV closed cohort. Consistent gap improvement with a useful service tradeoff supports this minimal gate change; earlier recovery followed by repeated preemption redirects the remaining question to resource evolution/service allocation. A cost-dominated or inactive gate ends this formulation. Small sign reversals require a discriminating analysis or a separately accepted bounded follow-up, not hidden repeats. The next regime expansion is a separate root decision, not part of this package.

Payload provenance: all inherited `pkg/` files match the prior package's manifest except `run.sh`, `run_probe.py` and `staged_store_rotation.py`; `saved_kv_analysis_base.py` reuses its completed-group analyzer unchanged. `manifest.json` hashes every frozen payload file and is checked by the controller before creating its launch-once receipt. The archive SHA is reported outside the archive to avoid self-reference.

After explicit root package acceptance and prior group release, the sole executor runs on the new host:

```sh
cd /root/autodl-tmp/saved-recovery-start-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

This command runs in the foreground. `group-status.json`, `campaign.log` and `results/<cell>/` are the execution entries. An interrupted or stopped directory is never reused. Retain all raw outputs and transfer them as one readback bundle. After readback, run from the repository root; analysis writes a fresh output path:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01/analyze_saved_recovery_start.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01/execution_weste_26862/analysis.json
```

There are no GPU results yet. Root owns the single current experiment record and paper argument outside this immutable package.
