# Natural saved recovery: one open-population qualification

Status: **PREPARED / GPU UNRUN**. Root is the sole execution owner. This package neither launches nor queues work. Run only after root accepts this exact version and the preceding whole-host group releases the authorized GPU pair and lock. Failed occupancy queries or another GPU process abort; no retry, process termination, pressure scan or extra arm belongs to this package.

Question: can the current saved-KV backend execute recovery and expose a real start opportunity when complete natural articles arrive over time and requests finish at an unknown EOS? This is one diagnostic qualification, not a performance comparison or a claim about production traffic. The preceding cooldown comparison is closed; this package fixes cooldown20 and does not repeat that search.

**One cell: `diagnostic-current`.** Reuse all 64 complete natural WikiText articles and their original order from `refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/preparation/pkg/inputs`. Prompt lengths remain334–3011 without new truncation or padding. The sole new arrival point is fixed before execution at0.2seconds/request, spanning12.6seconds. EOS is allowed (`min_tokens=0`, `ignore_eos=False`);1024 is only the declared maximum output count. Actual stop reasons, output lengths and zero-new-output terminal notices remain visible. Future EOS is never supplied to the policy or used to choose a step.

The previous0.5second point is retained without a rerun. Its four original cells completed64requests each; each recorded58length stops,6EOS stops and0forced rotations. Those native/no-host-offload and most-output/no-host-offload diagnostics establish a boundary under their own contract. They are not contemporary saved-KV repeats, a timing baseline for this cell, or proof that the change in arrival rate alone causes any new outcome. Evidence entry: `refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/execution/readback/results`.

Resources are fixed: one authorized GPU0 `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`;4096usable GPU KV blocks×2MiB=8GiB plus one null block. Thus the engine KV argument and actual tensor storage must be **8,592,031,744bytes**, not merely8GiB rounded. Native host KV is16GiB, checked as8192capacity blocks and17,179,869,184unique allocated bytes after deduplicating load/store aliases. The runner records parent cgroup memory limits/charge and process RSS/HWM; these overlap the KV allocation and do not establish a separate process-tree hard cap. Missing `memory.peak` is retained as unknown. The second GPU remains unused, with whole-host process isolation checked before initialization and before measurement.

Use only the existing verified offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, vLLM0.26 environment and model cache. The native source manifest is checked before GPU initialization. Running cap32, model length4096, scheduled-token budget1024, full initial-history reservation, FCFS, no APC, synchronous one-rank execution and native offload remain fixed. There is no model download or second GPU workload.

The online policy remains current most-output victim selection and longest-observed-absence target selection, with the inherited deterministic tie breaking and30/20/30/.90/8guards. Progress uses observed computation and the configured cap, not a natural-EOS prediction. Preparation saves only already materialized complete16-token KV blocks of its selected victim; the victim still decodes once. The next-boundary commit rechecks identities, state, physical blocks and funding. Native incremental store, flush-before-free, validity/eviction, host lookup/load and recomputation fallback remain unchanged. Protection reserves the target's needed GPU capacity until an actual new output; native token allocation and the existing growth guard are unchanged.

The minimal population change removes the requirement to first collect32pure-decode requests with an empty queue. Requests may enter and terminate, and ended identities leave the tracking maps. A protected terminal request is released explicitly; termination without a new output is not called successful output progress. The existing all-pure-running and empty-skipped-queue qualification gates remain. If a new prefill or skipped request invalidates these conditions between preparation and commit, the commit is cancelled through the existing cancellation path, without changing victim ranking or forcing unqualified concurrent recovery. Pending native stores remain under native ownership/lifecycle rules.

The diagnostic records resource funding and qualification gates separately: no preempted waiter, mixed prefill, native recovery underway, skipped/pending-load queue, active protection, commit state and current selector reasons. It also records the legacy closed-activation predicate without using it as a gate. Funded observations use the same longest-absence target and fixed most-output victim; they are not a search over targets/victims or a sustainable-window Oracle. Therefore “no action” can be separated into no observed resource opportunity versus an observed opportunity blocked by policy/backend qualification.

Measured budget: at most64requests, each with output cap1024, one180-second capture. Keep the same three inherited warmups:32short/cap16,32short/cap32,2long/cap2, each requesting16outputs and individually capped at120seconds. Initialization, each warmup raw, warmup drain/reset, the measured interval, later transfer drain and shutdown are recorded separately. The66warmup requests are outside the64measurement requests. No measured preemption, saving or rotation count is required for a valid complete diagnostic. Zero action is retained and ends this version; do not increase pressure until action appears.

Analysis preserves completed store/load jobs and bytes, cache snapshots, actual recovered work, L/E/S/F, new outputs before re-preemption or completion, and actual stop/length counts. E is observed full-history funding, not complete dispatch readiness. S is host load submission or a recovery-bearing engine-call entry, not physical DMA/kernel start. Engine-return output is not client receipt. Diagnostic wall, throughput, mean completion and gaps are descriptive with instrumentation costs; they establish no comparative service gain. Incomplete execution stays incomplete, never a problem-level NO-GO.

A complete cell with committed rotations and completed native save/load establishes this open-population path's qualification only. Recovery with no such action retains its precise resource or qualification boundary. This package contains no eager arm, new window, predictor, alternative victim rule or runtime matrix; root chooses any later study after seeing this single result.

After version acceptance and prior-group release, unpack on the new host and use the sole foreground entry:

```sh
cd /root/autodl-tmp/natural-saved-recovery-gate-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

The controller verifies `manifest.json` before creating its launch-once receipt. Execution entries are `group-status.json`, `campaign.log` and `results/diagnostic-current`. Retain the entire cell on failure or no action. After readback, reproduce the CPU analysis from the repository root with a fresh output path:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/analyze_natural_gate.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/execution_weste_26862/analysis.json
```

The previous packages, current experiment register and paper argument are not modified by this preparation. Preparation and CPU fixtures are neither GPU qualification nor a paper contribution.
