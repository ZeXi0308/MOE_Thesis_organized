# Natural save scope: balanced complete-request timing

Status: **PREPARED / GPU UNRUN**. The recovery-execution owner prepared this package and will execute/archive/analyze it only after root accepts this exact version in `CURRENT_EXPERIMENT.json` and formally delegates the available GPU window. Preparation does not launch or queue work. There is one execution owner for the whole four-cell group; root does not duplicate its counts or measurements.

Question: under the same current scheduler and fixed GPU/host budget, how does selected-victim saving compare with unrestricted native incremental saving on complete natural requests? D and E already qualified the two native paths. Their diagnostic timings are not performance arms, are not merged into repeats, and are not rerun here. This group adds no diagnostic qualification, new controller, cooldown scan, window, victim rule or pressure point.

**Frozen order: `block0-selected → block0-native_full → block1-native_full → block1-selected`.** Retain both contemporaneous pairs independently, including unfavorable or small sign changes. No best-repeat selection or automatic extra repeat. Maximum measured budget is256requests,64per cell, each with configured output cap1024; four180-second measurement limits. Each cell starts a fresh engine and policy state.

Inputs and resource configuration are reused byte-for-byte from qualified D/E:64complete natural articles, prompt334–3011, source-order0.2s arrival interval, EOS allowed. The runtime explicitly sets `ignore_eos=False`, `min_tokens=0`;1024 is the configured maximum, never an actual-EOS model or online future label. Actual request output counts and stop reasons may differ between arms and remain part of the result. Warmup/input JSON retains its preparation provenance; execution identity/order is this README, controller and accepted manifest.

Hardware/runtime: authorized GPU0 `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`,4096usable16-token GPU KV blocks plus1null block, actual tensor/engine bytes8592031744;16GiBunique native host KV/8192blocks. Verify both actual allocations. Use only the already verified cloned vLLM0.26 environment and offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; no downloads or second GPU workload. Whole-group lock `/root/autodl-tmp/moe-research-gpu.lock`, source hashes and physical GPU process checks remain mandatory at initialization and measurement boundaries. Busy state or failed query aborts without terminating another process.

The only experimental factor is `store_scope=selected|native_full`; saving is enabled in both. Selected keeps the qualified preparation-only materialized-prefix limiter. Full leaves the original `_calc_num_offloadable_tokens` class method untouched and uses `offload_prompt_only=False`; it does not promise permanent host residency. The same current30/20/30/.90/8guards, most-output victim, longest-absence target, prepare→next-boundary commit, physical funding checks, cancellation, pending-load treatment, protection until actual new output/terminal release and native token/growth allocation remain. Running cap32, scheduled-token budget1024, model limit4096, FCFS/full initial-history reservation/no APC and synchronous one-rank execution are fixed. No actual future EOS or learned request identity enters the action.

Both arms use the identical existing `request_measurement.py` lightweight capture with **sparse `_preempt_request` recording enabled**. It chains the real method and records host entry/return plus last actually returned output count/time; the original return value, failure and existing hooks are preserved. That cost remains in both measured paths. Full schedule/KV/memory tracing, eligibility snapshots, per-job source/prefix diagnostics, metadata logs and lookup/worker/completion observers are disabled in both arms. Native calculation, store/load, flush, completion handling and policy control still execute normally. Transfer/job counts, exact recovery-start times and recomputation are therefore `NOT_MEASURED`, not zero; their path qualification is inherited from D/E. The legacy `policy_hooks_modified` raw flag denotes this observation wrapper, not a different controller policy.

Initialization and warmups stay matched:32short at cap16,32short at cap32,2long at cap2, each requesting16outputs; then native drain and connector-cache reset. Each warmup retains its120-second limit. These66warmup requests per engine are outside the256measured requests. Preserve initialization/warmup intervals and warmup raws. Host snapshots are restricted to initialization, before capture, request end and after drain, retaining actual cache state, VmHWM/RSS and parent cgroup readings. Lifecycle/cgroup peaks may include warmups or prior shared activity and are not called exact measured-episode peaks; missing values stay unknown. Memory views overlap and must not be summed.

Primary objective remains maximum and per-request engine-return generation gap. Report output/request throughput, mean completion, TTFT, completed-request count, output volume/length and stop distribution alongside it. Show the tradeoff without demanding every metric improve or selecting a new SLO threshold. Natural EOS can change the amount of generated work, so an elapsed-time difference alone is not equal-work acceleration. Undefined gaps for0/1-output requests remain unknown; multi-token chunks are not interpolated. Legacy reference-SLO fields in inherited metrics do not determine acceptance.

Sparse analysis reports actual new outputs between successful preemptions and until completion. A1–2-output segment followed by another preemption directly identifies brief delivered service after recovery; zero returned outputs does **not** establish that an expensive recovery executed. Exact load/recompute/start causes are not inferred from sparse events. Existing short-segment counts are retained without choosing a new threshold. EOS terminal without a new output remains distinct from useful service.

Capture wall includes all live policy, saving/loading and measurement-hook costs. Native post-request drain is separately reported. Also show `capture_wall + drain_seconds` and the corresponding output rate as a non-overlapping resource-cost view; exclude intervening host snapshots/serialization from that sum and never add overlapping transfer intervals. Pending or policy incompatibility retains an incomplete cell and raw data; it does not authorize loosening common guards or silently rerunning.

Once the exact version is accepted and the window delegated, the sole foreground command on the authorized host is:

```sh
cd /root/autodl-tmp/natural-save-scope-timing-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

The controller verifies `manifest.json` before its launch-once receipt. Entries are `group-status.json`, `campaign.log` and the four `results/<cell>` directories. Retain all raw/failure artifacts, never reuse a stopped directory, and make one readback bundle. The execution owner runs the single primary analyzer from the repository root with a fresh output path:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/analyze_scope_timing.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/execution_weste_26862/analysis.json
```

The owner produces the main request/cost analysis; root uses it for research decisions. Preparation does not modify D/E, shared source, the current-experiment register or paper argument. Package readiness is not a new GPU result.
