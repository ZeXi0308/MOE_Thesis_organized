# Natural recovery cadence with native system reference

**PREPARED / GPU UNRUN.** This is the sole current package for this group. The earlier five-cell draft was never accepted, packaged or run. Root must accept this exact package hash before the delegated recovery-execution owner starts it. No GPU initialization, upload, queue or background task is part of preparation. Existing C/D/E/F artifacts remain immutable.

Question: on F's selected/current baseline, does removing the global cooldown convert waiting into earlier new outputs while retaining a useful complete-service tradeoff? A native-scheduling/full-saving reference also tests whether either custom scheduling arm improves the same-resource system baseline. It is not a cooldown-only counterfactual.

F completed both selected/full pairs with current scheduling. Native_full had longer maximum gaps in both, with mixed efficiency effects, so selected/current remains the pause-oriented reference for the cadence question. F did not measure full native saving without added rotation. The old0.5s/no-offload episode is not that baseline. This group closes that specific missing reference without changing F's identity or explanation.

The model's existing `20260915_joint_growth_decision_r01/START_CADENCE_DECISION.md` identifies201 D/current snapshots across18 old switching intervals where timer removal can affect a mature, fixed-most-funded proposal. These are not READY or feasible-action counts:3 also exceed free capacity after common prepare growth. They justify a bounded intervention, not a predicted speedup. Existing online selection uses currently observed absence, physical KV, progress and guards; it does not consume these offline row identities, future EOS or chosen steps.

## Single accepted run order and budget

| Order | Cell | Saving / scheduler | Measurement |
|---:|---|---|---|
| 1 | diagnostic-eager | selected / existing open adapter, cooldown0 | one new combination qualification |
| 2 | block0-native_full_native | native_full / no rotation adapter | sparse timing |
| 3 | block0-current | selected / current cooldown20 | sparse timing |
| 4 | block0-eager | selected / eager cooldown0 | sparse timing |
| 5 | block1-eager | selected / eager cooldown0 | sparse timing |
| 6 | block1-current | selected / current cooldown20 | sparse timing |
| 7 | block1-native_full_native | native_full / no rotation adapter | sparse timing |

Maximum448measured requests:64per cell, one diagnostic plus six timing cells. Each capture retains its180s limit. Every cell uses a fresh engine/state; no retries, extra diagnostics, alternate order, parameter scan or best-repeat selection. The native/current/eager/eager/current/native order retains both contemporary blocks. A failed/incomplete cell stops the remaining group and preserves all artifacts.

One eager diagnostic is needed because C qualified cooldown0 only in a closed cohort, whereas D qualified the open-population adapter only with cooldown20. The combination can reach pending-load/new-arrival/protection states at different times. Reuse D's current diagnostic, C's existing0/20 primitive and D/E native-path qualification; do not repeat current qualification. The adapter source is byte-identical to F.

The diagnostic must complete64requests under the unchanged guards, contain committed rotations and completed native store/load jobs, observe at least one loaded recovery segment with an actual subsequent new output, and execute at least one successful READY commit less than20steps after the previous successful commit. READY events must match the actually applied rotation count; cancelled/proposed prepares do not count. Without a sub20successful-commit interval, retain `NO_NEW_CADENCE_ACTION` and stop all subsequent timing. This is evidence that cooldown removal affected real action cadence, not a benefit count. `analyze_cadence.py --qualification-only` records its result before timing. If action or productive loaded recovery is absent, retain that result and stop; do not manufacture pressure, weaken guards or silently continue. EOS ending a segment without new output remains a valid terminal outcome, but does not establish a productive loaded-return witness. Failure or pending incompatibility remains INCOMPLETE. This gate is a combination-path check, not a performance threshold.

Diagnostic L/E/S/F and work remain separate: E is observed necessary funding, not READY; S is a host submission/engine-call boundary, not a GPU kernel-start timestamp; F is a genuinely returned new output. Detailed store/load/flush and completion observers are used only here. Its costs and timestamps are not paired with historical D diagnostics or mixed into the timing result. It cannot prove a matched internal-start speedup by itself.

## Shared execution and single-factor boundary

Reuse F's64unaltered complete natural articles,334–3011prompt tokens, source-order0.2s arrival spacing, `ignore_eos=False`, `min_tokens=0`, configured output cap1024. The cap is not an actual-EOS model. Actual output tokens and stop reasons may differ and must be retained. Inputs/warmups keep their original preparation metadata; this README, controller and manifest own this group's order/budget and arm identity.

Use the existing offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, vLLM0.26.0 and the cloned environment. Fixed GPU0 UUID `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`;4096usable16token blocks plus1null,2MiB/block, actual tensor bytes8,592,031,744. Unique host KV16GiB/8192blocks must be checked at runtime. Same32running,1024scheduled tokens,4096model limit, FCFS, full initial-history reservation, synchronous1rank, APCoff, native offload kwargs and `VLLM_USE_SIMPLE_KV_OFFLOAD=0`. This is a controlled same-resource runtime, not a claim about unconstrained default parameter settings or production traffic.

Current/eager differ only in `min_steps_between_swaps`20/0, recorded both in config and the applied adapter config. They keep selected saving enabled, most-output victim, longest-observed-absence target, absence30/residency30/.90progress/8absence guards, prepare→next-boundary commit, cancellation, native pending/flush handling, token/growth allocation and target protection until actual new output or terminal. No commit-funding recheck, protection extension or new scheduler is installed.

`native_full_native` is a minimal runner mode that skips `install_rotation`. It leaves the original native scheduler and `_calc_num_offloadable_tokens` class methods intact, with `offload_prompt_only=False`; it does not install a replacement scheduler. Installation rejects pre-existing instance overrides/rotation hooks. It reuses E/F's qualified full native saving, F's arguments/resources and the same lightweight measurement. Rotation count/config/events are NOT_APPLICABLE; raw policy lists are empty because no policy observer exists, not because they measure zero native activity. Actual native preemptions remain observed. Native full-saving jobs are NOT_MEASURED in timing, never fabricated zeros. No extra native diagnostic is added.

## Evaluation and retained costs

Every timing cell calls the same F `request_measurement.py` with `record_preemptions=True`. The wrapper preserves the native method and actual returned-output counts. Detailed scheduler/KV, eligibility, prefix-source and transfer observers are disabled in all six timing cells. Native service/control costs and the common sparse hook remain in the measured wall. Exact recovery starts, job counts and recomputation are NOT_MEASURED in these cells. Sparse preemption-to-output intervals include waiting and other work; they are not removable recovery-cost bounds.

The fixed primary objective is maximum and per-request engine-return generation gap. Report actual output/request throughput, mean completion, TTFT, completed requests, output volume/sequence differences and EOS/length distribution alongside it. Preserve both pairs and every system-reference comparison; no new SLO cutoff, metric substitution, best-repeat selection or requirement that all metrics improve. Natural output changes prevent an equal-work speedup claim from elapsed time alone.0/1output gaps stay undefined; multi-token chunks are not interpolated. Report existing1–2output/repreemption segments without inventing their causes or a new intervention threshold. Preserve EOS relative to that request's own preemption/returned-output/terminal history; EOS after another request preempts does not qualify recovery-time EOS.

The primary analyzer has two separate outputs: `performance_comparisons` contains only eager−current, normalized solely for variant/cooldown identity; `system_reference_comparisons` contains native_reference−current and native_reference−eager for each block. The latter explicitly differ in both saving scope and added scheduling, so are system comparisons, not timer attribution. All other resource/model/request/arrival/cap/measurement configuration is compared. A stronger native reference can invalidate the practical motivation even if eager improves over current.

Initialize and warm up identically:32short at cap16,32short at cap32,2long at cap2,16outputs each, each warmup limited to120s. These66warmup requests/engine are outside448measured requests. Retain all warmup/init/drain/reset/teardown intervals and raw files. Host snapshots afterinit/beforecapture/requestend/afterdrain preserve unique storage, ready KV, pending work, RSS/VmHWM and parent cgroup readings. Lifecycle peaks may include warmup; missing cgroup peaks remain UNKNOWN. Host views overlap and must not be summed; boundary occupancy is not an exact time-series peak.

Capture ends after request completion and includes required final stores/control on that path. Report native post-request drain separately and `capture_wall + drain_seconds` plus output rate as a non-overlapping cost view, excluding intervening snapshot/serialization. A tiny post-request drain does not imply stores were free. Exact observed GPU/host allocations and failure raws remain mandatory.

## Unique execution and analysis entry

Only after root accepts the exact archive and delegates the window, stage a new directory `/root/autodl-tmp/natural-recovery-cadence-20260915-r01`. The sole foreground entry is:

```sh
cd /root/autodl-tmp/natural-recovery-cadence-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

The controller checks the manifest and creates launch-once before the serial script. The common whole-group flock is `/root/autodl-tmp/moe-research-gpu.lock`. Verify both GPUs and installed source hashes at initialization/measurement boundaries; other compute processes, failed query or occupied lock abort without interference. GPU1 stays unused. Inspect group/cell status and diagnostic-qualification, not controller exit alone; the inherited controller records STOPPED even if its child failure does not propagate to its own exit code.

The delegated execution owner uniquely executes, archives, verifies readback, releases the window and performs main analysis. Root does not duplicate counting or readback. Keep one complete immutable archive including unsuccessful attempts, with a SHA receipt. Use `datetime` with `timezone.utc` for human-readable UTC timestamps; unix times retain execution identity.

From repository root, use a fresh output path:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/analyze_cadence.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/execution_weste_26862/analysis.json
```

Evidence ceiling: one controlled arrival regime, native in-process host measurements, not a production SLO or novelty claim. No internal-work proxy replaces complete service. Positive, mixed, absent-action and unfavorable-reference outcomes are all retained for root's next research decision; preparation does not establish any of them.
