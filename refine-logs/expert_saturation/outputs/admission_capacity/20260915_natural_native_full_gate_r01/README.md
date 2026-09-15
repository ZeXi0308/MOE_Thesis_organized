# Natural native-full saving: one qualification cell

Status: **PREPARED / GPU UNRUN**. Root is the sole execution owner. No launch, background queue, model transfer or download occurs in preparation. The sole cell is `diagnostic-native-full`; start only after root accepts this exact archive and the preceding whole-host group releases the authorized GPU pair and lock. Failed occupancy queries or another GPU process abort without killing or replacing that process.

Question: does unrestricted native incremental saving remain compatible with D's current recovery scheduler, and does it actually save/load KV outside selected-victim preparation? This establishes the stronger saving baseline's path, not a timing comparison, service-gain claim or new controller contribution. D's selected-saving diagnostic is reused under its own contract, never relabeled as this cell. The older `20260914_native_offload_baseline_r01` is prompt-only and does not qualify native-full.

**Only the saving scope changes.** `store_scope=native_full` leaves the original native `_calc_num_offloadable_tokens` class method untouched; `offload_prompt_only=False` is inherited from D. A pre-existing instance override is rejected. The selected branch retains its previous limiter, but this run never selects that branch. Original `_build_store_jobs()` considers scheduled and finished requests, native complete chunks, existing store progress and host allocation/eviction. Full means unrestricted native incremental eligibility; it does not mean every historical KV block remains cached, every possible job succeeds, or all generated positions are saved before they are computed. Neither actual future EOS nor an outcome-selected step enters the mechanism.

D's workload and resources are byte-for-byte reused:64complete natural articles, prompt334–3011, fixed source-order0.2s arrivals, EOS allowed with `min_tokens=0` and `ignore_eos=False`, output cap1024,4096usable16-token GPU blocks plus1null block (engine/tensor bytes8592031744),16GiBunique host KV/8192capacity blocks. The runtime verifies actual GPU and host allocation. Running cap32,1024scheduled tokens, model limit4096, FCFS/full-history reservation/no APC, one synchronous rank, current30/20/30/.90/8guards, most-output victim and longest-absence target remain fixed. The1024cap is a configured maximum, not a natural-EOS prediction. Input JSON retains its D preparation provenance; runtime `config.json` explicitly records `store_scope=native_full`.

The same prepare→next-boundary commit remains in both saving scopes. Victim selection, one preparation decode, target priority, funding checks, mixed-prefill/skipped-queue cancellations, pending-load handling, protection until actual new output/terminal release and other-request growth allowance are unchanged. Full saving may create more store/load jobs, host eviction or waits; these are real consequences, not reasons to restrict full saving back to the selected prefix or relax its scheduler. Pending/queue incompatibility is retained as `INCOMPLETE`.

The selected-only `inspect_store_delta()` prefix bound is not applied in native-full. A new read-only observer validates each native store's registered request/job, logical source/key mapping, and range under the actual native calculation. At adapter return, native `_update_after_schedule()` has already advanced computed tokens; the observer never adds scheduled tokens a second time. Thus a block completed by the preparation step can legitimately exceed the old pre-step selected prefix. Observation does not modify metadata, truncate jobs, call a load lookup or alter allocation.

Evidence includes all native source blocks/key representations, jobs created outside current victim preparation, chunks containing decode positions, any preparation-boundary extension, known-finished-request jobs, accepted/completed job-ID joins and flush metadata. The worker's native deferred-store submission/wait ordering is retained. Full-range checks do not claim numerical KV fidelity or full residency. On observer failure the attempted source metadata and error remain in the partial artifact. D's generic natural request/recovery analysis is reused; scope qualification and recovery qualification remain separate.

Initialization uses only the existing verified offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5` and cloned vLLM0.26 environment on GPU0 `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`. Keep the existing three warmups (32short/cap16,32short/cap32,2long/cap2;16outputs each), then drain and reset the connector. Source hashes and isolation are checked before initialization/measurement. The second GPU stays unused; `/root/autodl-tmp/moe-research-gpu.lock` covers the whole group.

Budget: one64-request diagnostic, output cap1024, maximum180seconds for measured capture; inherited warmups each retain their120-second cap and66warmup requests are outside the measured64. No eager arm, repeat, pressure adjustment or parameter scan belongs to this version. No action or no full-scope witness is retained and ends the cell, not converted into a problem-level NO-GO.

Request timestamps, internal work, host load dispatch, first new output and complete-service descriptors remain separate. This full diagnostic may incur more observation cost because it creates more jobs; it is not a lightweight timing arm. Preserve initialization/warmup intervals, actual stop/lengths, final stores, post-request drain, host valid entries, RSS/HWM and parent cgroup readings. Missing peaks remain unknown; overlapping memory views are not summed. EOS terminal with `F_s=None` has no observed first new output even if the inherited completed-recovery count increments. Neither D's individual long gaps nor this cell's transfer times are a removable-cost upper bound.

After root accepts the version and verifies prior-group release, the sole foreground entry on the new host is:

```sh
cd /root/autodl-tmp/natural-native-full-gate-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

The controller checks `manifest.json` before its launch-once receipt. Result entries are `group-status.json`, `campaign.log` and `results/diagnostic-native-full`; retain all failure/raw artifacts and never reuse a stopped directory. After readback, reproduce analysis from the repository root into a fresh file:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/analyze_native_full_gate.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/analysis.json
```

Preparation modifies no prior package, shared source, current-experiment record or paper argument. CPU checks and archive readiness do not establish native-full GPU qualification.
