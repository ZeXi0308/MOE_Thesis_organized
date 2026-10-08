# Natural cadence: independent document and longer arrival holdout

**CPU PREPARED / GPU UNRUN.** This is H's only package and execution list. Root must accept its exact archive identity before delegating the sole foreground execution. No upload, GPU initialization, queue or background execution belongs to preparation. G and all previous raw artifacts remain immutable; CURRENT remains root-owned.

Question: does the selected/eager0 pause-oriented choice made using G retain a useful full-service tradeoff on128 previously unused complete articles and a longer finite arrival episode? G is selection data; H independently checks new documents. This is not a new policy, independent runtime/model, production arrival model, or observation of host-history turnover.

## Frozen inputs and provenance

Use the same pinned Wikitext-103-raw-v1 **train** Arrow shard, dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, SHA `0c22278eac8186f5dd6060fcf84e4bcb4aa2266c51c77a279169764814f9a99e`. `input_builder_base.py` is the byte-identical prior streaming builder; H imports only its unchanged complete-article reconstruction and hashing helpers. A valid title is an exact raw ` = title = ` row with no internal equals and empty neighboring rows. Concatenate unchanged original rows through the next valid title, excluding an unterminated shard tail. Tokenize the whole text with the same pinned tokenizer, `add_special_tokens=True`, without truncation or padding.

`used_documents.json` freezes224 distinct prior train articles: the five earlier32-document sets plus all64 D–G inputs. The finite-arrival pool, initial16 full natural articles, and current warmups are subsets; their source receipts are included. This conservatively excludes materialized selection/warmup inputs even when their exact GPU use is unknown. The preceding repository inventory was reused, not rescanned during preparation. Test-split row IDs and fixtures are separate namespaces; their row numbers never establish train-row overlap or train-document independence. The concrete exclusion coverage is the repository's materialized224 train articles, not all text ever seen by the pretrained model or unrecorded external sessions.

Starting from the beginning of the pinned shard, exclude those document identities, full-text hashes, overlapping train row intervals and known prompt-token hashes. Select the first128 remaining complete articles with the **pre-existing fixed256–3072 full-token eligibility**. No latency, actual EOS, preemption/action frequency or results enter selection. Stream until128 qualify; do not resample any document after execution.

The scan inspected350 closed articles through row20861:221 prior articles skipped, one outside the fixed length range,128 selected. Selected article intervals run from row8173 to20861 with gaps and do not overlap any prior article, even though the prior set's maximum row end is21131. Actual full lengths are460–3064,247295 input tokens total. `pkg/inputs/INPUT_STATS.json` retains every scanned decision and source receipt. Actual lengths are observed inputs, not newly chosen eligibility bounds.

Freeze source-order arrivals at `i*0.2s`,128 requests,25.4s finite span. EOS is allowed, `ignore_eos=False`, `min_tokens=0`, configured output cap1024; the cap is not an actual-EOS model. No future EOS or request-identity selection enters the online policy. Actual output lengths/sequences and stop reasons may differ and are retained.

Offline input reproduction from repository root uses a **fresh** output directory and the existing local Arrow; it downloads nothing and loads no model weights:

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r01/prepare_inputs.py \
  --source-root . \
  --dataset-arrow "$HOME/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow" \
  --output-dir /private/tmp/natural-cadence-holdout-input-reproduction
```

## Only six cells, fixed order and budget

| Order | Cell | Saving and scheduling | Measurement |
|---:|---|---|---|
|1|block0-native_full_native|native full incremental; native scheduler without added rotation|sparse timing|
|2|block0-current|selected/save-on; global cooldown20|sparse timing|
|3|block0-eager|selected/save-on; global cooldown0|sparse timing|
|4|block1-eager|selected/save-on; global cooldown0|sparse timing|
|5|block1-current|selected/save-on; global cooldown20|sparse timing|
|6|block1-native_full_native|native full incremental; native scheduler without added rotation|sparse timing|

Maximum768 measured requests,128 per cell,180s capture limit per cell. Each has fresh engine/state and identical existing warmups:32 short requests at cap16,32 short at cap32,2 long at cap2;16 generated outputs each and120s per warmup. These396 group warmup requests are separate from768 measured requests. Keep initialization, warmup, reset, drain and teardown costs. Failed/incomplete cells stop the rest and preserve raw artifacts; no silent retries, alternate order, extra diagnostic, parameter scan or best-repeat selection.

Reuse G's accepted package `f1a3bda18a22f7f500024c632cbb6a14b81da391dddd017fd43d5291fe727027` and actual eager/open native saved-recovery qualification. `qualification_reuse.json` points to its raw qualification hash:58 loaded recoveries with subsequent new output and43 sub20 successful commit intervals. These are G evidence, not H observations. No H qualification cell is added. No action in H is a legitimate operating-domain result, not permission to increase pressure.

## Shared resources and execution paths

Same offline OLMoE-1B-7B-0924 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, tokenizer revision, seed20260905, vLLM0.26.0 and cloned environment. GPU0 UUID `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`;4096 usable16-token blocks plus1 null block,2MiB/block, actual tensor bytes8,592,031,744. Unique host KV allocation16GiB/8192blocks must be checked at runtime. Same32running,1024scheduled tokens,4096model limit, full initial-history reservation, FCFS, synchronous single rank, APCoff and `VLLM_USE_SIMPLE_KV_OFFLOAD=0`. Host KV budget is not a process-wide RSS cap; record actual RSS/VmHWM and available cgroup limits/peaks.

Current/eager differ only in global cooldown20/0, including applied `min_steps_between_swaps`. Keep selected/save-on, most-output victim, longest-observed-absence target, absence30/residency30/.90progress/8absence guards, prepare then next-boundary commit, cancellation, pending/flush handling, token/growth allocation, and target protection through the first actual new output or terminal. No commit-funding recheck, window change or new controller.

The native system reference skips `install_rotation`, retaining the original native scheduler and `_calc_num_offloadable_tokens`, with `offload_prompt_only=False`. It changes both saving scope and added scheduling relative to selected/current or selected/eager; do not attribute that system difference to cooldown. Rotation fields remain NOT_APPLICABLE. Native offload jobs in timing remain NOT_MEASURED, never fabricated zeros.

All six use the unchanged G/F `request_measurement.py` with `record_preemptions=True`; preserve native control and sparse observer costs in the measured wall. Detailed eligibility/KV/job/source observers are disabled. Host snapshots retain unique storage, valid KV/pending state, RSS/VmHWM and available parent-cgroup data at init, capture start, request end and drain end. Missing peaks remain UNKNOWN; lifecycle peaks include warmup and overlapping host views are not summed. Boundary occupancy is not an exact continuous host-occupancy peak.

Required final stores/control on the completion path remain in capture wall. Report native post-request drain separately and the non-overlapping `capture_wall + drain_seconds` cost view, excluding intervening snapshot/serialization. A tiny final drain does not make earlier stores free. Raw/failure/drain paths are unchanged from G.

## Evaluation declared before H outcomes

Primary objective remains maximum and per-request engine-return generation gaps. Keep G's complete metrics: actual output/request throughput, mean completion, TTFT, completed requests, output amount/sequence differences, EOS/length distribution, sparse short-service segments, host usage and drain. Do not replace these with internal recomputation or recovery-start proxies.0/1-output gaps remain undefined; multi-token chunks are not interpolated. Exact recovery starts/load jobs/recomputation remain NOT_MEASURED.

**Research cost budget:** for each eager/current pair, allow at most3% loss in output tokens/s and at most5% increase in mean completion. Inside that budget, require maximum gap to decrease. **Both pairs must pass** to accept a stable tradeoff in H's input domain; one pass is MIXED_UNCONFIRMED and no pair is discarded. Undefined/incomplete comparisons remain NOT_ASSESSABLE. These budgets were chosen after G and before H; they are finite service-cost allowances, not business SLOs, noise bounds or statistical-significance claims. No requirement that every metric improve. A no-action episode remains a valid domain result regardless of the numerical comparison.

`performance_comparisons` contains eager−current with only cadence identity normalized. `transfer_budget` applies the above declared limits without choosing thresholds after results. `system_reference_comparisons` separately retains native−current and native−eager for both blocks, with the saving/policy difference explicit. Keep the complete three-arm tradeoff; beating current does not mean beating every baseline. Different natural output amounts prevent an equal-work speedup claim. Sparse1–2-output segments indicate brief delivered service, not their causes or removable recovery-cost bounds. EOS after another request's preemption does not establish EOS during that request's recovery.

The model owner's separate predeclared late64 prediction consumes existing per-request outputs at arrival≥12.8s; it adds no observer, changes no primary metric and does not select requests by their outcomes. H supports only new-document and longer finite-arrival transfer. Host-history turnover and recovery-time EOS remain unverified unless existing data directly establish them; no additional diagnostic is authorized here.

## Unique execution and analysis entry

After root accepts this exact package and delegates the window, stage only a fresh `/root/autodl-tmp/natural-cadence-holdout-20260915-r01`. Sole foreground entry:

```sh
cd /root/autodl-tmp/natural-cadence-holdout-20260915-r01
export PATH=/root/autodl-tmp/expert-saturation/vllm-0.26/bin:/root/miniconda3/bin:$PATH
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python controller.py
```

Manifest verification and exclusive launch-once precede the serial controller. Take the common whole-group flock `/root/autodl-tmp/moe-research-gpu.lock`; check both GPUs and pinned installed sources at initialization/measurement boundaries. Busy lock/GPU or failed queries ABORT without touching others; GPU1 stays unused. Old SSH receipts never imply current availability. Inspect group and all cell statuses, not controller exit alone: inherited controller can record STOPPED while returning zero.

The delegated owner alone executes, preserves failures, archives the whole group, verifies complete readback, explicitly releases resources, runs primary analysis and writes RESULTS. Root does not duplicate raw counting. Human timestamps use `datetime` with `timezone.utc`; retain unix identities.

From repository root, use a fresh analysis output:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r01/analyze_cadence.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r01/execution_weste_26862/readback/results \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r01/execution_weste_26862/analysis.json
```

Evidence ceiling: request-level native in-process host measurements in one controlled model/runtime/resource domain. CPU preparation and inherited G qualification do not establish H results, production guarantees, host turnover, statistical significance or novelty. The only next action after acceptance is this six-cell group; the results determine transfer or a domain boundary, not an automatic controller extension.
