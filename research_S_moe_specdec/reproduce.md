# Reproduction entry

## Environment

Measured target: single RTX PRO 6000 Blackwell Server Edition, vLLM 0.26.0,
PyTorch 2.11.0+cu130. This initial campaign uses eager execution, synchronous V1,
Triton modular MoE, BF16 OLMoE-1B-7B-0924-Instruct, full resident weights and 6 GiB
KV cache. SM120 identity alone does not establish transfer to RTX 5090.

The relevant installed source snapshot is in `runs/environment_20261016/`.
The process-local patch is `code/vllm_s_plugin.py`; it never edits site-packages.
The remote working directory is `/root/autodl-tmp/research_S_moe_specdec`.
No credentials are stored here.

## Process-local integration and switches

Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `VLLM_USE_V2_MODEL_RUNNER=0`
before importing vLLM or calling `install(config)` from
`code/vllm_s_plugin.py`. Install **before engine construction**, then call
`runtime.attach_engine(engine)` on the returned runtime. The current integration
uses a single-GPU V1 engine, `enable_multiprocessing=False` and
`async_scheduling=False`; it does not establish support for the default
multiprocess `vllm serve` path. Calls to `runtime.start_episode(label, trace=False)`
and `runtime.end_episode()` delimit observation records. Nothing edits
site-packages or requires an additional installer.

For adaptive control, `run_matrix.py` constructs
`AdaptiveGammaPolicy(CalibrationTable.from_json(path), AcceptanceEWMA())` and
passes that policy and its same acceptance object to `install`, with
`mode="adaptive"` and `max_gamma=7`. The measured table must match the model,
draft source, execution mode, resource domain and full production budget7.
Once the compiled preparation has produced that table, the existing locked
runner provides the shortest complete entry:

```bash
python code/run_campaign.py \
  --config configs/olmoe_ngram_compiled_controller_evaluation.json \
  --run UNIQUE_COMPILED_ADAPTIVE_RUN --gammas 7 --mode adaptive \
  --proposal-budget 7 \
  --calibration runs/controller_preparation_compiled_r01/calibration/calibration.json \
  --skip-trace
```

`mode="native"` disables plugin truncation while retaining scheduler/acceptance
observations and vLLM's native batch table. `static` uses `static_gamma` as the
scheduled cap; `dense_heuristic` uses that cap up to `disable_above_batch` and
zero above it. `adaptive` selects the cap from measured costs and online state.
Plugin `mode="disabled"` means **cap0**, not removal: with the default plugin
`proposal_budget="max"` and an engine budget7, the engine still produces drafts
that are discarded. AR instead requires an engine with no speculative budget.
For complete removal of the Python patches, first shut down the engine, then
call `uninstall()`; never uninstall from a live engine.

The runner sets `strict=False` for compiled performance with `trace=False` and
native route return disabled. Graph route diagnostics require a separate engine
with plugin `native_route_return=True` and engine
`enable_return_routed_experts=True` set before construction; their timings are
instrumented. Optional `native_capture_padding=True` additionally records
physical graph rows and requires the GPU route backend, single TP/PP/DP, and
EPLB/DBO disabled.

## Prepare and run

Use the existing complete model at
`/root/autodl-tmp/moe-research-20261002/model` (revision
`7f1c97f440f06ce36705e4f2b843edb5925f4498`). ShareGPT source revision is
`192ab2185289094fc556ec8ce5ce1e8e587154ca`; the preparer verifies the complete file
SHA256 before selecting original-order prompts of 32–256 tokens, without truncation.

```bash
python code/prepare_workload.py \
  --source /root/autodl-tmp/research_C_format_sched/datasets/serving_sources/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /root/autodl-tmp/moe-research-20261002/model \
  --output configs/sharegpt_olmoe.json
python code/run_campaign.py --config configs/olmoe_ngram_matrix.json \
  --run UNIQUE_NAME --wait-lock-seconds 1800
```

The campaign claims the existing shared GPU lock for the entire group, verifies
no compute processes at engine boundaries, snapshots its code, and stops after
the first implementation failure or budget limit. It retains prior failures;
use a new run name after fixing their cause. GPU lock duration includes loading
and warmup and is charged against the 20-hour budget. Never start a second
controller when a previous one is still running.

The first 64 selected prompts are identical across all cells. A batch-B cell
processes them as 64/B synchronous waves. Each request generates 128 tokens with
EOS ignored; this is controlled-workload generation, not natural task completion.
Two independent execution repeats use forward/reverse batch order within each
gamma engine. Gamma engines run sequentially, so remaining order drift must be
examined before confirmation. Warmup uses disjoint prompts 80–143.

`perf-r*-b*.json` records full wave timings, every final output, chunk receipts,
native scheduler acceptance and draft counts, and energy. The headline
throughput includes prefill and request submission. Pure-decode calibration
must join `host_steps` to the same episode's `plugin.steps`, filter pure decode
and actual batch, and retain the full host step including drafting. Chunk
timestamps do not yield individual token latency; effective TPOT is explicitly
first-to-last receipt elapsed time divided by output-token count minus one.

Route diagnostics are separate `routes-b*.json` episodes with GPU histogram
instrumentation and a final batched D2H drain. Those timings never enter the
performance heatmap. Counts describe selected parameter storage, not hardware
HBM bytes. Prefix-cache reuse is disabled.

## Optional dense OLMo draft source

The CPU-only downloader below uses `configs/olmo_dense_draft.json` and the fixed
`allenai/OLMo-1B-0724-hf` revision `d7cbab742d80589e714b1a2d7f838dcd21cbe143`.
It reconciles the full file inventory and hashes, resumes `.part` transfers,
and writes only under this project's `models/olmo_1b_0724_draft/`. The complete
repository is 5,121,310,602 bytes (FP32 files); runtime BF16 weights are roughly
2.56 GB, plus separate draft KV and workspace. This command does not load a GPU.

```bash
python code/download_olmo_draft.py --endpoint https://hf-mirror.com --resolve-only
python code/download_olmo_draft.py --endpoint https://hf-mirror.com \
  --target-path /root/autodl-tmp/moe-research-20261002/model
python code/download_olmo_draft.py --check-only \
  --target-path /root/autodl-tmp/moe-research-20261002/model
```

The final command refreshes local compatibility without networking or rehashing
weights. It reads the existing target's `config.json` and `tokenizer.json`
without changing them. Missing target files produce `pending_local_target`;
incompatible config/token IDs produce a failed check. The ordinary 50,280-token
mapping is shared, but the target additionally registers `<pad>` at ID 50280.
Preserve the target chat template and check this extra padding token is absent
from actual prompt content during the native smoke run. A matching report is
structural evidence, not a runtime, acceptance-rate, or performance result.
Use the reported `draft_path` with `method=draft_model`; ordinary OLMo loads
through vLLM 0.26's Transformers backend. The EAGLE downloader remains separate.

## Analysis

```bash
python code/plot_results.py runs/UNIQUE_NAME --output figures/UNIQUE_NAME
python code/analyze_routes.py runs/UNIQUE_NAME/gamma0/routes-b1.json \
  --experts 64 --top-k-without-replacement --output runs/routes_q.json
python -m unittest discover -s tests -p test_plugin.py -v
python -m unittest discover -s code/tests -p test_cost_and_analysis.py -v
```

PowerSampler is attributed to the July reference in `ref_july/SOURCES.json`,
through the existing workspace implementation. Prefer the device energy counter;
otherwise integrate 50 ms power samples. Energy is whole GPU board energy,
including idle and host-induced GPU stalls; it is not process-only dynamic energy.
Two repeats expose raw run variation but do not establish significance.

`paper/main.tex` is a live manuscript. Missing experiments remain explicitly
unmeasured; analytic illustrations are not substituted for measurements.

## Measured route prediction and layer cost

For a new completed static matrix, create local F2 and measured-window energy
heatmaps without matplotlib:

```bash
python code/plot_results.py runs/DOMAIN_reduced_r01 --no-plot \
  --output runs/DOMAIN_matrix_analysis
python code/plot_matrix_pgf.py --analysis runs/DOMAIN_matrix_analysis \
  --output figures/F2_DOMAIN
```

The renderer rechecks original performance sources and preserves the analysis
values, full run provenance, and raw repeats. It separates model/draft/resource
domains and prompt populations. Missing repeats or an incomplete AR denominator
remain blank. Throughput, TPOT ratio and measured-window J/token use separate
figures; wave-window energy does not become complete-population energy.
`figures/F2_olmoe_ngram_r02_reproduced` reproduces the existing 112 records and
56 cells; its 280 summary-value checks have zero difference.

Compose separate measured domains with one shared color scale per metric:

```bash
python3 -B code/plot_matrix_panels_pgf.py \
  'OLMoE eager=figures/F2_olmoe_ngram_r02' \
  'OLMoE compiled=runs/olmoe_ngram_compiled_matrix_analysis_r01' \
  --output /private/tmp/moe_s_matrix_panels_preview_reproduced --columns 2 \
  --title 'Export layout preview: OLMoE eager and compiled'
```

This local export check uses 224 existing runs and 112 cells; all 560 reused
summary comparisons match, and three rendered metrics have been visually
checked. It adds no new experiments. Each explicit `LABEL=ANALYSIS_DIR` input
must contain one domain, with its own B/gamma axes and same-B AR denominator.
Once the six compiled model/source domains are measured, provide their six
analyses in the same form. Missing/incomplete cells stay blank; raw repeats,
original source hashes and common-scale limits accompany the PDFs/SVGs/PNGs.
Common colors do not establish controlled comparisons across models, precision,
draft sources, workloads or execution domains. Repeated runs share the recorded
engine lifetime. Rendering uses local TeX/Poppler and the existing local plot
helpers, without GPU access; use a fresh output directory for each export.

Fit token-inclusion marginals only on AR route episodes, then predict the
executed nonzero-gamma episodes using their actual token slots. The same prompt
subset appears in both, so this is a held-out execution-condition check, not
independent-workload validation. Nominal B(gamma+1) is retained as a separate
comparison; histograms are never described as exact token-adjacency replay.

After deriving a single-domain route analysis, its MAE panel can be rendered
locally without matplotlib:

```bash
python code/plot_route_prediction_pgf.py --analysis runs/route_prediction_r02 \
  --output figures/F1_olmoe_ngram_r02_reproduced --title 'OLMoE / n-gram'
```

This reads existing statistics without fitting again, checks source hashes and
model/precision/draft/execution identity, and keeps scheduled valid-row D
separate from physical-padding D. Future domains first use the validator below
with `--no-plot`, then pass their own analysis directory to this renderer.

```bash
python code/validate_route_prediction.py \
  --fit-routes 'runs/olmoe_ngram_matrix_r02/gamma0/routes-b*.json' \
  --test-routes 'runs/olmoe_ngram_matrix_r02/gamma[1-7]/routes-b*.json' \
  --top-k-without-replacement --nominal-comparison \
  --out-dir runs/route_prediction_r01 --plot-dir figures/F1_routes_r01
python code/run_kernel_campaign.py --run UNIQUE_KERNEL_RUN \
  --model /root/autodl-tmp/moe-research-20261002/model \
  --routes 'runs/olmoe_ngram_matrix_r02/gamma*/routes-b*.json' \
  --include-token-extremes --profile
python code/fit_kernel_cost.py \
  --inputs 'runs/UNIQUE_KERNEL_RUN/measurements.jsonl' \
  --out-dir runs/UNIQUE_KERNEL_FIT --plot-dir figures/UNIQUE_KERNEL_FIT
```

The microbenchmark uses median-D snapshots plus measured minimum/maximum token
counts, so a low-coverage n-gram cell does not contribute only its common AR
steps. For a tied token count it selects the median-D event, without looking at
timing. Cell grouping uses the engine's nominal gamma; per-step realized gamma
and actual slots remain in the output. Histograms preserve expert loads but
synthetically realize token co-occurrence unless exact top-k IDs were captured.
Five timed repeats and every call are retained. CUDA event intervals can include
host launch gaps; profiler kernel sums are a separate instrumented diagnostic.
The fitted max(aD,bS)+c is a single-layer component model with a fixed batch
holdout, not complete-step latency or measured HBM bandwidth.

The completed graph replay is `runs/olmoe_kernel_graph_r01`. Its 154 cases
match r03's weights, measured histograms, selected source events, tile settings,
seed and main-timing cache-turnover protocol. Reproduce its independent fit
and the descriptive paired graph locally:

```bash
python code/fit_kernel_cost.py \
  --inputs runs/olmoe_kernel_graph_r01/measurements.jsonl \
  --out-dir runs/kernel_cost_fit_graph_r01
python code/plot_kernel_diagnostics.py \
  --fit-dir runs/kernel_cost_fit_graph_r01 --figures figures/kernel_cost_graph_r01
python runs/kernel_cost_fit_graph_r01/build_paired_comparison.py
```

Do not pool eager and graph samples into one fit. Each retains the original
88/66 calibration/held-out split and five within-process timing repeats.
Capture and first replay are recorded outside those repeats. This fixed-input
single-layer replay has no model-level graph padding. The two campaigns ran
in separate time windows, so their contrast does not isolate CPU launch cost.
The legacy eager profiler omitted turnover; the new graph profiler includes
it. Their kernel-duration sums must not be subtracted to infer overhead.

Additional domains use compiled route-off performance matrices. Their separate
`DOMAIN_compiled_routes.json` plans set `repeats=0`, `skip_trace=false`,
`native_route_return=true`, and `native_capture_padding=true`, with the same
64-prompt route population at each B and 32 output tokens per request. Run them
without `--skip-trace` or a repeat override. The existing runner performs normal
loading/warmup and then route episodes, producing no `perf-r*.json`. Diagnostic
episodes still record timing/energy, which are excluded from performance and
calibration. Native capture remains active for that entire diagnostic engine;
it is never enabled for the F2 engine. Original r02 route episodes retain one
wave per batch, a different declared population.

## Controller selection and independent execution

`code/select_controller_baselines.py` reads measured effective TPOT and emits a
single best static gamma, a native unrestricted batch table, and a native
single-threshold table. The configured objective weights B1/4/16/64 equally.
These are development choices, not measurements of the resulting policies.
Run both tables through `--mode native --gammas 7 --native-table TABLE.json`;
run the selected single gamma in its own native engine. The post-proposer
`dense_heuristic` mode is only an execution-cost ablation.

Adaptive post-proposer truncation requires separate measured full-budget costs:

```bash
python code/run_campaign.py --config configs/olmoe_ngram_fullbudget_calibration.json \
  --run UNIQUE_FULLBUDGET_CALIBRATION --mode static --proposal-budget 7 --skip-trace
python code/build_calibration.py runs/UNIQUE_FULLBUDGET_CALIBRATION \
  --output runs/UNIQUE_FULLBUDGET_CALIBRATION/calibration
python code/run_campaign.py --config configs/olmoe_ngram_controller_evaluation.json \
  --run UNIQUE_ADAPTIVE_EVALUATION --mode adaptive --proposal-budget 7 \
  --calibration runs/UNIQUE_FULLBUDGET_CALIBRATION/calibration/calibration.json --skip-trace
```

The evaluation plan uses prompts 64–127; development uses prompts 0–63.
Configuration and execution mode must match the calibration. Native direct
costs cannot substitute for paying budget7 and truncating after proposing.

The first main-domain evaluation is prepared without reselecting its policies:

```bash
python code/prepare_controller_evaluation.py \
  --calibration-run runs/olmoe_fullbudget_calibration_r01 \
  --direct-run runs/olmoe_ngram_matrix_r02 \
  --output runs/controller_preparation_eager_r01
```

This CPU command requires all 32 measured full-budget cells and the original
64 development records used for selection. It writes five explicit commands
in `evaluation_commands.json`; it does not itself start GPU work. Commands
use the held-out input offset 64, with both general and shape warmup at offset
0. Their 6-hour lock-wait allowance is waiting time, not a GPU allocation.
The table, source and environment checks must pass before executing them.

For neural and EAGLE smoke runs, verify that drafts were actually scheduled
and verified before extending the measurement:

```bash
python code/check_native_draft_smoke.py \
  --run runs/qwen3_awq_eagle3_smoke_r01 --draft-source eagle3
```

The independent report retains every smoke cell and the raw count totals;
zero accepted drafts is permitted. Missing execution or OLMo semantic-token
boundary violations require an implementation investigation. Passing is not
a quality or performance finding. Actual MoE backend choice is read from the
engine log, not inferred from the requested `kernel_config`.

For each additional domain (`olmoe_dense_draft`, `qwen3_awq_ngram`,
`qwen3_awq_eagle3`, `granite3b_ngram`, `granite3b_neural_draft`), use its own
native development matrix and separate 12-cell budget7 cost calibration:

```bash
python code/prepare_reduced_evaluation.py --domain olmoe_dense_draft --describe
python code/run_campaign.py \
  --config configs/olmoe_dense_draft_reduced_calibration.json \
  --run olmoe_dense_draft_reduced_calibration_r01 --gammas 0 3 7 \
  --mode static --proposal-budget 7 --skip-trace --wait-lock-seconds 21600
python code/prepare_reduced_evaluation.py --domain olmoe_dense_draft
```

Replace the domain consistently for another target/source; never reuse a
cost table across domains. Preparation requires the complete native
`runs/DOMAIN_reduced_r01` and complete cost
`runs/DOMAIN_reduced_calibration_r01`, then emits the same five role-specific
commands under `runs/DOMAIN_controller_preparation_r01/`. The adaptive table
and simple-baseline choices are limited to gamma `{0,3,7}`. Cost calibration
has one repeat and 64 output tokens; policy evaluation has two repeats and
128 output tokens. General and shape warmup use offset0, development is 0–63,
and evaluation is 64–127. These are experiment protocols, not measured gains.

After copying a completed domain's preparation and raw runs locally, derive
its F3/F4/F6 tables without a GPU:

```bash
python code/compare_reduced_evaluations.py \
  --preparation runs/olmoe_dense_draft_controller_preparation_r01 \
  --output runs/olmoe_dense_draft_controller_comparison_r01
```

The reader relocates recorded S-project paths, verifies the original inputs,
and keeps both raw repeats. It reports missing cells as pending and leaves
comparison medians blank until both repeats are present. Complete-population
NVML counter-span energy remains separate from summed wave-window energy.
This local analysis does not submit work or create figures from pending data.

Render a completed main- or reduced-domain comparison locally with PGFPlots:

```bash
python code/plot_policy_comparisons.py \
  --comparison runs/olmoe_dense_draft_controller_comparison_r01 \
  --output figures/olmoe_dense_draft_policies_r01 --include-wave-energy
```

The five policy roles retain separate labels. F3 uses effective request TPOT;
F4 uses a validated complete-population cumulative-counter span. The optional
measured-window energy figure remains separate. F6 reports actual timed policy
selection and the fraction of available draft tokens discarded. Each figure
exports its raw points, complete-repeat summaries and source provenance.
Missing cells stay blank; a single repeat receives a raw point but no curve
summary. Rendering requires local TeX/PGFPlots and Poppler, not a GPU.

## Independent compiled controller comparison

The compiled OLMoE native matrix supplies a separate baseline selection. Its
costs cannot be borrowed from eager execution. The added calibration uses
B1/4/16/64, caps0--7, fixed proposal budget7, one 64-token repetition; the five
evaluation arms each use two 128-token runs on records64--127.

```bash
python code/prepare_compiled_controller_evaluation.py \
  --direct-run runs/olmoe_ngram_compiled_r01 \
  --calibration-run runs/olmoe_compiled_fullbudget_calibration_r01 \
  --output runs/controller_preparation_compiled_r01
```

This CPU preparation only emits commands after both input campaigns complete.
It verifies all 56 native cells, uses the protocol's four B values for selection,
and requires 32 measured budget7 cost cells. Static gamma0 and all-zero native
tables are legal; their engine budgets remain distinct and each is executed.
Historical native general warmup used records80--143, while its measured
selection population was0--63. That overlap with evaluation prompts is recorded;
the whole pipeline is not claimed prompt-unseen. New calibration/evaluation
warmup explicitly begins at0. The existing main comparison reader and five-role
plotter also accept this compiled domain when supplied its own selection/costs.

The other five target/draft domains use compiled configuration consistently in
smoke, native matrices, calibration, and evaluation. Code/summary matrices also
use compiled route-off execution. The already measured eager OLMoE results and
its matched policy/admission follow-ups remain separate. These choices precede
the new campaigns' performance results; compatibility failures are inspected,
not used to choose the faster-looking domain after measurement.

## Finite-population admission diagnostic

The matched plans below use the same 64 ShareGPT requests, 128 generated tokens,
eager execution, 6 GiB KV and an actual scheduler cap of 16. Both arms execute
the same wave and queued shape warmup. All requests are externally ready at
the common time origin; wave admission holds later groups locally, whereas
queued admission submits all 64 before the first engine step and permits the
scheduler to fill freed slots. The cap does not imply every decode step has B16.

```bash
python code/run_campaign.py --config configs/olmoe_wave_cap16_diagnostic.json \
  --run UNIQUE_WAVE_CAP16 --skip-trace
python code/run_campaign.py --config configs/olmoe_backfill_diagnostic.json \
  --run UNIQUE_QUEUED_CAP16 --skip-trace
```

Report `population_wall_s`, the existing sum-of-windows `wall_s`, and
`population_inter_episode_gap_s` together. The last includes meter handling,
runtime summaries and preparation between waves; it is not subtracted from
the population window. Request release/submission/first-output/completion
timestamps share the population origin; original chunk timestamps remain
episode-local. Energy still describes the measured episode windows, so do not
call their sum complete-population energy including inter-wave gaps. These
plans are prepared diagnostics, not yet GPU results. The B64/64-request case
has no waiting population and cannot establish a backfill benefit.

The additional LongBench-derived LCC and multi-news plans retain full prompts
and reserve 24 GiB target KV. For the first 64 selected prompts, a conservative
16-token-block calculation including 128 output and 7 speculative positions
requires 124,272 and 93,792 slots, respectively (15.17 and 11.45 GiB under the
OLMoE target geometry). This is a capacity check, not observed KV occupancy;
actual scheduler preemptions must still be checked in the measured run.

## Granite3B target with two draft sources

`configs/granite3b_target_source.json` pins the official Granite3B base snapshot
`d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a` (40 experts, top-8). The CPU-only
assembler verifies the eight complete B-project assets before copying them as
independent regular files into S, and fetches only the missing first shard and
`.gitattributes`. It never changes B. The complete S snapshot is 6,599,724,398
bytes; the expected network transfer is 4,998,541,007 bytes. Partial downloads,
fixed metadata, and final file hashes stay under S's `models/granite3b/`.

```bash
python code/download_granite_target.py --endpoint https://hf-mirror.com
python code/download_granite_target.py --check-only
python code/run_campaign.py --config configs/granite3b_ngram_smoke.json \
  --run UNIQUE_GRANITE3B_NGRAM_SMOKE --mode native --skip-trace
python code/run_campaign.py --config configs/granite3b_neural_draft_smoke.json \
  --run UNIQUE_GRANITE3B_NEURAL_SMOKE --mode native --skip-trace
```

The download command requires CPU/network only; campaign commands require the
existing shared GPU lock and budget. After successful native smoke checks,
the corresponding `_reduced.json` plans use B1/4/16/64 and gamma0/3/7, with two
runs per cell. Both sources reuse the fixed 160-record
`configs/sharegpt_granite1b.json` workload: the pinned 1B and 3B tokenizers
reproduce the same raw-continuation token IDs, so no new preparation is needed.
This is a base-model continuation workload, not an instruction-following
quality benchmark. The independent neural draft is the complete, read-only
B-project Granite1B snapshot `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
(32 experts, top-8). Native loading, draft acceptance, joint KV allocation,
kernel selection, and target/draft trace separation require actual execution.
Granite3B replaces the unmeasured Granite1B auxiliary target; Qwen remains
planned. This choice uses available compatible assets, not measured speed.

The CPU coordinator `code/run_following_campaigns.py` waits for
`olmoe_kernel_graph_r01/status.json` to report success, then runs
`olmoe_ngram_compiled_r01` (`olmoe_ngram_compiled.json`, `--skip-trace`) and
`olmoe_fullbudget_calibration_r01` (`olmoe_ngram_fullbudget_calibration.json`,
`--mode static --proposal-budget 7 --skip-trace`) in that order. It never holds
or queues for the GPU lock itself; each campaign uses the existing lock and
cumulative budget. Any incomplete predecessor/stage stops continuation for
inspection. Do not rerun while its original PID is active or reuse its fixed
output names. Its status is in `runs/olmoe_following_campaigns_r01/status.json`.

The subsequent CPU coordinator is now PID45296, with status under
`runs/olmoe_evaluation_sequence_r02`. It was launched with
`--run olmoe_evaluation_sequence_r02 --include-compiled-evaluation` after checking
that the old PID38597 was still waiting with no children; the old status and
`supersession.json` retain that replacement. No GPU experiment was stopped.
It waits for the predecessor and verifies its PID, then executes compiled costs,
five policies and natural-stop checks, the original eager comparisons, native
draft smoke checks, reduced matrices with separate route-only diagnostics,
admission/workload diagnostics, and each additional domain's costs/policies.
It never holds the GPU lock itself. Every child preserves the cumulative
20-hour budget; a failed or incomplete stage stops continuation for inspection.
A live process check is required after connection loss. Do not resubmit this
coordinator or reuse its existing experiment names.

For the remaining eager and reduced-domain controller evaluations and natural checks,
`shape_warmup_offset=0` initializes shapes and the continuous EWMA on development
requests; measured requests begin at offset64. `evaluation_initial_state.json`
records the actual acceptance snapshot before measurement. State is not reset
between measured batches, waves or repetitions. This differs from historical
matrix configurations, whose default shape warmup population remains unchanged.

Recover F5 from those actual online states after the policy runs are available:

```bash
python code/analyze_online_acceptance.py \
  --preparation runs/controller_preparation_eager_r01 \
  --output runs/F5_online_olmoe_ngram_r01
```

The same interface accepts each reduced-domain preparation. It follows the
recorded batch/repeat/wave order, checks logged updates and final snapshots,
and never supplies a future state to a missing earlier prediction. Per-run
exposure-weighted errors and raw repeats are in `online_tracking_runs.csv`;
`tracking_steps.csv.gz` supplies the chronological alpha/progress plotting
points. Conditional acceptance, prefix exposure and proposal availability
remain separate. Historical r02 lacks the required initial snapshot and uses
an older estimator, so its cold-prior replay is not an actual-online F5 result.

The retrieved `controller_preparation_eager_r01/evaluation_commands.json`
selects static gamma6, the unrestricted table `[[1,3,7],[4,15,5],[16,64,6]]`,
and threshold table `[[1,64,6]]`. Its calibration is the separate 32-cell eager
full-budget-G7 table. These are selected actions, not measured policy results.
The eager AR campaign completed at 09:43:54 on 2026-10-16 (Asia/Shanghai), and
its eight complete 64-request/8192-token records have been retrieved; all eight
population cumulative-counter spans validate. At 09:43:59 the original
coordinator had entered static6 (PID74026). SSH subsequently disconnected;
the 10:11 reconnect attempt returned Connection refused. Later remote stage,
process, lock and resource states are unknown. Check that original coordinator
and its current child before resubmitting anything.
After all five campaigns report complete, use their exact recorded paths:

```bash
python code/compare_controller_policies.py \
  --ar runs/olmoe_controller_eval_ar_r01 \
  --static runs/olmoe_controller_eval_static6_r01 \
  --native-unrestricted runs/olmoe_controller_eval_native_table_r01 \
  --native-threshold runs/olmoe_controller_eval_native_threshold_r01 \
  --adaptive runs/olmoe_controller_eval_adaptive_r01 \
  --selection runs/controller_preparation_eager_r01/selection/regime_10a33701f2/selection.json \
  --calibration runs/controller_preparation_eager_r01/calibration/calibration.json \
  --workload configs/sharegpt_olmoe.json \
  --output runs/olmoe_eager_controller_comparison_r01
```

This pending output is not a completed comparison. Do not substitute the
compiled selection, costs, or static0 run for these eager inputs.

## Prepared common code and summarization workloads

The original 80-record auxiliary inputs remain separate. New inputs in
`configs/{lcc,multi_news}_{olmoe,qwen3_awq,granite3b}_common128.json` support
64 development and 64 held-out requests for every target/draft-source pair.
They preserve complete prompts, source order and references. Each task uses
the same original source IDs across all three target tokenizers; OLMoE/Qwen
retain native chat wrapping, while Granite uses explicit raw continuation.
Equal source identity does not imply equal tokens or chat/base-model behavior.

`runs/aux_workload_feasibility_r01/` contains the actual source scan, tokenizer
and model-config hashes, counts, complete candidate records and CPU execution
times. The common eligible counts at prompt caps 2048/3072/3584 are
LCC 81/280/337 and MultiNews 71/123/143. The smallest feasible listed cap for
128 requests is therefore 3072 for LCC and 3584 for MultiNews. The builder
copies the first 128 common eligible records unchanged; it does not re-tokenize,
truncate, duplicate, shuffle, select using references, or run generation.
`runs/auxiliary_population_preparation_r01/manifest.json` pins all six outputs.

The scan ran at 09:45:47.276–09:46:14.281 on 2026-10-16 (Asia/Shanghai),
while eager static6 was in progress. Single-CPU affinity, nice19 and disabled
CUDA do not establish zero host interference. Preserve this 27.005-second
window alongside the eventual eager comparison; no exact per-cell overlap is
inferred without compatible clock anchors, and no estimated time is subtracted
or performance record retrospectively excluded.

`runs/auxiliary_resource_plan_r01/resource_plan.json` gives the complete
target-plus-draft KV geometry and population-based capacity calculation. It
reserves each full prompt, 128 output tokens and seven speculative positions,
with block rounding at 16/32/64. The largest calculated B64 payload is
37.90625/12.07056/15.53809 GiB for OLMoE/Qwen/Granite; the fixed worker KV
candidates are 48/24/24 GiB, identical across the two sources within a family.
These are capacity estimates, not measured allocation, peak memory, guaranteed
concurrency or fixed total resident VRAM. Inspect actual cache groups, block
size, peak occupancy and B64 behavior at the first execution. Each auxiliary
domain uses its own native baseline and complete-cost calibration under this
resource setting; do not reuse the short-prompt ShareGPT tables.

For a fresh project copy without these generated files, the construction is:

```bash
python code/build_common_aux_workloads.py
python code/prepare_auxiliary_campaigns.py
```

Both commands refuse to overwrite existing products. The capacity report is
an input to the second command, and must refer to the exact population-manifest
hash. The checked-in `configs/auxiliary_campaigns.json` pins 36 domain configs,
six workloads, the population/resource reports and the reduced protocol.
`prepare_reduced_evaluation.py --domain NAME` accepts a safe filename-component
name and applies the same existing execution/source/population checks.

The supplemental sequence is **prepared locally, not deployed or started** as
of the last connection failure. After restoring SSH, checking the original
process state and copying/verifying the new inputs, its entry is:

```bash
python code/run_auxiliary_sequence.py --describe
python -u code/run_auxiliary_sequence.py \
  --run auxiliary_evaluation_sequence_r01 \
  --predecessor runs/olmoe_evaluation_sequence_r02/status.json
```

Start it only once. It first waits for the original sequence to report complete,
checking a live predecessor's PID and run identity. It then runs all 12 native
matrices before each domain's separate G7 cost calibration and five actual
policies. Thus the planned total is 84 GPU campaigns, not completed results.
Each child retains the existing shared GPU lock and the same aggregate
72,000-second budget across all prior S runs. The plan does not guarantee that
all pending work fits within the remaining budget. A failed, incomplete or
budget-limited stage leaves its state and halts continuation without retry.
Reconnection alone never authorizes duplicating an existing process.

Analysis uses the existing reduced-domain comparison and online-state readers
with the corresponding `DOMAIN_controller_preparation_r01`. The new native
grid is B1/4/16/64 × gamma0/3/7, two repeats with 128 forced output tokens;
G7 complete-cost calibration uses one repeat with 64 output tokens. Five-policy
evaluation uses held-out entries64–127, two repeats and 128 output tokens.
General and shape warmup use development entries0–63. All runtime costs remain
charged, and these forced-length runs do not establish natural task quality.

## Completed compiled OLMoE matrix

`runs/olmoe_ngram_compiled_r01` completed all 56 cells and 112 performance
records, using route-off V1 compiled execution. Its GPU lock duration was
2,223.78645 seconds. It uses per-batch/prompt-shape warmup and the same measured
64-request, 128-token population as eager r02. The two execution domains ran
in separate time windows with different warmup coverage; their difference does
not isolate CUDA graph causality. The independent compiled routing diagnostic
has scheduler cap32; the performance matrix has cap64.

```bash
python code/plot_results.py runs/olmoe_ngram_compiled_r01 --no-plot \
  --output runs/olmoe_ngram_compiled_matrix_analysis_r01
python code/plot_matrix_pgf.py \
  --analysis runs/olmoe_ngram_compiled_matrix_analysis_r01 \
  --output figures/F2_olmoe_ngram_compiled_r01 \
  --metric speedup tpot_speedup joules_per_token \
  --title 'OLMoE / n-gram / compiled'
python code/analyze_online_acceptance.py runs/olmoe_ngram_compiled_r01 \
  --output runs/F5_online_compiled_matrix_r01
```

The F2 throughput and energy panels retain the original sum-of-wave-window
boundary. The source records also contain complete-population wall time and
inter-wave gaps. Cumulative NVML counter spans can cover those gaps, but their
endpoint timestamps are distinct from the runner's wall-clock endpoints.
F5 covers all eight actual engines, starts from each recorded post-warmup
state, and reconstructs 240,254 logged steps. Native static runs record no
adaptive decision-progress predictions, and their availability EWMA is not
updated; neither is converted into claimed controller performance.

The supplemental complete-population summary is reproducible with
`python runs/olmoe_ngram_compiled_matrix_analysis_r01/summarize_population.py`.
It exports both original repeats and same-B AR values without replacing the
wave-window heatmap metric. Render the complete actual-online F5 panels with
`python runs/F5_online_compiled_matrix_r01/render_local.py`; the fixed displayed
caps are 1, 4 and 7, and the illustrative continuous trajectory is gamma4,
repeat0, B64 including drain. The prior partial figures remain unchanged.

## Complete-budget cost and single-step reference

The eager G7 calibration `runs/olmoe_fullbudget_calibration_r01` is complete:
32 measured cells, one run per cell, with 64 output tokens per request and
shape warmup. Its independent local analysis does not replace the coordinator's
later `controller_preparation_eager_r01` artifacts.

```bash
python code/build_calibration.py runs/olmoe_fullbudget_calibration_r01 \
  --output runs/olmoe_fullbudget_cost_analysis_r01
python code/diagnose_stepwise_prefix_oracle.py \
  --trajectory runs/olmoe_ngram_matrix_r02/gamma7 \
  --calibration runs/olmoe_fullbudget_cost_analysis_r01/calibration.json \
  --output runs/derived/stepwise_prefix_eager_r01
python runs/olmoe_fullbudget_cost_analysis_r01/compare_and_plot.py
```

The last script preserves raw native and fixed-budget cost values and renders
two separate diagnostics under `figures/fullbudget_cost_eager_r01` and
`figures/stepwise_prefix_eager_r01`. Native r02 differs in output length,
warmup, observation code and execution window, so cost differences are not a
causal drafting tax. The perfect-prefix reference uses future information on
the same fixed trajectory and calibrated costs. It does not execute changed
trajectories, estimate service TPOT, or establish an end-to-end upper bound;
keep it separate from measured F3 curves. Its two raw run summaries coincide
under identical prefix histories, cold replay initialization and the same table.

After an actual adaptive campaign finishes, check transfer of its selected
complete-step cost predictions with:

```bash
python code/analyze_policy_cost_prediction.py \
  runs/olmoe_compiled_controller_eval_adaptive_r01 \
  --output runs/olmoe_compiled_policy_cost_prediction_r01
```

This local reader uses the campaign's copied calibration file, validates its
measured domain and source population, and matches prediction/measurement on
the same event. It retains raw run errors and excluded-step reasons. It does
not infer costs for unselected actions or turn prediction accuracy into a
service-performance claim. The compiled adaptive input and the named analysis
are complete: eight valid runs, 19,191 eligible same-event cost comparisons.
Actual eager and other-domain adaptive inputs remain pending.

## Saved engine memory observations

```bash
python code/summarize_engine_resources.py \
  runs/olmoe_ngram_matrix_r02 runs/olmoe_ngram_compiled_r01 \
  runs/olmoe_fullbudget_calibration_r01 \
  --output runs/engine_resource_summary_r01
```

This local reader exports one row per engine and preserves every matching
startup log report with context. The 24 existing observations are initialized
PyTorch allocated-memory snapshots, before application warmup, plus the explicit
KV cap and separately scoped model-loading/graph-capture log values. They do not
measure NVML residency, peak memory or a memory-component decomposition. Missing
reports stay null; multiple model-loading reports are neither summed nor assigned
to target/draft models without evidence. Reuse the same reader after the neural
draft campaigns complete.

## Compiled complete-budget table and diagnostic

The compiled G7 calibration finished all 32 cells in 760.84267205 lock-seconds.
The coordinator's actual table and five evaluation commands are in
`runs/controller_preparation_compiled_r01`; its table SHA256 is
`305a7835a12caaa2dc68bb48fbebd07d772eb63e1b85012280c6973220d57271`.
Reproduce the separate local analysis and figures with:

```bash
python code/build_calibration.py runs/olmoe_compiled_fullbudget_calibration_r01 \
  --output runs/olmoe_compiled_fullbudget_cost_analysis_r01
python code/build_calibration.py runs/olmoe_ngram_compiled_r01 \
  --output runs/olmoe_compiled_fullbudget_cost_analysis_r01/native_direct
python code/diagnose_stepwise_prefix_oracle.py \
  --trajectory runs/olmoe_ngram_compiled_r01/gamma7 \
  --calibration runs/olmoe_compiled_fullbudget_cost_analysis_r01/calibration.json \
  --output runs/derived/stepwise_prefix_compiled_r01
python runs/olmoe_compiled_fullbudget_cost_analysis_r01/compare_and_plot.py
```

The local table matches the coordinator's values; machine-specific source paths
may differ. The comparison uses compiled sources only. Runner/plugin/cost-model
source hashes match between these campaigns, but output lengths, general warmup,
configured budgets and time windows differ. Do not treat their cost difference
as an isolated truncation cost. The fixed-trajectory future-prefix reference
remains a diagnostic, separate from actual F3 curves.

The compiled development selection is static gamma0, threshold cutoff0, and
native unrestricted table `[[1,15,0],[16,63,4],[64,64,7]]`. Keep all five actual
roles: AR and static0 use separate G0 engines; the all-zero native threshold
still has configured G_eng7, but its native table overrides a V1 FULL-containing
graph configuration to PIECEWISE. It is neither the independent G0 AR engine
nor the FULL_AND_PIECEWISE graph domain of post-proposer G7 calibration.
Zero-action choices are not omitted. These tables were selected from
direct-gamma development runs, not exhaustive same-graph-domain tuning. The selected population is disjoint from
the evaluation population; historical native general warmup exposed 48 of the
evaluation prompts, which prevents a whole-pipeline prompt-unseen claim.

The resolved CUDA Graph mode is part of the execution domain, not determined
by G_eng alone. In the installed vLLM source,
`runs/environment_20261016/source/vllm/config/vllm.py:819–836`
(`_maybe_override_dynamic_sd_cudagraph_mode`, called at 1273–1274) changes a
V1 FULL-containing mode to PIECEWISE when a native batch table is configured.
The dynamic predicate is `num_speculative_tokens_per_batch_size is not None`
(`runs/environment_20261016/source/config/speculative.py:1337–1338`).
`config/compilation.py:1466–1474`, preserved in
`runs/compiled_route_summary_r01/native_source_excerpt.txt`, applies the
G_eng+1 rounding only for V1 FULL decode with uniform query length >1.
The maximum capture budget still depends on G_eng; that does not imply the
same rounded grid.

Actual startup evidence is in
`runs/olmoe_compiled_controller_eval_native_unrestricted_r01/gamma7.log`:
the override is logged, the grid contains 1/2/4, and capture completes with
51 PIECEWISE sizes, no FULL capture, 6s and 0.17GiB. The G7 post-proposer cap0
calibration log `runs/olmoe_compiled_fullbudget_calibration_r01/gamma0.log`
records 48 PIECEWISE and 48 FULL captures, 3s and 0.21GiB. These are startup
records, not timed dispatch fractions or service costs. Trace-off performance
records do not establish actual P1 versus P8 execution, so neither padding cost
nor a performance cause is inferred from this contrast.

## Actual held-out policy comparison

All five compiled campaigns are complete. The canonical comparison contains
40 accepted run records (five roles, four batches, two repeats) and zero
exclusions. Each run completes the same evaluation entries 64–127 with 128
forced tokens/request. Shape warmup uses development entries; the historical
general-warmup exposure caveat above remains. Recompute the actual comparison
and its F3/F4/F6 figures with:

```bash
python code/compare_controller_policies.py \
  --ar runs/olmoe_compiled_controller_eval_ar_r01 \
  --static runs/olmoe_compiled_controller_eval_static_r01 \
  --native-unrestricted runs/olmoe_compiled_controller_eval_native_unrestricted_r01 \
  --native-threshold runs/olmoe_compiled_controller_eval_native_threshold_r01 \
  --adaptive runs/olmoe_compiled_controller_eval_adaptive_r01 \
  --selection runs/controller_preparation_compiled_r01/selection/regime_e7cb8fe905/selection.json \
  --calibration runs/controller_preparation_compiled_r01/calibration/calibration.json \
  --workload configs/sharegpt_olmoe.json \
  --output runs/olmoe_compiled_controller_comparison_r01
python code/plot_policy_comparisons.py \
  --comparison runs/olmoe_compiled_controller_comparison_r01/comparison.json \
  --output figures/F3F4F6_olmoe_compiled_controller_r01 --label OLMoE
python code/compare_generations.py --workload configs/sharegpt_olmoe.json \
  --run ar=runs/olmoe_compiled_controller_eval_ar_r01 \
  --run static=runs/olmoe_compiled_controller_eval_static_r01 \
  --sample-workload-offset 64 \
  --output runs/olmoe_compiled_baseline_generation_check_r01
```

The latter is a fixed-length token-ID comparison, with no tokenizer decoding or
natural-stop quality claim. Both baseline configurations are identical; all 512
cross-arm sequence pairs match. Their performance differences describe local
sequential-run variation, not a strategy effect or a transferable noise floor.

The actual plots are
`figures/F3F4F6_olmoe_compiled_controller_r01/F3_effective_tpot_92cb309cdb53.pdf`,
`F4_counter_span_energy_92cb309cdb53.pdf`, and
`F6_control_92cb309cdb53.pdf` (each also PNG/SVG/TeX).
`comparison.json`, `policy_runs.csv`, `policy_summary.csv`,
`weighted_objective.csv`, `control_runs.csv`, and `control_summary.csv` retain
all accepted raw values and scope metadata. Plot medians summarize the two
resident-engine repeats; they are not independent engine repetitions or CIs.
Native tables use PIECEWISE and zero-budget production returns before drafting,
whereas adaptive uses full-budget production followed by truncation. Thus the
comparison includes actual policy execution/configuration costs and does not
isolate a selection rule, graph padding, or a pure drafting-budget effect.

The declared equally weighted B1/4/16/64 effective-TPOT objective is
7.714910/7.740541/7.917877/8.141005/8.989071 ms for
AR/static0/native-unrestricted/native-threshold/adaptive. Adaptive/static ratios
for TPOT are 2.227642/1.273748/1.033728/0.959581; for complete-population
throughput 0.460511/0.745835/0.916799/0.948388; for counter-span J/token
2.393173/1.350436/1.055272/1.083353. Adaptive does not win the aggregate
objective here; B64's lower TPOT coincides with worse throughput and energy.
These are descriptive values in this n-gram domain, without significance or
universal-method claims.

Complete-population throughput includes inter-wave handling gaps. The validated
NVML counter span includes gaps too, but its first-wave-start/last-wave-end
endpoints differ from the runner population wall clock. Wave-window energy is
retained separately; neither measure subtracts idle energy or covers the whole
system. Adaptive decision-call means span 39.70–52.14 us across eight runs.
Discarded/available fractions for B1/4/16/64 are 0/0,
0.422666/0.422666, 0.731301/0.689627, and 0.430243/0.430243.
Decode counts include terminal/drain steps; CPU means use actually timed calls.
Missing native/fallback call timings are not zero. Nested decision/wrapper/
schedule times cannot be summed or subtracted from wall time, and discarded
counts are not measured wasted GPU time or energy.

Future-prefix fixed-trajectory references remain separate diagnostics and must
never enter F3 as executed policies, oracle speedups, or theoretical bounds.
Natural-stop artifacts are in
`runs/olmoe_compiled_natural_generation_check_r01/summary.json` and
`inspection.md` plus `manual_inspection.md`. The smoke completes 12 cells: AR/static/adaptive at B1/B8,
two repeats, and the same 16 requests per cell. Failed, unfinished and missing
counts are all zero. At B1 each AR/static run has EOS4/cap12, versus adaptive
EOS2/cap14; at B8 every arm/run has EOS3/cap13 (256-token cap). The fixed first
eight workload entries from offset64 at B1/repeat0 are selected independently
of outcomes.
Inspection finds shared instruction violations/unsupported personal achievements and capped
incomplete code, so neither task correctness nor quality/distributional
equivalence is established. The planned neural-source, other-target, and
code/summarization evaluations continue; their availability remains to be measured.

Actual adaptive state/progress analysis is separately retained in
`runs/F5_online_compiled_adaptive_r01/`: 19,759 alpha-after checkpoints and
170 wave-end states match; 19,419 of 19,589 decision steps are scored.
Run-level request-weighted progress MAE spans 0.060344–0.231269
tokens/request-step against realized observations, not latent probabilities
or a service-performance claim.
