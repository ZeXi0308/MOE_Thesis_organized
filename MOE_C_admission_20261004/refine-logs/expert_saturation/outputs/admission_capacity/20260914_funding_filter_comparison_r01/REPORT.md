# Funding-filter component comparison

Status: `CPU_PREPARED_GPU_UNRUN`. Preparation is not a measured result.

This six-cell comparison reuses the frozen heterogeneous cohort, OLMoE/vLLM backend, 6,656-block pool, capture path, arrival trace, thresholds and native recovery semantics from the context calibration. No workload, pressure point or threshold was selected after observing this package.

The frozen order is `most_output / least_progress / least_feasible / least_feasible / least_progress / most_output`. `least_feasible` is the existing least-progress selector with one present-state component enabled: after the original target and victim guards, it removes victims for which `free_blocks + released_blocks < target_required_blocks`. The target, cooldown, progress, residency and absence guards are unchanged. `most_output` and `least_progress` explicitly set the flag false.

The progress fraction is measured against each request's declared `P + 1024` generation upper bound. It is not known EOS distance, true remaining length, or evidence that a protected request will soon release KV; the corresponding current-source comment correction does not change packaged behavior.

The primary view is the per-request maximum-ITL distribution against complete-episode request throughput. Secondary views are completion distribution, overall ITL distribution, recomputed positions, restore waiting and actual selected/applied rotations. Every complete or unfavorable cell is retained.

If filtering improves only over `least_progress` and does not exceed `most_output`, it is a simple-baseline repair and not an independent contribution. CPU qualification proves current-state selection and native fixture behavior only; it does not predict execution time, output equality or an alternate future.

Package archive SHA256: `27a392a8e46e590426416d8aee305a3cf270a0bcf9275e756365404e18f858bc`. Remote path reserved by the copied execute entry: `/root/autodl-tmp/moe-funding-filter-comparison-20260914-r01`. GPU execution remains `UNRUN`.

The exact producer used for this archive is retained at `preparation/producer.py`; its SHA256 matches `preparation/preparation.json`. After freezing, the working selector changed only the explanatory upper-bound comment above, with an identical Python AST. The current preparation CLI can reproduce from the qualified frozen implementation by passing `--implementation-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01/preparation/pkg`; this does not modify the staged archive or metadata.


## Transfer completed, execution waiting for the existing group

`execution/execution.json` records successful staging and verification of all20 package files on the existing authorized westc:53036 host. No GPU controller or engine was started for this group. The earlier registered Qwen r03 lifecycle began with monitor77578 and owns the shared lock through loading and execution. This group waits for its complete terminal state and explicit release, with another live process/lock/resource check before launching. Staging is not runtime qualification or a measured result. No background waiter or automatic restart was installed.


## Analysis readiness

`analyzer_cpu_checks.json` records four actual legacy trajectories replayed with the default selector, compilation of the filtered replay and six missing/forged-flag rejections. `analysis_r01.json` retains six UNRUN cells and zero comparisons. The filtered replay reads actual before-state owned blocks and retains native release, reservation, new-output, request identity and full-cost checks. The analyzer implementer is not an independent reviewer of this new code; the prior group's review is not reused as repair acceptance.

The one next experiment is the frozen six-cell group after Qwen releases. It tests whether the observed rank-then-check gap affects complete-service tradeoffs, rather than only validating that a different current-state candidate fits. No new threshold, workload or additional arm is selected while waiting.


## 2026-09-14 measured addendum: six cells complete on replacement GPU

Current status: `MEASUREMENT_ONLY`, six actual cells / 192 completed request instances / 196,608 returned output token IDs. The preparation and old waiting sections above are historical; `analysis_r01.json` remains the original UNRUN artifact. Current measured source: `analysis_r02.json`.

The authorized westc:53036 endpoint recovered with a different GPU UUID, `GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc`. All six cells ran on this RTX 5090 with the same pinned runtime and actual 6,656 usable KV blocks, block size 16, requested KV bytes 13,960,740,864. The previous CTX calibration ran on `GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9`: it is the input/structural source, not a timing pair, noise baseline or same-device reproduction. Every timing comparison below is internal to the new six-cell group.

An authorized assisting session first launched the original untouched 20-file package at 1789396410.3845625 (controller1917/shell1918). The original executor observed that same group and did not start another driver. It completed with exit0 at 1789396822.5909417; at 1789396869.3064802 both control PIDs were absent, GPU compute processes empty and shared flock available. The previous Qwen task had no surviving process on the replacement instance; its old RUNNING record and missing terminal receipt were preserved, without inferring an exit cause or restarting it.

| Cell | Wall s | req/s | Mean completion s | Max request ITL s | Mean TTFT s | Mean TPOT s | Recomputed positions | Forced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block0-most_output | 23.871104 | 1.340533 | 21.981054 | 2.112613 | 0.345683 | 0.021149 | 93044 | 22 |
| block0-least_progress | 24.553422 | 1.303281 | 21.046706 | 1.931141 | 0.277276 | 0.020302 | 83510 | 20 |
| block0-least_feasible | 25.315591 | 1.264043 | 21.706353 | 2.039727 | 0.291850 | 0.020933 | 89761 | 21 |
| block1-least_feasible | 24.836317 | 1.288436 | 21.311881 | 1.990427 | 0.275688 | 0.020563 | 89761 | 21 |
| block1-least_progress | 25.152222 | 1.272253 | 21.554351 | 2.008486 | 0.272394 | 0.020803 | 83510 | 20 |
| block1-most_output | 23.570727 | 1.357616 | 21.735523 | 2.152757 | 0.274120 | 0.020979 | 93044 | 22 |

Positive completion/ITL delta means worse; positive throughput delta means better. All six within-block comparisons are retained.

| Block / action vs baseline | Throughput delta | Max ITL delta s | Mean completion delta | Requests with worse max ITL / later completion | Equal output sequences |
|---|---:|---:|---:|---:|---:|
| block0-least_progress vs most_output | -2.779% | -0.181473 | -4.251% | 6/32, 6/32 | 31/32 |
| block0-least_feasible vs least_progress | -3.011% | +0.108586 | +3.134% | 29/32, 32/32 | 30/32 |
| block0-least_feasible vs most_output | -5.706% | -0.072886 | -1.250% | 15/32, 13/32 | 30/32 |
| block1-least_progress vs most_output | -6.288% | -0.144271 | -0.834% | 6/32, 15/32 | 31/32 |
| block1-least_feasible vs least_progress | +1.272% | -0.018059 | -1.125% | 28/32, 0/32 | 30/32 |
| block1-least_feasible vs most_output | -5.096% | -0.162330 | -1.949% | 15/32, 9/32 | 30/32 |

The funding filter has not established a repeatable full-request improvement over least-progress: throughput changes −3.011% / +1.272%, mean completion +3.134% / −1.125%, and maximum ITL +0.108586 / −0.018059 s. Relative to most-output, filtering has lower worst gap and mean completion but throughput costs of 5.706% / 5.096%; this is a measured tradeoff, not dominance or an independent method contribution.

Fresh prefill is 90,112 and fresh decode 32,736 positions per cell. Compared with least-progress, filtering adds 6,251 recomputed positions in each block (83,510→89,761), one forced preemption (20→21), and seven engine calls (1,445→1,452). Mixed recompute calls can also emit fresh peer output, so these counts are not recoverable time, predicted speedup, or a pure-decode denominator. Each cell retains the nonoverlapping time sum `scheduler inclusive + engine remainder + outside engine = wall`; decision timing is nested, not added again.

Each role uses the same documents in forward/reverse order. This is not two independent workload repeats or a calibrated noise bound. Same-role block0→block1 throughput drifts are +1.274% (most), −2.381% (least), +1.930% (filtered); all same-role output sequences match32/32, while cross-policy equality is30–31/32. Do not convert token/request observations into independent repetitions or output equality into task quality. All reference TTFT/TPOT SLOs pass, so reference goodput equals throughput and provides no additional separation.

Archive: `execution/readback.tar.gz`, 133,554,053 bytes, SHA256 `7ad278ed5df66db7729338f534cd522c60f76958800ec60fc9ec52b5cf3a8329`. Full raw remains both in this bundle and the original remote directory; no failed cell or unfavorable block was dropped. The frozen analyzer accepts all six cells and produces nine predeclared comparisons. A fresh targeted review and the already-prepared structural model validation are in progress; no audit acceptance or model-match claim is inherited from the previous group.


### Frozen structural model checked against the new GPU executions

`model_check_r01.json` validates all six independent trajectories from their first qualified before-state at step90: **7,702 scheduler steps matched**, including scheduling/computed/output counters, actual KV allocation/free counts, forced/natural preemption, resumes, new-output events and request completion. The model source was frozen before these GPU results; max_steps=6000 is a predeclared constant and observed futures are validation labels only. This is a structural check, not a wall-time predictor, unknown-EOS model, tensor/quality proof or latency-ranking validation.

Both actual least-progress cells reject the same restore target (source0020902) at steps891–908 because their selected victim0020958 owns200 blocks: free31+200<required234. Filtering chooses the eligible victim0020484 with211 blocks, so31+211≥234, and removes these18 consecutive rejected decisions in each block. The target's first new output occurs at894 instead of916; scheduling recovery at891 is not itself a new output. This is one blocked restore episode per block, not18 independent events.

The local fix changes later requests: the worst pause shifts from source0020902 in least-progress to source0020794 in filtered. Target0020902's per-request maximum ITL falls by0.208906/0.318821s, while0020794's rises by0.293867/0.149979s. More capacity-feasible action does not by itself improve the full-request tradeoff. The model also matches total recomputation83,510→89,761 and last scheduler step1,444→1,451 in both blocks; it does not convert these differences into elapsed time.

The old CTX block1 conditional prediction is not the same initial state: its step90 free count was921, whereas both new least/filtered states are922 and some request counters differ. Its predicted86,779/1,447 values must not be substituted for the new89,761/1,451 results. No model coefficient or transition was fitted to make the new result match.


### Shared lifecycle localization: why early recovery was short-lived

The separate service-window analysis of these same raw files is reused directly, not counted as a second experiment: [lifecycle/source-localization report](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_context_r01/funding_r01/REPORT.md), with its `event_localization.json`. Both filtered blocks have one two-output re-discard and one zero-output re-discard, neither present in the same-group most/least cells.

For source0020902, recovery at891 produces new outputs at894/895, then native allocation pressure preempts it at896. The first-output guard had already ended normally at895; the30-step residency guard restricts adapter-selected victims, not native allocator eviction. Later it naturally resumes at1026 without an adapter recovery target; at1029 it needs49 blocks for the remaining781 positions but earlier peer allocations leave48, so its2,991-position partial recompute is discarded before any new output. The old prefix was reused between chunks before being discarded; neither the prefix count nor a mixed recovery-call span is a pure time-saving estimate.

These are two distinct boundaries—protection ending after its promised first output, and a naturally resumed request outside the adapter's explicit-swap protection—not a demonstrated violation of an enabled protection contract or a published-system flaw. Extending protection still requires a full-service comparison. The single-victim branch diagnostic below/next changes only the first choice, so it can test whether another legal victim avoids the later loss without immediately adding a service-window controller.


### Single-victim structural branch: immediate discard cost misses a later loss

`action_branches_r02.json` retains all27 currently guarded, fundable candidates at the same modeled step891 state; only that victim changes, and all later decisions use the frozen least-feasible rule. Every branch independently evolves from the same step90 before-state, with identical prefix to891 and fixed horizon6000. `r01` is preserved; r02 binds qualification/model/selector/raw hashes and adds the explicitly exploratory selection record, without changing any candidate metric.

All27 candidates produce the target's first new output at894, so this target-only indicator cannot distinguish their later consequences. Total recomputation ranges86,794–89,773, final completion step1,448–1,451, and the largest per-request new-output interval84–93steps (common output anchor89, earlier intervals/TTFT excluded). These are structural model values, not measured milliseconds, request goodput or a complete action Oracle.

Eleven candidates strictly improve allthree structural coordinates over default filtered: total recomputation, final completion step and maximum output interval steps. The default source0020484 minimizes immediate discarded computed positions among all27 (3,375positions;211blocks). An exploratory candidate source0017453 instead discards3,436positions and releases215blocks, yet predicts total recomputation89,761→86,902, final step1,451→1,448 and maximum output interval90→84steps. The target's first output stays894. Immediate discard minimization therefore does not minimize the modeled later recomputation in this event.

The candidate was selected after inspecting the complete range by minimum max-gap steps, then minimum total recomputation, then source ID. It is a next diagnostic candidate, not a preregistered win, held-out policy, learned online selector or runtime improvement. No runtime controller or additional GPU run implements this branch yet. The next smallest experiment is a native one-decision counterfactual that preserves the input, actual pool, later selector and full cost accounting, withmost retained as the strong simple reference; it must test whether this modeled ordering survives real batch/compute costs.

`selected_branch_localization_r01.json` records the causal diagnostic and producer/input hashes without replacing either 27-candidate table. At step1029, the default has50 free blocks before two peer allocations and48 available for the target's49-block growth; it discards its2,991-position partial prefix with no new output. In the selected branch,54 becomes52, enough for the same49-block allocation, and the target returns a new output. The branch has independently evolved: its target has703 prior outputs versus700 in the default, so this is not a fixed-future replay or a claim that the two states remain otherwise equal.

The selected branch's recomputation change is −2,859 positions: +6,909 for the new victim, −6,787 for the former victim, −2,985 for the target, and +2 each for two peers. One natural target preemption at1029 disappears. This localizes a candidate dependence between the first victim's released capacity and a later recovery completion; it does not establish a wall-time gain or prove that changing four blocks alone explains every downstream difference. The default event is supported by the shared actual lifecycle analysis; the alternate outcome remains model-only until the one-decision runtime probe.

### Targeted GPU integrity review complete

`EXPERIMENT_AUDIT.md/json` reports `WARN`, no P0/P1, same-family/provisional review. Its acceptance covers only the six actual GPU cells, packaged execution/analysis code, request and full-work metrics, current-state action replay, KV conservation, and measured addendum lines32–66. The structural model, shared lifecycle analysis and candidate branches are outside its review. The measured section's earlier “review in progress” sentence is historical. The pre-measurement `safe-cap-qualification.json` label `NOT_YET_RUN` is preserved; each adjacent terminal status and the group receipt are COMPLETE.

The reviewer recommends a fresh cohort to confirm the funding filter's performance hypothesis. That hypothesis remains unconfirmed. The immediate research task is instead the already localized one-decision diagnostic: distinguish whether the alternative victim's predicted later recovery completion survives native execution and full costs. This is exploration of a concrete causal omission, not a substituted confirmation test or permission to claim a post-hoc candidate as a new method. A stable benefit and contribution claim will still require fresh data and stronger coverage.
