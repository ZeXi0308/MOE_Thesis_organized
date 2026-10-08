# Context dispersion and victim choice

Status at preparation: `STAGED / GPU_UNRUN`. The authoritative execution receipt is `execution/execution.json`; preparation is not measurement.

The fixed-KV question remains: how can a started request regain useful output without transferring excessive pauses or complete-service cost to other requests? This iteration asks whether generated progress remains an adequate victim ordering when prompt lengths differ. It does not propose a new controller.

The current strong evidence is the cohort3 native/most/fit/residual eight-cell group. Most has better maximum ITL and throughput than residual in both blocks, while residual has better mean completion. The service-window reanalysis also found no residual advantage over the measured native/most Q(g) envelope. Fixed-shape protection tuning is therefore stopped. The current group did not contain least-progress rotation; native or component configuration fields named `least_progress` are not that arm. Old d6 persistent least/most showed a mean-completion–throughput exchange, and the previous step329 single-event branches did not stably rank their conditional futures. Those old branches are not repeated here.

## Smallest model and discriminating change

At a scheduler boundary the online state is free physical blocks F, owned blocks b_i, prompt length p_i, emitted output o_i, computed positions c_i, active/waiting membership, previous preemption/resumption steps, and time of last new output. For a fully decoded resident, c_i = p_i + o_i − 1. Current block-table ownership is observed rather than inferred from the declared output cap.

A waiting target needs h_t = ceil((p_t + o_t)/16) − b_t blocks to fund its existing history. The current one-victim rotation tests F + b_v >= h_t after selecting v by output/progress. If feasible, actual native preemption releases v's blocks; the adapter protects t's remaining history and holds peers when their allocation would consume that reserve. Protection ends only after a new output is observed. The 30-step absence/residency and 20-step swap interval remain unchanged; these are not a per-request wall-clock waiting guarantee.

When all prompts have equal length and residents are in decode, generated progress, computed positions and KV footprint are ordered together up to block rounding. With unequal prompts those rankings can disagree. More reclaim may discard more work; fewer discarded positions need not save elapsed time in a shared batch. This calibration measures that separation and its actual request consequences before adding a cost-aware choice or feasibility fallback. No fixed per-request restoration-time constant, known EOS distance or counterfactual future trace is used.

## Frozen six-cell calibration

| Setting | Value |
|---|---|
| Inputs | All 32 cohort3 documents in the existing source order; even positions use first 2560 tokens, odd positions retain 3072 |
| Output and arrivals | Fixed 1024 outputs with EOS ignored; existing 50 ms arrival spacing |
| Runtime and physical pool | Existing OLMoE BF16, vLLM 0.26, synchronous native recompute, APC off, 6656 usable blocks / 13960740864 configured KV bytes |
| Engine bounds | Original 4096 model context, cap32, work budget1024; no context extension |
| Arms and order | native / most_output / least_progress / least_progress / most_output / native |
| Primary view | Per-request maximum generation gap versus complete-episode request throughput |
| Other costs | TTFT, TPOT, every request's completion and maximum ITL, recompute positions, restore waiting, held work, nested scheduler/decision cost |

The summed full-length demand is 7680 blocks versus the previous homogeneous 8192. This is an upper-demand sum, not realized pressure. The actual timeline, minimum free blocks, preemptions, rotation activation and unsuccessful funding proposals are measured separately. A complete no-action cell is retained as unexposed calibration; it is not rewritten as implementation failure or dropped to select a favorable workload. No alternative pressure point is scanned based on this group's outcomes.

This is exploratory reuse of previously observed documents and a deliberately changed operating regime. It is not a fresh confirmation cohort, matched dynamic pressure, true-EOS validation or independent repeated workload. Comparisons are within the six cells. Hardware, resource pool, backend, model semantics and observation are common across those cells. The primary view is not replaced by whichever secondary metric improves.

## Implementation and pre-run checks

`prepare_context_victim_calibration.py` copies the already executed cohort3 package. Only its runner changes: it verifies each prompt length, exposes the existing least-progress arm, supplies the per-request vector and its conservative maximum to the existing drained-engine resource qualifier, and retains zero-action episodes. Native capture, allocation, rotation, preemption and protection sources are byte-identical. The qualifier's 256-block value is explicitly a worst-request bound; the actual vector sum and actual live pool are recorded separately.

`analyze_context_victim_calibration.py` reuses the qualified native work/state accounting. The causal selector replay now uses the actual arm and each request's p_i + 1024 cap instead of fixed `most_output` and 4096. All requests and repeat directions remain in the analysis. Missing cells produce six `UNRUN` entries and zero comparisons. The previous homogeneous most/least traces exercise both replay paths; intentionally mismatched victim order is rejected. These CPU checks do not establish heterogeneous runtime correctness or benefit.

Package: `03b3be341fd30405689d249e711ead4990d70ce3118150ef7234cc31d7e686de`.
Remote: `/root/autodl-tmp/moe-context-victim-calibration-20260914-r01` on the existing authorized westc:53036 host.
No new instance, model download or GPU driver has been started by staging. The group waits behind the already registered B second-cohort group; execution requires another live process/lock check and holds the common lock across all six engine initializations and measurements.

## Interpretation rule and continuation

If the workload exposes preemption and candidate rank disagreement, inspect whether the actual selected victim cannot fund a target that another eligible victim could fund, or whether both feasible orders mainly transfer cost. Only then choose one observed decision or one minimal fallback for a native causal comparison. A static feasible candidate is not a speedup claim. If all tested simple arms cover the useful tradeoff, retain them as stronger baselines. If no recovery action is exposed, the result limits this workload's diagnostic value; it does not falsify fixed-KV recovery scheduling.

The next executable step is this exact six-cell group after the existing queue releases. The current contribution boundary remains `MEASUREMENT_ONLY`: no verified scheduling defect has yet produced a new method advantage over the strongest simple baseline.

## Executed result — 2026-09-14
`MEASUREMENT_ONLY / OBSERVED_UNFUNDED_SELECTION_GAP`. All six original cells completed: 192 request instances and 196608 emitted tokens. These are two ordered repeats of the same 32-document workload, not 192 independent workloads. The final readback receipt is `execution/recovery.json`; the earlier `execution/execution.json` intentionally retains `UNKNOWN_REMOTE` after transport loss. Controller65959/shell65960 finished at1789349373.497525 and were later verified absent. No cell was restarted. The complete archive SHA is `7e83a4483e37f65c4ad83ecebc3f522eefbd7263687ac4e10de86be92b1e9f5d` (133095054 compressed bytes,155 members). GPU release was recorded in the shared coordination file.
`analysis_r02.json` is the valid complete analysis. It checks all recorded request/receipt/step identities, executed positions, actual block release and6656-block conservation, selector replay, source/resource identity, saved metrics and all warmups. `analysis_r01.json` is retained: an analyzer pressure-summary field assumed a native observer contained `forced_preempted`; using an empty list for that absent rotation-only field fixed the two false invalid labels. Raw results did not change.
| Arm / block | Wall s | Requests/s | Mean completion s | Maximum request ITL s | Recomputed positions | Forced / all preemptions |
|---|---:|---:|---:|---:|---:|---:|
| native / 0 | 24.462045 | 1.308149 | 20.306862 | 9.396252 | 17820 | 0 / 5 |
| most_output / 0 | 23.874605 | 1.340336 | 22.031640 | 2.104871 | 93044 | 22 / 26 |
| least_progress / 0 | 24.948554 | 1.282639 | 21.478581 | 1.925519 | 83510 | 20 / 24 |
| least_progress / 1 | 25.442964 | 1.257715 | 21.842406 | 2.005216 | 83517 | 20 / 24 |
| most_output / 1 | 23.449806 | 1.364617 | 21.627211 | 2.122336 | 93044 | 22 / 26 |
| native / 1 | 24.830598 | 1.288733 | 20.571991 | 9.774883 | 17820 | 0 / 5 |

Every cell reached the actual6656-block limit. This establishes pressure in this episode; it does not establish a matched pressure trajectory against the old homogeneous cohort. Fresh prefill is90112 positions and fresh decode32736 positions in every cell. Recompute is measured separately; it is not a time-saving denominator.
Most versus native improves throughput by2.461%/5.888% and reduces the maximum generation gap by7.291/7.653s, while mean completion increases8.494%/5.129%. Respectively29/28 requests finish later, and17/17 have worse own maximum ITL. Thus the global worst gap reduction transfers costs to other requests. Least versus most has0.179/0.117s smaller global maximum ITL but4.305%/7.834% lower throughput; its mean-completion difference changes sign,−2.510%/+0.995%. No method winner or significance is inferred.
The largest restore-wait component is9.224/9.647s for native,1.946/1.964s for most and1.758/1.836s for least. These are observed wait intervals after the preempting call, not fixed per-request restoration constants. Least block0 has a0.494s maximum recovery-service span versus0.151s in block1; its cause is not attributed. Native and most repeat outputs match32/32; least repeats match21/32, with83510/83517 recomputed positions. Timing drift and changed generated text remain in the results; output equivalence is not a quality evaluation.
### Actual decision gap
`victim_candidates_r01.json` replays4671 actual proposal calls across the four rotation cells. In each most cell22/22 proposed rotations execute, with no funding rejection;21/22 proposal events have eligible candidates whose output and owned-block rankings strictly disagree. Thus prompt heterogeneity actually breaks the old fixed-P ordering relationship.
In each least cell,38 rotation proposals yield20 actual exchanges and18 funding rejections. All18 rejections occur at consecutive steps891–908 for the same target `memory-train-article-0020902` and selected victim `memory-train-article-0020958`; they are not18 independent target failures. Each rejected proposal has another then-eligible candidate satisfying the unchanged physical funding test. At891, target need is234 blocks and free is31: selected victim owns200, giving231 and rejection. Candidate `memory-train-article-0020484` owns211, giving242. At908 another eligible candidate `memory-train-article-0020734` can still fund the target. The actual failed-victim computed positions are state at risk, not work actually discarded: no forced eviction happened for these proposals.
This local failure is in the tested least-progress rotation plus rank-then-check adapter. It is not established for most, Andes, LTR or a complete published system. The static alternative is feasible in the observed before state; its full future and net service cost have not yet been executed. The next smallest change is a current-capacity filter before the existing victim ranking, preserving target, cooldown, protection, backend and physical resources. It is a stronger component baseline. It must be compared with both least and most in new native executions before any benefit or contribution claim.
### Boundary
Evidence tier: `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE`. Request-window costs include actual scheduler, observation, normal/recomputed work and waiting; engine initialization, warmup, post-measurement serialization and network/client transport are outside this window. The two order blocks do not provide independent workloads, a noise bound or production SLO evidence. Maximum output is fixed and EOS ignored; no known-EOS-distance or MoE-specific novelty claim follows. The bounded failure category is **an unfulfillable selected victim despite an eligible feasible alternative**, with net effect of repair still `UNRUN`. The fixed-KV recovery problem remains open.

### CPU repair and fresh review

The optional funding filter is implemented in the worktree selector/native adapter, separately from this group's frozen package. With the option absent,4671 calls and120 proposals from all four actual rotation cells replay exactly. Applied separately to the36 rejected before-states, the filter retains each original target and selects a victim satisfying the current funding test. At891 it chooses the211-block candidate. Native allocator fixtures retain actual preemption and first-new-output protection. `funding_filter_cpu_r01.json` preserves these checks and the then-stale test signature whitelist; the later regression test update belongs to the next prepared comparison. These checks do not advance alternative futures or measure repair latency.

The [fresh integrity review](EXPERIMENT_AUDIT.md) found no P0/P1: A–D PASS, E/F WARN, same-family provisional. It independently recalculated the six request and work totals. The accepted finding remains the observed local funding gap; repair benefit and superiority over most_output remain unmeasured.


### CPU structural model: earlier output can increase episode work

`model_check_r02.json` validates the reused discrete KV/token model from each actual cell's first qualified active state at step90. All32 requests are resident in pure decode, waiting is empty, and prior preemption/resume/forced history is empty. Across the four most/least cells, all4978 modeled scheduler steps match actual scheduling order, token counts, computed/output counts, free blocks, preemption, resumption, new-output events and completion steps. Recompute totals also match. Exact model source is `experiments/admission_capacity/model_source.py`; validation entry is `validate_context_progress_model.py`.

Only after this qualification, a single callsite adapter enables the frozen funding filter and advances an independent candidate state. The first difference is step891 in both blocks. Selecting the211-block victim funds target0020902 and starts recovery then; it produces the first new output at894, rather than immediately at891.

| Initial-state block | Target first new output step: least → filtered | Recomputed positions: least → filtered | Last episode step: least → filtered |
|---|---|---|---|
| 0 | 916 → 894 | 83510 → 89761 | 1444 → 1451 |
| 1 | 915 → 894 | 83517 → 86779 | 1444 → 1447 |

These are **conditional CPU predictions**, not new GPU measurements or a latency ranking. Both filtered predictions have21 forced exchanges versus20. Earlier target output can coexist with more total recomputation and later completion steps. Mixed calls still serve fresh peer outputs; token/call counts are not converted into time savings. The unchanged six-cell native comparison remains necessary to measure complete-service tradeoffs against most_output.

This model covers the controlled fixed-output closed cohort, not tensor equality, output quality, unknown EOS, continuous arrivals or elapsed time. `model_check_r01.json` used observed last_step+2 as a validation horizon; its values are preserved, but that was not a clean future-independent input contract. r02 uses a predeclared6000-step limit for all simulations. All previous validation and prediction fields are exactly equal; the new nested field records the target's first new output. GPU raw/package/metrics were unchanged.

Reproduce from the worktree: `python3 -B refine-logs/expert_saturation/experiments/admission_capacity/validate_context_progress_model.py --output /private/tmp/NEW-context-model.json`. The simulator uses only the initial resource/request state and prior controller history to advance state; actual future events are validation labels.
