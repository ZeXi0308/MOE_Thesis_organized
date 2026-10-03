# Experiment integrity review: rotation holdout r01 / execution03

**Date:** 2026-09-13  
**Overall verdict:** PASS  
**Integrity status:** pass  
**Reviewer:** gpt-5.6-sol, ultra, fresh read-only same-family reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  

This is an integrity PASS for the bounded native-serving measurement and its current report. It is not method acceptance, statistical confirmation, semantic-quality validation, or a GO verdict.

Path aliases used below:

- `E` = `refine-logs/expert_saturation/experiments/admission_capacity`
- `H` = `refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_holdout_r01`

## A. Ground-truth provenance: PASS

The workload is real dataset input, not a reference synthesized from model output. Preparation loads a pinned WikiText Arrow shard and pinned tokenizer/model revision, verifies their hashes, reproduces the old 32 inputs, and then selects the next 64 complete articles before splitting them into two document-disjoint cohorts (`E/prepare_rotation_holdout.py:50-68`, `E/prepare_rotation_holdout.py:70-108`). The retained provenance identifies WikiText-103 train revision, Arrow hash, selection and reconstruction rules, and explicitly limits independence to document/row identity rather than population independence (`H/inputs/inputs_report.json:533-551`). I independently hashed the retained Arrow file to the recorded `0c22278e...f9a99e`, checked 64 unique document and prompt hashes, checked 3,072-token prefixes, and confirmed disjoint row intervals.

There is no semantic ground truth and no quality metric. Generated sequences are model outputs; sequence equality is reported only as an execution observation. The report explicitly says that differing outputs are not errors and matching sequences are not a quality guarantee (`H/REPORT.md:53-55`). Native performance has a different evidentiary basis: host receipt clocks, request completion, scheduler actions, and executed work are directly measured by the real vLLM runtime.

## B. Metrics and denominators: PASS

Request accounting checks one-to-one output ID/timestamp alignment, monotonic host times, completion after the last token, and defines ITL as adjacent receipt gaps, TTFT from arrival to first receipt, and mean TPOT across the request's output span (`H/execution03/frozen/metrics.py:78-121`). Throughput is completed requests divided by `observation_end_s - min(arrival_s)` and the denominator includes all arrived requests (`H/execution03/frozen/metrics.py:136-155`). Work accounting reconstructs mutually exclusive first-time prefill, first-time decode, and recomputed request-position intervals from successful engine calls (`refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/analyze_native_preemption.py:35-88`). Host time closes as `wall = scheduler-inclusive + engine-non-schedule + outside-engine`, with decision time retained only as a scheduler subset (`E/analyze_headroom_cost.py:18-38`); work-call classes are disjoint and reject simultaneous prefill/recompute classification (`E/analyze_headroom_work_cost.py:23-44`).

I independently recomputed all 20 raw cells. Each has 32 unique requests, 32 completed, zero failed/unfinished, 32,768 output IDs and aligned receipt times. The saved elapsed time, throughput, mean completion, TTFT/TPOT distributions, and maximum ITL match before report rounding. The independently rerun primary and recovery analyzers both returned `MEASUREMENT_ONLY`; their substantive JSON is recursively equal to the published analysis after treating unordered bookkeeping collections as unordered.

The headline values are correct. Across cohort0/block0, cohort0/block1, cohort1/block0, and cohort1/block1, rotate versus native throughput is `+0.475%, -0.619%, -0.269%, +0.001%`; mean completion is `+1.109%, +2.284%, +1.901%, +1.713%`; maximum ITL falls by `3.467430, 3.433087, 3.449905, 3.461144 s`. Rotate versus headroom improves throughput by `1.943%-2.781%` and mean completion by `4.257%-5.067%`; these match `H/analysis/report.md:30-55` and `H/REPORT.md:11-18`. Safe29 has only 29/32 reference-SLO passes while the other arms have 32/32, and the report states this rather than conflating goodput and throughput (`H/REPORT.md:20-29`).

No A/A maximum is treated as a statistical noise bound or subtracted from policy effects. Request/token/step observations are not counted as independent repeats (`H/REPORT.md:33-42`; `H/analysis/report.md:66-70`). There is no significance or noninferiority claim.

## C. Evidence existence and frozen identity: PASS

The frozen manifest contains exactly two cohort roots and 20 unique cohort/block/role cells (`H/execution03/frozen/campaign.json:5-17`, `H/execution03/frozen/campaign.json:17-179`). The execution record is terminal `COMPLETE`; all 20 cells are `READ_BACK`, return code zero, terminal `COMPLETE`, and 32 requests completed (`H/execution03/execution.json:1-24`, `H/execution03/execution.json:299-333`). I checked every manifest label for `status.json`, `config.json`, `engine_args.json`, `environment.json`, `raw.json`, `metrics.json`, `headroom-decisions.json`, `memory-before.json`, `safe-cap-qualification.json`, and `gpu-after.json`; none is missing.

The provenance checker binds run order to the frozen manifest, verifies the uploaded archive and each read-back archive hash, checks execution-source hashes and runtime versions, and records the same GPU identity at the observed boundaries (`H/analysis/execution_checks.py:12-36`). Its retained 20 raw hashes and runtime fingerprints are in `H/analysis/execution_checks.json:1-181`. A representative rotate terminal records 32/32 complete, ten actual native recompute preemptions, and `MEASUREMENT_ONLY` (`H/execution03/gpu_results/cohort0-block0-rotate/status.json:1-8`). Earlier failed attempts remain separately retained and do not enter the 20-cell analysis (`H/REPORT.md:76-82`).

Each `safe-cap-qualification.json` intentionally records the pre-measurement snapshot field `measurement_status: NOT_YET_RUN`; the runner writes qualification before the episode and writes raw/metrics/terminal state afterward (`H/execution03/frozen/run_probe.py:114-134`, `H/execution03/frozen/run_probe.py:167-186`). It must not be read as the terminal campaign status; terminal/raw/execution records agree on COMPLETE.

## D. Dead code, online state, and future leakage: PASS

The executed runner installs rotation only for the rotation cells, installs headroom/native modes for the other cells, runs the real episode, uninstalls the adapter, and retains every decision attempt (`H/execution03/frozen/run_probe.py:147-182`). Rotation decisions use current running progress, tracked current absence, current free blocks, and current recovery need; the selector contains no later route, completion, or other-policy trajectory (`H/execution03/frozen/absence_rotation.py:114-180`). The adapter physically removes the chosen victim from running, calls vLLM's native `_preempt_request`, promotes the selected preempted request, reserves its full recovery, and checks retained KV/progress for held requests (`H/execution03/frozen/rotation_native.py:94-123`, `H/execution03/frozen/rotation_native.py:125-205`). A retained applied decision shows the actual forced victim, resume target, current free/released/required blocks, and scheduled action (`H/execution03/gpu_results/cohort0-block0-rotate/headroom-decisions.json:49536-49555`).

Native capture wraps the actual scheduler, records scheduled position ranges and exclusive recompute/prefill/decode work, then advances each policy through its own `engine.step()` calls and cumulative outputs (`H/execution03/frozen/native_capture.py:64-100`, `H/execution03/frozen/native_capture.py:104-177`). All four rotation cells record eight forced plus two natural native preemptions. Recovery accounting confirms native reset/prefix preservation, first recovery schedule, and first new token; it is a rerun over actual policy-specific traces rather than offline masking (`refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/analyze_native_preemption.py:91-108`; `E/analyze_rotation_recovery.py:17-67`). No defined metric/policy function relied on by the report is dead or bypassed.

## E. Baseline fairness and scope: PASS

The preregistered protocol fixes two cohorts, two blocks, five roles per block, the randomized order, the primary native label, and the rule that native A/A cannot replace the primary native baseline (`H/execution03/frozen/DECISIONS.md:13-24`). The analyzer enforces one cell per cohort/block/role, exact resource/input matching within each pair, all-20 completion before comparisons, and keeps A/A and same-role block repeats separate (`E/analyze_rotation_holdout.py:29-80`, `E/analyze_rotation_holdout.py:125-155`). Engine arguments match across all 20 cells; safe29 changes only the admission cap while the engine maximum, actual KV bytes, token budget, model, input and runtime stay fixed.

The narrative retains all four block-level signs, including the native-throughput sign flip and uniformly worse mean completion, and reports request-level harms and cross-policy output divergence (`H/REPORT.md:9-18`, `H/REPORT.md:44-55`). It describes initialization/pre/post process observations as boundary checks, not continuous GPU isolation or clock control (`H/REPORT.md:76-82`). It limits the claim to two new document cohorts in one model/length/arrival/KV regime, states the lack of formal nearest-prior-art comparison, Oracle, semantic quality, significance and noninferiority, and keeps the verdict `MEASUREMENT_ONLY` (`H/REPORT.md:93-108`). This scope is adequate for the stated descriptive transfer result and inadequate for broader claims, which the report does not make.

## F. Evaluation type: PASS

**Classification:** `self_supervised_proxy` on the semantic-evaluation axis, because no dataset quality target or human judgment is used. This label does not turn the performance result into a simulation or model-generated reference. The systems-performance portion is **direct native measurement**: OLMoE BF16 in vLLM 0.26 on one RTX 5090, with complete request histories, real native preemption/recomputation, and host token-receipt timing (`H/execution03/gpu_results/cohort0-block0-rotate/config.json:2-15`, `H/execution03/gpu_results/cohort0-block0-rotate/engine_args.json:1-19`, `H/REPORT.md:95-103`). Semantic quality remains unmeasured, so output equality supports only sequence consistency in the observed execution pairs.

## P0/P1 findings and claim impact

- **P0:** none.
- **P1:** none.
- The statement `20/20 COMPLETE, 640/640 formal requests` is supported.
- The rotate-versus-headroom throughput, completion and max-ITL ranges are supported as descriptive measurements in these four blocks.
- The rotate-versus-native max-ITL reduction and mean-completion penalty are supported; native-throughput preservation remains unconfirmed because the four deltas change sign.
- Throughput noninferiority, statistical significance, semantic equivalence, population generalization, MoE specificity, component necessity, novelty and method GO remain unsupported and are explicitly outside the report's claim ceiling.

## Non-blocking reproducibility notes

- A fresh reanalysis is numerically and structurally equal to the published analysis after unordered collections are normalized, but it is not byte-identical because `held_accounting` iterates Python sets when serializing per-request dictionaries and interval rows (`E/analyze_completion_headroom.py:39-52`). This changes ordering only and has no metric or claim impact.
- GPU process checks occur at initialization and measurement boundaries; continuous isolation and fixed clocks were not verified. The report states this limit.
- The Arrow file hash and retained prompt identities were independently checked; this audit did not rerun full tokenizer reconstruction from the Arrow shard. The preparation code and retained hashes support provenance, but this is not a separate re-tokenization replication.

## Audited immutable report hash

`H/REPORT.md` SHA256: `54af24825eb86c61fad46035d7f750b60ba13633d36f4d1a93bb66249bd6c658`.
