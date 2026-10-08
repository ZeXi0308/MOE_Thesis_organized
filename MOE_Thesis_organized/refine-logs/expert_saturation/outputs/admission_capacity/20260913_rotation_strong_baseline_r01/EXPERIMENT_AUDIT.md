# Experiment Audit Report

**Date:** 2026-09-14 (Asia/Shanghai)  
**Campaign:** `20260913_rotation_strong_baseline_r01`  
**Measured run:** `execution02_westc_53036`  
**Auditor:** GPT-5.6-Sol ultra, fresh read-only same-family reviewer  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Repository HEAD inspected:** `c4ae4f5daaea929e1a4358862296d832f1cc67ab`  
**Workspace state:** dirty before this audit; all pre-existing changes and experiment artifacts were treated as immutable.  
**Trace:** no `.aris` trace was created, as required by the explicit audit scope.

## Overall Verdict: WARN

The eight measured cells, policy actions, retained request/token records, and reported arithmetic are authentic and internally consistent. No P0 or P1 integrity defect was found. The result remains `WARN` because this is one model, one GPU, one 32-document cohort, and two order-confounded blocks separated by a 22.28-minute controller interruption. The retained A/A observations and reverse order expose drift, but they are not controlled independent repeats or a noise bound. One large same-path runtime stall remains unexplained; JIT/cache, temperature, CPU scheduling, and kernel/tactic causes are unverified.

This audit supports the campaign's bounded `MEASUREMENT_ONLY` interpretation. It does not support task-quality equivalence, statistical generalization, production SLO claims, method GO, or causal attribution of the block-to-block timing difference.

## Severity Findings

- **P0:** 0.
- **P1:** 0.
- No fake ground truth, self-normalization, phantom result, dead reported action, omitted failed cell, selective recovery rerun, overwritten canonical result, or numerical substitution was found.

## Evaluation Type and Scope

| Dimension | Audited scope |
|---|---|
| Evaluation type | Direct paired systems-performance observation without task-quality ground truth; output equality is a paired execution-identity proxy only |
| Evidence tier | `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE` |
| Scientific status | `MEASUREMENT_ONLY`; primary native-relative throughput question remains open |
| Model/runtime | `allenai/OLMoE-1B-7B-0924`, BF16, vLLM 0.26.0 |
| Hardware | One NVIDIA RTX 5090, one recorded GPU UUID |
| Workload | One fresh 32-document WikiText-derived cohort; 3,072 input and 1,024 output tokens per request; 50 ms steady arrivals; seed `20260905` |
| Design | Four roles (`native`, `native_aa`, `headroom`, `most_output`) × two reverse-order blocks = eight fresh engine executions |
| Comparisons | Six within-block policy pairs, two native A/A pairs, four same-role cross-block observations |
| Excluded | HTTP/client/network latency, multi-GPU EP, task-quality ground truth, production workload/SLOs, independent population, significance/noninferiority, and device-level kernel attribution |

The workload and evidence ceiling are frozen before GPU results in `inputs/config.json:2-40` and `execution02_westc_53036/frozen/DECISIONS.md:3-25`. The eight-cell order and roles are fixed in `execution02_westc_53036/frozen/campaign.json:11-83`.

## A. Ground Truth and Provenance: PASS

- Model/revision, tokenizer revision, request dimensions, seed, reference SLOs, dataset revision, Arrow shard/hash, selection rule, reconstruction rule, and workload hash are explicit in `inputs/config.json:2-40`.
- The reviewer independently reconstructed the pinned local source selection. All 32 current documents, prompt texts, 3,072-token prefixes, IDs, hashes, and row intervals matched. Across the 96 prior identities plus this cohort, 128/128 document hashes and prompt hashes were unique and source intervals were disjoint. The retained receipt records these checks and the source hash at `execution02_westc_53036/frozen/fresh_inputs_receipt.json:706-729`.
- Frozen and current input copies and declared hashes matched. The preflight records the three model shard hashes, GPU identity, and empty process list at `execution02_westc_53036/preflight.json:5-23`.
- This is a new document cohort from the same corpus, length, arrival, and runtime regime, not an independent population. That limitation was frozen at `execution02_westc_53036/frozen/DECISIONS.md:11-17`.
- No external task-quality reference exists. Generated-token equality is valid only as a paired execution-identity check. The reference TTFT/TPOT thresholds are not ground truth and do not constrain maximum ITL; `RESULTS_WESTC_ADDENDUM.md:40-42` states this boundary.

**Unresolved limitation:** provenance establishes fidelity to the retained local Arrow artifact. It does not independently authenticate WikiText upstream or establish semantic output quality.

## B. Score Normalization and Metric Denominators: PASS

The frozen metric code includes every arrived request, rejects duplicate request IDs, checks timestamp/token alignment and monotonicity, and declares the denominator as all arrived requests at the actual observation end (`execution02_westc_53036/frozen/metrics.py:10-29`, `:62-105`). Exact definitions are retained at `:106-155`:

- TTFT = first output timestamp − arrival.
- TPOT = `(last output timestamp − first output timestamp) / (output tokens − 1)`.
- ITL = each consecutive output-token timestamp difference.
- Request completion latency = completion − arrival.
- Throughput = completed requests / `(observation_end − earliest arrival)`.
- Goodput uses SLO-passing completed requests over the same duration.

Deterministic checks covered, per cell, 32 request samples, 32 TTFT values, 32 TPOT values, and 32,736 ITLs. All 32 requests completed in every cell, so throughput equals `32 / observation_duration`; every request passed the reference TTFT/TPOT checks, so goodput equals throughput.

The reviewer independently recomputed all six policy-pair deltas and both A/A deltas from raw records. They match `analysis02_westc_53036/report.md:18-32` to floating-point precision:

| Pair | Throughput Δ | Mean completion Δ | Max ITL Δ |
|---|---:|---:|---:|
| block0 native → headroom | −4.793% | +10.067% | −3.029202 s |
| block0 native → most_output | −1.362% | +5.962% | −3.051666 s |
| block0 headroom → most_output | +3.604% | −3.730% | −0.022465 s |
| block1 native → headroom | −1.958% | +6.352% | −3.194041 s |
| block1 native → most_output | +3.241% | +1.031% | −3.739974 s |
| block1 headroom → most_output | +5.303% | −5.003% | −0.545933 s |
| block0 native → native_aa | +0.582% | −0.433% | +0.127070 s |
| block1 native → native_aa | +1.113% | −1.064% | −0.070028 s |

No model-output self-normalization, proposed-saving denominator, request/token mismatch, favorable averaging, or double-counted decision time was found. The accounting identity and decision-time inclusion are explicit in `analysis02_westc_53036/report.md:34-40`.

**Unresolved limitation:** maximum ITL is the literal maximum of 32,736 intervals per cell. It is sensitive to one runtime stall and is neither production p99 nor a long-pause SLO.

## C. Result Existence and Numerical Fidelity: PASS

Exact retained evidence checked:

- 8/8 cell directories and raw files.
- 8/8 metrics, config, environment, status, and execution-receipt sets.
- 8/8 stdout/stderr pairs.
- 24/24 warmups, all `COMPLETE`; each cell retained 32-request, 32-request, and 2-request warmups with 16 output tokens per request.
- 256/256 planned requests completed; 0 failed, unfinished, or skipped.
- 262,144/262,144 requested output-token IDs and timestamps retained.
- Every measured request had exactly 1,024 output IDs and monotonic token times.
- All scheduler work, engine calls, decisions, preemption events, prefill/decode/recompute totals, and KV-pool conservation checks reconciled.
- Every actual paired request comparison had 32/32 identical output-token sequences. This proves identity for these executions only.

All eight cells and their completion/preemption statuses are present in `execution02_westc_53036/execution.json:2-133`; all eight measured rows and every favorable and unfavorable primary comparison appear in `analysis02_westc_53036/report.md:5-36`. The analyzer requires all eight cells, common execution configuration, action qualification, and matched workload identities before emitting numeric pairs (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_strong_baseline.py:82-143`; imported helper `analyze_completion_headroom.py:104-202`).

The retained execution archive hash is `7cbd2f0e595327ddcba427a3d3599f46594e3c67ca96b45b4ec6f1e1a591d9fe` (`execution02_westc_53036/execution.json:150`). All eight local cell archives independently matched their per-cell receipt hashes at `execution02_westc_53036/execution.json:11-124`.

No missing claimed metric key, omitted failed cell, overwritten raw result, favorable-result selection, or baseline substitution was found.

## D. Action and Independent Policy Evolution: PASS

- The runner enforces frozen role/cap/input settings, refuses an existing output directory, records source hashes, starts a fresh engine, installs the selected policy, and only then captures the episode (`execution02_westc_53036/frozen/run_probe.py:40-118`, `:152-194`).
- `headroom` decides from current scheduler/KV state and holds requests in the actual scheduling path (`execution02_westc_53036/frozen/completion_headroom.py:144-231`).
- `most_output` selects eligible victims from current generated-output counts, without future outputs or an offline completed trace (`execution02_westc_53036/frozen/absence_rotation.py:133-205`).
- Forced rotation invokes the native preemption path, keeps independent held/recovery state, and reconciles requested actions with scheduler outcomes (`execution02_westc_53036/frozen/rotation_native.py:94-205`).

Actual retained paths differ as the policies require:

| Role, per cell | Forced / natural preemptions | Recomputed positions | Held-request steps |
|---|---:|---:|---:|
| headroom | 0 / 0 | 0 | 9,598 |
| most_output | 9 / 2 | 43,471 | 0 |
| native / native_aa | 0 / 2 | 7,685 | 0 |

These differences, together with distinct engine-call trajectories, show that the policy functions changed real execution rather than writing dead diagnostic records. All eight custom-source hash sets matched the frozen files, engine/config signatures were common apart from the intended policy/victim-order fields, and the seven recorded third-party vLLM source-hash maps were identical across arms.

**Unresolved limitation:** the exact third-party vLLM source files were not retained locally, so the recorded remote vLLM hashes cannot now be independently re-derived from source bytes.

## E. Scope, Retention, and Environmental Comparability: WARN

The frozen design is one model, one RTX 5090, one fresh 32-document cohort, two reverse-order blocks, four roles, eight engines, and one seed (`execution02_westc_53036/frozen/campaign.json:11-83`; `inputs/config.json:2-16`). It is a descriptive repeated execution, not eight independent workloads.

The interrupted-controller recovery is intact. The third cell completed remotely; recovery read it back and executed only the five untouched suffix cells, with no rerun (`execution02_westc_53036/controller_recovery/1789315156441299000/recovery.json:2-16`). The remote precheck records the terminal processes absent, source/archive agreement, and only the already-created prefix artifacts (`.../remote_precheck.json:125-160`).

The third cell finished at `1789313826.032673` and the fourth began at `1789315162.842915`, a gap of 1,336.810242 seconds (22.28 minutes), visible at `execution02_westc_53036/execution.json:38-59`. Other coordinated work used the GPU during that interval. Per-cell preflight found no competing process at initialization, and all cells used the same GPU UUID and software/source hash maps, but this does not establish one uninterrupted thermal/cache/runtime state. The later handoff process is explicitly separated at `gpu_handoff.json:2-15`.

Environmental evidence has these limits:

- Every cell logged a fused-MoE JIT warning before measured capture; representative evidence is `execution02_westc_53036/gpu_results/cohort2-block0-native.stdout.log:37-42`.
- Every cell logged that the device-specific MoE configuration was missing and the default might be suboptimal; representative evidence is the same stdout at line 28.
- Every stderr begins with two SM12/CUDA capability warnings; representative evidence is `.../cohort2-block0-native.stderr.log:1-2`.
- No continuous temperature, clocks, CPU scheduling, compile/cache activity, or process telemetry proves identical state throughout each measured window.

Native A/A drift reaches +1.113% throughput and −1.064% mean completion. The two otherwise identical `most_output` executions differ by +4.160% throughput, −3.858% mean completion, and −0.499378 seconds maximum ITL from block0 to block1. The targeted read-only diagnostic localizes most of the difference to the same logical step-839 engine call: 0.766318 versus 0.031701 seconds, while scheduler time differs by only 0.000265 seconds (`diagnostics/most_output_block_difference.md:16-24`). The diagnostic cannot distinguish JIT/cache, temperature, tactic selection, CPU delay, or another runtime cause.

Therefore A/A and reverse order are useful drift observations, but neither is a controlled noise bound, an independent repeat, or removal of order/time confounding. No retained evidence justifies subtracting or discarding the slow call.

## F. Evaluation Classification and Claim Scope: WARN

**Primary classification:** direct paired systems measurement without task-quality ground truth. Host timestamps, scheduler actions, request identity, and token IDs are direct execution records. Output-sequence equality is a self-comparison proxy, not an external correctness label.

**Evidence tier:** `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE`. This is native vLLM engine execution with policy-specific state evolution. It excludes network/client overhead, multi-GPU EP, production traffic, external task quality, and device-level attribution.

The new claim source is faithful within that ceiling. Its result/cost tables match raw-derived analysis; it preserves both unfavorable native-relative outcomes, rejects quality and long-pause-SLO upgrades, retains the slow call, leaves the internal cause unresolved, and discloses the interruption and claim limits (`RESULTS_WESTC_ADDENDUM.md:40-42`, `:44-74`, `:78-89`). The mutable future GPU scheduling statement at line 91 was not accepted as experiment evidence.

Same-family review cannot establish different-family reviewer independence. This audit therefore remains provisional even with no P0/P1 defect.

## Claim Impact

| Claim | Impact |
|---|---|
| All eight cells completed and the published values/pair deltas match retained raw | Supported |
| The policies executed their declared actions and evolved independent runtime state | Supported |
| `most_output` shortened maximum ITL relative to native in both observed blocks | Supported only as a descriptive observation in this campaign |
| `most_output` beat `headroom` on throughput and mean completion in both blocks | Supported only as a descriptive observation in this campaign |
| `most_output` preserved native throughput or mean completion | Unsupported; native-relative throughput changes sign and mean completion worsens in both blocks |
| Matching outputs establish task-quality equivalence | Unsupported |
| A/A/reverse order establish a noise bound or controlled repeat | Unsupported |
| JIT, temperature, or a particular kernel caused the step-839 stall | Unverified |
| Generalized method benefit, production SLO benefit, or method GO | Unsupported |

## Unresolved Limitations

1. One 22.28-minute controller interruption with intervening GPU use.
2. Two order-confounded blocks, not controlled independent repeats.
3. One large same-path stall whose internal cause is unverified.
4. JIT/cache, temperature, clocks, CPU scheduling, and tactic effects are unverified during measurement.
5. One model, one GPU, one same-corpus cohort, one length/arrival/KV regime, and one seed.
6. No external task-quality ground truth and no production long-pause SLO.
7. No complete action-space Oracle or problem upper bound.
8. Same-family reviewer independence only.

## Final Research Contract

| Field | Audited conclusion |
|---|---|
| Verdict | `WARN`; integrity checks pass, experimental independence and scope remain limited |
| Evidence type | Direct request-level native in-process systems measurement; output equality is an identity proxy |
| What was measured | Eight matched engine executions, all request/token completions, host timing, actual holds/preemptions/recompute, full observed wall accounting, and per-request effects |
| What was not measured | Task quality, independent production SLOs, independent workloads/seeds, bursty or heterogeneous regimes, second model/GPU class, multi-GPU EP, device-level timing cause |
| Strongest baseline | Predeclared native plus fast `completion_headroom`; `native_aa` is drift observation only |
| Oracle/headroom status | No full action-space Oracle or problem upper bound; only measured strategy tradeoffs |
| Claim ceiling | One-model, one-GPU, one-cohort, steady-arrival descriptive tradeoff |
| Failure category | Native-relative throughput direction changes sign; runtime-state cause of the large same-path timing difference is unverified |
| Resurrection condition | The question is open, not dead; require an uninterrupted controlled repeat and interpretable cost evidence before expanding the claim |
| One next smallest experiment | One uninterrupted, frozen four-engine group on the same cohort/runtime/warmups: `native → most_output`, then `most_output → native`; preserve every slow call and observe whether the aligned first-recovery call and native-relative direction recur |

**Direct answer:** the retained campaign supports its exact descriptive numbers and action-path observations with no P0/P1 integrity defect. It does not establish a stable native-relative benefit or a method claim; that question remains open under the stated repeat condition.
