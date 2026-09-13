# Experiment Audit — first-swap weste six cells

Date: 2026-09-13. Auditor: fresh GPT-5.6-Sol ultra, read-only, same-family/provisional.
Overall verdict: **PASS**. P0=0, P1=0. Independent numerical discrepancies: 0.
Reviewer task: `/root/first_swap_integrity`.

R = `execution02_weste_23478`; F = this experiment directory; E = `refine-logs/expert_saturation/experiments/admission_capacity` relative to the repository root.
This records the reviewer's judgment. It is not a method GO, statistical acceptance or cross-family review.

## A. Reference / GT provenance — PASS / N/A

Real native-runtime measurement has no accuracy ground truth. Frozen config records WikiText input, model/tokenizer revisions and workload identity (R/gpu_results/cohort0-block0-least_progress/config.json:2–40); model shard sizes/hashes are in R/preflight.json:2–23. Model output tokens are used for output accounting and cross-policy equality, not as accuracy reference labels.

Tokens/completion originate in actual engine.step and host receipt timing (R/frozen/native_capture.py:133–177).

## B. Denominator / accounting — PASS

Throughput uses completed requests divided by observation_end_s minus earliest arrival; ITL and TPOT derive from raw token times (R/frozen/metrics.py:40–59,75–155). No self-max/min/mean normalization was found. Relative treatment changes use the measured baseline (experiments/admission_capacity/analyze_completion_headroom.py:168–190).

Diagnostics sums actual engine-call host duration for pure-decode width buckets (F/diagnostics/diagnose_paths.py:85–101,126–128). Its report explicitly rejects independent additive causal saving and hard-bound interpretations (F/diagnostics/measured02_weste_23478/report.md:12).

## C. Artifacts / completion / independent recomputation — PASS

The reviewer found all requested result artifacts. Recomputed hashes for six cell tar archives and the execution archive match R/execution.json:11–115. All six cells are READ_BACK, return code0 and COMPLETE (R/execution.json:2–100).

Frozen campaign/source and preparation source match byte-for-byte. Engine args match; normalized configs match after removing only rotation_victim_order. The 32 request/document/arrival/prompt-length/prompt-hash identities match between prepared workload and all raw files. The manifest declares the single treatment key (R/frozen/campaign.json:67–69).

Independent recomputation from all six raw files:

| Cell | Completed | Outputs | Duration s | Requests/s | Mean completion s | Max ITL s |
|---|---:|---:|---:|---:|---:|---:|
| block0 least | 32 | 32768 | 22.779435411 | 1.404775817 | 20.747717550 | 1.009063169 |
| block0 first-most | 32 | 32768 | 22.741770795 | 1.407102388 | 20.741039660 | 1.010541799 |
| block0 most | 32 | 32768 | 22.438916729 | 1.426093799 | 20.964271838 | 0.981988566 |
| block1 most | 32 | 32768 | 22.480406631 | 1.423461796 | 21.002802813 | 0.979499810 |
| block1 first-most | 32 | 32768 | 22.825782742 | 1.401923446 | 20.824178322 | 1.009927103 |
| block1 least | 32 | 32768 | 22.728727856 | 1.407909858 | 20.690988400 | 1.009009473 |

All192 requests have1024 output tokens and monotonic aligned token times; total196608. Values exactly match saved metrics and primary analysis before report rounding (F/analysis02_weste_23478/report.md:5–12). All six complete output-sequence hashes match.

Diagnostics primary_input hash matches the primary analysis, and all six report rows match diagnostics JSON (F/diagnostics/measured02_weste_23478/diagnostics.json:44114–44118). run_dir_checks only validates CLI routing and rejection of missing/incomplete execution; it explicitly contains no GPU-result claim (F/diagnostics/run_dir_checks/checks.json:2–26).

## D. Action / metric execution and transition semantics — PASS

The actual path installs rotation, executes measured capture_episode, saves raw/decisions, then derives metrics (R/frozen/run_probe.py:151–191). Forced actions invoke the native scheduler's _preempt_request and check protected recovery/held state (R/frozen/rotation_native.py:94–105,179–205).

Independent trace checks yielded A/C8 forced+2 natural and B9+2 in each block. Raw successful preemption totals10,10,11,11,10,10 close exactly. The first successful forced action is step836 in all six; C uses most_output only there and least_progress afterward. Only successful forced actions increment the counter (R/frozen/absence_rotation.py:101–109; rotation_native.py:203–205; E/analyze_rotation_first_swap.py:49–71).

Each cell creates/closes an independent LLMEngine (R/frozen/run_probe.py:117–118,198–200). Six initial GPU-boundary records are empty; source and scheduler hashes agree (representative environment.json:7–30).

## E. Independence / scope — PASS

One reused cohort, 32 documents, two reversed-order blocks, six engine executions; these are not six independent workloads (F/analysis02_weste_23478/analysis.json:336949–336958).

The primary report explicitly limits same-role drift, significance, fairness, quality equivalence, noninferiority and method GO (report.md:33–37). C/A throughput changes sign (+0.166%/−0.425%); role-repeat drift is0.185%–0.368%. These support descriptive same-workload evidence only.

## F. Evaluation classification — PASS

Real single-GPU native vLLM in-process runtime measurement, no GT task.
Evidence: NATIVE_SERVING_INPROCESS_HOST_CAPTURE (representative raw.json:18239418).
Claim ceiling: NATIVE_INPROCESS_CAPACITY_QUALIFICATION (representative config.json:42–54).

Scope is OLMoE BF16/vLLM0.26/RTX5090, fixed32-request cohort with3072/1024 tokens. Multi-GPU EP, production arrival, cross-model generalization and statistical superiority are outside scope.

## Unverified scope and claim impact

The reviewer did not rehash remote Arrow/model shards through SSH, rerun GPU experiments, or reconstruct full worker-side KV/block contents. It did not independently recompute every diagnostics width bucket from raw; it checked input hash, source/denominator path, result existence and report–JSON agreement.

No corrective action was required for the current MEASUREMENT_ONLY result. These limits prohibit upgrading to a method GO or general performance claim. The final RESULTS_WESTE_ADDENDUM.md is an executor synthesis; the reviewer assessed the primary analysis and diagnostics files listed above.

No .aris trace was created, following repository instructions; this report and its JSON retain the review result.

