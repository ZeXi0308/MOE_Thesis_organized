## Gate 0 Producer Integrity Audit

Overall verdict: **FAIL**  
Audit state: **INVALID/BLOCKED — concurrent mutation**

The last fully reviewed snapshot changed again before certification:

- Fully reviewed: producer `d631b650…`, test `968d2b2f…`
- Current at final check: producer `7c09ef67…`, test `58142572…`
- Config remained `7c66a63b…`

Therefore the `67/67 PASS` run applies only to the earlier snapshot and cannot certify the current files.

### A. Ground Truth Provenance — PASS

The producer derives routes from native model `router_logits`, then recomputes softmax/top-k rather than using dataset labels ([capture_continuous_decode.py:465](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:465>)). Its reference is a same-model serial cached-decode rerun, explicitly labeled `scientific_ground_truth=false` ([capture_continuous_decode.py:861](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:861>), [line 954](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:954>)).

Classification: `self_supervised_proxy / same-model engineering equivalence`, not `real_gt`. No self-generated target is presented as dataset ground truth.

### B. Score Normalization — PASS

No Gate-0 metric is divided by prediction maxima, means, or other prediction statistics. Token/route fractions become `1.0` only after fail-fast exact comparisons; raw audited request/step counts accompany them ([capture_continuous_decode.py:877](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:877>), [line 948](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:948>)).

The relative-error metrics in `capture_moe.py` are unrelated to this producer path; the legacy capture disables those diagnostics ([capture_native_routes.py:394](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_native_routes.py:394>)).

### C. Result File Existence — FAIL

No Gate-0 formal `results/`, `outputs/`, `RUN_STATUS.json`, or `CAPTURE_COMPLETE.json` exists. The README requires both completion files and still fixes all eligibility flags false ([README.md:91](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/README.md:91>)).

The machine preregistration has:

- unresolved arrival-trace and dataset revisions;
- `formal_execution_authorized=false`;
- five explicit formal blockers.

Evidence: [gate0_continuous_decode_v1.json:9](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json:9>), [line 44](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json:44>).

The tracker also links `gate0_audit_2026-08-02.md`, but that file does not exist ([current README:64](</Users/leandrozhao/Desktop/毕设论文资料/docs/current/README.md:64>)). The originally listed Markdown preregistration is also absent; the README now identifies the JSON config and experiment card as replacements.

### D. Dead Code / Exercised Path — WARN

The transient snapshot passed 67 unit tests. Tests exercise:

- mutable batch sizes 1/2/3 and serial route/token parity;
- cache stacking/splitting indirectly;
- identity conservation;
- manifest rejection;
- unauthorized and non-canonical formal preregistration;
- output-class isolation.

Evidence: [test_continuous_decode.py:61](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:61>), [line 148](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:148>), [line 187](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:187>).

Not exercised end-to-end:

- formal CUDA path;
- real-model native-route closure;
- CLI creation of partial/complete bundles;
- crash-between-status-and-sentinel behavior;
- CUDA synchronization;
- EOS occurring exactly on the final allowed step.

### E. Scope — WARN

Executed scope is one randomly initialized two-layer tiny OLMoE fixture, three requests, three decode steps, one seed ([test_continuous_decode.py:25](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:25>), [line 61](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:61>)).

Preregistered formal scope is two pretrained models, 128 requests per cell, max 16 steps, BF16, CUDA ([config:5](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json:5>), [line 21](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json:21>)). None ran.

The seed is bound and initialized, but under frozen prompts plus greedy decode it is provenance, not evidence from repeated independent runs.

### F. Evaluation Classification — PASS

- Candidate development test: `self_supervised_proxy / model-derived engineering equivalence`
- Legacy smoke fixtures: `simulation_only`
- Future natural pretrained capture: natural-input/model-output provenance, still not dataset ground truth
- Human evaluation: none
- Real dataset ground truth: none

## Blocking Issues

1. **Formal execution is not authorized.**
2. **No formal manifests, CUDA cells, completion sentinels, or results exist.**
3. **The code changed after testing and after line-by-line review; current hashes are uncertified.**
4. In the last fully reviewed snapshot, serial parity compared token IDs and expert identities only—not gate weights, full logits, or cache tensors ([capture_continuous_decode.py:931](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:931>)). This does not establish the current authority document’s `exact-output replay` requirement ([current README:99](</Users/leandrozhao/Desktop/毕设论文资料/docs/current/README.md:99>)).
5. EOS loses to `max_decode_steps` when both occur on the same final step, misclassifying terminal accounting ([capture_continuous_decode.py:828](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:828>)).
6. The completion summary omits explicit EOS/max-step reason counts and router-invocation denominators; these are only recoverable from lower-level ledgers.

## Non-blocking Strengths

- Native router provenance rather than patched-route output.
- Prefill routes are not emitted as decode.
- Request/decode-step/token/layer/top-k/expert fields are explicit.
- CUDA calls are synchronized around timing.
- Arrival trace and prompt bytes are hash-checked.
- Canonical preregistration, clean Git state, new output directory, formal/development isolation, and fail-closed flags are implemented.
- Partial runs begin as `INCOMPLETE`; the completion sentinel is written last.

## Claim Impact

- “Candidate producer implementation exists”: **supported for the reviewed snapshot**.
- “Tiny random CPU mechanics pass”: **supported only for the transient tested snapshot**.
- “Gate 0-A passed”: **unsupported**.
- “Formal natural pretrained producer qualified”: **unsupported**.
- “Exact-output replay established”: **unsupported**.
- “Gate 1, BCRD/DEPA result, TPOT/P99, EP, or system claim”: **unauthorized**.

**Formal CUDA run authorized? No.** The only authorized next action is to stabilize/commit the producer, freeze the unresolved workload inputs, rerun tests on that exact commit, independently audit it, and only then consider changing the preregistration authorization.
