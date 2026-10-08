# Context calibration: recovery lifecycle

**MEASUREMENT_ONLY / REANALYSIS_OF_NATIVE_INPROCESS_SERVING.** In both repeats, native and most-output still avoid resumed intervals invalidated with zero or only 1–2 new outputs. Least-progress also has neither case. Thus this six-cell calibration does not reopen that specific short-service residual or support adding a longer protection window.

All six raw captures and runner statuses are COMPLETE: every cell completes all 32 requests and returns 32768 tokens. The unchanged MAIN `analyze_effective_recovery_service.analyze` accepts every cell; no alignment failure occurred. `lifecycle.json` retains every residency, all six raw paths and SHA256 values, configs, reused-helper hash, runtime source hashes and source-workload hash. Raw files and shared sources were not modified.

| Arm / block | Requests complete | Resumed intervals | Served to completion | Output then discarded | Zero-output discard | 1–2-output discard | Recomputed positions | Recovery-prefix positions reexecuted next interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native / 0 | 32 | 5 | 5 | 0 | 0 | 0 | 17820 | 0 |
| most_output / 0 | 32 | 26 | 18 | 8 | 0 | 0 | 93044 | 28151 |
| least_progress / 0 | 32 | 24 | 6 | 18 | 0 | 0 | 83510 | 62272 |
| least_progress / 1 | 32 | 24 | 6 | 18 | 0 | 0 | 83517 | 62278 |
| most_output / 1 | 32 | 26 | 18 | 8 | 0 | 0 | 93044 | 28151 |
| native / 1 | 32 | 5 | 5 | 0 | 0 | 0 | 17820 | 0 |

Every preempted interval eventually resumes in these cells; none is left unresumed or censored. All resumed intervals retain a partial restored prefix across later execution calls, so partial recovery has actual internal reuse.

Native's five recovered requests produce respectively 562, 445, 338, 205 and 81 further outputs and finish without another invalidation. Most-output's eight closed intervals produce 954 outputs in total in each repeat, distributed as `{11:2, 15:1, 57:1, 154:1, 194:1, 238:1, 274:1}`. Their 28151 recomputed prefix positions are later actually reexecuted. Least-progress's 18 closed intervals produce 603 / 602 outputs, with ranges 5–59 / 5–58; their recovery-prefix reexecution totals are 62272 / 62278.

These counts show different amounts of useful service before discarding state. They do not establish whether any interval economically amortized recovery: recovery call duration includes shared execution and host/scheduler work, and the repeated-prefix count overlaps the recomputation accounting. Neither quantity is a predicted saving in milliseconds. The parent analysis evaluates full-request latency and efficiency; this artifact makes no throughput or method-GO claim.

Scope: the same 32 cohort3 documents with alternating prefix lengths 2560 / 3072, fixed 1024 output tokens, 6656 usable GPU KV blocks, APC off and native recompute. This is reused-document calibration, with different dynamic KV pressure from the homogeneous case; it is not an independent cohort or an unknown-EOS result.

The owner's contemporaneous `analysis_r01.json` rejects native cells for a missing `forced_preempted` field. Native uses the existing completion-headroom recorder, which legitimately lacks that rotation-only field. This is a separate analyzer-schema issue, not a raw execution or lifecycle failure. This artifact neither edits that analysis nor uses its invalid labels.

Reproduce the core accounting with the unchanged helper, supplying a new output directory:

```sh
python3 '/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/experiments/admission_capacity/analyze_effective_recovery_service.py' \
  --workspace /private/tmp/moe-a-recovery-components-20260914 \
  --campaigns 20260914_context_victim_calibration_r01 \
  --expected-cells 6 --output-dir PATH_TO_NEW_DIRECTORY
```

The next decision remains the parent full-request tradeoff interpretation. A new protection action requires an uncovered request-level residual after native/most; absence of a 0–2-output failure in these six cells is not a global recovery-scheduling result.
