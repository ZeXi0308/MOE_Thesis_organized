# C sustained output token-pattern diagnostic

Post-hoc development diagnostic, computed from the eight already verified original raws. This does not alter the frozen native recomputation pilot or remove requests.

For each completed output, find the longest contiguous span with period 1–16 token IDs and at least two repetitions. The 256/512 thresholds were chosen for this diagnostic after viewing one cell; they are not confirmatory quality metrics.

| Cell | ≥256-token repeat | ≥512-token repeat | Entire output is one repeated token ID |
|---|---:|---:|---:|
| selected_observe_1 | 45 | 43 | 11 |
| native_full_1 | 46 | 46 | 11 |
| static64_1 | 48 | 48 | 10 |
| guarded_static64_1 | 43 | 42 | 11 |
| guarded_static64_2 | 47 | 45 | 10 |
| static64_2 | 43 | 43 | 11 |
| native_full_2 | 45 | 45 | 11 |
| selected_observe_2 | 47 | 47 | 11 |

All cells completed 128 requests. Across 1,024 executions, 359 have a short-period contiguous repetition of at least 512 tokens; each cell has 42–48 such executions. The 86 entire-single-token outputs all have 1,024 tokens. This strengthens the workload-validity limitation already suggested by 971/1,024 length-cap terminations. Real source prompts and permission to terminate at EOS do not establish representative generated content.

This is an exact token-pattern observation, without decoding, semantic-quality labels or an independent cohort. The serving/cost measurements remain measurements of this specific development workload. They cannot support useful-output goodput claims. No request is excluded from existing or future metric denominators; the native pilot remains a missing-reference check on the same workload.

[Analyzer](C_SUSTAINED_TOKEN_PATTERN_DIAGNOSTIC.py); [per-request JSON](sustained_token_pattern_diagnostic_v1.json). Source raws are pinned through the existing eight-cell audit. Tiny algorithm checks cover a repeated token, a period-two sequence and a nonrepeating sequence.
