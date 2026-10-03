# First dispatch: KV protection and execution-token allocation

The first-output obligation succeeds, but its `-2` execution priority excludes 30 ready resident decodes at steps 406 and 407. A single-step check of the unchanged frozen planner confirms that the actual step-406 prestate admits 994 restore tokens plus those 30 decode tokens without an additional victim and while reserving the complete current restore history. This supports separating execution-token allocation from the KV reservation as the next smallest causal test. It does not establish alternate-run latency or completion benefits.

Both repeats have the same first real action divergence at step 406, after 11,295 returned output tokens. Elapsed time already differs by +0.414703 s and −0.389886 s before this divergence. Earlier timing differences remain separate from the changed action.

| Step-406 resource | Value |
|---|---:|
| Actual free blocks | 148 |
| Restore 3640 current history / already owned | 205 / 63 |
| Remaining complete-history reservation | 142 |
| Other resident requests, each pending one decode token | 30 |
| Their additional blocks, requests 628/2733/3299 | 3 |
| Joint reservation / remaining slack | 145 / 3 |
| One-step scheduling budget | 994 + 30 = 1024 |
| Diagnostic additional victims | 0 |

The CPU check first reconstructs the real plan, obtaining exactly `{3640: 1024}` and no victims. It then calls the same pure frozen planner with a diagnostic per-request chunk cap of 994: every other selected request has pending=1, so this checks the intended split at this one state. Priority, arrival order, full-history block reservation, capacity, and all request states remain the actual recorded inputs. The runtime package and configuration are unchanged. No alternate state is advanced past step 406.

In the real on runs, 406 and 407 each select 1024 tokens for restore 3640 and zero of the 30 ready residents. Step 408 selects the final 232 restore tokens and all 30 resident decode tokens. Those residents' common next-output gap is 85.780 ms / 73.575 ms on, versus 18.488 ms / 18.521 ms off. These are observed outcomes of separate runs, not predicted milliseconds saved by the one-step diagnostic.

The first returned new token for restore 3640 is output 203 at step 408; the next schedule releases its obligation at 409. It receives no further service at 409–410. At step 411, free blocks are zero and the selected 30 resident decodes require three additional blocks; the actual victim is the now-unprotected 3640, releasing 205 blocks, leaving 202 after allocation. This one-output re-preemption is distinct from the earlier two calls of execution-token exclusion. The total +21,973 repeated positions across the complete on trajectory cannot be assigned entirely to either event.

The broader request-level findings remain in `REPORT.md` and `analysis.json`: 27/27 on obligations release after real new output, zero interrupted obligations, four one-output re-preemption residencies, and 27/32 requests with worse maximum ITL in each pair. The longest request gap falls in both repeats, while throughput changes sign. Those facts establish delivery and cost/delay redistribution, not net efficiency or quality.

Next smallest test: preserve the existing KV reservation and first-output endpoint, reserve one scheduling token for each following ready resident with pending=1, and give the restore the remaining positive token budget. Keep the original 200/10 counters. This diagnostic provides a directly feasible first action for that test; it does not authorize conclusions about its unexecuted continuation or solve the separate post-release amortization question.

Reproduce: `python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_completion_r01/analysis/localization/first_dispatch.py`. Full source hashes, token maps, actual 405–411 events and each resident's aligned next-output gap are retained in `first_dispatch.json`. Only derived localization files were written; no raw, frozen package, original analyzer, primary analysis or GPU state changed.
