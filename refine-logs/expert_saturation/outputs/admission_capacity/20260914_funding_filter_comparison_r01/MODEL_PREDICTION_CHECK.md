# Frozen least-feasible prediction check

**Verdict: PARTIAL DIRECT CONFIRMATION.** The pre-GPU model_check_r02.json correctly qualified the funding-filter action and its complete structural trajectory for the start state that actually occurred. Both new least_feasible cells map exactly to the old block0 model-visible start and match its frozen prediction. The distinct old block1 start did not recur, so its 86,779 / step 1,447 branch was not directly tested.

The prediction was registered in RESULT_LEDGER.md:316 while this group was still STAGED/GPU_UNRUN. Its SHA256 is 5e5a235a57b25d79eadf8d84cc063343da453e0e2652072a08a9dbdf09b81c6f. The recovered original six-cell archive is 7ad278ed5df66db7729338f534cd522c60f76958800ec60fc9ec52b5cf3a8329. No model coefficient, threshold, or transition was changed, and no observed future was fed back into the predictor.

## Realized start and prediction

The qualifier is the first active all-32 pure-decode, no-wait, pre-forced before-state defined by validate_context_progress_model.py:41-71. I aligned per-run internal IDs through raw.internal_to_source and compared stable request/document identity, prompt hash, arrival, prompt/output/computed counters, allocated blocks, request status/preemptions, running order, waiting set, free blocks, and pre-cutoff event history.

Both funding-block0-least_feasible and funding-block1-least_feasible equal context-block0-least_progress on every model-visible field at step 90: 32 running pure-decode requests, no waiting/history, and 922 free blocks. All 32 ephemeral internal suffixes differ, but stable source identities and their lexicographic order are preserved.

| Metric | Frozen block0 prediction | New block0 | New block1 |
|---|---|---:|---:|
| first changed action | step 891 | step 891 | step 891 |
| target / victim | 0020902 / 0020484 | same | same |
| free + released / required blocks | 31 + 211 / 234 | same | same |
| target first new output | step 894 | step 894 | step 894 |
| target completion | step 1356 | step 1356 | step 1356 |
| recomputed positions | 89,761 | 89,761 | 89,761 |
| final scheduler step | 1,451 | 1,451 | 1,451 |
| forced rotations; preempt/resume | 21; 26/26 | 21; 26/26 | 21; 26/26 |

The existing source-owner model_check_r01.json (SHA256 5ffcbe38243f9ceef88577a0f99156d9227d5a53b860dca2a4cdd8ff59b906e7) compares each new cell's frozen-model trajectory with its actual scheduler/output/completion events. It reports 1,362 matched calls per filtered cell, first_mismatch=null, and equal completion maps. Because both actual inputs equal the pre-GPU block0 input, there is no first future structural difference through terminal step 1,451. This addendum did not repeat that full replay.

## Why old block1 is not a second prediction test

The old block1 qualified state already differs at step 90: 921 free blocks instead of 922; 15 requests are each one output/computed token ahead; request memory-train-article-0018032 owns 165 rather than 164 blocks. Stable document identities, prompt hashes, arrivals, running order, waiting set, and history match. This is an input-state difference, so its frozen prediction—86,779 recomputed positions, final step 1,447, and 25 preemptions/resumes—cannot be substituted for either new cell. The new block1 observation matching the old block0 branch is not a miss of the old block1 model.

This closes the action question at the structural level: the frozen CPU probe correctly predicted the realized target/victim action, first output, recomputation, and terminal step for the state that occurred, twice. It supported the pre-run experiment decision; the runtime controller used current state and did not consume this offline prediction. It does not validate wall time, latency ranking, generated-token identity, KV tensor equality, an Oracle, or a method win.
