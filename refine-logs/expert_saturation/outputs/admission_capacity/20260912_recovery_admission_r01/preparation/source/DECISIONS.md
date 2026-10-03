# Same-pool native recovery-admission comparison

2026-09-12. Question: does the existing full-history reservation guard cause avoidable recovery wait, or does relaxing it cause thrashing?
Fixed physical KV 16089350144 bytes, 7671 usable16-token blocks; cap32, OLMoE pinned BF16,3072in/1024out,50ms arrival,token1024. Same original3warmups.
Order full→chunk→chunk→full. Only scheduler_reserve_full_isl differs. Max_seconds120 unchanged; retain all failed/unfinished requests. Stop on incomplete cell, never overwrite/relaunch.
Primary exploratory objective: reduce existing request maxITL while retaining throughput; show TTFT, mean completion and all shifted waits. No posthoc SLO selection.
Hypothesis support: earlier recovery with fewer long pauses and no throughput loss. Refutation of this config: earlier partial recovery re-preempts before new token, raising wasted work or shifting pauses. Either outcome stays within the same fixed-resource service-progress research question.
This native configuration baseline is not a new method. No expert paging/quality/production claim.
