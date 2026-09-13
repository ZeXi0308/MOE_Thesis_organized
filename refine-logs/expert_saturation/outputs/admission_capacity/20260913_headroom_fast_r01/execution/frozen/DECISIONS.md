# Same-policy observer optimization: native versus fast headroom

Status PREPARED_UNRUN. Hypothesis: avoid per-step full block-ID conversion for
all running requests while preserving every online leader/held decision and
exact held-ID checks. CPU recorded-state replay proves adapter choice only,
not policy performance or physical GPU KV state.

Same OLMoE BF16 revision, vLLM0.26, one RTX5090, 16089350144-byte KV,
7671 usable blocks, 32 requests, 3072 input/1024 output, 50ms arrivals,
maxbudget1024 and original three warmups. Full-history admission stays enabled.
Native and headroom use the same fast observer setting. Actual decisions only
use current request/allocator state and declared output bounds; no replay or
future trace enters the GPU policy. Each arm owns independent future state.

Order native -> headroom -> headroom -> native; fresh engine/process per cell.
The old checked-headroom campaign is retained as history, not a same-block
performance arm. Compare new fast-headroom against its new native controls.
Normalize source IDs and compare new policy action paths to old checked only
as a separate state/action consistency check, never as an unexecuted timing
counterfactual. Preserve all outputs, failures and warmups; stop on incomplete.

Primary goal unchanged: reduce incumbent pauses while retaining full-cohort
throughput. Report per-request max-ITL distribution, TTFT, mean completion,
wall, held intervals and all failures. Expose shifted waiting. Costs use
exclusive host timing buckets; do not subtract observation time to claim a
speedup. No threshold/leader/workload/metric changes. If runtime cost remains
negative, locate residual rather than re-run for a favorable sample.
