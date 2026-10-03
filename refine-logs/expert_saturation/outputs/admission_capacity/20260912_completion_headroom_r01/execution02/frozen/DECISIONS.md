# Same-KV retained completion headroom probe

Status: PREPARED_UNRUN. No performance or quality result exists.
Question: can retaining KV while protecting one request's completion headroom
reduce incumbent max-ITL without losing complete-cohort throughput?

OLMoE pinned BF16/vLLM0.26, RTX5090, 16089350144-byte KV, 7671 usable blocks,
32 requests, 3072 input/1024 output, 50ms arrivals, budget1024. Full-history
reservation stays enabled in both arms. Same three warmups, fresh engine per cell.
Order: native -> headroom -> headroom -> native. All attempts and failures retained.

Activate only once the 32-request closed cohort is pure synchronous decode.
Use current progress, allocated/free blocks, and declared output upper bounds.
Before pressure preserve the native schedule. At pressure lock the request
with the smallest remaining declared output bound (FCFS tie); reserve its full
remaining KV growth. Other requests advance only if their next blocks leave
that reservation intact; zero-block progress is allowed. Held requests retain
RUNNING status, computed tokens and KV ownership. No WAITING admissions after
activation. Do not share policy future routes, KV, outputs or completion states.

Main metrics: whole-cohort throughput and per-request max-ITL. Also report TTFT,
mean completion, total wall, every held interval, destroyed/recomputed tokens,
all failures and scheduler/capture overhead. Do not replace these by a favorable
reference SLO. Fewer preemptions alone is not success. Shifted waiting is charged.
Stop on an incomplete/invalid cell, failed pool/config identity or unavailable
headroom. A negative performance result changes this mechanism, not the problem.
GPU correctness, output quality and method novelty remain unverified.
