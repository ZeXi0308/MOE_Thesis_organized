# One first-fit upper-bound admission pilot

Registered after the native/bound matched block, before first-fit GPU execution.

**Hypothesis:** Selecting later waiting requests that fit the unused reservation capacity can reduce FCFS head-of-line delay enough to improve complete-episode joint goodput, without causing KV preemption. The previous pilot showed such followers in 2,723/4,921 no-head-admission steps; this was only an opportunity count. The matched block makes bound FIFO a competitive reference: 64/67 joint-qualified and 0.7377/0.7751 requests/s, with about 4% higher mean flow than native.

**Single action:** Scan the currently arrived native waiting list in FCFS order. Select each request whose known `ceil((prompt_tokens + max_tokens)/16)` reservation fits the remaining 4,096-block budget and 32-sequence limit; continue scanning after a non-fitting request. During the synchronous original schedule call expose the selected requests in their original order and limit admission to that selection. Restore all unadmitted requests to their original relative FIFO order before returning. Keep native allocation, completion and release ownership. No future arrival, actual EOS, future output length or post-hoc slack is an online input.

**Fixed run:** One new pilot with the same viewed 128 articles, external arrivals, model revision, greedy EOS/cap-1,024 generation, physical KV budget, 25 CPU threads, batch 1,024, engine maximum 32, no offload and application warmups. Only the measurement adapter changes. Retain every external arrival and drain the entire episode. Use the existing bounded child launcher and shared lock. Fresh root: `/root/autodl-tmp/c-research-20260930/c-native-bound-first-fit-pilot-v1/native_first_fit_1`.

**Measurements:** Actual overtaking admissions, reservations and zero-preemption prediction; all completion/output/finish counts; the existing 20-point frontier and primary 20 s TTFT / 4 s maximum host-return gap goodput; TTFT, gap and flow distributions; per-request improvements and regressions relative to both matched bound runs and native references. Explicitly retain the waiting/flow cost to bypassed heads. Allow zero overtaking as a zero-action result; do not force a favorable action or drop requests.

**Decision:** If no meaningful overtaking occurs or goodput does not improve, report the failure of this formulation without adjusting a threshold or seed. If the pilot appears useful, freeze a new FCFS/first-fit/first-fit/FCFS block before comparison claims. Either outcome remains a simple scheduling reference, not a new algorithm or full WAIT/CacheOPT reproduction. Long-run starvation/fairness, useful generated content and held-out generalization remain unproven.

Sources: `C_NATIVE_BOUND_FIRST_FIT_ADMISSION_V1.py` (CPU-ready at `0809d0b`), derived cell and tiny launcher to be frozen before dispatch. No additional GPU experiment is queued.
