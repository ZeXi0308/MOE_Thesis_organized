# Native APC allocation qualification on a long shared-prefix BBH task

## One question

Does the existing native prefix cache remove the apparent capacity pressure from summing per-request prompt lengths for BBH geometric_shapes? This is a strongest-simple-baseline allocation check, not a new controller, speed benchmark, or confirmation cohort.

The prior27-task qualification is complete:6/27 correct,23 explicit blank-line stops,4 caps,0 EOS. Full-corpus geometry then showed the first32 geometric_shapes known-cap reservations sum4993blocks, above4096, while the task-wide common prefix contains115 complete16-token blocks. This task is selected for that structural long-prefix case, not for a favorable model answer. Every first32 official example is retained. Its index0 was already included in the27-task development qualification.

## Fixed native pair

- Same pinned BBH source/model/tokenizer/runtime and official3-shot base prompt. First32 geometric_shapes examples in original order; all arrive at0. Greedy512 cap, blank-line stop, EOS enabled. No altered prompt, repeated sample, output filtering or parser rescue.
- Native32seq, batch1024, context4096,4096usable KV blocks; no offload or custom admission gate.
- Fixed adjacent order **APC off, APC on**, one run per arm. The only inference configuration change is enable_prefix_caching. The two process runs initialize independently. Existing shared nonblocking lock covers both cells through exit/release; one bounded pair, no retry queue.
- Application warmup is separate. The on arm resets its prefix cache after warmup and verifies an empty active-request state before measurement. The off arm has no prefix cache. The measured cohort starts without a warmup-seeded prefix advantage.
- Record all32 outputs/golds/finish and stop reasons per arm, actual preemptions, prefix lookup hits, and block references on completed native scheduler calls. Count unique physical block IDs once; distinguish inactive cached/reclaimable blocks from blocks referenced by active requests. Preserve shared-reference observations and all failures.
- The preallocated KV tensor pool has the same byte size in both arms. Unique active blocks measure consumption of that pool's capacity; a lower active-block count is not a reduction in the preallocated GPU tensor bytes. Per-step census and allocation hooks are enabled in both arms and included in descriptive host time.

## Decision and limits

Evidence of actual shared live blocks plus substantially lower unique active occupancy would support the expected native APC explanation. Merely counting common prompt tokens is insufficient. Zero hits/sharing or unchanged pressure would refute this explanation for the fixed cold simultaneous batch and motivate inspecting allocation timing, not immediately constructing a new controller. Any output changes or incorrect answers remain visible.

Both arms are development diagnostics; no inferential confidence interval, equal-work speedup, production arrival model or independent algorithm novelty. Per-step allocation observation and any JIT are exposed costs, and host timing is descriptive only. Scores use the already-fixed extraction/normalization; all32 remain in each denominator. This is not the complete BBH benchmark.

## Source grounding

Pinned [vLLM0.26 KVCacheManager](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py#L189-L221) finds computed prefixes only when caching is enabled. Allocation accounts for cache hits, and [BlockPool touch/free](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/block_pool.py#L647-L681) increments/decrements references to shared physical blocks. A prefix must actually be computed and discoverable when another request is admitted; simultaneous arrival alone does not guarantee the ideal sharing bound.

The [Past-Future author queue](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/req_queue.py#L887-L943) accumulates per-request tuple lengths and subtracts a separate prompt-cache occupancy item. That inspected queue does not itself deduplicate live shared-prefix lengths. This is a code-path observation, not a claim that no prior system handles the coupling. C's previous port explicitly disabled APC, so the old measurements are not retroactively relabeled as APC comparisons.
