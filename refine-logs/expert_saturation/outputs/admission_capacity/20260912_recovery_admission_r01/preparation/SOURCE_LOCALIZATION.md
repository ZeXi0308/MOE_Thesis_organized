# Fixed-KV native recovery source localization

Evidence: `NATIVE_SERVING` raw plus exact installed vLLM 0.26.0 source. No runtime intervention or performance counterfactual executed by this analysis.

The installed files in `/private/tmp/moe-native-v026-recovery-source/` have the same SHA256 as **all four** `20260912_kv_budget_r01/gpu_results/*/environment.json` measurements:

- `scheduler.py`: `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`
- `kv_cache_manager.py`: `3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf`
- `block_pool.py`: `202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661`

## Why 237 freed blocks do not permit immediate recovery

Actual budget90 is 7,671 usable blocks, block size 16, one full-attention cache group, no prefix caching, speculative tokens, or external KV transfer. The request's history includes generated tokens already returned to the user.

Installed source pointers:

- `scheduler.py:562`: RUNNING requests allocate first; on failure FCFS takes `running.pop()` at line 601, then preempts.
- `scheduler.py:666`: WAITING is considered only when no preemption occurred in this step. A preemption therefore creates at least a one-step recovery delay.
- `scheduler.py:841`: waiting/recovery token demand is `request.num_tokens - num_computed_tokens`, including prior outputs.
- `scheduler.py:935`: WAITING allocation passes `full_sequence_must_fit=self.scheduler_reserve_full_isl` at line 944; failed head allocation breaks the waiting loop at line 956.
- `scheduler.py:1212`: preemption frees KV, resets computed count to zero, preserves generated output, and prepends the victim to WAITING at line 1233. A later preemption moves ahead of an older recovery victim.
- `kv_cache_manager.py:411`: before allocating the requested chunk, the full-sequence gate asks whether all current `request.num_tokens` fit. The comparison at lines 425–427 may reject a request whose next chunk fits.
- `kv_cache_manager.py:429`: chunk-only allocation still enforces actual free-block capacity at lines 462–466. Disabling the full-sequence gate does not allow illegal memory allocation.

In repeat0, victim 0003640 is preempted at step 806 after 3,780 computed tokens and 709 outputs. Its 237 blocks are freed, but the remainder of the existing decode pass leaves 236. At step 807, the next existing decode pass leaves 234. The requested recovery chunk is 993 tokens = 63 blocks, but the full history is 3,781 tokens = 237 blocks; admission fails. Free space then decreases as the remaining decodes grow.

At step 928, victim 0003571 frees 245 blocks and becomes the new queue head. At steps 929–932, head-allocation free space is 243/242/238/237: too small for its 3,906-token history (245 blocks), although the older victim's 237-block history would fit. The first completed request frees 256 blocks after step 1025; 0003571 then recomputes at steps 1026–1029. Another completion admits 0003640 at steps 1030–1033.

For **each** of the two low-budget repeats, all 219 failed WAITING allocations divide into:

- **187** next chunks fit while full histories do not;
- **32** even the next chunk does not fit.

There are also two failed RUNNING allocations, which trigger the two preemptions. Counts are derived from each failed allocation's actual free blocks and requested tokens, not schedule-start free blocks. Current full-attention/no-sharing/no-lookahead assumptions make the block arithmetic exact. Historical telemetry omitted the gate flag and watermark, so the inferred full-gate reason should be captured explicitly in the next run.

Repeat0 maximum token pause remains 4.590643 s = 4.473617 s before the first recompute call + 0.117025 s across four recompute calls. The latter contains mixed-batch service and host overhead; it is not isolated GPU recompute cost.

## One next causal experiment: native full-history reservation off

Use the existing native implementation with `scheduler_reserve_full_isl=False` against `True`, at the **same actual 7,671 usable blocks**, cap32, 1,024-token engine budget, exact OLMoE revision/BF16, prompt/output lengths, arrivals, warmup, and capture. Fresh engine per arm; preserve every result. This native configuration baseline precedes any custom controller. Print the resolved configuration and scheduler attribute; keep runtime/model source hashes fixed.

Question: does whole-history admission conservatism create an avoidable pause, or does releasing it expose insufficient future growth and repeated recompute? The first incomplete recovery chunk is a useful source-level discriminator, not a reason to reject the entire research problem.

Both arms preserve legal allocation and ordinary decode priority. A plausible failure mode is that the resumed victim is appended to the running tail, then cannot allocate its last recovery chunk and is preempted again before a new token is returned. This is a hypothesis, not a measured outcome. Smaller recovery chunks alone do not remove the `True` arm's full-history gate.

Small observation addition at the existing `manager.allocate_slots` wrapper, for waiting/preempted calls only:

1. Capture before original call: request id/status, `num_tokens`, `num_computed_tokens`, allocated blocks, requested chunk tokens, actual free blocks, `full_sequence_must_fit`, watermark blocks, reserved blocks, lookahead, and relevant cache/transfer flags.
2. Under the qualified single full-attention/no-prefix/no-lookahead regime, compute pure arithmetic `chunk_needed=ceil((computed+requested)/16)-allocated`, `full_needed=ceil(num_tokens/16)-allocated`, then count chunk-fit/full-not-fit, neither-fit, and both-fit. Store the small event/counters and actual allocation success. Do not call `allocate_slots` a second time to probe a hypothetical action: it mutates KV state.
3. Existing preemption/computed-interval/output ledger identifies resumed execution followed by re-preemption before any new returned token, failed recovery count, and recompute tokens. Retain these even if run-limit or watchdog fires. No new full route capture is required.

Compare all-request completion/wall, mean completion, TTFT, per-request max ITL, and total recomputed tokens. Earlier first recompute or fewer allocation failures alone is not success; the mechanism must reduce complete-request harm without merely moving it to other requests.

## What is unavoidable versus not yet shown unavoidable

At this resource level, future full KV for all 32 requests cannot fit simultaneously (32 * 256 = 8,192 > 7,671). Some execution must finish/release, wait, or discard and recompute history. Actual destroyed KV requires real recomputation; reducing an admission check does not remove that physical cost. The particular 4.47 s recovery wait, victim concentration, and repeated-head rejection are policy-dependent and are **not** a proven lower bound. The full-history-off A/B directly tests the cheapest available adjustment before considering retained-KV stalls, recovery reordering, or completion headroom.
