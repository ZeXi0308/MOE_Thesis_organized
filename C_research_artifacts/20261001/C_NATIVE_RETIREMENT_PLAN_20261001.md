# One conditional cap-retirement FIFO pilot

Registered before GPU execution. First-fit failed its fixed primary target (59 qualified, .6738 requests/s versus bound FIFO .7377/.7751), so it receives no threshold tuning or matched repeat. This is a different action: retain FIFO, change the capacity calculation using already observed decode progress.

## Hypothesis and evidence

The full-bound gate assumes all resident requests reach their maximum KV footprint simultaneously. Already-decoding requests can instead reach their known output caps and release in different scheduler calls. On the two existing bound traces, 4,789/4,799 capacity-blocked calls have all residents decoding; a conditional cap-retirement envelope admits the current FIFO head in each of those states (68/67 distinct heads). Median reservation reduction is 356/367 blocks. These correlated observed-state counts are neither replay nor benefit evidence. They justify one native pilot, without changing input, arrival rate, target, cap or seed.

## Action and conditions

Only when every existing resident is a pure decoder (`O>0`, `C=P+O−1`, no in-flight/speculative/placeholder tokens) may a new FIFO prefix be admitted. For each existing decoder let `R=M−O`, where `M` is its known max_tokens. Before calling native schedule, calculate

`E = max_t sum_i ceil((P_i + O_i + t)/16) * 1[t <= R_i]`.

The maximum only needs `t=0` and each `R_i`: occupancy increases between retirements. The cap-reaching allocation remains counted; `P+O` includes a conservative extra token relative to current computed KV. Add the full `ceil((P+M)/16)` bounds of candidate new requests and require the sum ≤4,096. Keep FIFO order, native allocation/free ownership and the 32-sequence limit. While any resident still prefills, admit no new requests. Newly admitted requests keep their full bound until a later all-decoding decision.

The fixed synchronous, single-GPU, non-speculative native scheduler serves existing decoders before appending new prefills. With at most 32 existing one-token decoders and a 1,024-token budget, their progress cannot be displaced by those new prefills. The pinned worker's normal one-token sampling path does not discard pure-decoder outputs. Check the protected prefix actually schedules one token and then advances output count by one or completes through native free. Abort and mark the cell invalid on a violated condition; do not silently claim the envelope held. This is a conditional implementation, not a general guarantee for asynchronous, speculative, paused, multimodal, PP or interleaved-prefill schedulers. EOS is unknown and may only release earlier.

Source basis: pinned `scheduler.py` running loop 469–620, append at 1011, output/stop/free path 1650–1815; `gpu_model_runner.py` non-spec sample/discard path 2059–2081, 2210–2218 and 3707–3730. Local endpoint calculation agrees with exhaustive occupancy over 120 small CPU cases. No new audit framework is required.

## Fixed experiment and decision

Run one cell in `/root/autodl-tmp/c-research-20260930/c-native-retirement-pilot-v1/native_retirement_1`, using the existing shared-lock/child-cleanup launcher. Same 128 viewed articles, external arrivals, OLMoE revision, natural EOS and max1,024 output tokens, 4,096 usable GPU KV blocks, no offload, fixed 25 CPU threads, batch1,024 and prior warmups. Keep all requests through complete drain.

Report actual admissions beyond the full-bound gate, physical/envelope/full-bound peaks separately, preemption and progress violations, all completion/output/finish counts, primary TTFT20s/gap4s goodput, the existing 20-point frontier, TTFT/flow/gap distributions, per-request costs and output differences. Episode time includes the online calculation; decision_seconds measures only the pre-schedule adapter portion, not all instrumentation overhead. Compare with both native and both bound FIFO runs already completed. Any deadline curve is descriptive and does not change the fixed primary.

Zero incremental admissions is a zero-action result. Violated runtime conditions invalidate this formulation's cell. A valid pilot with no service improvement is retained as negative, with no threshold rescue. A useful pilot requires a separately frozen bound/envelope/envelope/bound matched block before a policy comparison claim. Novelty relative to prior resource-aware scheduling and confirmation on unseen data remain separate open questions.
