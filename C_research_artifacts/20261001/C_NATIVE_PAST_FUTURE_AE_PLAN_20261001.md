# Author-AE Past-Future core: one native development pilot

## Why this next measurement

The direct prior already covers the resident completion-peak formula; independent formula novelty is withdrawn regardless of further performance. The complete fixed lower-rate control found no capacity-driven action or material service benefit. The next useful native asset is the closest prior's causal history/conditional-sampling/peak admission core, with native preemptions and complete-request accounting retained. This is a baseline implementation pilot, not a new C candidate or a performance rescue.

Select the author's AE code at commit `d93ff69c07d4097b0a6a4ee67ea355e8780e3095`, explicitly distinguish it from the paper: history40, startup20 copies of1024, minimumsamples200, maximumlists5, reserve0.05, resident conditional interpolation, waiting history choice, maximum sampled peak, strict `<` capacity check, at most one new request per pass. Dedicated Python/NumPy RNG seed20261001; no model RNG mutation. Each actual completed measured request appends its output length once; no previous runs of this cohort seed history. Keep the source's interpolation rounding behavior, including a sample equal to already generated count.

## Explicit changed execution domain

The existing batch1024 runtime can split a prompt across steps; raw vLLM partial-prefill state must not be passed as an already-decoding LightLLM NormalReq. This pilot uses **max_num_batched_tokens=4096**, explicitlong_prefill_token_threshold=0, with32sequence slots,4096usable16-token KVblocks and the same modelmax4096. Candidate admission additionally requires all running requests to be puredecoders and their one-token needs plus the head's complete outstanding prefill/recompute tokens to fit the4096 joint budget. Native gets at most that single waiting head. Thus a newly admitted prompt completes its prefill in that iteration; a preempted request can wait until its complete recomputation fits. This compatibility guard is specific to a co-batched vLLM port; it is not the original separate-prefill LightLLM engine.

AE WAIT_IN_QUEUE and PAUSED_AND_OFFLOAD conversion must preserve the source's distinct one-token endpoint offsets. Native allocation, victim choice, freeing and recomputation remain operational; preemptions are measured outcomes, not qualification failures. A partial-prefill/progress conversion violation fails the pilot and preserves output without using it as performance evidence. Physical paged allocation remains native; the author peak score is token-level with its5% reserve, not a new hard safety guarantee.

Keep now-viewed article ranks832–959 and original fixed Poisson arrivals(scale1), naturalEOS/cap1024, greedy modelseed20260905, private runtime,25CPUthreads,nooffload/APC/spec/async and original warmups. Init/warmup excluded;128requests and full drain included,300smeasurement/900sownedchild limits. No arrival/seed/SLO tuning. Keep20/4primary andall20frontier points.

## One bounded execution and decision

One native Past-Future AE core pilot under the existing GPU UUID/shared inode lock. Source/plan/hash freeze before upload; nonblocking busy means GPU_DEFERRED with no child. No repeated polling or secondqueuedjob. Release before local analysis.

First test: the causal history updates, conditional samples, actual admissions/resumptions, allocator preemptions and completion lifecycle execute through128/128drain. Preserve source-input tuples and admission decisions in the native trace for later action diagnosis. Native contract violation stops the test; performance-negative or preempting outcomes do not authorize tuning the reserve/window/seed.

**No matched performance conclusion from this pilot.** Oldbatch1024 results are not matched references. After a valid pilot, obtain a separately frozen same-batch4096 native/Past-Future comparison (and only add a cap-envelope reference if it answers a remaining question). The port is not a full LightLLM system reproduction or the paper's1000history configuration. It supplies the direct prior core on the current native backend and makes its compatibility choices reviewable.

GPU `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`; lock `/root/autodl-tmp/moe-research-gpu.lock`, inode2304:29005388732. Output root `c-native-past-future-ae-pilot-v1`. User machine-hour authorization is shared across lines. Existing C data retained; no publication/push.
