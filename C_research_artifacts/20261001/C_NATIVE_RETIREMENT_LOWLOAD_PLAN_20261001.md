# Fixed lower-arrival-rate control

## Question and prediction

The unchanged retirement policy improved the frozen fresh-cohort20/4 goodput against both FIFO references, with target-sensitive native tradeoffs. The cap-donor comparator then executed22 real transfers but had lower primary goodput than all four FIFO/retirement references. Preserve that negative result without parameter rescue.

The next question is whether the retirement controller adds cost when offered load is lower. Freeze **arrival_scale=5.0** on the existing Poisson trace: nominal rate1 request/s, actual arrival span113.411645s. Keep the same128 articles, order, model, generation seed/naturalEOS/max1024,32sequence slots,4096usableKVblocks,1024batch,25CPUthreads,nooffload, policies and warmups. This is an intended low-load control, not a claim that the trace is known to be pressure-free. Do not redraw arrivals or adjust the scale if bursts still cause pressure.

Prediction: native preemptions and retirement admissions beyond full-bound should become absent or much less frequent. If no capacity-driven action remains, large systematic service gains are not expected; complete-request differences expose overhead, the policy's prefill admission restriction, output variation and measurement noise. Unexpected pressure, cost, or benefit is retained and explained from the trace, not removed.

## One comparison block

Run **native1 / retirement1 / retirement2 / native2** on the same replacement GPU under the existing shared nonblocking lock. All128 arrivals and complete drain stay in each denominator. The measurement cutoff remains300s and child timeout900s; the fixed slowed trace fits the current cutoff. No other GPU task is queued.

The unchanged fresh-input contract first validates the original files. The measurement uses the existing capture's `arrival_scale=5.0`; output checks receive a separate copy of the source arrival list multiplied by the same factor, without extra rounding. Record base input hashes, the explicit transform and the effective arrival hash/config. Never rewrite the original fresh input or call this now-viewed cohort a new holdout. Warmups retain scale1.0.

## Outcomes and decision

Keep the primary20/4 experimental target and all20existing frontier points. Report both paired directions, both same-arm repeats, native preemptions, retirement incremental admissions and holds, physical/conditional peaks, decision cost, full output counts/IDs/lengths, finish reasons, TTFT/gap/flow distributions and individual regressions. Include all runs even if the expected absence of pressure does not occur. This is a diagnostic control of operating scope, not a new candidate selected for this trace.

If the two arms remain close with no capacity-driven action, retain this as a limited negative control. If the controller adds cost, report the cost and its cause. If substantial pressure persists, report the actual lower-rate regime; do not change scale to force a clean negative control. A capacity/progress/completion error stops the block and is preserved as invalid execution. No deadline/seed/scale tuning after results.

Device `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`, existing lock `/root/autodl-tmp/moe-research-gpu.lock` inode2304:29005388732. Source/plan frozen before execution; busy lock means no child. One bounded block under the shared user-authorized machine budget, followed by release and local analysis. No publication or claim of full CacheOPT reproduction.
