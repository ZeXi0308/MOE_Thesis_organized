# One fixed repeat after a sign flip

Frozen before any repeat GPU execution on 2026-09-08. The paired-r01 forward and
reverse steady comparisons changed sign; bursty did not materially exceed the best
static point. This invokes the repeat clause of ../20260908_capture_ladder_paired_r01/DECISIONS.md.

Repeat exactly the original execution.tar.gz (SHA256 below), same 32 WikiText rows,
128/128 tokens, arrivals, TTFT200ms/mean-TPOT9ms, source, engine configuration,
common 15 warmups per engine, 16 measured episodes per engine, forward and exact
reverse ordering. No selector, threshold, cap, seed, input, warmup or model change.
Both repeat engines are retained. Original paired-r01 forward remains canonical;
this repeat is additional stability evidence and cannot replace earlier runs.

One additional two-engine campaign only, 32 measured episodes and 30 warmups.
After this repeat, no third campaign on these unchanged inputs is authorized by
this repeat decision. If signs remain inconsistent, conclude no reproducible
request benefit for the tested formulation and localize the earliest differing
observable state from already retained data before proposing any new action.
If both repeats retain improvement versus old feedback but not the strongest
static baseline, report an engineering improvement only. No expert contribution
or method GO can follow from this repeat alone.

Use two fresh directories/processes under /root/autodl-tmp/moe-capture-ladder-paired-repeat-20260908-r01.
Run forward, immediately download and verify all raw including warmup, then reverse.
Retain source/config/commands/env/stdout/stderr/exit status and all failures.
Remote originals must remain until local verification completes.

Execution package SHA256: 8fb235b3502c69d884df13c8ca5127481c7cfba4b516726286ef1a0043defdf6
