# Native / static64 matched development result

All four registered cells completed 128/128 requests in native–static–static–native order. Each child exited normally and released its GPU state.

| Cell | Joint 20/4 requests | Goodput (req/s) | Episode (s) | Output tokens | Preemptions |
|---|---:|---:|---:|---:|---:|
| native_recompute_1 | 58 | 0.7244 | 80.068 | 125304 | 38 |
| static64_1 | 62 | 0.6477 | 95.729 | 125150 | 306 |
| static64_2 | 63 | 0.6667 | 94.493 | 126105 | 307 |
| native_recompute_2 | 60 | 0.7471 | 80.308 | 124329 | 41 |

## What changed

The matched block reverses the impression from the earlier one-cell pilot: native recomputation has higher observed primary goodput in both paired directions. Static64 qualifies 3–4 more requests, but its episode is about 15 seconds longer. Native is now the strongest measured simple reference for this primary objective on this development workload. The initial native pilot (45 joint, 0.5464 req/s) remains in the record; its lower goodput was not reproduced.

Static64 keeps all 128 requests below the 4-second observed-gap limit; native has 117/128 below it in both runs. This protection has a broad cost: static worsens maximum gap for 113/128 requests in each pair, and mean completion flow is 22.8–23.3% higher. Native maximum-gap medians are about 0.037 seconds, versus 1.43–1.80 seconds for static; native tail maxima are 11.22–12.64 seconds, versus 2.96–3.50 seconds for static.

## Remaining question

Each native run has 11 gap failures, with eight request IDs shared across repeats. Every longest failing gap brackets one native preemption. Five failures per run meet the TTFT target, leaving a narrow conditional opportunity to improve continuity without losing native throughput. Fixing only these gaps at unchanged TTFT and episode time would add five qualifying requests; this is a conditional accounting observation, not an achievable policy bound.

The next minimal test is a stronger simple baseline on the native no-offload path: reserve each admitted request’s known prompt-plus-max-output KV bound and defer the FCFS head when those bounds would exceed the physical pool. It uses no actual future output length. This tests whether ordinary conservative admission already covers the remaining gap cases before inventing a new method.

## Limits and artifacts

Two runs per arm on one viewed development cohort. Native uses no CPU KV offload storage; static uses 16 GiB and a different selected-save path, so these are complete-configuration comparisons. Adjacent pairs differ in 59 and 66 output-ID sequences; same-arm repeats differ in 59 native and 54 static sequences. Workload repetition and source limitations remain. No causal, equal-work, held-out or paper-readiness claim.

[Main analysis](native_static_matched_analysis_v1.json); [gap context](native_preempt_gap_context_v1.json). Original archive: `C_research_artifacts/20261001/c-native-static-matched-dev-v1-complete.tar.gz`, SHA-256 `49277d348f9d754086fa487ce55d6132a428d56c88b2a92dc9737d0c49648b32`. Runner commit `2ce90bf`.
