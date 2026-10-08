# Native maximum-bound admission: completed development pilot

## Result

The simple native reservation rule completed all 128 requests, removed native KV preemptions and every observed gap above 4 seconds. It is now a competitive simple reference. This is one development pilot, not a new algorithm or a confirmed performance improvement.

| Policy/run | Joint 20 s TTFT / 4 s gap | Goodput, requests/s | Episode, s | Mean flow, s | Preemptions |
|---|---:|---:|---:|---:|---:|
| Previous matched native 1 | 58 | 0.7244 | 80.068 | 34.900 | 38 |
| Previous matched native 2 | 60 | 0.7471 | 80.308 | 34.678 | 41 |
| New maximum-bound pilot | 67 | 0.7768 | 86.254 | 36.537 | 0 |

Each resident request reserves `ceil((prompt_tokens + max_tokens)/16)` blocks; only a fitting FCFS prefix enters the native scheduler. The sum peaked at 4,095 of 4,096 usable blocks. All 128 admissions and genuine native completion releases were recorded. The gate held waiting requests in 5,041 of 6,517 schedule calls; maximum effective concurrency was 23. This is conservative admission, not physical preallocation or an eviction mechanism.

All 128 requests passed the 4 s gap target. Maximum observed host-return gap was 0.0426 s, median TTFT 19.647 s and p90 TTFT 48.736 s. Relative to the two earlier native runs, the pilot had higher goodput at 16/20 previously specified frontier points, but mean completion flow increased 4.7–5.4% and output throughput fell 6.9–7.9%. TTFT worsened for 81 and 85 requests respectively. Thus improved tail continuity still has a queueing cost.

Actual output was 124,325 tokens: 121 requests hit the length cap, seven stopped. Output ID sequences differed for 57/128 and 64/128 requests versus the two native references, with three/four length differences. The previous eight-cell repetition concern remains. These are measured generated outputs, not validated useful-output goodput or equal-work speedup.

## Execution and next decision

The adapter/cell/launcher were frozen at commit `61edda3`; the plan predates execution. The run held the existing shared GPU lock from loading through child exit, finished at 2026-09-30 19:49:57 UTC, and released all requests/KV. GPU usage after exit was 2 MiB with no compute process. The inherited native launcher's `pressure_qualified=false` means the deliberately prevented 32-running-plus-preemption condition was absent; the admission contract itself passed.

Raw source: `/root/autodl-tmp/c-research-20260930/c-native-max-bound-pilot-v1/native_max_bound_1/raw.json`. Local originals: `/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/c-native-max-bound-pilot-v1`. Transferred raw SHA256 equals the launcher receipt: `0cc5876bd972b209c682dc5a8407d0e849e4577452900078d1656e1a0a537803`. No archive was generated remotely because another job held the lock; files were copied with a bandwidth limit.

Analysis: `C_NATIVE_MAX_BOUND_ANALYZE_V1.py` → `native_max_bound_analysis_v1.json`. The next and only GPU experiment is the preregistered native/bound/bound/native repetition below. Prior native pilot goodput 0.5464 remains retained; the observed variation is why the new single-run advantage is not a conclusion.

## Observed queueing opportunity while the repetition was GPU-deferred

A small CPU analysis (`C_NATIVE_BOUND_HOL_OPPORTUNITY_V1.py` → `native_bound_hol_opportunity_v1.json`) reconstructs the waiting queue from actual engine submissions and first native scheduling; all 6,517 waiting counts match. Of 4,921 steps where no FCFS head could enter, 2,723 had at least one later request whose known upper bound fit the remaining reservation budget. These involve 78 distinct followers; choosing only the earliest fitter at each observed step involves 27 distinct requests. Their first-opportunity TTFT slack medians are 14.973 s and 0.876 s respectively. No sequence-slot limit caused these holds.

This is an observable head-of-line opportunity, not an action result: repeated steps are correlated, a fitting request can still miss TTFT, and bypassing the head increases somebody else's wait. The sole registered next GPU experiment remains the native/bound matched repetition. If it completes, a simple queue-selection comparison is more directly motivated than a length predictor. In this pilot, 121/128 outputs reached the cap; summing per-request `ceil((prompt+actual_output)/16)` gives 23,815 blocks versus 24,238 using the upper bound (1.75% less). Those sums are neither concurrent physical usage nor a performance oracle.
