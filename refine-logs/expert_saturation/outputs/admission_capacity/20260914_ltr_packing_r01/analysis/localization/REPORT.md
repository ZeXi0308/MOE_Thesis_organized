# Packing first-divergence and recovery localization

MEASUREMENT_ONLY_ACTUAL_TRAJECTORY_LOCALIZATION

All measurements are retained policy-specific executions. Common actual token allocations and returned prefixes do not establish full-engine or KV-byte checkpoint equality. Recorded prestate includes counters, block counts and pool state only. No future trace is reused as a counterfactual and timing differences are not wholly attributed to packing.

| pair | first action difference | common returned outputs fit/prefix | elapsed fit/prefix s | prefix max gap s | 0/1/2 outputs fit → prefix |
|---|---:|---|---|---:|---|
| block0-d6-packing-fit_scan / block0-d6-packing-rank_prefix | 406 | 11295 / 11295 | 8.297852 / 8.588199 | 6.732402 | {'0': 2, '1': 8, '2': 3} → {'0': 6, '1': 0, '2': 0} |

block0-d6-packing-rank_prefix: longest gap request memory-train-article-0003345, output 541→542, steps 642–979, 338 calls, max consecutive nonselected 200. Partial-recompute selections: [(775, 0, 997, 0), (976, 0, 1024, 0), (977, 1024, 2048, 0), (978, 2048, 3072, 0), (979, 3072, 3613, 1)]. Recorded common prestate equality=True, available output-prefix equality=True. 
| block1-d6-packing-fit_scan / block1-d6-packing-rank_prefix | 406 | 11295 / 11295 | 8.611777 / 8.359053 | 6.576921 | {'0': 2, '1': 8, '2': 3} → {'0': 6, '1': 0, '2': 0} |

block1-d6-packing-rank_prefix: longest gap request memory-train-article-0003345, output 541→542, steps 642–979, 338 calls, max consecutive nonselected 200. Partial-recompute selections: [(775, 0, 997, 0), (976, 0, 1024, 0), (977, 1024, 2048, 0), (978, 2048, 3072, 0), (979, 3072, 3613, 1)]. Recorded common prestate equality=True, available output-prefix equality=True. 

At the first difference, the shared30 decode selections reserve3 growth blocks, leaving145 of the initial148 free. Waiting request3571 needs212 blocks; even releasing later resident3640 owned63 gives208, four blocks short. Resident3640 itself needs142 more blocks. Fit-scan continues its recovery with994 tokens; rank-prefix stops after the shared30 decode tokens. These are observed matching prestate counts and actual actions, not a reused future trace.

Rank-prefix longest gaps split into133 nonselected calls, one997-position partial restore, then200 nonselected calls, then four boosted restore calls. The partial restore resets idle133→0, is held40 calls, and is preempted again before producing a new output. The later boosted restore succeeds.

Rank-prefix removes the measured 1/2-output re-preemption segments but leaves six 0-output re-preemption segments per run, versus two in fit-scan. It therefore does not eliminate unfinished-restore churn in this common backend.
Elapsed time already differs before the first differing action: prefix minus fit is +0.290346s in block0 and -0.252724s in block1. These offsets are retained; later wall/completion differences cannot all be assigned to the differing packing actions.

The direct residual supports a bounded next causal test at an observed zero-output recovery: preserve a selected recovery only until its first new output, then release the obligation, independently execute the future, and count costs to every other request. This is an untested action hypothesis, not evidence of feasibility, net benefit or full-LTR failure. The earliest post-start starvation/blocking state should be checked before allocating another run.
The complete action boundaries, ranked candidates, both prefix longest-gap timelines, and every retained 0/1/2-output segment are in analysis.json.
