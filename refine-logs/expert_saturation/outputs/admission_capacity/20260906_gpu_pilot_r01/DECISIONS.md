# Decisions before follow-up GPU execution

2026-09-06, local branch agent/publish-current-moe-code@2a37765; execution uses
one exported code version and the pinned OLMoE cache. All executions are retained.

- First execution: exactly the previously prepared 24-cell cap=2/4/8 scan,
  TTFT <= 5 s and mean TPOT <= 0.2 s, 16 requests, 128 prompt / 16 output tokens,
  steady/bursty, OFF/ON, two counterbalanced repeats. No original configuration changed.
- After those cells exposed the TTFT versus TPOT tradeoff between cap=4 and 8,
  freeze one follow-up bracket: cap=4/6/8 with all other workload, SLO and repeat
  parameters identical. Adding the midpoint tests whether a stronger fixed cap
  removes apparent need for a route controller. Anchors 4 and 8 rerun in this
  bracket to expose drift. This is adaptive exploration, not a preregistered
  holdout or independent workload. No favorable-run replacement.
- U/C remain post-action telemetry; these scans cannot establish action-choice
  increment beyond ordinary state or signal survival during drain. No new
  predictor or Controller is justified by their mean trend alone.

- After cap=6 won every OFF bracket comparison, freeze the final actuator probe:
  A = [[0,6],[1.5,6],[3.0,6]] (sham callbacks); B = [[0,6],[1.5,8],[3.0,6]].
  The first intervention time is the final planned arrival (1.5 s); the second
  is twice that value (3 s), not chosen from a favorable outcome or U/C.
  Run whole episodes sequentially A/B/B/A, each with steady/bursty and OFF/ON
  (16 cells total), fresh model process and independent KV/queue/output each
  time. Inputs and both SLOs unchanged. Model/process warmup is outside serving
  wall time for every arm. Across arms this is controlled end-to-end rerun, not
  exact prestate branching; timing may change the pre-action frontier. Report
  that alignment explicitly. Primary purpose: measure real non-preemptive
  actuation lag and ask whether this one temporary cap increase improves on
  the strongest tested simple static cap. No search over pulse times; no U/C
  policy claim; a failed pulse does not kill dynamic concurrency control.

- Controlled pulse repeat decision: the first A/B/B/A produced bursty ON
  goodput deltas of -7.84% and +73.15%, with the latter driven by the final
  hold control dropping to 8/16 joint passes. Before any mechanism claim,
  rerun the exact four-process A/B/B/A, all original steady/bursty and OFF/ON
  cells, in a new actions-repeat directory. Workload, SLO, pulse times,
  model/runtime and warmup remain unchanged. Do not replace actions/ with
  this repeat. This is one diagnostic repeat, not new independent text or
  predictor training. Execute after the existing GPU cohort queue finishes;
  no simultaneous model execution. Report both runs even if signs differ.
