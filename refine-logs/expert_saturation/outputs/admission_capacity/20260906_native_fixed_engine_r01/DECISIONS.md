# Fixed-engine native admission probe, frozen before execution

2026-09-06. Previous native transfer completed 32 episodes. Fast-arrival cap8
won all eight descriptive comparisons, but engine max_num_seqs also changed
the graph capture plan and all requests passed the old SLO. This experiment
separates admission cap from the engine's static capacity/configuration.

- Both arms use identical EngineArgs.max_num_seqs=8, max_model_len=256,
  max_num_batched_tokens=1024, memory utilization=.70, compiled graphs,
  synchronous in-process FCFS, FA2/Triton MoE, BF16 pinned OLMoE and greedy output.
  Keep VLLM_USE_FLASHINFER_SAMPLER=0 for the already diagnosed toolkit issue.
- Set only scheduler.max_num_running_reqs to 6 or 8 when frontend and native
  queues/requests are empty. Preserve scheduler_config.max_num_seqs and worker/
  graph configuration. This is a static admission experiment, not online drain.
- Reuse the original 16 real texts, 128 prompt tokens and 16 output tokens.
  All future state is independently executed. Only fast arrivals: scale=.02,
  steady and bursty, each with the same 0–.03 s arrival horizon.
- Four fresh engine processes: 6/8/8/6. Two reversed condition repeats per
  process give 4 measured episodes each, 16 total. Retain all measured repeats.
- Before measurement, retain the original all-at-zero warmup and then one
  warmup episode for each actual steady/bursty condition. Record phase starts/
  ends so remaining JIT warnings can be assigned to warmup or measurement.
  Warmup is outside serving timing, with compact status/width summaries retained.
- The main exploratory thresholds were declared in the previous report:
  TTFT<=.20 s and request mean TPOT<=.009 s. They come from pooled p75 of all
  256 previous fast-arrival request records, rounded upward to 10 ms / 1 ms.
  They are not business SLOs or effect-selected per-cap thresholds. Also compute
  the unchanged reference limits (5 s / .2 s) from each new raw episode.
- Keep host timing, actual scheduler counts, all output tokens and failures.
  Verify common engine arguments/graph sizes, actual cap exposure, no preemption,
  no token accounting drift, and whether SLO outcomes now separate. Existing
  counterbalanced repeats are retained even if noisy; no auto-retry for gain.

The outcome can qualify a static admission response and expose remaining noise.
It cannot establish U/C residual, online action value, task-quality equivalence,
an Oracle, multi-GPU benefit, or a paper-ready method. If a clear ranking fails
to persist, the next action is one unchanged controlled repeat, not a predictor.
