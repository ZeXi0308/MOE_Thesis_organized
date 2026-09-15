# Native transfer: first bounded request-level probe

2026-09-06. After the custom-runtime cohort repeat retained unstable cap rankings,
the next question is whether the native backend exposes a reproducible capacity
response. No pressure-aware policy is introduced.

- Use the original prepared 16 real WikiText rows, exact 128-token prompts,
  16 fixed output tokens, pinned OLMoE BF16 revision. Prefix caching and routed
  expert export are disabled; each episode executes its own future state.
- Native vLLM 0.26.0, synchronous in-process V1 engine. Async scheduling is
  explicitly disabled so scheduled token identity and host output are observable.
  Use normal compiled/CUDA-graph execution first (`enforce_eager=False`), common
  max_model_len=256, max_num_batched_tokens=1024, memory utilization=0.70 and FCFS.
  These settings are shared by both cap arms. Native prefill scheduling differs
  from the earlier custom one-prefill loop; cross-backend speedup is not a claim.
- Four fresh model processes in A/B/B/A order: caps 6/8/8/6. Within each, run
  steady/bursty at arrival scale 1 and 0.02, then reverse condition order for
  repeat 1. This yields 8 episodes per process, 32 total if all complete.
- Scale 1 retains the earlier 0–1.5 s arrival horizon. Scale 0.02 gives a 0–0.03 s
  horizon and tests faster demand without modifying model routes or experts.
  Both loads are declared before native timing. Keep the original exploratory
  TTFT <=5 s and mean TPOT <=0.2 s for this first measurement; all-pass results
  imply a need for separate native-load/SLO calibration, not controller success.
- Submit all due requests to the engine; max_num_seqs is the native admission
  limit. Do not add a second client inflight limiter. Record actual scheduler
  running/waiting, scheduled prefill/decode tokens and preemption events.
- Store scheduled arrival, client submission and host token receipt on one
  perf_counter origin. Preserve core monotonic timestamps separately; never
  subtract an epoch timestamp from a core monotonic value. Cumulative output
  handling retains real chunk receipt times, with no invented interpolation.
- Buffer the scheduler trace in memory; its overhead is included in host timing.
  Warmup/loading/compilation are excluded from serving episodes. GPU process
  checks cover episode boundaries only. Retain every failure and repeat.

If the initial native engine or observation path fails, fix the concrete issue
and use a new output directory. Do not relabel initialization failure as a
scientific NO-GO. If native caps are underfilled or SLOs all pass, report that
boundary and freeze the next calibration separately. No U/C, native method,
multi-GPU or submission-ready claim follows from this transfer probe alone.

## Compatibility retry after attempt 1 (zero measured episodes)

User approved transfer and execution. Attempt 1 loaded/compiled OLMoE but failed
during engine initialization in FlashInfer sampling's architecture check. The
log reports `SM 12.x requires CUDA >= 12.9`; actual Torch reports CUDA 13.0 and
device capability 12.0. No serving episode ran; retain the failure as attempt 1.

For attempt 2, use vLLM's supported `VLLM_USE_FLASHINFER_SAMPLER=0` opt-out for
both caps. This selects native sampling and leaves compiled model execution,
FlashAttention and Triton MoE unchanged. Real requests use temperature zero;
the inspected sampler returns greedy tokens before stochastic sampling.
Record the switch in environment.json. Keep all workload/SLO/cap/repeat settings
unchanged and use a fresh remote directory. This is a compatibility repair, not
an effect-selected experiment configuration or a model precision change.
