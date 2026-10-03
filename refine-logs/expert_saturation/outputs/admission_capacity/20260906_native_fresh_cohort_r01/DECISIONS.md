# Fresh-input fixed-engine admission probe, frozen before GPU execution

2026-09-06. Previous fixed-engine native run completed 16 episodes on the
original cohort. Cap 8 beat cap 6 on throughput and main goodput in all eight
descriptive pairs. The current question is whether that ordering persists on
one new set of inputs, with all execution settings and thresholds unchanged.

- The original source manifest contains only 48 rows reaching 128 tokens;
  all 48 were used by admission_capacity cohorts 0/16/32. Therefore select
  from the same cached WikiText-103-raw-v1 test Arrow bytes, in dataset order
  strictly after the original manifest's largest row index (204).
- Select the first 16 nonempty rows whose pinned OLMoE tokenizer output reaches
  128 tokens, excluding prior cohort text and token hashes and duplicate new
  hashes. Selection reads no routes, logits, timings or outcomes. Preserve raw
  text and use the same add_special_tokens / right-truncation settings.
- Freshness means distinct rows/text/token IDs from the 48 inputs already used
  in this admission research line. It is not article-disjointness, another
  dataset, or a repository-wide never-used holdout. Record the Arrow SHA and
  current derived dataset fingerprint separately.
- Run the byte-identical native runner, capture helper, metrics and campaign
  from the previous fixed-engine experiment. Engine max_num_seqs is 8 in all
  arms; empty-episode admission cap is 6/8/8/6 in four fresh sequential processes.
- Keep 16 requests, 128 prompt / 16 output tokens, the same seed/model revision,
  BF16, compiled FA2/Triton MoE, synchronous FCFS, token budget 1024,
  max model length 256, memory utilization .70, and prefix cache disabled.
- Keep both original arrival traces at scale .02 (0–.03 s). Each process runs
  steady/bursty twice in reversed order, for 16 measured episodes total.
  Keep the same three warmups per process and the FlashInfer sampler opt-out.
- Main SLO remains TTFT <= .20 s and request mean TPOT <= .009 s. Reference
  SLO remains 5 s / .2 s. Do not recalibrate using the new inputs or outcomes.
- Retain every run and independently advance all future state. Report all
  eight paired rankings and any sign reversal; do not replace an unfavorable
  run. If signs reverse, the next experiment is one unchanged controlled repeat,
  not a predictor. If cap 8 still dominates, retain it as the stronger tested
  static baseline; this is not proof of global optimality or zero dynamic Oracle.

The evidence ceiling remains a single-model, single-GPU native in-process
static admission measurement. No U/C, dynamic policy, Oracle, task-quality,
multi-GPU or sustained-capacity claim is authorized by these measurements.
