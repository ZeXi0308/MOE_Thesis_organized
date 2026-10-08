# Direct prior changes the retirement candidate's novelty status

## Primary evidence

[SOLA §5.1](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf) attributes its peak-memory admission mechanism to LightLLM. [Past-Future §3.3, equations2–4](https://arxiv.org/html/2507.10150#S3.SS3) evaluates batch memory at predicted completion endpoints; Algorithm1 applies that check to each FIFO candidate. This is a closer prior than the previously tested single-donor partial comparator.

Official [LightLLM queue](https://github.com/ModelTC/LightLLM/blob/6c9e03b14ed3e0da8d4c867258be7a90e46d941d/lightllm/server/router/req_queue/chunked_prefill/impl.py) sorts `(has_tokens, remaining_steps)` in descending remaining order and takes the maximum of cumulative existing tokens plus prefix size times the corresponding remaining steps. The author's [AE repository](https://github.com/WuSiYu/lightllm-ae/tree/d93ff69c07d4097b0a6a4ee67ea355e8780e3095) retains this formula in both conservative and Past-Future queue sources. Its request conversion already has maximum-output branches. Thus merely replacing predicted EOS length with a cap is not an independent conceptual contribution.

## Algebra and implementation scope

Our own derivation: for residents with H_i=P_i+O_i and R_i=M_i−O_i, sort R descending. The token-level expression `max_i(sum_{j≤i}H_j+i*R_i)` equals `max_t sum_{j:R_j≥t}(H_j+t)` over retirement endpoints, with the last member of each equal-R group retaining all simultaneous retirees. The paging-aware version puts a per-request ceil-to-16 inside each sum. Current C adds a conservative endpoint token, explicitly protects one-output-per-call synchronous progress, and fully charges each new/partial-prefill request rather than jointly projecting it. These change scope/conservatism; they do not establish new peak-at-retirement insight.

The native traces saved aggregate peaks but not complete per-resident states at every gate call. The [142-line CPU check](C_NATIVE_RETIREMENT_PRIOR_PEAK_CHECK_V1.py) and [result](native_retirement_prior_peak_check_v1.json) verify seven synthetic integer states using the unchanged C function: exact equality after endpoint alignment and per-request paging. In the 16-resident example both give3544blocks while rounding the token peak once gives3538; the six-block difference is paging. There are zero nonempty native state comparisons; no missing states are reconstructed by assumption. Existing physical-capacity/progress checks and GPU outcomes retain their original status.

## Consequence and unique next direction

Retain all prior N/F/R and donor results. Complete the already-running fixed lower-rate block unchanged. Downgrade the independent novelty hypothesis for the current envelope; do not rename it or seek a favorable seed/threshold to rescue that claim. The measured R/F service improvements remain evidence about a conservative native implementation in this narrow domain.

Next compare the actual Past-Future scheduling core on the same backend, with documented paper/code differences. Do not call the current cap-envelope or donor a full reproduction. A native adaptation must include causally updated completed-output history, conditional resident estimates, FIFO candidate-wise future peak and the stated reserve; preserve natural EOS, all requests, preemptions/recomputation and full drain. Cold-start history may use the author's maximum-length initialization. Online completions from the ongoing episode are valid past information; preloading outcomes from a prior full run of the evaluation cohort is not a fresh causal baseline.

The saved AE code differs from the paper: WINDOW_SIZE40, MINIMUM_SAMPLES200, MAXIMUM_LISTS5, reserve0.05, twenty initial maximum-length samples, conditional interpolation for residents, and at most one new request per queue pass. The paper describes a 1000-request history and simpler conditional sampling. Choose and freeze a named implementation before running; do not merge favorable pieces or label either configuration universal. Timing/indexing, paging, vLLM chunked-prefill progress and preemption recovery need an explicit small port, not an unrelated framework rewrite.

Scope: this is an action-level collision and next-baseline decision, not a performance verdict on full LightLLM, CacheOPT or Nested WAIT. No new GPU job is queued by this source check. The paper remains WORKING_DRAFT and is not READY_FOR_HUMAN_REVIEW.

## Saved source references

Local stable source excerpts and URL/SHA receipt: `C_research_artifacts/20261001/lightllm-primary-source/`. Author AE commit `d93ff69c07d4097b0a6a4ee67ea355e8780e3095`; official LightLLM inspected commit `6c9e03b14ed3e0da8d4c867258be7a90e46d941d`. These files were read, not installed or executed. No remote runtime changes.
