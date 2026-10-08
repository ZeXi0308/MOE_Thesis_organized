# Native128 development observation — frozen question, not a policy comparison

## Evidence and one hypothesis

The fixed16 Instruct GSM8K unit completed with16 genuine EOS,0caps,1571tokens and5/16correct. All16 lengths were49–129. Full1319 input geometry proves the inherited32seq regime has no possible known-cap pressure (largest32 full-cap sum3586<4096). Source check77e1142 establishes that the same directEngineArgs entry resolves128 when max_num_seqs is unspecified; publicLLM()256 is a different entry.

One next question: **does a complete batch at the same entry's128 sequence limit encounter actual KV pressure while serving natural, scored outputs?** This is a development observation of a reasonable native concurrency limit, not32-vs128 performance, not a production arrival trace, and not a new admission mechanism.

## Fixed unit

Use the first128 official test rows in source order, including the prior16; all retained. Same pinned Instruct revision,8shot chat+Answer prompt,greedy seed20260905,EOS-only,max1024,min0; all arrivals0 and full drain. Keep BF16,batch1024,4096 usableKVblocks,APCon,no offload,async or speculation. Explicitly set max_num_seqs128 and capture the resolved runtime value. Only count/concurrency and additional light schedule-state observation change from the fixed16 qualification; do not describe all manual resource settings as vLLM defaults.

Observe physical used blocks before/after native schedule, running/waiting, scheduledtokens and actual native preemptions. Preserve all128 texts/tokenIDs/finish reasons/host delivery times; apply the identical historical numeric extraction to the full128 denominator. Two16-token first/last warmups, then APC reset. Existing spare lock and bounded1560s launcher cover load,warmup,generation,drain,child exit and stage cleanup. No retry queue or parameter sweep.

## Prediction and stopping decision — before128 outputs

Source-first128 fullcap geometry is13784 independent blocks,8831 with exact input-prefix block deduplication; these are whole-cohort bounds, not measured usage. If every output were at most the prior16's maximum129, and all requests reused the common39 fullprefixblocks, the corresponding geometry would be1668blocks. That is a **hypothetical extrapolation**, not an online bound: future EOS is unknown and native cache materialization may differ.

The short pilot therefore predicts little or no pressure at128. Test that prediction once. Native preemptions or failed allocations with actual pool exhaustion would establish a concrete state for further causal work, not a strategy benefit. A no-preemption run with ample measured headroom rejects this current GSM8K capacity opportunity: stop this direction at these reasonable settings. Do not raise concurrency further, shrink KV, change max_tokens, select long outputs, alter prompts/seeds, or rerun until pressure appears. Retain incorrect answers and report quality regardless of result. All first128 are development data after inspection, not fresh confirmation.

No followup policy is selected or promised by this plan. The overall paper still requires a distinct mechanism, strong comparable baselines, complete costs and held-out confirmation; this observation does not substitute for those requirements.
