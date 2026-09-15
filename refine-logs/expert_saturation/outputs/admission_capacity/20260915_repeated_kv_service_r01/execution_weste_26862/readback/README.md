# Repeated native KV saving: complete service value

Status: PREPARED, GPU UNRUN. This package supersedes the measurement contract of the original r02 pair; it does not overwrite or merge its results.

Question: with the same staged most_output policy and actual GPU/host limits, does native repeated saving improve complete-request delay versus service efficiency? No window-controller change is included.

Run order, fixed before outcomes: diag-off, diag-on, block0-off, block0-on, block1-on, block1-off. All cells execute serially on GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5, under the existing whole-group exclusive lock. The two diagnostic cells are separate from the four primary timing cells. Both balanced pairs are retained; no best-repeat selection. All requests and failure artifacts remain present.

Both arms: OLMoE-1B-7B-0924 revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5, the existing cloned vLLM 0.26 environment, 32 long requests (3072 prompt, forced 1024 outputs), GPU KV allocation 13,960,740,864 bytes / 6656 usable 16-token blocks, native host cache configured 16 GiB, offload_prompt_only=False. The host parent memory.max is recorded separately; native KV allocation, valid cached entries, process RSS/HWM and parent cgroup usage are overlapping accounting views. No new model download or second GPU experiment.

Off: forbid all measured-episode native store jobs; retain normal request/output history, discard preempted GPU KV, use native recomputation. Warmup connector state is reset before measurement.

On: the identical selector first prepares its chosen victim, which still decodes one token, then commits at the next scheduling boundary after rechecking resources. Only preparation permits a native store, capped at floor(computed_at_prepare/16)*16 materialized positions using their owned physical GPU blocks. The unmaterialized decode position and incomplete trailing block are excluded. Native incremental saving may emit only missing blocks or no job; no job is not proof of a hit. Native pending-store flush prevents block reuse before completion. Native lookup/load handles validity and eviction; all remaining history is recomputed. The same target protection ends after an actual new output in both arms.

Primary measurement: engine-return timestamps of new output tokens, full-cohort makespan/output rate, mean completion, TTFT and generation-gap distributions; no service SLO threshold is selected from outcomes. All policy execution and native recovery costs remain in the live engine path. Post-request offload drain is separately recorded and cannot be added to overlapping transfer timings. Full step/KV/lookup/dispatch instrumentation is confined to diagnostic cells.

Diagnostic timestamps: last returned new token L; first resource-feasible boundary E (direct free blocks, or fixed most_output victim funding, distinguished from queue/policy eligibility); first actual native load host dispatch or first recovery-bearing engine call S; next returned new token F. Host dispatch is not device DMA start, and engine return is not client receipt. Report E-L, S-E and F-S separately, retaining unknown boundaries. A resource fit alone does not assert native queue dispatch or a sustainable future window. Output waiting is never reset by internal scheduling.

The internal-work account reports native store/load bytes and completion, actually allocated unique CPU storages, valid cache entries, recovered prefix coverage, remaining executed recomputation, new outputs until re-preemption/completion and state discard. It never calls inclusive engine time pure recomputation tax or equates less work with a service improvement.

Interpretation: if saving improves the measured delay/efficiency tradeoff consistently, adopt it as a strong baseline. If recomputation falls but long pre-start wait dominates, the next question is recovery start timing. If losses occur during recovery before useful service despite funded admission, return to recovery resource sustainability. This pilot is fixed-length, closed-cohort evidence, not an unknown-EOS or general serving claim.
