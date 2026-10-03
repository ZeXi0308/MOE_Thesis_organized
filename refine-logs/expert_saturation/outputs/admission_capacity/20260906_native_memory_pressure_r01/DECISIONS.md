# Natural context memory pressure: frozen before GPU execution

Question: with resident BF16 OLMoE on the existing RTX5090, do legal long-context
requests create actual KV-capacity constraints and request-level consequences?
This is operating-regime qualification, not an expert-paging method or Oracle.

Use 32 real WikiText103 train articles, selected in source order; matched short
128-token and long3072-token prefixes, both with1024 fixed generated tokens.
The pinned model supports4096 total tokens. Existing cached test rows were too
short and only27 reconstructed test articles met the initial3968-token length;
the cached train shard provides32 complete articles without repetition/padding.
This is not a holdout or task-quality evaluation. Input details are retained in
`inputs_preparation/README.md` and `prepared/inputs_report.json` below it.

Before any run, the initial3968+128 idea was changed to3072+1024: a short decode
could finish while the remaining long prefills are still entering, leaving no
sustained active population. This is a pre-run operating-regime decision, with
no new GPU metrics consulted. All source identities and arrivals match between
the two length conditions. Arrival spacing is50ms, a synthetic steady episode
of32 arrivals, followed by full completion or an explicitly recorded boundary.

Fixed common engine: vLLM0.26, existing PyTorch2.11/CUDA13 environment, pinned
OLMoE weights/tokenizer, BF16, max_model_len4096, engine max_num_seqs32,
max_num_batched_tokens1024, chunked prefill enabled, prefix cache disabled,
FCFS, synchronous in-process stream_interval1, compiled mode, seed20260905.
GPU budget is the ordinary0.90 engine setting, not the previous short-run0.70;
it is identical in all arms. No artificial tiny expert cache, offload, quantization,
route/top-k modification or expert relocation is introduced.

Eight primary episodes, all in fresh processes: short16 / long16 / short32 /
long32, then the exact reverse. Every process uses the same three warmups:
short32 requests at cap16, short32 at cap32, and long2 at cap2, with16 outputs
and simultaneous arrival. The pressure population is not consumed as warmup.
All outcomes, including failures, remain canonical; no favorable-run selection.

The active-request cap controls admission only. Native decode progress and
unscheduled eligible decode IDs are recorded. To enforce the no-preemption
constraint, a guard stops the episode on the first `_preempt_request` attempt,
before KV is freed or computed state reset. The native caller already removes
its victim from a running list, so the attempt is not a usable resumed engine
state. Capture the pre-schedule population and failed attempt separately, save
all requests, shut down this process, and continue the campaign with a fresh
engine. No GPU forward executes for the failed scheduling attempt. Such a cell
is `CAPACITY_BOUNDARY_STOP`, not a successfully completed policy or measured
full-horizon throughput. Unexpected/OOM/warmup failures remain incomplete.

Primary measurements: actual/target active counts, KV total/usable/free blocks,
per-request block-table lengths, allocation failures, attempted preemption,
completed/unfinished requests, continuous TTFT/TPOT and completion timeline.
Record unique parameter/expert storage and unique physical KV storage plus
Torch allocated/reserved/peak bytes and NVML boundary observations. KV occupied
blocks are a subset of the preallocated KV pool, not extra memory to add to it.
The remaining live allocation is not automatically named workspace or graph
memory. All telemetry cost lies inside request timing; no per-step CUDA sync.

Reference SLO5s TTFT/200ms mean TPOT is retained as a secondary diagnostic only.
Do not apply the previous9ms short-workload threshold to select a winning arm.
Do not compare truncated boundary-cell rates as completed-episode throughput.

Continue only if actual capacity constraints are exposed in this natural
configuration and simpler token/KV-budget admission leaves a useful tradeoff.
Ordinary KV saturation alone is not MoE novelty. Full residency also means this
probe cannot measure expert transfers or prove a paging opportunity. Reopen an
expert action only with an executable existing runtime path and measured full
cost. If all cells fit and the caps do not bind, record that domain boundary;
do not iterate thresholds or arbitrary cache restrictions to obtain a result.

Evidence ceiling: native synchronous in-process request/capacity measurement,
single model/GPU, fixed output lengths, synthetic arrivals. GPU is UNRUN until
the frozen campaign actually executes. No CCF-B result or new method is claimed.
