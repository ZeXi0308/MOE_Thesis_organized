# Native admission capacity knee: frozen before GPU measurement

Question: with one native engine configuration, does sustained arrival expose a
repeatable throughput-versus-request-latency tradeoff across admission caps?
If not, do not build a controller just because custom cap6/8 once differed.

Use pinned BF16 OLMoE on the existing single RTX5090/vLLM0.26 environment.
Engine max_num_seqs32, max_model_len256, token budget1024, utilization.70,
compiled/FA2/Triton MoE, native FCFS, no prefix cache, no async scheduler,
no preemption or request skipping. Only change cap on completely drained engine.
All policies regenerate future KV, batch, route, outputs and completion timeline.

32 real WikiText rows,128 prompt and128 fixed output tokens. Frozen source has
only48 eligible rows: initial offset64 preparation failed with0 GPU cells.
Use offset16/count32, already explored in custom runs, first used here in native
admission runs. This is not fresh article-disjoint holdout. Preserve failure.

Steady gap.05s (20req/s); burst groups8. Both first/last arrivals0–1.55s.
Caps4/8/16/32, common engine32. Two fresh engines: forward caps4,8,16,32 and
reverse caps32,16,8,4. Per cap forward steady/bursty, reverse bursty/steady.
One condition repeat per engine:16 primary episodes total. Additional steady
underload scale8 (gap.4s;0–12.4s), only endpoint caps4/32 in each process:
4 negative-control episodes,20 measured episodes overall,640 request executions
but only32 source rows. If active reaches cap4 with queue, qualify that control
as not sufficiently underloaded; do not assume it was a null intervention.

Both engines warm all caps in the same ascending order, first all-at-zero then
both actual primary arrival conditions, before measurement. Phase logs distinguish
warmup from measured cells. Cap changes preserve compiled engine configuration.
GPU process check before initialization and before/after every measured episode.

Keep primary TTFT<=.20s / mean TPOT<=.009s, and reference5s/.2s. Predeclare a
descriptive full grid TTFT .2/.5/1.0s × mean TPOT .009/.012/.016s; show every point,
never select a favorable grid point as canonical. These are exploratory thresholds,
not business guarantees. Judge continuous latency/throughput alongside pass counts.
All queueing, host submission, sampling delivery and trace overhead count in the
episode denominator; initialization and matched warmup are outside serving time.

Primary inference unit is episode/cohort, not adjacent decode steps. Record target,
actual active, decode width, post-schedule waiting, submission lag, output identity,
and zero preemption. submission lag is not native queue wait.

If all meaningful caps are dominated by the largest, stop this domain's controller.
If largest cap never binds, report that the knee is not covered. If primary SLO
all passes/fails, retain it and interpret continuous curves, not a system NO-GO.
If a repeat changes ranking near noise, only one unchanged controlled repeat is
allowed before interpreting it. A stable tradeoff authorizes one simple ordinary-
latency feedback probe in this same chain; no U/C model unless residual remains.

No exact snapshot Oracle, second model, full audit, or extra predictor is required
for this exploratory experiment. Five existing native capture checks are sufficient
unless raw evidence identifies a concrete error. No claim of steady-state capacity,
native routed-expert telemetry, joint-SLO policy improvement, or CCF-B method yet.

Execution note: first startup passed an empty-device precheck, but a separate native cohort campaign acquired the GPU during initialization. vLLM rejected insufficient free memory, before any warmup or measured episode. Preserve this0-cell blocked attempt. Only after that campaign and GPU process exit, rerun the identical source/workload in a new remote directory. Concurrent work also means these retained source rows must not be promoted to a fresh-native holdout claim.
