# Fixed-frontier admission probe (before GPU execution)

Question: after six completed decode calls at cap6, do original and offset32
text cohorts exhibit different action responses under comparable ordinary state?
The original wall-time pulse was not reproduced. Do not search pulse return times.

Use only hold6 versus one rise to8 after completed decode call6, no return action.
Use original cohort offset0 and the retained offset32 cohort (already explored;
not fresh holdout), 16 real WikiText rows each, 128 prompt / 16 output tokens.
Identical arrivals (steady and bursty, 0–1.5s), TTFT<=5s, mean TPOT<=0.2s.
Keep OFF/ON independent trajectories. Match warmup widths6 and8 for every arm.
Do not change the runtime's one-prefill-per-iteration / full active decode rule.

Eight fresh model processes in palindrome order:
  a0_c0_hold, a0_c32_hold, b0_c32_up, b0_c0_up,
  b1_c0_up, b1_c32_up, a1_c32_hold, a1_c0_hold.
Each process runs steady OFF/ON and bursty OFF/ON:32 cells,32 unique source rows.
No overlapping GPU tasks; process checks before load and each cell before/after.
Keep all raw runs; same Torch thread setting as prior environment (OMP/MKL25).

Primary comparison is within each cohort/regime/repeat/telemetry, hold versus up.
Compare actual action-frontier membership/progress/KV lengths/queue and future
counts, plus waiting ages/recent model/iteration/ITL. Across cohorts compare shapes
and ordinary values; different request IDs/text are intentional. Use latest
completed step per-layer U/C separately from the four-step mixed-width history.
No predictor, no offline masking, no dynamic oracle, no exact-KV-snapshot claim.
Same visible discrete frontier does not establish equal hidden tensors or clocks.

Stop after this probe unless a specific execution failure needs repair. If
frontiers differ, latency variation overwhelms action responses, or no consistent
increment beyond hold6 appears, retain the exact local limit and prefer the
already-prepared native-runtime existence check over more custom predictors.
Any future native transfer remains a separate experiment, not parallel work.

Execution note: first launch found another native vLLM task at the before-load
GPU check and exited with0 cells (UNRUN). Retain that blocked attempt. Only after
both the GPU process and native campaign parent had exited was the identical
probe started under a new results-r02 directory; no foreign process was killed.

Mechanism boundary identified before result interpretation: with at most one
prefill per iteration, cap6 has not blocked admission during the first6 decode
calls. Raising to8 at this frontier therefore corresponds to the static-cap8
continuation under this runtime's admission semantics. The new measurement is
frontier qualification and controlled response, not a new scheduling method.
