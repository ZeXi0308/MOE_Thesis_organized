# Mixed-step CPU/GPU attribution, frozen before execution

Question: does the long host interval aligned with 4 decode + 512 prefill tokens
mostly contain GPU execution or host-side engine delay? The arrival-swap control
showed 15/24/37 ms first intervals; synchronous request submission explained only
3–10%. A host sampling function may wait for preceding GPU execution.

Use the arrival-swap archive's original 16 requests, 128 prompt / 16 output tokens,
cap8 / engine8, BF16 OLMoE, all engine arguments unchanged. Retain its three
warmups (all-zero, steady, bursty), then run only one bursty .02 episode per
fresh process. Four processes run profiler off/on/on/off. Both modes include
the same small host boundary wrappers; only ON activates CPU+CUDA tracing.
No per-step CUDA synchronization or policy change. Preserve every result.

Record step IDs for engine/core/executor/forward/sample/D2H boundaries and
associate them with native scheduled token counts. CUDA activity correlation,
not host function names, determines GPU attribution. Nested CPU spans and
overlapping CUDA activities must not be added. Trace export occurs after the
measured episode. Profiler overhead can alter arrivals and state, so compare
actual scheduling geometry and label unmatched intervals. Do not estimate a
performance gain or exact action counterfactual from profiler-off/on differences.

Continue only if the trace closes a concrete attribution question and identifies
an actionable residual beyond known prefill/decode interference. If ordinary
prefill cost or instrumentation/state drift explains the observation, stop this
mechanism formulation; do not add another predictor. No U/C, isolated action
Oracle, quality improvement or CCF-B method claim is permitted by this probe.

These are diagnostic native request executions, with profiling perturbation
explicitly retained. Failure to trace CUDA or reproduce the relevant shape is
an incomplete diagnostic, not a scientific NO-GO.
