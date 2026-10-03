# Arrival-slot swap: stage association without stable SLO failure

Verdict: **MEASUREMENT_ONLY**. The long first token interval follows the first
arrival group after swapping texts, but the SLO failure does not reproduce
consistently. These data do not establish a content-specific mechanism or a
useful scheduling action.

Evidence: native vLLM 0.26, OLMoE BF16 on one RTX 5090, 16 requests per episode,
128 prompt / 16 generated tokens, fixed engine/admission cap 8. The primary
condition is bursty arrival; steady arrival is supplementary. This is native
request-level host timing and scheduler alignment, not isolated kernel timing.

All eight episodes completed (128/128 requests). Main SLO attainment was
124/128 at TTFT <= 200 ms and mean TPOT <= 9 ms. The reference 5 s / 200 ms
SLO passed all requests. All four steady episodes passed the main SLO.

| Bursty process | First-group source rows | Mean TPOT, ms | First ITL, ms | Main SLO pass | Goodput, req/s |
|---|---|---:|---:|---:|---:|
| Original A0 | 205, 206, 210, 211 | 7.477 | 14.950 | 16/16 | 62.444 |
| Swapped B0 | 215, 219, 223, 227 | 8.992915 | 36.867 | 16/16 | 51.096 |
| Swapped B1 | 215, 219, 223, 227 | 8.244 | 23.827 | 16/16 | 57.590 |
| Original A1 | 205, 206, 210, 211 | 9.223 | 36.756 | 12/16 | 37.777 |

B0 remains below 9 ms; rounding it to a failure would change the conclusion.
The first group's maximum interval is token 1 to 2 in all four processes,
aligned with a step scheduling four decode tokens plus 512 prefill tokens.
Post-schedule waiting counts differ (0, 8, 4, 8), so equal scheduled token counts
do not imply an identical full engine state. Swapped original texts, now in
the second arrival group, have first intervals of 7.694–8.252 ms.

Summing disjoint synchronous `add_request` calls within that first interval
accounts for 0.480, 3.690, 1.403 and 3.756 ms respectively, or 3.21–10.22%.
Client submission calls therefore explain only a small part of the gap.
The remainder includes engine work and GPU execution; it is not a removable
time bound. In particular, a host sampling call can wait for earlier GPU work.
Derived values are retained in `analysis/timing_partition.json`.

The strongest control is the unchanged original ordering, repeated in fresh
processes using A/B/B/A order and identical engine arguments, source bytes,
arrival arrays and warmups. Every policy regenerates its own state. This is
an ordering control, not an exact pre-step-state intervention. No U/C telemetry,
future-information Oracle, alternative controller or isolated GPU span was
measured. No method headroom is established.

Failure category: unstable request-level threshold crossing plus unresolved
engine/GPU attribution. This rejects a claim that these four fixed texts
reliably cause the observed failures; it does not reject a batch-stage or
MoE interference family. Reopen the mechanism claim only after attribution
and controlled repetition show a stable, actionable residual beyond ordinary
prefill/decode interference.

One next experiment: a bounded profiler-off/on bursty diagnostic with the
same frozen runtime, labeling engine step, forward and sampling boundaries
without inserting per-step GPU synchronizations. Use it to distinguish GPU
execution from host waits; profiler-on timing cannot support performance gains.

Reproduction: `run_campaign.py` and the frozen `execution.tar.gz` retain the
execution contract; `analyze_swap.py` independently recomputes the main metrics
and verifies source/engine identity. Existing raw artifacts are unchanged.

Research question answered: arrival stage is associated with the long interval,
but text identity does not reliably determine SLO failure in these repeats.
