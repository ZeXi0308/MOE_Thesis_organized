# Follow-up: content variation at fixed workload geometry

Decision made after the retained cap 2/4/8 scan and cap 4/6/8 bracket, before
executing either new cohort. This is adaptive exploration, not a confirmatory
holdout or an online policy evaluation.

The bracket's four telemetry-OFF comparisons all favor cap 6 over caps 4 and 8
(goodput 1.7899–1.9553 requests/s). The earlier cap 8 ranking changed across
repeats near the mean-TPOT threshold. A stronger static cap is therefore the
current baseline. U/C vary within a batch width, but the original cohort's
visible trajectories repeat across arrival regimes and repeats. That evidence
does not establish incremental action information.

Next question: with the same prompt length, output length and arrival schedules,
does changing natural text content change the cap 6 versus cap 8 response?

- Select the next two consecutive cohorts from the existing source manifest,
  after its existing prompt-length filter: offsets 16 and 32, 16 rows each.
  Selection does not inspect routing or outcomes. These are distinct source rows;
  article-disjointness is not certified and no independent-test claim is made.
- Keep 128 prompt tokens, 16 fixed output tokens, TTFT <= 5 s, mean TPOT <= 0.2 s,
  0.1 s steady gap, burst size 4, the same 0–1.5 s arrival horizon, BF16 OLMoE
  revision, eager attention and one-prefill-per-iteration FCFS execution.
- Compare static caps 6 and 8, telemetry OFF and ON, steady and bursty, with two
  counterbalanced repeats: 16 cells per new cohort. Run cohorts sequentially.
- Keep every run. Report within-cohort cap response and each repeat; do not pool
  adjacent decode steps as independent samples or attach ON telemetry to OFF
  trajectories. OFF/ON remain actual rerun diagnostics, not an isolated tax.
- For pressure comparisons use ON trajectories and separately describe steps
  with prefill. The first full-batch no-prefill interval before any completion
  supplies a consistently selected diagnostic window, if present.
- If cap 6 continues to dominate, retain that negative evidence against a need
  for content-dependent cap selection in this regime. If rankings differ, repeat
  the affected comparison before explaining an expert-specific mechanism.

No new Controller or predictor is introduced. The source extension is a shared
deterministic cohort-offset selector, with old prepared inputs defaulting to
offset zero. This experiment cannot establish pre-action U/C usefulness,
non-preemptive drain-time signal survival, native-serving transfer or a paper GO.

## One controlled repeat after observing a sign change

The first two new cohorts completed (32 cells). In the 12 OFF comparisons across
the original bracket and new cohorts, cap 6 wins 11. The sole cap 8 win is offset
32 / bursty / repeat 1: 2.2571 versus 1.6372 requests/s. The same cohort/regime's
repeat 0 instead gives 0.9257 versus 1.3794. ON also has one different sign
change (offset 32 / steady), while the first full-batch pressure window repeats
the same U/C values. Do not interpret this as a content-dependent cap rule.

Freeze exactly one additional fresh-process execution of the unchanged offset
32 prepared inputs: all 16 cells, both arrival regimes and telemetry settings,
the same cap order reversal, workload, SLO and warmup. Retain it as
`offset32_repeat`, never replacing the first run. Start only after the currently
observed separate actuator-repeat queue (PID 3739) finishes and the GPU is free.
This repeat addresses timing/ranking stability, not a new mechanism. If signs
remain unstable, stop interpreting static ranking differences and investigate
runtime timing or obtain a representative-runtime control before policy search.
