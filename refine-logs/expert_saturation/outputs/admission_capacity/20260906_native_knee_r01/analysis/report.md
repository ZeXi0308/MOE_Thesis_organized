# Native capacity scan: MEASUREMENT_ONLY

Completed 20/20 episodes; qualified 20; completed request executions 640.

| Cell | Scale / regime / repeat | Cap | Status | Pass | Throughput | Goodput | TTFT / TPOT p50 s | Max active / decode / waiting after |
|---|---|---|---|---|---|---|---|---|
| forward/cell-000 | 1.0 / steady / 0 | 4 | COMPLETE | 4/32 | 5.5969 | 0.6996 | 1.7943 / 0.0054 | 4 / 4 / 21 |
| forward/cell-001 | 1.0 / bursty / 0 | 4 | COMPLETE | 4/32 | 5.8569 | 0.7321 | 1.6603 / 0.0052 | 4 / 4 / 20 |
| forward/cell-002 | 1.0 / steady / 0 | 8 | COMPLETE | 8/32 | 8.0306 | 2.0076 | 0.8669 / 0.0073 | 8 / 8 / 16 |
| forward/cell-003 | 1.0 / bursty / 0 | 8 | COMPLETE | 8/32 | 8.9572 | 2.2393 | 0.6308 / 0.0068 | 8 / 8 / 16 |
| forward/cell-004 | 1.0 / steady / 0 | 16 | COMPLETE | 4/32 | 11.0072 | 1.3759 | 0.2026 / 0.0090 | 16 / 16 / 8 |
| forward/cell-005 | 1.0 / bursty / 0 | 16 | COMPLETE | 32/32 | 12.1598 | 12.1598 | 0.0506 / 0.0083 | 16 / 16 / 8 |
| forward/cell-006 | 1.0 / steady / 0 | 32 | COMPLETE | 4/32 | 12.2359 | 1.5295 | 0.0199 / 0.0101 | 27 / 27 / 0 |
| forward/cell-007 | 1.0 / bursty / 0 | 32 | COMPLETE | 32/32 | 12.4789 | 12.4789 | 0.0253 / 0.0083 | 24 / 24 / 0 |
| forward/cell-008 | 8.0 / steady / 0 | 4 | COMPLETE | 32/32 | 2.5022 | 2.5022 | 0.0150 / 0.0029 | 2 / 2 / 0 |
| forward/cell-009 | 8.0 / steady / 0 | 32 | COMPLETE | 32/32 | 2.5034 | 2.5034 | 0.0151 / 0.0029 | 2 / 2 / 0 |
| reverse/cell-000 | 1.0 / bursty / 0 | 32 | COMPLETE | 32/32 | 12.6564 | 12.6564 | 0.0231 / 0.0081 | 24 / 24 / 0 |
| reverse/cell-001 | 1.0 / steady / 0 | 32 | COMPLETE | 7/32 | 12.3536 | 2.7024 | 0.0180 / 0.0097 | 26 / 26 / 0 |
| reverse/cell-002 | 1.0 / bursty / 0 | 16 | COMPLETE | 32/32 | 12.2906 | 12.2906 | 0.0391 / 0.0082 | 16 / 16 / 8 |
| reverse/cell-003 | 1.0 / steady / 0 | 16 | COMPLETE | 6/32 | 11.1322 | 2.0873 | 0.1879 / 0.0089 | 16 / 16 / 8 |
| reverse/cell-004 | 1.0 / bursty / 0 | 8 | COMPLETE | 8/32 | 9.0072 | 2.2518 | 0.6206 / 0.0067 | 8 / 8 / 16 |
| reverse/cell-005 | 1.0 / steady / 0 | 8 | COMPLETE | 8/32 | 8.1574 | 2.0393 | 0.8251 / 0.0071 | 8 / 8 / 16 |
| reverse/cell-006 | 1.0 / bursty / 0 | 4 | COMPLETE | 4/32 | 5.8372 | 0.7297 | 1.6541 / 0.0053 | 4 / 4 / 20 |
| reverse/cell-007 | 1.0 / steady / 0 | 4 | COMPLETE | 4/32 | 5.5778 | 0.6972 | 1.8094 / 0.0054 | 4 / 4 / 21 |
| reverse/cell-008 | 8.0 / steady / 0 | 32 | COMPLETE | 32/32 | 2.5050 | 2.5050 | 0.0143 / 0.0028 | 1 / 1 / 0 |
| reverse/cell-009 | 8.0 / steady / 0 | 4 | COMPLETE | 32/32 | 2.5052 | 2.5052 | 0.0142 / 0.0028 | 2 / 2 / 0 |

All cap comparisons (high relative to low); descriptive paired reruns:

| Block / scale / regime | Low → high | Status | Throughput Δ% | Goodput Δ% | TPOT p50 Δ% |
|---|---|---|---|---|---|
| 0 / 1.0 / bursty | 4 → 8 | DESCRIPTIVE_PAIRED_RERUN | 52.9341 | 205.8683 | 29.5924 |
| 0 / 1.0 / bursty | 4 → 16 | DESCRIPTIVE_PAIRED_RERUN | 107.6147 | 1560.9178 | 58.2056 |
| 0 / 1.0 / bursty | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | 113.0645 | 1604.5159 | 59.2609 |
| 0 / 1.0 / bursty | 8 → 16 | DESCRIPTIVE_PAIRED_RERUN | 35.7543 | 443.0174 | 22.0794 |
| 0 / 1.0 / bursty | 8 → 32 | DESCRIPTIVE_PAIRED_RERUN | 39.3178 | 457.2713 | 22.8937 |
| 0 / 1.0 / bursty | 16 → 32 | DESCRIPTIVE_PAIRED_RERUN | 2.6249 | 2.6249 | 0.6671 |
| 1 / 1.0 / bursty | 4 → 8 | DESCRIPTIVE_PAIRED_RERUN | 54.3064 | 208.6129 | 28.3374 |
| 1 / 1.0 / bursty | 4 → 16 | DESCRIPTIVE_PAIRED_RERUN | 110.5569 | 1584.4548 | 55.7099 |
| 1 / 1.0 / bursty | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | 116.8224 | 1634.5790 | 55.0596 |
| 1 / 1.0 / bursty | 8 → 16 | DESCRIPTIVE_PAIRED_RERUN | 36.4537 | 445.8148 | 21.3285 |
| 1 / 1.0 / bursty | 8 → 32 | DESCRIPTIVE_PAIRED_RERUN | 40.5142 | 462.0566 | 20.8218 |
| 1 / 1.0 / bursty | 16 → 32 | DESCRIPTIVE_PAIRED_RERUN | 2.9757 | 2.9757 | -0.4176 |
| 0 / 1.0 / steady | 4 → 8 | DESCRIPTIVE_PAIRED_RERUN | 43.4835 | 186.9670 | 34.6883 |
| 0 / 1.0 / steady | 4 → 16 | DESCRIPTIVE_PAIRED_RERUN | 96.6683 | 96.6683 | 67.0720 |
| 0 / 1.0 / steady | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | 118.6210 | 118.6210 | 85.9432 |
| 0 / 1.0 / steady | 8 → 16 | DESCRIPTIVE_PAIRED_RERUN | 37.0668 | -31.4666 | 24.0434 |
| 0 / 1.0 / steady | 8 → 32 | DESCRIPTIVE_PAIRED_RERUN | 52.3666 | -23.8167 | 38.0545 |
| 0 / 1.0 / steady | 16 → 32 | DESCRIPTIVE_PAIRED_RERUN | 11.1623 | 11.1623 | 11.2953 |
| 1 / 1.0 / steady | 4 → 8 | DESCRIPTIVE_PAIRED_RERUN | 46.2468 | 192.4937 | 31.2153 |
| 1 / 1.0 / steady | 4 → 16 | DESCRIPTIVE_PAIRED_RERUN | 99.5794 | 199.3691 | 63.1332 |
| 1 / 1.0 / steady | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | 121.4775 | 287.5857 | 78.4992 |
| 1 / 1.0 / steady | 8 → 16 | DESCRIPTIVE_PAIRED_RERUN | 36.4675 | 2.3506 | 24.3248 |
| 1 / 1.0 / steady | 8 → 32 | DESCRIPTIVE_PAIRED_RERUN | 51.4409 | 32.5108 | 36.0354 |
| 1 / 1.0 / steady | 16 → 32 | DESCRIPTIVE_PAIRED_RERUN | 10.9721 | 29.4675 | 9.4194 |
| 0 / 8.0 / steady | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | 0.0482 | 0.0482 | -0.4082 |
| 1 / 8.0 / steady | 4 → 32 | DESCRIPTIVE_PAIRED_RERUN | -0.0080 | -0.0080 | 0.3277 |

Finite episodes and two counterbalanced trials are not a steady-state capacity estimate or independent workloads.
Comparisons pair the same input, arrival, trial block and within-trial repeat; every cap executes its own future state.
The nine-cell SLO grid is descriptive and cannot select the canonical policy or replace the frozen main SLO.
Step histograms describe exposure; adjacent decode steps are not independent statistical samples.
queue_s is host submission lag. Native queue exposure is separately measured after scheduler admission.
Host token receipt includes capture cost. Core timestamps are used only within the core clock domain.
No dynamic policy, expert signal, Oracle, hidden-state equality or semantic quality is tested.
GPU isolation is checked at episode boundaries, not continuously; small-sample tail quantiles are descriptive.

