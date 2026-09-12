# Retained capture-aligned ladder analysis

MEASUREMENT_ONLY: 24/24 episodes; 768 request executions.

- Single model/GPU, 32 reused texts; 768 request executions are not 768 independent workloads.
- F1 fixed historical milliseconds cannot falsify the staircase; actual width exposure is explicitly reported.
- F2/F3 use historical campaign comparisons, not a contemporaneous old-ladder intervention arm.
- No step-independent significance, causal Oracle, expert-signal increment, or production-capacity claim.
- Capture padding is inferred from logged bucket sizes and actual pure-decode width, not measured HBM or executed CUDA kernels.
- Best measured static is a descriptive hindsight reference, not an Oracle or held-out deployable selector.
- Successful warmup has summaries but no retained full raw trajectories in inherited runner.

| Cell | Regime/arm | Status | Joint pass | Goodput | TTFT ms | TPOT ms | Active/decode/wait max | Pure width p50 | Padding waste | Width17–24 n / p50 ms |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| forward/cell-000 | steady/static8 | COMPLETE | 8 | 2.0300 | 820.1505 | 7.1733 | 8/8/16 | 8.0000 | 0.0213 | 0 / — |
| forward/cell-001 | bursty/static8 | COMPLETE | 8 | 2.2543 | 618.2544 | 6.7458 | 8/8/16 | 8.0000 | 0.0000 | 0 / — |
| forward/cell-002 | steady/static16 | COMPLETE | 11 | 3.8415 | 189.8823 | 8.8407 | 16/16/8 | 16.0000 | 0.0920 | 0 / — |
| forward/cell-003 | bursty/static16 | COMPLETE | 32 | 12.1721 | 49.1544 | 8.2310 | 16/16/8 | 16.0000 | 0.0014 | 0 / — |
| forward/cell-004 | steady/static24 | COMPLETE | 5 | 1.8909 | 18.3997 | 9.7638 | 24/24/2 | 14.0000 | 0.1385 | 107 / 9.2945 |
| forward/cell-005 | bursty/static24 | COMPLETE | 32 | 12.4473 | 26.2699 | 8.4138 | 24/24/0 | 16.0000 | 0.0024 | 10 / 9.2911 |
| forward/cell-006 | steady/static32 | COMPLETE | 4 | 1.5309 | 18.1054 | 9.9076 | 26/26/0 | 14.5000 | 0.1668 | 79 / 9.3226 |
| forward/cell-007 | bursty/static32 | COMPLETE | 32 | 12.4782 | 28.4688 | 8.4095 | 24/24/0 | 16.0000 | 0.0024 | 11 / 9.3668 |
| forward/cell-008 | steady/shadow32 | COMPLETE | 3 | 1.1443 | 20.0005 | 10.0340 | 27/26/0 | 15.0000 | 0.1699 | 74 / 9.3166 |
| forward/cell-009 | bursty/shadow32 | COMPLETE | 32 | 12.3990 | 26.9450 | 8.4168 | 24/24/0 | 16.0000 | 0.0024 | 9 / 9.2730 |
| forward/cell-010 | steady/feedback32 | COMPLETE | 14 | 3.9721 | 202.9971 | 8.6780 | 16/16/15 | 9.0000 | 0.2066 | 0 / — |
| forward/cell-011 | bursty/feedback32 | COMPLETE | 32 | 11.9690 | 46.4226 | 8.4413 | 24/24/8 | 16.0000 | 0.0027 | 2 / 9.5166 |
| reverse/cell-000 | bursty/feedback32 | COMPLETE | 32 | 12.0718 | 27.1650 | 8.2183 | 16/16/8 | 16.0000 | 0.0027 | 0 / — |
| reverse/cell-001 | steady/feedback32 | COMPLETE | 13 | 3.6275 | 21.8517 | 8.6431 | 17/17/15 | 9.0000 | 0.2516 | 21 / 9.2655 |
| reverse/cell-002 | bursty/shadow32 | COMPLETE | 32 | 12.5509 | 27.6883 | 8.2228 | 24/24/0 | 16.0000 | 0.0023 | 8 / 9.4522 |
| reverse/cell-003 | steady/shadow32 | COMPLETE | 4 | 1.5355 | 18.7875 | 9.8003 | 26/26/0 | 14.0000 | 0.1589 | 91 / 9.3540 |
| reverse/cell-004 | bursty/static8 | COMPLETE | 8 | 2.2441 | 620.7931 | 6.7738 | 8/8/16 | 8.0000 | 0.0000 | 0 / — |
| reverse/cell-005 | steady/static8 | COMPLETE | 8 | 2.0250 | 789.5183 | 7.2233 | 8/8/16 | 8.0000 | 0.0256 | 0 / — |
| reverse/cell-006 | bursty/static16 | COMPLETE | 32 | 12.2677 | 32.0555 | 8.1717 | 16/16/8 | 16.0000 | 0.0014 | 0 / — |
| reverse/cell-007 | steady/static16 | COMPLETE | 7 | 2.4258 | 185.4835 | 8.9318 | 16/16/8 | 16.0000 | 0.0936 | 0 / — |
| reverse/cell-008 | bursty/static24 | COMPLETE | 32 | 12.5988 | 28.8743 | 8.2637 | 24/24/0 | 16.0000 | 0.0024 | 8 / 9.2941 |
| reverse/cell-009 | steady/static24 | COMPLETE | 6 | 2.2955 | 19.7336 | 9.6635 | 24/24/2 | 14.0000 | 0.1396 | 107 / 9.2927 |
| reverse/cell-010 | bursty/static32 | COMPLETE | 32 | 12.5416 | 27.0300 | 8.3362 | 24/24/0 | 16.0000 | 0.0024 | 9 / 9.4549 |
| reverse/cell-011 | steady/static32 | COMPLETE | 4 | 1.5296 | 18.0998 | 9.9376 | 26/26/0 | 14.5000 | 0.1643 | 81 / 9.3875 |

## Within-engine feedback versus every measured static

- forward/steady: feedback 3.9721, best static ['forward/cell-002'] = 3.8415; feedback beats all static: True.
  - versus forward/cell-000: goodput change 95.6680%.
  - versus forward/cell-002: goodput change 3.3992%.
  - versus forward/cell-004: goodput change 110.0579%.
  - versus forward/cell-006: goodput change 159.4645%.
- forward/bursty: feedback 11.9690, best static ['forward/cell-007'] = 12.4782; feedback beats all static: False.
  - versus forward/cell-001: goodput change 430.9440%.
  - versus forward/cell-003: goodput change -1.6688%.
  - versus forward/cell-005: goodput change -3.8424%.
  - versus forward/cell-007: goodput change -4.0804%.
- reverse/steady: feedback 3.6275, best static ['reverse/cell-007'] = 2.4258; feedback beats all static: True.
  - versus reverse/cell-005: goodput change 79.1309%.
  - versus reverse/cell-007: goodput change 49.5405%.
  - versus reverse/cell-009: goodput change 58.0261%.
  - versus reverse/cell-011: goodput change 137.1494%.
- reverse/bursty: feedback 12.0718, best static ['reverse/cell-008'] = 12.5988; feedback beats all static: False.
  - versus reverse/cell-004: goodput change 437.9283%.
  - versus reverse/cell-006: goodput change -1.5968%.
  - versus reverse/cell-008: goodput change -4.1828%.
  - versus reverse/cell-010: goodput change -3.7456%.

## Historical F2/F3 diagnostics only

Comparisons below do not isolate ladder from campaign time or engine state.

- forward/steady: waste 0.3014 → 0.2066; F2 True, F3 True; TPOT delta ms -0.2220.
- forward/bursty: waste 0.0014 → 0.0027; F2 False, F3 False; TPOT delta ms 0.2494.
- reverse/steady: waste 0.1570 → 0.2516; F2 False, F3 True; TPOT delta ms -0.2327.
- reverse/bursty: waste 0.0387 → 0.0027; F2 True, F3 True; TPOT delta ms -0.0441.

## Source and environment

All readback execution sources match manifest and both environments: True.
Engine arguments equal: True; executed metrics equals analysis metrics: True.
Full per-request metrics, causal checks, width counts, source paths/hashes and historical environments are retained in analysis.json.
