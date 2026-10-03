# Joint-SLO goodput as an estimator in this operating regime

64 sealed episodes. Latencies are re-decided against moved
thresholds; nothing is re-simulated and no policy is re-run.

SLO under test: TTFT <= 200 ms, mean TPOT <= 9 ms.

## Where the population sits relative to the threshold

| regime | n | TPOT p50 / SLO | requests within +/-20% of TPOT SLO | verdicts flipped by a 1% threshold move |
|---|---:|---:|---:|---:|
| steady | 20 | 0.99x | 100% | 2.0 of 32 |
| bursty | 20 | 0.91x | 100% | 0.0 of 32 |

## Does the cap create joint-SLO passes, or only relocate failures?

| campaign | engine | regime | caps | joint pass by cap | TTFT fails | TPOT fails | pass swing | TTFT-fail swing | TPOT-fail swing |
|---|---|---|---|---|---|---|---:|---:|---:|
| ired_r01 | forward | bursty | [8, 12, 16, 24, 32] | [8, 16, 32, 32, 32] | [24, 16, 0, 0, 0] | [0, 0, 0, 0, 0] | 24 | 24 | 0 |
| ired_r01 | forward | steady | [8, 12, 16, 24, 32] | [8, 7, 8, 6, 7] | [24, 20, 16, 0, 0] | [0, 14, 10, 26, 25] | 2 | 24 | 26 |
| ired_r01 | reverse | bursty | [8, 12, 16, 24, 32] | [8, 16, 32, 32, 32] | [24, 16, 0, 0, 0] | [0, 0, 0, 0, 0] | 24 | 24 | 0 |
| ired_r01 | reverse | steady | [8, 12, 16, 24, 32] | [8, 7, 7, 8, 6] | [24, 20, 16, 0, 0] | [0, 15, 13, 24, 26] | 2 | 24 | 26 |
| peat_r01 | forward | bursty | [8, 12, 16, 24, 32] | [8, 16, 32, 32, 32] | [24, 16, 0, 0, 0] | [0, 0, 0, 0, 0] | 24 | 24 | 0 |
| peat_r01 | forward | steady | [8, 12, 16, 24, 32] | [8, 11, 14, 5, 5] | [24, 20, 16, 0, 0] | [0, 1, 2, 27, 27] | 9 | 24 | 27 |
| peat_r01 | reverse | bursty | [8, 12, 16, 24, 32] | [8, 16, 32, 32, 32] | [24, 16, 0, 0, 0] | [0, 0, 0, 0, 0] | 24 | 24 | 0 |
| peat_r01 | reverse | steady | [8, 12, 16, 24, 32] | [8, 11, 7, 5, 7] | [24, 20, 16, 0, 0] | [0, 1, 11, 27, 25] | 6 | 24 | 27 |

## Reading

- steady: joint pass count swings 4 of 32 across the whole cap sweep, while TTFT failures swing 24 and TPOT failures swing 26.
- That is the signature of an exchange, not a gain: the cap moves requests
  from one failure mode to the other and leaves the joint count nearly fixed.
- With the TPOT median at 0.99x the threshold, the surviving variation is threshold noise, not scheduling signal.

