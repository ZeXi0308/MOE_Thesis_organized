# Mean TPOT versus request-max ITL: post-hoc readout

- Post-hoc description of the same 32 documents and two repeats; no independent-token inference.
- TTFT <=5s is fixed. The generation threshold applies separately to mean TPOT or maximum ITL, both inclusive.
- All observed values from both metrics and all four cells form the exact breakpoint union; none are selected for a desired winner.
- Goodput uses the full original episode duration; no stage cost is added or removed.
- This readout authorizes no new SLO, threshold or policy and contains no budget95 measurement.

| Cell | Mean-TPOT 200ms passes | Max-ITL 200ms passes | Mean-rule goodput | Max-rule goodput |
|---|---:|---:|---:|---:|
| repeat0-native32 | 32/32 | 30/32 | 1.381630 | 1.295278 |
| repeat0-safe | 29/32 | 29/32 | 1.070307 | 1.070307 |
| repeat1-safe | 29/32 | 29/32 | 1.062563 | 1.062563 |
| repeat1-native32 | 32/32 | 30/32 | 1.372738 | 1.286942 |

## All winner intervals for the common generation threshold

Intervals are left-inclusive and right-exclusive; infinity means no further observed breakpoint changes the winner. These are descriptive comparisons, not threshold recommendations.

| Repeat | Threshold interval (s) | Mean-TPOT winner | Max-ITL winner |
|---|---|---|---|
| 0 | [0.0, 0.01799420974529617) | TIE | TIE |
| 0 | [0.01799420974529617, 0.020134036652270416) | safe | TIE |
| 0 | [0.020134036652270416, 0.08188015222549438) | native32 | TIE |
| 0 | [0.08188015222549438, infinity) | native32 | native32 |
| 1 | [0.0, 0.01814244966498801) | TIE | TIE |
| 1 | [0.01814244966498801, 0.02025267761837935) | safe | TIE |
| 1 | [0.02025267761837935, 0.09172966703772545) | native32 | TIE |
| 1 | [0.09172966703772545, infinity) | native32 | native32 |

138 exact observed breakpoints and all 128 per-request values are retained in readout.json.
At 200ms, mean TPOT hides two long-paused native32 requests in each repeat; max ITL removes those two passes. Native32 still has higher full-episode goodput under both definitions.
