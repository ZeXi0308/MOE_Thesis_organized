## Bridge classification

NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK

## Result table

| Condition | Recovered | Harmed | Net |
|---|---:|---:|---:|
| proxy-background same-rank | 31 | 6 | 25 |
| native-background same-rank | 0 | 0 | 0 |
| native-background matched random | 0.2500 | 0.6250 | -0.3750 |
| native-background oracle | 1 | 0 | 1 |

## Bridge metrics

same-rank bridge transfer = 0.0000; oracle bridge transfer = 0.0278; native specificity gap = 0.3750. Same-rank cells positive/zero/negative = 0/33/0; positive document coverage = 0/8.

## Direct cost

Target-MoE-stage native total latency median 5.728160 ms (p10-p90 5.444445-5.922333); native + replay 5.815904 ms (5.531952-6.009526). Paired delta 0.089520 ms (0.043164-0.133930), relative overhead 0.0157, per protected action 0.089520 ms, dummy ratio 7/8 = 87.5%. Per-recovered-route cost is reported only for the exact frozen timing subset: None ms; it is not extrapolated to the full policy.

## Mechanistic interpretation

The strict bridge executed every native expert group before issuing the selected C8 replay, so unselected rows retained their native batch shape and raw outputs. The native baseline exposed 18 downstream route opportunities across 14 cells. The replay result is classified as NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK under the pre-latency decision order. Fixed C8 remains a canonical arithmetic state rather than M1, M64, FP32, or ground truth. The measured delta includes duplicate expert compute, seven dummy rows, replay launch, replacement, and the unchanged gate/scatter/combine path.

## System implication

Canonical ShapePatch + profile/witness policy

## Scope

This is one frozen OLMoE revision on one RTX 5090 with 33 proxy-selected cells. Route recovery is not model-quality improvement, and target-MoE-stage CUDA-event cost is not TTFT, TPOT, queueing, fragmentation, lost batching opportunity, or a serving SLO. No ridge-selector outcome was used.

## Next minimal experiment

仅用 native-background trace 冻结一张 per-regime static rank map，然后在 held-out cells 上测一次 replay 转移。
