# RouteFidelity-EP P0-B 严格验证结果

> Phase: `calibration`  
> Verdict: **CALIBRATION_ENGINEERING_ONLY**  
> Boundary: teacher-forced logical C0/C1 records only; not backend frames, wire bytes, latency, TTFT, TPOT, or P99.

## Primary problem gate

| model | seeds regret>=5% | seed median regret | Holm lower bound | Holm p | H-P |
|---|---:|---:|---:|---:|---|
| olmoe | 11/20 | 5.87% | 0.00% | 0.3768 | FAIL |
| llmjp | 0/20 | 0.00% | 0.00% | 0.8637 | FAIL |

H-P 要求两个模型均达到 16/20 seeds regret>=5%、seed median>=5%，且 Holm-adjusted one-sided lower bound >0。任何一项失败即杀死 CCF-C 主线。

## Controls and method boundary

- `olmoe`: C0 all seeds/placements exact; S3 C1 regret=0 is constructional; bit-packed size ratio=109.85%; method gate=`NOT_RUN_BY_PROTOCOL`.
- `llmjp`: C0 all seeds/placements exact; S3 C1 regret=0 is constructional; bit-packed size ratio=106.62%; method gate=`NOT_RUN_BY_PROTOCOL`.

## Interpretation

- 若 verdict 为 `KILL_CCFC_MAINLINE`，证据表示 request-conditioned exact expert degrees 已足以在此 C1 配置池中保持 placement 决策；不能靠 architecture-only 或 P99 maximum 重新包装该主线。
- 若为 `EXACT_REPLAY_ONLY / KILL_METHOD_NOVELTY`，说明问题存在，但紧凑 S3 表示没有达到预注册的 size/decision gate。
- 只有 `PROMOTE_TO_GPU_P1` 才允许实现真实 backend adapter；即便晋级，本实验本身仍不证明系统加速。

Run validity: `True`.
