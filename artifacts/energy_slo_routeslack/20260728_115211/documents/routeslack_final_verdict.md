# RouteSlack-MoE 最终裁决

## Verdict

MEASUREMENT_ONLY

当前形成了可复现的 measurement contract、batch-size-one cached-decode development capture、identity conservation、Oracle/online 隔离和 fail-closed dry-run；但 physical sample=0，不能证明或否证真实 RouteSlack Energy–SLO headroom，也不具备 8×A100 候选资格。

## 核心假设状态

- H1 Energy variation：`[Blocked]`。0/2 冻结模型完成 natural continuous-batching service-energy surface；raw energy/latency CI=N/A。
- H2 Actionability：`[Blocked]`。fixed replica、bounded seal、tier transition 和 dispatch ordering 未在真实 deadline/EP executor 上执行。
- H3 Residual headroom：`[Blocked]`。conservative Oracle net saving=N/A；formal strongest-simple CaptureRatio=N/A。

没有 H1–H3 的正结果，也没有足够物理数据触发 actionability<20%、Oracle<10% 或 simple capture≥90% 等科学 KILL threshold。

## 为什么不是 KILL

Gate 0 FAIL 来自 continuous batching/executor/meter 未闭合及当前 CUDA/NVML 硬件不可用，而不是 H1–H3 已被物理反证。把“没有测到”写成负结果会制造不存在的科学结论。

这不授权继续实现 controller：Gate 0 的硬停止仍禁止 Experiments A–E 和 Gate 4。

## 为什么不是 8xA100_CANDIDATE

- 两模型 AND：0/2；
- natural workload phenomenon：未测；
- Oracle net ≥10%：N/A；
- strongest simple capture <90%：N/A；
- physical accounting：未通过；
- controller overhead budget：未测。

## 可保留内容

仅可作为 methodology/characterization：

- prefill + KV + one-token decode 的 development correctness 与 v2 route schema；
- exact contribution identity 和四阶段 conservation assertion；
- cache/accounting/thermal/fallback/Oracle isolation 的 CPU contracts；
- 10 baselines + Oracle 的接口级 dry-run；
- manifest、raw artifacts、test logs、SHA-256 和 fail-closed verdict 流程。

不得把 tiny model、synthetic cost、60% fixture CaptureRatio、CPU no-op 数字、测试通过数或历史 5090/FP8 characterization 写成 RouteSlack 系统收益。

## Gate table

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

Gate 1–3 的 FAIL 表示因 Gate 0 顺序停止而未达到 PASS，不是相应物理假设已失败。

## Artifact

主 evidence bundle：`artifacts/energy_slo_routeslack/20260728_114917/`。它记录 95 个 protocol-critical tests、tiny development capture、GPU fail-closed probes、synthetic dry-run、host-only no-op raw trials、环境、命令、git diff 与文件 hashes；`formal_result=false`。`20260728_114933/` 只补充 BCRD `SMOKE_ONLY` 全链。

## 最终动作

保留为 characterization。
