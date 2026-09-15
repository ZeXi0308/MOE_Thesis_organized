# StableBatch

## Current verdict

`STOP_PREACTION_STABLEBATCH`（2026-08-10）。

StableBatch 的数值传播现象和 hindsight action opportunity 成立，但当前系统机制不成立：在 16 个全新 document-disjoint requests、240 cells、1,920 actions 的冻结 Selectability Gate 中，Outcome Oracle reward 为 `57`，matched shuffle 为 `-4`；Static Compatibility Map 与 Online Observable Ridge 均为 `-7`，Recovered Oracle Gap 均为 `-0.04918`，LODO 均为 `0/16`。

这意味着：问题存在，但 calibration 后冻结的执行前信息没有选中值得保护的 row。原始 compatibility-aware coalescing 停止，row-conditioned 版本也不进入 planner/controller 实现；不得通过改 B、阈值、特征、seed 或新增第三个手工 selector 救活。

## What remains valid

- execution shape 的数值变化可以传播到 downstream routing；
- fresh action space 有显著 hindsight upper bound；
- 这些是 self-supervised route-proxy、单模型、单 RTX 5090 的有界证据。

它们不构成 serving、模型质量、自然 prevalence、跨模型或多 GPU/EP 结果。

## Evidence entry points

- Frozen plan：`../../../refine-logs/EXPERIMENT_PLAN.md`
- Formal run：`experiments/outputs/selectability_decomposition_20260810_run02/`
- Result card：`experiments/outputs/selectability_decomposition_20260810_run02_audit/PILOT_RESULT.md`
- Independent aggregation：`experiments/outputs/selectability_decomposition_20260810_run02_audit/INDEPENDENT_RECOMPUTE.json`
- Independent raw-route verifier：`experiments/outputs/selectability_decomposition_20260810_run02_audit/RAW_ROUTE_RECOMPUTE.json`
- Integrity audit：`experiments/outputs/selectability_decomposition_20260810_run02_audit/EXPERIMENT_AUDIT.md`

## Stop rule

不继续做 MaxGate-v2、static-map v2、feature search、partition planner、controller 或 vLLM 集成。未来若采用执行后 shadow probe / verification，它属于新的机制与新的证据链，不能复用 StableBatch 名称宣称当前 Gate 已通过。

