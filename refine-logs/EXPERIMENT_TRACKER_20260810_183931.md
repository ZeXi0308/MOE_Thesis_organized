# D10 Fixed-C8 Continuous-Decode Gate 跟踪器

> 更新时间：2026-08-10 20:44:45 +08:00  
> 对应计划：`refine-logs/EXPERIMENT_PLAN_20260810_183931.md`  
> 状态：`COMPLETE / NO_GO_D10_HEADLINE_COST / INDEPENDENT_RECOMPUTE_PASS`

| Milestone | 状态 | 完成条件 |
|---|---|---|
| D10-C0 protocol | PASS | 冻结三臂协议与 claim ceiling 未变 |
| D10-C1 implementation | PASS | runner/config/test 已实现；未修改 correctness run02 |
| D10-C2 local qualification | PASS | targeted tests、py_compile 与 source hash closure 通过 |
| D10-C3 GPU preflight | PASS | RTX 5090、模型、workload 与 fresh output closure 通过 |
| D10-C4 GPU execution | PASS | 1 warmup + 2 measured repeats × 3 arms 完整结束 |
| D10-C5 independent audit | PASS | raw ledgers 独立重算 PASS |
| D10-C6 authority update | PASS | 本 tracker、`docs/current`、SemanticFence README 与 MANIFEST 已更新 |

## 冻结判定摘要

- C8 policy 内任一 raw/route/final mismatch：`NO_GO_D10_C8_CURRENT_STACK`。
- correctness 通过，但 C8 expert GPU time 未比 serial-M1 低 20%，或 token-step p99 比 native 高超过 5%：`NO_GO_D10_HEADLINE_COST`。
- 两个成本门均通过：`PROVISIONAL_SUPPORT_VS_SERIAL / EXTERNAL_BI_OPEN`。
- 官方 vLLM BI 当前不可执行；不得用替身补数，也不得声称完成完整四臂 Gate。

## 最终结果

- fixed-C8 within-policy repeat：raw / route / final-logit mismatch 均为 0。
- fixed-C8 / serial-M1 expert GPU-time ratio：0.8491007，失败于 <=0.8 门。
- fixed-C8 / native token-step p99 ratio：1.4693813，失败于 <=1.05 门。
- fixed-C8 padding fraction：0.7740559。
- 最终裁决：`NO_GO_D10_HEADLINE_COST`；correctness 证据保留，但 universal fixed-C8 headline cost claim 停止。

## Evidence ledger

- 上一道 correctness authority：`docs/ideas/stablebatch/experiments/outputs/shape_lane_correctness_20260810_run02/`
- 冻结 workload：`docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json`
- COMPLETE GPU result：`docs/ideas/stablebatch/experiments/outputs/shape_lane_continuous_cost_20260810_run02/`
- independent recompute：`docs/ideas/stablebatch/experiments/outputs/shape_lane_continuous_cost_20260810_run02_audit/INDEPENDENT_RECOMPUTE.json`
