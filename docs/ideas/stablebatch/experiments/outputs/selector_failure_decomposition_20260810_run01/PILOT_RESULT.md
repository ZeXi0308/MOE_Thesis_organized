# Selector Failure Decomposition — exploratory result

## Decision: `STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`

本轮复用两套已经看过 outcome 的完整 C8 surface，仅做纯 CPU 回顾性 failure decomposition；它不能确认新 policy。

## Exact decomposition

`u=recovered-harmed`, `mu_c=mean_rank(u)`, `delta_c,r=u_c,r-mu_c`。实现用整数 `residual8=8*u-sum_rank(u)`，所有 cell 均通过零和闭合检查。

| Effect | broad 16-fold LODO | fresh transfer |
|---|---:|---:|
| Cell-head selection gain | -3.86563 | -0.439063 |
| Rank-residual ridge gain | +0.625 | +3 |
| Hierarchical profile rank gain | -1.375 | -5 |
| Harm-head exact harm avoidance | +0.625 | +7.125 |

## Fresh fixed-B outcomes

| Policy | Recovered | Harmed | Net |
|---|---:|---:|---:|
| Global matched random, exact | 4.82969 | 7.39062 | -2.56094 |
| Cell-head cells + uniform rank, exact | 8.125 | 11.125 | -3 |
| CellGate + RankResidualRidge | 8 | 8 | 0 |
| CellGate + HierarchicalProfile | 7 | 15 | -8 |
| CellGate + MinPredictedHarm | 9 | 4 | 5 |

## Head diagnostics

- Cell head fresh MSE skill: `-0.04787776222867368`; exact cell gain `-0.439063`.
- Rank-residual ridge fresh MSE skill: `-0.0047815803260538026`; rank gain `+3`.
- Profile fresh MSE skill: `-0.01997423300054857`; rank gain `-5`; positive documents `6/16`.
- Harm head fresh MSE skill: `0.023257565024297322`; exact harm avoidance `+7.125`; effective=`true`.

## Interpretation boundary

Fresh primary profile rank gain is non-positive. Under the pre-result two-branch rule, do not create a harm-only rescue policy; stop supervised selector iteration and move to WitnessPatch / budgeted probing.

任何 Hybrid 选择只代表下一套全新 document-disjoint cohort 的预注册候选；不得把本结果称为 online dynamic observability、模型质量、serving SLO 或生产证据。
