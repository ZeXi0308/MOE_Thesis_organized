# Oracle action-sweep integrity verdict

**Overall: WARN**  
**P0: 0**  
**P1: 1**

数值结果与产物完整性通过；但完全 outcome-naive 的冻结验证不成立，因此只能作为 rapid-exploration、same-family provisional 证据。

## P1

源 observable run 先完成并暴露 A0/MaxGate `-3` 与 frozen shuffle `+3`；oracle config 随后冻结并写入这两个 closure。影响：不破坏本次穷举结果或预算计算，但 frozen 只能解释为剩余 action sweep 前冻结，不能作为全候选结果未知的 confirmatory preregistration。

## Core checks

- `RUN_STATUS=COMPLETE`；manifest 12 files 的 size/hash、runner/config/lock/source bindings 全匹配。
- 独立遍历 `cell_results.jsonl` 240 rows，summary mismatch 0。
- 每 cell 均有 R/U/A0..A7；1920 candidate actions；33 positive actions 均有 confirmation。全部 action 执行后才读取 outcome 选 oracle。
- R 是显式 self-supervised proxy；scope 仅 one RTX 5090、one OLMoE revision、16 documents、240 same-cell prompt-forward cells。

## Recomputed decisive metrics

- `D(U,R)=43`
- no intervention `0`
- uniform random over all 240 cells `9/4 = 2.25`
- forced/abstaining oracle `37/37`
- action budget `33`
- remaining distance `6`
- recovery `37/43 = 0.8604651163`
- budget-matched global random `99/320 = 0.309375`
- oracle advantage over global random `11741/320 = 36.690625`
- conditional random on selected cells `39/2 = 19.5`
- oracle advantage over conditional random `35/2 = 17.5`
- positive cells/victims `33/8`
- full restoration cells `31`
- MaxGate/shuffle closures `-3/+3`
- all four frozen strong checks true

## Claim impact

支持：该 frozen single-contribution same-cell proxy action space 存在显著 hindsight oracle action value。  
不支持：online selector、serving、EP、跨模型、prevalence 或泛化。

`reviewer_model=gpt-5.6-sol`  
`reviewer_reasoning=ultra`  
`review_independence=same-family`  
`acceptance_status=provisional`
