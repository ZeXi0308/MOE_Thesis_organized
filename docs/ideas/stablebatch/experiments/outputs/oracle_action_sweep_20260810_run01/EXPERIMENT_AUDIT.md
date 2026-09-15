# Oracle Action Sweep Experiment Audit

**Date**: 2026-08-10  
**Auditor**: GPT-5.6-Sol ultra（fresh same-family、read-only、provisional）  
**Project**: StableBatch single-contribution oracle action sweep  
**Overall Verdict**: `WARN`  
**Integrity Status**: `warn`  
**P0 / P1**: `0 / 1`

## Checks

### A. Ground Truth Provenance: PASS

`R` 是显式标注的 all-M1 `self_supervised_proxy`，不是 dataset ground truth。结果只解释为 retrospective route-agreement upper bound。

### B. Score Normalization: PASS

使用 raw signed reward、raw route distance 与精确分数；没有用模型自身 max/min/mean 对分数归一化。

### C. Result Existence and Binding: PASS

- `RUN_STATUS.json` 为 `COMPLETE / scientific_result_eligible=true`。
- manifest 中 12 个文件的实际 size/hash 全匹配。
- runner、config、frozen lock 与 source hashes 和 `run_request.json`/`static_bindings.json` 一致。

### D. Decisive Path Execution: PASS

- 240 cells 每个均有 `R/U/A0..A7`。
- 8 个 candidate ranks × 240 cells = 1920 candidate actions。
- 33 个 positive oracle actions 均有 repeat confirmation。
- runner 先执行所有 arms，之后才从 outcome 选择 best rank/oracle。

### E. Scope and Freeze: WARN

源 observable run 已暴露 A0/MaxGate `-3` 与 frozen shuffle `+3`，oracle config 随后冻结并写入这两个 closure。该事实不影响完整 action sweep 的执行与聚合数值，但禁止声称“全候选 outcome-naive preregistration”。

证据只覆盖 one OLMoE revision、单 RTX 5090、16 documents、240 same-cell prompt-forward cells。

### F. Evaluation Type

`self_supervised_proxy + retrospective_oracle_upper_bound`

## Independent Recompute

fresh reviewer 直接遍历 `cell_results.jsonl` 240 行，没有调用 summary 聚合函数。结果与 `summary.json` 0 mismatch：

- `D(U,R)=43`
- forced/abstaining oracle `37/37`
- action budget `33`
- recovery `37/43 = 0.8604651163`
- global budget-matched random `99/320 = 0.309375`
- conditional budget-matched random `39/2 = 19.5`
- oracle advantages `36.690625 / 17.5`
- positive cells/victims `33/8`
- full restorations `31`
- MaxGate/shuffle closures `-3/+3`

## Claim Impact

- “当前 single-contribution action surface 存在强 hindsight oracle value”：**supported with retrospective/proxy qualifier**。
- “MaxGate 失败不能推出 action space 无效”：**supported**。
- “已经找到 online selector”：**unsupported**。
- serving、quality、EP、跨模型或 prevalence：**unsupported**。

## Action Item

只需修正证据表述；不修改结果、不调整 threshold、不重跑 rescue experiment。

`reviewer_model=gpt-5.6-sol`  
`reviewer_reasoning=ultra`  
`review_independence=same-family`  
`acceptance_status=provisional`

