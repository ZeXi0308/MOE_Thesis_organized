# RCBA Oracle Gate Preparation Tracker

> 更新时间：2026-08-10 23:05:14 +08:00  
> Candidate：`Route-Conditioned Barrier Amplification Boundary / PRIMARY_NEXT_CANDIDATE / UNVALIDATED`  
> Preparation verdict：`BLOCKED_PROTOCOL_AMBIGUITY`  
> Formal Gate：`NOT_RUN`  
> GPU / SSH：`NOT_USED`

| Stage | Purpose | Status | Evidence / impact |
|---|---|---|---|
| RCBA-P0-AUTHORITY | 恢复冻结状态、模型与输入来源 | DONE | JoinStream/CriticalSplit frozen verdicts unchanged；两个 model/revision 与 BCRD frozen manifests 明确 |
| RCBA-P0-PROTOCOL | 检查 regime/metric/barrier/dependency/capacity | BLOCKED | metric 已按任务合同锁定；common natural regime、barrier whitelist、full dependency semantics、capacity 和 decorrelation scope/seed 未唯一冻结 |
| RCBA-P0-TRACE | 检查双模型 identity-complete trace | BLOCKED | OLMoE NO；LLM-jp NO；manifest/code capability 不能替代 completed canonical artifact |
| RCBA-P0-SURFACE | 检查双模型 measured service surface | BLOCKED | OLMoE NO；LLM-jp NO；稀疏 single-expert/aggregate timing 不覆盖 full DAG |
| RCBA-P0-EVALUATOR | 实现 full-request DAG evaluator 与三 replay | SKIPPED_FAIL_CLOSED | 任务要求 protocol ambiguity 时不得继续实现；evaluator=false，replay=0/3 |
| RCBA-P0-TEST | 运行 14 项 contract tests/tiny replay | NOT_RUN | tests `0/14`；没有用 synthetic fixture 形成候选证据 |
| RCBA-P0-ARTIFACT | 写 preparation-only bundle | DONE | `artifacts/rcba_oracle_gate/preparation/20260810_230514/`；无 `COMPLETE.json` |

## Decision-changing blockers

1. **P0**：common natural regime 未唯一冻结；选择不同 population/cell 会改变 `J`、`H` 与 10% verdict。
2. **P0**：barrier/dependency/capacity 合同未冻结；误删真实依赖或错误并发会虚增/压低 earliest-feasible headroom。
3. **P0**：两个模型的 identity-complete trace 与 complete measured surface 均不存在；补造 identity/duration 会改变 work、tail ordering 和 verdict。

当前不授权 `RUN_ORACLE_GATE`。在允许的下一步枚举中选择 `RETURN_TO_CANDIDATE_DISCOVERY`；若继续 RCBA，须先以新的 protocol-definition qualification 明确上述合同，不能把本 tracker 当作 run lock。

