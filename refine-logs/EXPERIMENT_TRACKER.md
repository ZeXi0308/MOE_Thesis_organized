# Current Decision Tracker — JoinStream Freeze and Next-Idea Reset

> 更新时间：2026-08-10 23:05:14 +08:00  
> 当前状态：`JOINSTREAM_FROZEN_WEAKENED / NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`  
> Next Primary：`Route-Conditioned Barrier Amplification Boundary / ORACLE_FIRST / UNVALIDATED`  
> RCBA preparation：`BLOCKED_PROTOCOL_AMBIGUITY / FORMAL_GATE_NOT_RUN`  
> 本轮动作边界：协议与输入盘点后 fail closed；未运行 CPU/GPU 实验，未实现 evaluator 或下一机制。

## JoinStream evidence ledger

| Stage | Evidence | Verdict | Status |
|---|---|---|---|
| CPU exact Oracle | `artifacts/joinstream_pilot/20260810_184136/` | `SUPPORT_ACTION_SPACE / CPU_EXPLORATORY_SIGNAL` | historical evidence retained |
| Synthetic single GPU | `artifacts/joinstream_gpu_pilot/20260810_202548/` | `WEAKEN_TAX_DOMINATES / SINGLE_GPU_EXPLORATORY_MICROBENCHMARK` | historical evidence retained |
| Realistic MoE-tail | `artifacts/joinstream_real_moe_tail/20260810_205953/` | `WEAKEN_UPPER_BOUND_TOO_SMALL / GATING_INSUFFICIENT / WEAKENS` | final freeze; `0/4` safe benefit, `3/4` natural window |
| Final integrity | `artifacts/joinstream_real_moe_tail/20260810_205953/EXPERIMENT_AUDIT.md` | `PASS / P0=0 / P1=0` | complete |
| Final memo | `docs/current/JOINSTREAM_FINAL_FREEZE_2026-08-10.md` | `Paper viability: FREEZE` | authority updated |

## Next-Idea decision ledger

| Item | Status | Gate |
|---|---|---|
| Existing-pool mechanical merge | DONE | 12 candidates, no pre-jury quality filtering |
| Fresh candidate jury | DONE / same-family provisional | at most 3 candidates |
| Primary | `Route-Conditioned Barrier Amplification Boundary` | first prove charged end-to-end critical-path Oracle headroom on a near-real, identity-complete full request DAG |
| Preparation | `BLOCKED_PROTOCOL_AMBIGUITY` | common natural regime、removable barrier/dependency semantics 与 resource capacity 未唯一冻结；[RCBA tracker](EXPERIMENT_TRACKER_RCBA_ORACLE_GATE_PREPARATION.md) |
| Formal trace | OLMoE `NO` / LLM-jp `NO` | frozen manifests 和 producer code 不是 completed identity-complete artifact |
| Measured surface | OLMoE `NO` / LLM-jp `NO` | sparse single-expert / aggregate timing 不足以连接 full DAG |
| Evaluator / experiment | NOT IMPLEMENTED / NOT RUN | protocol-first fail-closed；不得计算正式 headroom |

## Historical tracker retained below

# Selector Failure Decomposition — Historical execution tracker

- `S0 DEFINE`: DONE — decomposition, heads, exact baselines, and two-branch policy decision frozen.
- `S1 IMPLEMENT`: DONE — pure-CPU runner plus 6/6 synthetic contract tests PASS; bounded code review found no P0/P1.
- `S2 LOCK`: DONE — v2 re-lock completed after the serialization-only fix and explicit finite-result checks; model, features, thresholds, inputs, and policy decision are unchanged.
- `S3 RUN`: DONE — authoritative CPU-only `run01` completed under v2 lock; primary fresh profile rank gain is negative and the frozen rule selects the supervised-selector stop branch.
- `S4 AUDIT`: DONE — fresh same-family ultra audit PASS, P0=0, P1=0; raw labels, all decisive exact fractions, predictions, V2 lock, output manifest, and failed-attempt isolation independently reproduced.

No GPU or model inference is permitted in this experiment. Both outcome surfaces predate this analysis, so all results remain retrospective and exploratory.

Final selected policy: `STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`. The positive fresh harm diagnostic is retained as a finding, not promoted into an unregistered harm-only policy.
