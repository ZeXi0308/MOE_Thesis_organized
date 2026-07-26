# RIC-Clean-v2 N1 Core Phase 4 Review

状态：**BLOCKED**  
范围：`n1_joint_milp.py` 与 `test_n1_joint_milp.py` 的 pure-core fixture；不含正式
pair producer、route/LUT parser 或 lock。

远端 SciPy/HiGHS 5/5 tests PASS 仅说明当前 fixture 内部自洽，不能开 Phase 5。

## Open P0

1. prior event 未完整约束在 decision time 前，current-ready release/stage 状态可伪造；
2. current action 用单一全局时钟串行累加四段 service，没有真实四资源流水/容量；
3. S/B allowlist 缺 age/slack/fairness、aggregate qdepth/port/cut/topology，且精确未来
   apply time 存在 look-ahead 风险；
4. S/B 采用 open-loop 全 schedule 相同，不是逐 observation-node nonanticipativity；
5. “独立 replay/enumeration”与 MILP 共享同一 replay/accounting，不能发现共同错误；
6. violation 的 absolute closure 与 latency budget 口径未统一；
7. flip 未硬绑定为两个指定 last-missing candidate；
8. route-derived join identity、outcome-blind 16-pair denominator、正式 artifacts/C/lock
   尚未实现。

## 已通过但不足以签字

- contribution identity 唯一性与已表示 event 的 stage precedence/capacity；
- RU-CVaR 公式；
- 三阶段 lexicographic freeze；
- solver OPTIMAL/gap 基础门；
- fix-and-resolve 基本框架。

下一步必须回 Phase 3 重写事件驱动四资源 replay、严格信息投影与动态 observation tree，
并提供真正独立的第二套会计检查。修复前不得生成 N1 lock，不得把 5/5 PASS 表述成
receiver necessity 证据。

