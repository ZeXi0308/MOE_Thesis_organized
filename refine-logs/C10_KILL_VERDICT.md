# C10 Final Verdict: KILL

> Candidate：Scheduler-Induced Popularity Endogeneity / CohortFence / HorizonFence  
> Status：`KILLED_AS_DOMINANT_PAPER_ROUTE`  
> Evidence tier：conceptual review only; `0` pilot results  
> Review：Round 1 `5.20/10 RETHINK`; Round 2 `4.60/10 KILL`  
> Acceptance：same-family provisional  
> Date：2026-07-29

## Verdict

C10 不再作为 CCF-B paper route 继续。不得实现 HorizonFence、加入 predictor/OPE/controller、改 workload/cohort/detector/horizon 或调 gate 抢救。

这不是“现象已被实验否定”。准确结论是：在投入 pilot 前，方法层已经出现终止性 novelty 与 soundness objection；所有自然现象、错误迁移比例和系统影响仍为 `UNVERIFIED`。

## Terminal reasons

1. 原 CohortFence 的 backlog conservation 与 partial identification 是 generic queue accounting / censoring；arrival-cohort attribution 不能直接判断 fast/slow action 是否错误。
2. 收缩后的 HorizonFence 变成通用 robust dominance rule：`sup feasible benefit < inf unavoidable cost`。MoE specialization 只有 future route-event cardinality bound，不足以形成方法级贡献。
3. 为保证 sound，trigger 后全部 external arrivals、action-dependent admission 和 full request-DAG fanout 必须进入最大收益上界，使其显著变宽。
4. optimized asynchronous migration 在合法状态下可以完全 overlap，completion-lateness 的 universal unavoidable cost lower bound 可为 `0`。
5. 因而存在结构性 fork：使用正经验 cost floor 会 unsound；使用 sound 的 `C_a^- = 0` 又使证书退化为 `UNRESOLVED`。
6. HorizonFence 的 veto inequality 不再使用 scheduler-induced backlog attribution，已偏离 C10 的核心因果贡献；再加入 causal sensitivity/full-DAG reachability 会构成新路线，而不是允许的一次 refinement。

## Preserved negative insight

可保留为内部审计规则，但不能包装成论文主方法：

- 对异步迁移，除非存在 state-specific、不可 overlap 的结构 witness，否则 universal cost lower bound 必须取 `0`；
- local service-time saving 不能逐 event 直接相加成 request completion/SLO benefit，因为 queue release、batch membership、barrier 与 admission 会产生 downstream amplification；
- 任何 local-window Oracle 在宣称动作收益前，必须证明 action ranking 对 full request-DAG 的 causal closure。

最后一条直接形成下一候选的 problem anchor，但不是 C10 的改名复活。

## Pivot boundary

下一候选是独立问题：**Causal-Closure Action-Ranking Certificate**。它不判断 demand 是否外生，也不提供 migration futility guard；它研究局部/截断 Oracle 的 action ranking 在遗漏 downstream causal cone 后是否仍可识别。

新候选必须重新走 novelty、method review 与 experiment plan，且仍保持 `EXPLORATORY / NOT_CURRENT_MAINLINE`。C10 的任何 threshold、workload 或预期结果不得迁移为新候选的 rescue condition。
