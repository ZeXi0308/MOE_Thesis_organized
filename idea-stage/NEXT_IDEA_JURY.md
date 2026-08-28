# Next-Idea Jury — Oracle-First Reset

> 最新冻结快照：[2026-08-10 22:28:05](NEXT_IDEA_JURY_20260810_222805.md)  
> 范围：只重筛本地已有候选池；未实现、未实验、未联网扩展 prior art  
> 评审：GPT-5.6-Sol fresh jury；`same-family / provisional`  
> 硬门：设计任何 GPU action/runtime 前，先在 near-real serving DAG、trace 或 workload model 上证明 material end-to-end critical-path Oracle upper bound。

## 裁决

`PRIMARY_NEXT_CANDIDATE = Route-Conditioned Barrier Amplification Boundary`

它是 Oracle-first 研究问题，不是机制实现。正结果出现前，不设计 scheduler、stream、priority、polling、notification 或 controller。

## 1. Route-Conditioned Barrier Amplification Boundary — PRIMARY

- **精确定义**：固定 arrivals、work 与自然 top-k routes，在 identity-complete request DAG 上比较 real barrier、no-barrier Oracle 与 route-decorrelated control，只归因未被 overlap 隐藏的 request-completion/SLO 差异。
- **当前证据**：`未验证`；缺 formal Gate-1 phenomenon、完整 denominator 与 full-request-DAG，局部单卡 expert 数据不能证明 barrier 暴露。
- **最大 novelty 风险**：退化为已有 load-imbalance/barrier profiling；必须证明跨模型边界不能被 max-load、CV 或简单 rank-tail 指标解释。
- **最小 Oracle Gate**：两个模型、自然 continuous-decode arrivals、完整 request/layer/step identity、实测 service surface；在相同 work 下比较 real-barrier、no-barrier 与 decorrelated replay。
- **正结果**：共同自然 regime 保留 `>=10%` charged full-request Oracle headroom，且 MoE route/barrier 变量在简单 max-load/CV 之外仍有稳定解释力；只授权随后寻找最小合法动作。
- **负结果立即冻结**：所有共同自然 cells `<5%`，或 actionable mass `<20%`，或 max-load/CV 在 holdout 达到同等预测能力；冻结 standalone direction，不换 synthetic skew、模型或 denominator。
- **资源**：CPU full-DAG replay；1×RTX 5090 仅采 fresh routes/service surfaces；正式 EP/通信结论才需要 8×A100。
- **综合评分**：`(8, 9, 8, 5, 7, 5, 7, 9)`，`7.25/10`。

## 2. SemanticFence-v2 End-to-End Qualification

- **精确定义**：不训练 selector，先在 fresh、document-disjoint 自然请求上计算 future-known、semantics-preserving M1/M2 assignment 相对 all-M1/native 的 charged full-request Oracle。
- **当前证据**：natural exact-M2 只有 `3.4034%` expert-stage projection；`26.2038%` 来自 enriched reused-calibration semantic shadow；不是 natural prevalence、fresh generalization 或 serving speedup。
- **最大 novelty 风险**：与 safe batching、numerical verification、verify/rollback、selective repair 强碰撞；没有执行前 certificate 时会退化为昂贵 shadow/replay。
- **最小 Oracle Gate**：fresh pre-outcome、两模型 continuous-decode trace 与完整 request DAG；比较 all-M1/native 和 future-known semantics-preserving assignment，计入 wait、packing、dispatch、fallback 与 semantic-check 成本。
- **正结果**：sealed route/top-k 或更强语义契约下仍有 `>=10%` full-request headroom；只证明 opportunity。
- **负结果立即冻结**：净 headroom `<5%`、无 natural eligible rows，或收益只存在于 enriched/reused calibration；不再换 selector、阈值或 M。
- **资源**：CPU full-DAG replay + 1×RTX 5090 fresh qualification；多卡仅用于后续 serving validation。
- **综合评分**：`(5, 6, 8, 3, 8, 7, 7, 9)`，`6.63/10`。

## 3. Exact Action-Surface Existence Certificate

- **精确定义**：把 assignment、bounded seal、admission、release 编译为“可见状态—可改对象—删除依赖—下游传播—完整成本”，合并等价动作后在完整 request DAG 上证明合法净空间是否存在。
- **当前证据**：`未验证`；只有 fail-closed harness/CPU correctness asset，BCRD Gate 2 仍为 `INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`。
- **最大 novelty 风险**：与 scheduling DSL、causal simulator、action-equivalence tooling 接近，也可能只形成 infrastructure work。
- **最小 Oracle Gate**：在一个 near-real identity-complete full DAG 上精确枚举合法动作，闭合 dependency-deletion certificate、replay、non-anticipation 与全部成本。
- **正结果**：至少一个 non-anticipative、semantics-preserving action class 保留 `>=10%` net full-DAG headroom；之后只选择该最小类。
- **负结果立即冻结**：所有动作类 `<10%`，或收益依赖 future leakage、zero cost、未闭合传播或语义改变；返回 `ACTION_SPACE_ABSENT`。
- **资源**：CPU exact enumeration + 1×RTX 5090 service qualification；正式通信成本才需要 8×A100。
- **综合评分**：`(7, 9, 6, 4, 6, 5, 6, 10)`，`6.63/10`。

评分顺序：end-to-end Oracle headroom、critical-path relevance、MoE-specific novelty、collision safety、verifiability、2–4 周 kill 成本、system/paper story、cheap freeze。

## 排除、Gate blocker 与非阻塞风险

CriticalSplit 同构、JoinStream gate/priority/polling/stream/notification 变体全部排除；optimized EP return-path 只保留为 8×A100 qualification measurement。

- **P0**：当前没有已验证的 identity-complete full-request-DAG evaluator；它会使 Oracle Gate 无法正确运行。若两周内不能在不引入 proxy denominator 的条件下闭合，改判 `NO_QUALIFIED_NEXT_CANDIDATE`。
- **预冻结 negative condition，不计当前 P1**：若 barrier Oracle 被 max-load/CV 完全解释，冻结 Primary，再评估 Exact Action-Surface Certificate。
- **P2**：本轮按约束未联网做全面最新 prior-art 复核；若未来发现同 action/counterfactual/claim 的直接碰撞，再取消 Primary。本项不阻塞当前方向筛选。

本 jury 只授权 `PREPARE_ORACLE_GATE`，不授权实现或运行实验。
