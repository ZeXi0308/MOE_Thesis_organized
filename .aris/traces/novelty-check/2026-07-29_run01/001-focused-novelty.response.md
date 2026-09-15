# Novelty Report

## 总裁决

结论很明确：N09、N06、N01 都不足以作为“新方法”进入资格 pilot，0/3 入选。

| 候选 | 查新分数 | 核心判断 | 最终裁决 |
|---|---:|---|---|
| N09 MaxRoute | 1.5/10 | max-plus 时间故障签名、主动输入设计、可诊断性与 probe selection 都有直接前作；换成 MoE route 是组合迁移 | ABANDON |
| N06 RECAP | 1.5/10 | 本质是 factorial interaction test、matching/common-support audit 与系统反事实实验的领域化 | ABANDON |
| N01 Frontier-Cut Bisimulation | 1.0/10 | bisimulation quotient、value-preserving abstraction、SMT certificate、witness/counterexample 均已有直接理论与工具前作；MoE cut 只是实例化 | ABANDON |

ABANDON 指放弃独立方法论文候选，不代表相关测量或工程检查毫无价值。三者都可以降级成验证工具或 characterization experiment，但不能再声称核心算法新颖。

## N09 — MaxRoute

### Atomic claims

| Atomic claim | 新颖性 | 判定 |
|---|---|---|
| 用 MoE fork-join/multi-stage max-plus 模型生成 expert/rank/link 的 E2E 时间故障签名 | LOW | max-plus timed-event 故障签名与定位已有直接前作；MoE 是目标系统替换 |
| 对 route pool 计算 k-diagnosability、不可区分类并选择最小 canary codebook | LOW | network tomography 已系统研究 identifiability、probe-matrix selection；max-plus active diagnosis 也已主动合成输入以细化定位 |
| 仅凭 E2E latency，fail-closed 地区分 expert/rank/link slowdown 与正常 queue/load skew | LOW | 一般情况下不可识别；最多输出由相同观测签名诱导的等价类 |

### 最接近工作与精确差异

| 工作 | 年份 | 直接重合 | 与 N09 的剩余差异 |
|---|---:|---|---|
| [Failure Detection and Localization for Timed Event Graphs in (max,+)-Algebra](https://doi.org/10.1007/s10626-020-00329-7) | 2021 | 用 max-plus/residuation 构造时间故障 indicator，并为每个 observable output 建立故障 signature matrix，输出可能故障集合 | 固定 Timed Event Graph，不是动态 MoE route/placement；但算法骨架已经存在 |
| [Active Diagnosis Algorithm for the Localization of Time Failures in (Max,+)-Linear Systems](https://doi.org/10.1016/j.ifacol.2022.10.354) | 2022 | 离线合成、测试和分析新输入流，使时间故障响应更可区分 | N09 从 natural route pool 选 canary；是输入约束变化，不是新诊断原理 |
| [deTector](https://www.usenix.org/conference/atc17/technical-sessions/presentation/peng) | 2017 | probe matrix coverage/identifiability 和 greedy path selection | Boolean/loss path，不是 max-plus fork-join latency |
| [NetBouncer](https://www.usenix.org/conference/nsdi19/presentation/tan) | 2019 | 主动路径探测、设备/链路假设搜索、应对不一致观测 | 网络路径语义不同，但 active hypothesis localization 已成立 |
| [GREYHOUND](https://www.usenix.org/conference/atc25/presentation/wu-tianyuan) | 2025 | 定位 slow GPU/communication link，并处理 contention、device degradation、network congestion | 训练场景且使用组件级机制，不是 natural MoE request inverse problem |
| [StriaTrace](https://www.usenix.org/conference/osdi26/presentation/wu-haonan) | 2026 | trace synchronization point、critical path，并用动态 roofline/regression/correlation 诊断推理异常 | 使用额外 tracing；说明区分 phase/rank/kernel 往往需要组件级 telemetry |

### 结构性不可识别

对 canary c，观测可写成各层可执行 branch 上 queue、service 与 communication 延迟之和，再对每层取 max。由此产生三个问题：

1. 非 critical branch 不可见：slowdown 小于 branch slack 时，max 输出完全不变。
2. 同暴露列等价：expert、承载 rank 与对应 link 若在全部 canary 中共同位于同一 critical branch，则 E2E signature 相同。
3. queue skew 可模拟故障：同一 critical chain 上的等待、expert service 与 link time 可产生相同 E2E 增量。

E2E-only 最多返回相同 signature 的 hypothesis equivalence class。增加 canary 只能在 exposure pattern 不同且能改变 criticality 时有帮助，不能突破结构等价。区分 expert/rank/link 需要 component span/counter、受控 component intervention，或强排他性故障与 queue 假设。

### 最强 kill objection 与裁决

N09 是 max-plus timed-event fault signature/localization、active diagnosis 和 tomography probe selection 在 MoE execution graph 上的直接组合；而且在 E2E-only 条件下存在结构性观测等价。

- 查新分数：1.5/10
- Recommendation：ABANDON
- 不进入资格 pilot。

仅可作为窄 characterization：在指定两个模型、冻结 route pool 和软件延迟注入下比较 max-plus 与 additive；不主张新诊断理论或真实 EP fault-type 可分性。

## N06 — RECAP

### Atomic claims

| Atomic claim | 新颖性 | 判定 |
|---|---|---|
| 拟合 route-incidence localizer 前检验 additive model adequacy | LOW | 标准模型检查和系统实验原则 |
| 用同 marginals/异 co-activation 或单 incidence difference 的 natural exact-route pairs 做 2x2 分解 | LOW | 标准 factorial design；natural matching 是 support 条件 |
| 输出模型成立的最大 route/load regime 与不可识别等价类 | LOW | overlap region、local validity 和 partial identification 均是成熟概念 |

### 最接近工作与精确差异

| 工作 | 年份 | 直接重合 | 与 N06 的剩余差异 |
|---|---:|---|---|
| [Coz](https://doi.org/10.1145/2815400.2815409) | 2015 | 真实执行中做受控性能干预，估计组件速度变化的整体因果作用 | 干预 code region，不是 route hyperedge |
| [CausalSim](https://www.usenix.org/conference/nsdi23/presentation/alomar) | 2023 | intervention 使原 trace 失效，并处理 counterfactual trace validity | 不检验 additive-vs-max，但覆盖反事实 trace validity 原则 |
| [On the Statistical Role of Inexact Matching](https://doi.org/10.1093/biomet/asac066) | 2022/2023 | exact/inexact matching、残余偏差与 misspecification | 非系统性能场景 |
| [Characterization of Overlap in Observational Studies](https://proceedings.mlr.press/v108/oberst20a.html) | 2020 | 只有 common-support 区域能无额外假设估计反事实 | N06 将 treatment/covariates 换为 route/co-activation/load |
| [Avoiding the Ordering Trap](https://www.usenix.org/conference/atc23/presentation/duplyakin) | 2023 | trial independence、carry-over 与顺序偏差 | 不编译 route pairs，但属于同类 adequacy/hygiene gate |

同 per-expert marginals、异 co-activation 不会自动控制 hidden state、prompt、gating score、KV length、batch history、queue age、共享资源状态等。要求全部匹配会使 support 极稀疏；放松会引回 confounding；人工改 route/batch 又可能改变模型路径和语义。natural exact-route matching 是可获得性假设，不是新识别方法。

### 最强 kill objection 与裁决

如果合法 natural pairs 存在，N06 是标准 exact matching 加 factorial interaction/model-adequacy test；如果不存在，它没有反事实识别能力。

- 查新分数：1.5/10
- Recommendation：ABANDON
- 不进入资格 pilot。

仅可在预声明 exact-support 子集内做 model characterization，不外推 unmatched cells，不宣称匹配生成了新的因果反事实方法。

## N01 — Frontier-Cut Bisimulation

### Atomic claims

| Atomic claim | 新颖性 | 判定 |
|---|---|---|
| canonical frontier state 推出 suffix bisimulation 和 action-ranking preservation | LOW | bisimulation quotient、state abstraction、value-preserving reduction 的直接实例 |
| machine-checkable event pairing proof 与 mismatch/witness | LOW | SMT/CEGIS certificate 和 witness/counterexample 已有直接前作 |
| sound 前提下比 full DAG 压缩至少 50% | LOW | 经验 gate，不是新定理；不能解决最坏 quotient 退化 |

### 最接近工作与精确差异

| 工作 | 年份 | 直接重合 | 与 N01 的剩余差异 |
|---|---:|---|---|
| [Bisimulation Learning](https://doi.org/10.1007/978-3-031-65633-0_8) | 2024 | 学习 state classifier/ranking functions，SMT 检查，失败返回 counterexample，成功得到 quotient | 通用 transition system；核心均已覆盖 |
| [Witnesses and Counterexamples for Timed Bisimulation](https://arxiv.org/abs/2606.16736) | 2026 | timed bisimilarity witness 和反例 | timed automata，不是 request DAG；但证书/首个不匹配并不新 |
| [Value Preserving State-Action Abstractions](https://proceedings.mlr.press/v108/abel20a.html) | 2020 | state-action abstraction 保留价值/策略的条件 | MDP，而 N01 是确定性 serving action ranking |
| [Frontier](https://arxiv.org/abs/2605.21312) | 2026 | scheduler-batch-engine loop、stateful request 与 serving dynamics | full simulation，不做 quotient；但展示 sound suffix state 的依赖面 |
| [Unity](https://www.usenix.org/conference/osdi22/presentation/unger) | 2022 | computation/parallelization/communication graph 与 theorem-proved substitution | training graph，不是 continuous serving action pair |

continuous decode/batching 的 sound state 至少要覆盖 queued/in-flight identities、remaining work、route/rows/merge readiness、batch/seal/release、resource availability、streams/communication/scheduler hidden state、future arrivals/cancellation/output length 生成状态，以及 tie-breaking 时间/随机状态。冻结 complete future ledger 时接近 full replay；对所有 future 普遍量化时 quotient 往往退化或验证爆炸。50% 自然 trace 压缩只是工程结果，不建立一般方法新颖性。

### 最强 kill objection 与裁决

N01 把成熟的 bisimulation/model reduction/counterexample generation 套到 MoE simulator；补齐 continuous serving soundness 后，cut state 很可能只是 full Markov frontier/full DAG 的另一种编码。

- 查新分数：1.0/10
- Recommendation：ABANDON
- 不进入资格 pilot。

仅可在冻结 future-arrival ledger 的小型 deterministic simulator 中报告工程性 replay 与压缩，不主张新的 bisimulation 理论。

## 资格 pilot 选择

没有任何候选足够新颖进入资格 pilot。若继续跑，只能标作已有方法在 MoE 上的 bounded characterization，不能因结果好看重新包装 novelty。

## 实际检索 query 清单

### N09

1. site:arxiv.org mixture-of-experts max-plus fault localization latency
2. max-plus fault diagnosis discrete event systems
3. site:arxiv.org fork-join max-plus fault diagnosis localization slow component
4. site:arxiv.org group testing codebook slow component diagnosis end-to-end latency
5. network tomography identifiability probe selection fault localization diagnosability
6. Active Diagnosis Algorithm for the Localization of Time Failures
7. site:usenix.org MoE inference fail-slow expert fault localization end-to-end latency
8. site:usenix.org 2023 2024 2025 GPU fail-slow inference fault localization end-to-end
9. site:arxiv.org mixture-of-experts inference reliability slow expert diagnosis 2025 2026
10. max-plus fault localization timed event graph diagnosis
11. site:usenix.org identifiability fault localization probe selection
12. site:arxiv.org mixture of experts slow expert rank link diagnosis route latency

### N06

13. site:arxiv.org mixture-of-experts route incidence additivity model adequacy counterfactual
14. site:usenix.org systems performance model adequacy counterfactual experiment factorial interaction
15. site:arxiv.org network tomography validate additive delay assumption queueing interaction
16. site:usenix.org exact matched pairs systems performance experiment natural trace counterfactual
17. site:arxiv.org causal matching exact covariate overlap positivity counterfactual support
18. site:arxiv.org MoE grouped GEMM co-activation interaction latency expert routing
19. site:proceedings.mlr.press factorial interaction model adequacy counterfactual matching overlap
20. Characterization of Overlap in Observational Studies AISTATS 2020 PMLR
21. site:usenix.org counterfactual performance systems intervention causal profiling factorial
22. On the statistical role of inexact matching observational studies 2022 DOI
23. Avoiding the Ordering Trap systems performance measurement ATC 2023 USENIX

### N01

24. site:arxiv.org MoE serving bisimulation frontier state request DAG suffix equivalence
25. site:proceedings.mlr.press bisimulation state abstraction model reduction action ranking
26. site:arxiv.org causal cut suffix equivalence discrete event simulation queueing
27. Bisimulation Learning SMT counterexample ranking functions 2024
28. site:arxiv.org trace bisimulation checker counterexample witness 2023 2024 2025
29. site:arxiv.org proof-carrying simulation event pairing certificate graph rewrite semantic equivalence
30. queueing network state aggregation lumpability bisimulation exact reduction future arrivals
31. site:arxiv.org Markov queueing state abstraction bisimulation sufficient state scheduler future arrivals
32. site:arxiv.org LLM serving simulator full request DAG future arrivals continuous batching
33. Frontier MoE closed-loop discrete-event graph 2605.21312
34. site:usenix.org Unity graph rewrite semantic equivalence OSDI 2022 DNN
35. site:proceedings.mlr.press value preserving state abstraction action ranking bisimulation

另执行了 NetBouncer、deTector、Gestalt、Coz、CausalSim、Frontier、Unity、max-plus active diagnosis 等题名/DOI 核验查询。

## 来源与评审边界

- 评审日期：2026-07-29。
- 来源：arXiv primary pages、USENIX 官方页、PMLR、Springer、ScienceDirect、Oxford Academic 和 DOI 元数据。
- 重点覆盖 2023–2026，并追溯 2015–2022 的直接经典前作。
- 某些付费论文只核验 publisher abstract/metadata；负面裁决依赖已找到的直接重合，不依赖“没有搜到”。
- 未把候选 jury 排序当作证据；未读取旧 IDEA_REPORT、refine-logs、旧裁决或 ARIS traces；未修改候选文件。
- fresh-agent；same-family provisional。
