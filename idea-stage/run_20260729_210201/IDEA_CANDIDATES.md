# MoE Serving Idea Candidates — Run 20260729_210201

> 状态：`EXPLORATORY / UNRANKED / NOT_CURRENT_MAINLINE`  
> 生成：3个只读 lens（causal certificate、single-GPU exact semantics、route-aware reliability）  
> 机械去重：10个候选，未因主观新颖性/影响力在本阶段删除  
> 证据边界：所有 RTX 5090 实验都只能是 `QUALIFICATION_PROXY`，不得写成 EP/NCCL/RDMA/TPOT/P99 证据。

## N01 — MoE Frontier-Cut Bisimulation Certificate

- **Problem anchor**：局部/截断 Oracle 只有在 cut 上的状态足以决定全部 suffix 时才能保持 full request-DAG 动作排序；当前缺少可检查证书。
- **Key insight**：若两动作在 cut 前累计成本相同，且 cut 上的 request frontier、expert queue、batch membership、KV-ready time 与 top-k merge readiness 相同，suffix 事件可做双模拟配对。
- **Method delta**：action-conditioned frontier canonicalizer + trace-bisimulation checker；成功时输出逐事件证书，失败时返回首个边界不匹配与可重放 witness。它不是另一个 what-if simulator。
- **MoE specificity**：top-k route、dispatch/combine fork-join、expert queue、rebatching 和 token merge 形成离散同步边界。
- **Minimum falsification**：在两模型小型多层/多步 DAG 上 exhaustive replay assignment/hold/release。任意 false certificate 立即 KILL；若自然资格 trace 都必须把全部未来放入 cut state，相对 full DAG 压缩 <50%，停止独立方法。
- **Formal proof needed**：native continuous decode、identity-complete all-arrival ledger、真实 EP dispatch/combine/NCCL/queue timing。
- **Closest/collision**：Frontier、CausalSim、Vidur、Unity；新意必须是可检查 suffix-bisimulation proof，而不是完整图仿真。
- **dedup_key**：`moe_frontier_cut_bisimulation_certificate`

## N02 — Non-Anticipative MoE Oracle Leakage Certificate

- **Problem anchor**：离线 placement/admission/batching Oracle 可能读到未来 route、output length 或尚未到达请求，使 clairvoyant gain 与可实现 headroom 混合。
- **Key insight**：用决策时信息集把“同 prefix/不同 hidden suffix”分成不可区分类；若 Oracle 在类内选不同动作，排序依赖未来信息。
- **Method delta**：proof-carrying decision ledger 记录读取字段/时间/provenance；paired-future completions 生成 leakage witness；输出 clairvoyant 与最优 non-anticipative action class 的 regret interval。
- **MoE specificity**：未来 expert identity 逐层、逐 decode step 才显现，placement/replication 对这些隐藏 route 尤为敏感。
- **Minimum falsification**：冻结 decision timestamp，依次遮蔽 next-layer route、future decode route、output length、future arrivals。若所有自然 cell 动作一致率 >95% 且净 headroom 差 <5%，或 ledger 检不出人工 future field，停止。
- **Formal proof needed**：真实 scheduler 字段读取审计、continuous decode、未完成/取消请求 denominator，多卡 SLO 影响。
- **Closest/collision**：Director、Mixture-of-Experts Serving、一般 online scheduling；差异必须是对任意 Oracle 生成 non-anticipativity witness，不是新 predictor。
- **dedup_key**：`non_anticipative_moe_oracle_leakage_certificate`

## N03 — Partially Identified MoE Action Dominance under Missing EP Costs

- **Problem anchor**：单卡能观测 route/local compute，却无 NCCL、dispatch/combine、migration 与跨-rank queue cost；点估计会把 proxy 写成多卡事实。
- **Key insight**：把未观测 EP component 表示为满足物理约束的可行集；只有 A 在所有可行成本实现中均优于 B 才认证 dominance，否则输出 `UNIDENTIFIED` 与两个反排序 witness。
- **Method delta**：route-conditioned symbolic event ledger + robust partial-order solver；还要给出“使排序首次可辨识”的最小额外测量集。
- **MoE specificity**：消息量由 top-k route、expert coalescing、replica assignment 与 combine fan-in 共同决定。
- **Minimum falsification**：用隐藏的合法 synthetic EP cost 只测 solver coverage。任意错误 dominance 立即 KILL；若所有动作都永远 `UNIDENTIFIED` 且无稳定最小测量集，停止。
- **Formal proof needed**：真实 optimized EP 的 topology/NCCL/migration/overlap/queue-spillover 测量以收窄集合，外加 full request-DAG。
- **Closest/collision**：Frontier/Vidur 点式仿真、Gimbal/Director 点式成本模型；风险是过于保守而始终无结论。
- **dedup_key**：`partially_identified_moe_action_dominance_missing_ep_costs`

## N04 — MoE Action-Ranking Stability Radius and Minimal Reversal Witness

- **Problem anchor**：即使 full-DAG replay 给出 winner，service noise、expert heterogeneity、batch threshold 与少量 route 变化也可能翻转排序。
- **Key insight**：固定 event order 内动作差是局部仿射成本，而 event-order/batch/top-k merge 改变形成分段边界；可求到最近的排序反转扰动。
- **Method delta**：piecewise event-order certificate + SMT/MILP minimal reversal search；输出 stability radius、critical edges 和两次 exact replay 可复核的 before/after witness。
- **MoE specificity**：rows service、sparse route、fork-join、coalescing threshold 会产生离散 event-order jump。
- **Minimum falsification**：对小型 full-DAG 动作对计算 radius，用 exhaustive 邻域验证最近 sign flip。任意漏检反转或 witness 不可重放即 KILL；若 witness 几乎总是完整 trace，无因果压缩，停止。
- **Formal proof needed**：双模型重复 service 测量、natural route variation、all-arrival denominator 与多卡 EP variance。
- **Closest/collision**：Frontier/CausalSim 参数敏感性与通用 robust optimization；差异必须是最小语义保持反转 witness。
- **dedup_key**：`moe_action_ranking_stability_radius_minimal_reversal_witness`

## N05 — ExpertCover: Exact-Route Canary Set Compiler

- **Problem anchor**：生产流量可能使某些 expert 长期共路由或低覆盖，即使 latency 无噪声也无法定位 silent slowdown。
- **Key insight**：natural cached-decode state 会生成稀疏 expert-set codeword；从真实 prompt/decode states 中选一个冻结小集，最大化覆盖、pairwise separation、冗余与 conditioning。
- **Method delta**：active-canary compiler；输出 per-layer/expert coverage、indistinguishable classes、minimum distance、leave-one-canary-out certificate 和 fail-closed measurement plan。
- **MoE specificity**：codeword 来自学习到的 top-k route，且随 layer/decode state 变化；OLMoE top-8/E64 与 llm-jp top-16/E32 的密度显著不同。
- **Minimum falsification**：双模型 document-disjoint discovery/test；编译多个 budget 的 canary set；冻结 5/10/20% 软件 expert delay。若 held-out route 不稳定、关键 expert 不覆盖、pair 不可分，或看 test 后才能成功，停止。
- **Formal proof needed**：real optimized EP、natural continuous batching、expert-worker/rank/link 真实 slowdowns、canary isolation 与 SLO 开销。
- **Closest/collision**：NetBouncer/deTector 和通用 set cover；若只是 set cover 则无新意。
- **dedup_key**：`exact_route_active_canary_codebook`

## N06 — RECAP: Route-Equivalent Counterfactual Additivity Probe

- **Problem anchor**：route-incidence localizer 通常假定 request residual 是 degraded expert cost 的可加和；grouped kernels、batching、barrier 与 co-activation 可使该假设错误。
- **Key insight**：exact cached-decode route 可编译“每-expert 边缘分布相同但 co-activation 不同”或“仅差一个 expert incidence”的配对 counterfactual。
- **Method delta**：不拟合 NNLS，而是生成 route-equivalent 2x2 反事实，分解 expert main effect、pairwise co-activation、rows 与 barrier/max effect，输出 inverse model 成立的最大 regime 与外部等价类。
- **MoE specificity**：top-k hyperedge、可变 rows 和 grouped execution interaction；top-8 对 top-16 构成两个密度 regime。
- **Minimum falsification**：同 KV/BF16 exactness 下注入 5/10/20% expert delay，在 sealed test 比较 additive、max/barrier、interaction model，带 route permutation/zero-delay 负控。若有效 counterfactual 太少、噪声淹没效应，或交互项在所有 regime 均主导，停止。
- **Formal proof needed**：real EP dispatch/grouped GEMM/combine/network queue/rank barrier 下的冻结复现。
- **Closest/collision**：compressed-sensing tomography、GEM service curves、一般 factorial causal experiment；新意必须是 exact-semantics model-adequacy boundary。
- **dedup_key**：`route_equivalent_counterfactual_additivity_audit`

## N07 — RouteSyndrome: Sparse Expert Localization for Silent Compute Corruption

- **Problem anchor**：expert 可以在无 error/无 latency spike 时返回错误数值；全量双路 inference 贵，而 aggregate output check 不能定位 expert。
- **Key insight**：冻结 same-KV canary 提供 exact BF16 reference contribution；重叠 route incidence 可构成 syndrome code，用低维 keyed sketch 定位被破坏 expert。
- **Method delta**：带最小 syndrome distance 的 natural-canary compiler + combine-boundary contribution sketch + 有界单/多 expert corruption decoder，报 collision 与 minimum-detectable-magnitude guarantee。
- **MoE specificity**：稀疏 route 提供重叠 expert incidence，并将 corruption 限制在命中该 expert 的 token/request。
- **Minimum falsification**：双模型 sealed canaries，注入 bit flip、scale/sign、block error、stochastic noise，扫 sketch width/budget。若有价值的小幅 corruption 系统性逃逸、tail expert 不可定位，或 sketch 接近 full-output cost，停止。
- **Formal proof needed**：real fault model/故障硬件、真实 EP compute/transport/combine boundary、优化 checksum kernel 与 serving overhead。
- **Closest/collision**：ABFT/checksum、SDC replay、group-testing syndrome；若只是对黄金输出做 hash 则无新意。
- **dedup_key**：`route_coded_silent_corruption_syndrome`

## N08 — TraceSeal: Exact-Semantics Observer-Effect Certificate

- **Problem anchor**：route tracing 可能改变 launch order、sync、batching 或 timing，创造它声称观测的 5% anomaly。
- **Key insight**：native/instrumented cached decode 可共享冻结 input/KV frontier，同时证明 semantics non-interference 和 exact-route/load 条件下的 timing non-interference。
- **Method delta**：telemetry ladder（route IDs、weights、timers、sketches）+ paired randomized ABBA + hash-closed BF16 parity + per-route observer-tax interval；输出可辨别最小效应预算。
- **MoE specificity**：telemetry 量按 layer/top-k 增长，hook 位于 sparse dispatch/expert/combine boundary，可改变 expert batching。
- **Minimum falsification**：双模型 request-disjoint ABBA，要求 logits/routes/weights/argmax/KV exact，并检验 5% 已知 delay 能否与 observer tax 分离。任意语义不等，或 tax upper bound 与 target effect 重叠，停止。
- **Formal proof needed**：在真实 EP backend 重新证明 tracer 对 NCCL/RDMA/CUDA graph/rank sync/continuous batching 的非干扰。
- **Closest/collision**：Dapper、StriaTrace、GREYHOUND overhead study；若只是 overhead benchmark 则不成立。
- **dedup_key**：`exact_semantics_route_telemetry_observer_certificate`

## N09 — MaxRoute: Max-Plus Route-Probe Diagnosis for Slow MoE Components

- **Problem anchor**：低于 timeout 的 slow expert/rank/link 会被正常 route skew 和 batch 变化混淆；additive path tomography 不符合每层 parallel fan-out/join 的 critical-path 语义。
- **Key insight**：对每个候选 fault hypothesis（expert compute、rank compute、link），由 exact route tensor、rows service 与 placement 生成一个 max-plus latency signature；只有 hypothesis signatures 在噪声集合下可分时才允许诊断。
- **Method delta**：MoE-specific max-plus hypothesis compiler + k-diagnosability certificate + minimal natural-canary codebook + fail-closed decoder。它先认证可辨识，再定位，并输出不可区分 fault classes；不用单一 NNLS。
- **MoE specificity**：每层是并行 expert/rank 的 max/join，层间和 decode 步是因果串联，rows、coalescing 与 placement 改变 hypothesis signature。
- **Minimum falsification**：先在两模型真 route 上冻结 64/128 个 canary 候选池，用 sealed test 验证 route stability 与 hypothesis separation；再注入 5/10/20% 单 expert 软件延时，检查 top-1/top-k localization 与置乱负控。若两模型都无法产生非平凡可辨识 codebook，或 max-plus 相对 additive 没有减少错误定位，停止。
- **Formal proof needed**：real EP 的 component spans/service curves、compute/link/rank 分类注入、continuous batching、TPOT/P99/goodput 和与 direct probe/StriaTrace/Tarragon 对比。
- **Closest/collision**：NetBouncer/deTector、delay tomography、GREYHOUND、StriaTrace；关键风险是 max-plus compiler 仍可能被审稿人视为领域迁移。
- **dedup_key**：`max_plus_moe_route_probe_diagnosis`

## N10 — DeltaRoute: One-Component Differential Canary Compiler

- **Problem anchor**：端到端 latency 中 attention、launch、KV、共享队列与温漂往往比 5–10% expert slowdown 更大，直接回归 route incidence 容易混淆。
- **Key insight**：从 natural cached-decode pool 中编译 matched canary pairs：在 token/KV length、大部分层路由、rows 与并发形状上匹配，但仅对目标 layer-expert/rank 的暴露程度显著不同；对内差分取消共享路径噪声。
- **Method delta**：constrained natural-state matcher + imbalance-bounded differential estimator + pair-separation certificate；对找不到合法 pair 的 expert 显式输出 `UNMEASURABLE`。
- **MoE specificity**：匹配对的处理变量是 layer/expert route exposure 和 routed rows，而非一般 prompt similarity。
- **Minimum falsification**：在两模型 discovery split 编译 pairs，test split 锁定；用 zero-delay、route-label permutation、非目标 expert delay 和目标 5/10/20% delay 负/正控。若覆盖 <50% expert、matched residual 仍与共享负载显著相关，或 false localization >5%，停止。
- **Formal proof needed**：真实 EP 中 matched pair 对 dispatch/combine/network/background load 的平衡证据和生产开销。
- **Closest/collision**：matched-pair causal inference、active diagnosis、canary testing；新意需来自 MoE route-constrained compiler 与 fail-closed measurable-set certificate。
- **dedup_key**：`one_component_differential_route_canary`

## 机械合并与预算 Gate

- 10个 `dedup_key` 均不同；N05/N06/N09/N10 为同一 reliability 问题的不同方法单元，但无评审前不主观融合或删除。
- 10/10 均可在 CPU + 至多 1×RTX5090、2小时/候选内做资格性证伪；正式结论所需多卡证据已分开标记。
- 所有候选均允许 `NO-GO`、`UNIDENTIFIED` 或 `UNMEASURABLE`；不允许在观察失败后更换 workload、threshold、denominator 或添加 rescue controller。

