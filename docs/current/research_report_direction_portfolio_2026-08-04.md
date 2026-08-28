# MoE 毕设探索方向全景整合与筛选

> 日期：2026-08-04
> 文档定位：跨全部 formulation 的组合审计与优先级筛选，**不是新的实验结果，也不自动改写Gate 顺序**。
> 状态标签：`PORTFOLIO_TRIAGE / NO_NEW_EMPIRICAL_RESULT`
> 与权威文档的关系：证据引用一律以 [`docs/current/README.md`](README.md) 和各方向 sealed verdict 为准；本文只做归并、去重和排序建议。若要把本文的排序变成执行顺序，必须显式写回 `README.md`。

## 执行摘要

仓库目前累计探索过约 30 个 formulation，可归为六族：receiver/return-path、energy-SLO、expert-pressure 调度（BCRD/DEPA）、精度与质量分配、诊断与可靠性、以及安全与数值一致性。截至今天没有任何一个取得系统级 GO；已形成的正式结论几乎全是窄负结果。

原定两条主线的阻塞性质完全不同，而且其中一条的诊断需要纠正。Receiver-awareness 的阻塞是硬件加已证明的效应上界：fixed RankLane 在冻结域内的最乐观 E2E 改善只有 4.1667%，要越过 5% 需要原始暴露 return fraction 达到 23.5294%，而这个分母单卡不可测，必须真实 8×A100。Energy-SLO 的阻塞**不是数据集不够**：WikiText-103 raw test（revision `b08601e`、Arrow SHA `2b8a3efa…`）与 BurstGPT 真实到达 trace（commit `d895a53`、CSV SHA `46fc9480…`）都已按行冻结并绑定哈希，两个模型 revision 也已固定；真正的阻塞是物理计量失败——formal physical strategy 的 latency 与 energy 样本数都是 0，NVML 299 点探针的最大采样间隔 10.925 ms、窗口内温度从 35°C 升到 58°C（Δ23°C）直接让 thermal gate FAIL，密封复跑的 32 个窗口只有 14 个有效（12 个因gap、6 个因 thermal 作废），OLMoE 更是只有 4/16 窗有效。这是仪器与热稳态问题，不是语料问题。

后续多方向探索（14 候选一轮、16 候选一轮，共两次fresh jury）几乎全灭：第一轮 3 个推荐里C10 已终审 KILL、C01 已终审 KILL；第二轮 16 个候选给出 15 KILL + 1 PHASE0_ONLY。RouteSieve、RouteGuard-KV 的 CCF-B 身份、RouteContract 也都已判死。

真正值得继续的是一条此前没有被当作方向、而是作为"故障"出现的线索：2026-08-03 在真实 RTX 5090 上，OLMoE 的 batched cached-decode 与 batch-1 serial decode 在 `gate0-000/ step 0 / layer 0` 出现了 top-k membership 翻转（`expected_only=[54]`、`observed_only=[33]`，rank-8/rank-9 边界），原始分类为 `UPSTREAM_LAYER0_ACTIVATION_DIVERGENCE`。它同时是 BCRD Gate-0A 的必修阻塞项，也正是 SpectatorRoute（N05）在16 候选中唯一存活主张的核心现象，并与 N05 在 5090 上跑出的 64/64 positive victims、8192/8192 joint-positive cells 相互印证。两处证据都还没通过完整性审计，但这是全仓库唯一两处在真实 GPU 上被复现的阳性信号，且完全不需要多卡即可证伪。

## 一、方向全景与当前状态

下表按族归并全部 formulation，状态一律取现存最新sealed/审计结论。

| 族 | Formulation | 当前状态 | 关键定量证据 |
|---|---|---|---|
| receiver / return-path | Gate-Aware Partial Combine（原方案 A） | 主线淘汰，保留为历史与 baseline | 静态 receiver group 不等于在线 receiver queue；BF16 不是合格 baseline |
| receiver / return-path | fixed RankLane / CPR | `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20` | 上界 `G=p(s_c-s_b)/(1-p s_b)`，取 `p=0.2, s_b=0.5, s_c=0.6875` 得 4.1667%；过 5% 需 `p_return≥23.5294%` |
| receiver / return-path | FJRC keyed Join-Deficit | `NO-GO` | 冻结 replay 未过跨模型门 |
| receiver / return-path | PhaseMap closed-pair | `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`，`holdout_opened=false` | 封闭双请求 work-conserving reorder 无 action headroom |
| receiver / return-path | 5090 multi-MoE inference-time 表征 | `SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE` | 16 层本地 MoE 累计占 profiled decode 82.8%–90.2%；`expert_loop` 74.9%–85.7%；context 128/512/2048 同 batch median 极差 ≤1.0191；自然/合成配对 A/B 差异 −0.052%–+1.130% |
| receiver / return-path | optimized EP return-path existence | `NOT_TESTED_REQUIRES_8XA100` | 只有 Gate 设计，无数据 |
| energy-SLO | Route-row动态 FP8 | `NO-GO` | OLMoE 仅 2 个expert 在 rows=4096 快 6.4%–6.5%，rows≤2048 全部 BF16 更快；LLM-jp rows=4096 仍慢 32%–34%；动态 activation 量化占 FP8 linear 时间 56%–78%；rows=4096 需 4096 个active decode token，单张 32GB 卡不可达 |
| energy-SLO | JouleQueue v1 | Phase 4 blocked | 缺真实 activation/surface producer、独立 input-event 能耗样本、闭合 serving oracle |
| energy-SLO | RouteSlack | `MEASUREMENT_ONLY / GATE0_FAIL`，Gate 0–3 全 FAIL | 124/124 protocol 测试 PASS，但 physical strategy latency/energy N=0；LLM-jp 11/16 窗有效，rows 1/8/32/128 的 raw J/expert-row 为 0.008206/0.001004/0.000280/0.0000856；OLMoE 4/16 窗，标记 `CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS` |
| expert-pressure | BCRD / BCRD-SC | `GATE0_A_PARTIAL / GATE0_B_PASS / FORMAL_NOT_RUN / REQUEST_DAG_OPEN` | Gate 0 六项中 A/D/F 为 PARTIAL、C/E 为 FAIL、仅 B PASS；72/72 CPU 测试通过；formal Gate 2 硬编码 `INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED` |
| expert-pressure | DEPA-MoE | `DEVELOPMENT_ONLY_NOT_SCIENTIFIC` | 四项formal capability 全 false，夹具固定 `scientific_result_eligible=false` |
| 精度 / 质量 | Rank-tail + FP8-first | Claim 1 跨模型 GO（仅结构性），Claim 2 `SUPERSEDED / NO-GO` | 尾部相对头部安全边际达数十倍量级；62.5% 仅探索性逻辑 payload 点，不含scale metadata/header/padding |
| 精度 / 质量 | ConfidenceGuard v3 / TriageAudit | `NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION` | 无跨模型 audit allocation 增量 |
| 精度 / 质量 | Quality debt / 质量隔离 | `NO-GO` | 16 值合成流点估计 11%–12%，未过预注册 20%；旧 best_proxy claim 因 leakage 失效 |
| 精度 / 质量 | RouteGuard-KV | `KILL_CCF_B / CONTINUE_R0A_KILL_PROBE_ONLY` | fresh 7D 均分 4.0/10（originality 3/10）；smoke 50/50、calibration 200/200 仅工程完整性；K-only@2048 router share 47.56%，non-tie flips 89.37% 低于 90% 门 |
| 诊断 / 方法 | C10 Scheduler-Induced Popularity Endogeneity | `KILLED_AS_DOMINANT_PAPER_ROUTE` | Round 1 5.20/10 RETHINK、Round 2 4.60/10 KILL，0 pilot |
| 诊断 / 方法 | C01 Causal-Closure Certificate | 终审 KILL | Round 1 4.70/10 |
| 诊断 / 方法 | RouteSieve | `KILL / NO_GPU_PILOT` | 主查新 4.8/10、敌对审稿 3.2/10 |
| 诊断 / 方法 | RouteContract | `G0 FAIL` | 只有 OLMoE 是真实预训练模型，tiny-random Qwen/Mixtral 不满足冻结 capsule 契约 |
| 安全 / 数值 | SpectatorRoute（N05） | `EXPLORATORY / PHASE0_ONLY`，run01 `INVALIDATED_PRE_AUDIT_DIAGNOSTIC_ONLY` | jury 16 候选中唯一存活，7D 均分 7.31（现象新 8.0 / 方法新 5.0）；run01 独立复算 64/64 positive victims、8192/8192 joint-positive cells、0 unstable |
| 安全 / 数值 | RouteShield-MoE | `PROTOCOL_ONLY`，Gate 0 `BLOCKED_PROTOCOL_NOT_AUTHORIZED` | 无tenant-qualified ledger、无物理 placement 证据 |
| 安全 / 数值 | BCRD Gate-0A router boundary（2026-08-03） | 原始 `UPSTREAM_LAYER0_ACTIVATION_DIVERGENCE`，审计后 `INVALID_FORENSIC_EVIDENCE`，卡片已消耗、禁止同卡重跑 | 真实 5090 上 batched 与 serial 在 layer 0 出现 rank-8/rank-9 top-k 翻转，`expected_only=[54]`、`observed_only=[33]`；23项 SHA256SUMS 全部校验通过，但两个 router-logit 张量缺 source-storage-byte hash、payload 缺 delta evidence |
| 归档 | CreditReduce、MassCover、TokenRace、PLTB additive、Prefetch、RouteFidelity、Graceful/QTree、QuotaEP、WaveCredit-EP、Mean-balance Placement、Residual-EP、Progressive Residual Transport | 已判死或未闭环归档 | 死因分五类：前提证伪、效应量天花板、固定开销吃掉收益、不赢最强简单基线、未闭环 |

## 二、两条原定主线的真实死因

Receiver-awareness 的问题不在证据量而在结构。它的核心分母是 optimized EP 下未被 overlap 隐藏的 exposed return fraction，这个量在单卡上原理性不可测；已完成的 5090 表征把完整 KV-decode 分母补齐了（本地 MoE 累计 82.8%–90.2%），但那82.8%–90.2% 里装的是 router、expert compute 和本地 combine，不含任何 return all-to-all，因此既不能升级为 receiver congestion 证据，也不能反过来救活 RankLane。同时 fixed actuator 的代数上界已经写死在 4.1667%，reopen 条件是明确的 `p_return≥23.5294%`。结论：这条线在没有真实 8×A100 之前不存在可推进的科学动作，且即使拿到硬件，也是先做 existence Gate 而不是做 controller。

Energy-SLO 的问题需要把诊断改正。数据侧已经很干净：两个模型 revision 固定、语料按行按哈希冻结、到达 trace 用的是公开真实 BurstGPT。真正卡住的是三件事。第一是计量分辨率：NVML 采样最大间隔 10.925 ms，对单个 expert-stage 窗口而言过粗。第二是热稳态：单窗温升Δ23°C 远超`maximum-temperature-range-c 2` 的门，导致大批窗口被 fail-close 作废，密封复跑 32 窗只剩 14 有效。第三是排他性：synthetic ABBA 两次都因检出竞争 CUDA 进程而整次拒绝，可接受 energy 样本合计为 0。附带地，route-row 动态 FP8 这条能耗机制已经独立 NO-GO——它唯一的加速区在 rows=4096，而这需要 4096 个 active decode token，单张 32GB 卡根本到不了这个 operating region。所以 Energy-SLO 若要继续，先修的是测量装置（更长窗口、锁功耗与时钟、强制预热到稳态、独占 GPU），而不是去找更多数据集；而且即使修好，它最多是第二贡献或评价维度，不足以承担主线。

## 三、后两轮多方向探索留下了什么

第一轮 14 候选中被推荐的三个已经全部退场或降级：C10 在两轮方法评审后终审 KILL，理由是收缩后的 HorizonFence 退化为通用 robust dominance 规则，且异步迁移在合法状态下可完全 overlap，使 universal cost lower bound 只能取 0，形成"要么 unsound、要么退化为 UNRESOLVED"的结构性 fork；C01 作为 C10 的 pivot 也在一轮后KILL；C14 本身自认只是强制 baseline，不是新方法。C10 留下的三条负面洞察值得作为内部审计规则保留，特别是"local service-time saving 不能逐事件相加成 request completion 收益"和"任何 local-window Oracle 必须先证明 actionranking 对 full request-DAG 的因果闭合"——后者正是 BCRD formal Gate 2 被硬编码为 INVALID 的同一件事。

第二轮 16 候选给出 15 KILL，死因高度一致：`Remove-MoE` 测试不通过（把 MoE 换成任意 fork-join、分片服务或 batched MLP 后方法完整保留），或直接撞上 vLLM Elastic EP、EEP、TBIK、CRAFT、UltraEP、DeepEP 等现成实现。唯一存活的是 N05，而且 jury 明确只承认它的**现象**（7.2分里现象新 8.0、方法新 5.0），canonical/margin fallback 方法本身撞 MarginGate 与 LLM-42，只能作机制消融。

## 四、唯一值得优先探究的交叉证据

把两件独立的事放在一起看，结论就变得清楚。

一件是 SpectatorRoute Phase-0A：在真实 5090、真实预训练 OLMoE hidden rows 上，独立复算得到 64/64 positive victims、8192/8192 joint-positive cells、0 unstable cells。它因为缺 GPU UUID/外来 PID 连续监控、raw BF16 bit 计数、watchdog 和最终 `COMPLETE.json` 而被降为 `INVALIDATED_PRE_AUDIT_DIAGNOSTIC_ONLY`，但数值本身内部一致。

另一件是 BCRD Gate-0A 的失败：同样在真实 5090、同一个 OLMoE 固定 revision 上，batched continuous-decode 与 batch-1 serial decode 在 layer 0 出现 rank-8/rank-9 边界的 top-k 成员翻转，expert 54 被33 取代。它因为两个 router-logit 张量缺 source-storage-byte hash、payload 缺 delta evidence 而被判`INVALID_FORENSIC_EVIDENCE`，卡片消耗、禁止同卡重跑。

这两件事在讲同一句话：**批组成会改变 MoE 的路由结果**。前者是主动构造的，后者是在完全无意的工程流程里撞出来的——后者的证据价值更高，因为它排除了"为了拿阳性而设计输入"的嫌疑。这条线的四个优点很突出：它是 MoE-specific 的（dense FFN 没有 spectator 可控的稀疏expert histogram，`Remove-MoE` 通过）；它单卡可完全证伪，不依赖 8×A100 或多卡 EP；它两周内能出 GO/NO-GO；而且修它本身就是 BCRD Gate-0A 的必经步骤，等于零额外机会成本。

它的致命风险也必须先摆出来：翻转可能只来自 kernel 非确定性、左填充实现或 tile 选择，而不是任何可利用的语义依赖。最强反例是 batch-conditioned refusal 那篇在 batch-invariant kernel 下 0/55 复现，以及 RaMP 只证明 histogram 改变 performance/tactic 选择而非 reduction 顺序。因此第一个实验必须是判死实验而不是确认实验。

## 五、筛选结果

第一档，值得作为毕设主线继续。合并 SpectatorRoute 的现象主张与 BCRD Gate-0A 的 router-boundary 故障，成为一条"连续批处理下 MoE 路由与输出的批组成敏感性"测量线。建议把叙事从"攻击面"改为"exactness 与可复现性"，因为后者不需要现实共驻威胁模型，也不需要多租户 EP，负结果同样可发表；攻击视角可以作为 implication 一节而不是主张。BCRD 的现有资产（冻结 manifest、identity ledger、route-v3 契约、72/72 测试、continuous-decode producer）正好是它的载体。

第二档，保留但降级为支撑。BCRD 的 Gate 0/1 基础设施保留为该主线的实验底座，但不再宣称 assignment+seal 机制为论文主贡献，因为 Gate 0 的 C（service surface）与 E（full-path denominator）仍是 FAIL，Gate 2 因缺 full request-DAG 强制 INVALID，机制闭环在毕设周期内不现实。Energy-SLO 保留为第二贡献候选，但前置条件是先修计量（窗口时长、锁时钟功耗、独占 GPU、强制热稳态），修不成就永久降为评价维度。Rank-tail 的 Claim 1 保留为 motivation。8×A100 return-path existence Gate 保留为条件分支，reopen 条件 `p_return≥23.5294%` 不变。

第三档，明确停止且不得改名复活。receiver controller 全族（RankLane、FJRC、PhaseMap、DDRC、CJC、RIC）、ConfidenceGuard v3、quality debt、route-row 动态 FP8、RouteSieve、C10 与 C01、RouteGuard-KV 的 CCF-B 路线、RouteContract，以及归档目录里的十余个已判死 formulation。

第四档，毕设周期内不启动。RouteShield（需要真实多租户 EP 与双租户 full-path 分母）、C09 silent slowdown（需要多GPU fault injection）、DEPA（action space 过宽，且 Gimbal/QLLM/AMoE/EDF 已覆盖大部分动作）、Guarded Expert Reuse 与 BufferLease（只作换题备案，各有独立前置 Gate）。

## 六、建议的最短执行路径

第一步，三天内补齐 2026-08-03 那次forensic 的两处缺口：为两个 router-logit 张量补 source-storage-byte hash，为 payload 补独立推导的 delta evidence。由于旧卡片已消耗且禁止同卡重跑，需要开新卡片、新 lock、新输出目录，跑一次拿到被审计接受的 first-divergence 分类。这一步同时解掉 BCRD Gate-0A 的阻塞。

第二步，三到七天做判死实验：在同一target 上开启 deterministic algorithms、关闭 TF32、固定 SDPA后端、并在可用时使用 batch-invariant 路径重测。如果翻转消失且 prevalence 归零，这条线立即死，回落到第二档，并把负结果写成一节。这一步必须在看到 prevalence 数据之前冻结阈值。

第三步，若翻转不消失，七到十四天做 prevalence 与传播测量：两个模型、128 个已冻结请求、全部层，统计 top-k membership flip 率、翻转在 rank 维度的分布（是否集中在 rank-k与 rank-k+1 边界）、以及传播到最终输出 token 的比例，并配matched-random batch 负控。判死门槛沿用仓库既有严格度：两模型共同自然 cell 的 flip prevalence 与output-flip rate 的 95% LCB 都必须非零且跨模型同向，否则记窄负结果停止，不调 workload、seed、dtype 或 detector 抢救。

与主线并行、但不占 GPU 主时段的低成本动作只有一项：修 NVML 能耗计量装置，为第二档的 Energy-SLO 保留可能性。

## 七、局限

本文是文档整合，没有产生任何新的实验数据。两处关键阳性信号（N05 run01 与 2026-08-03 router boundary）都尚未通过完整性审计，因此"批组成改变路由"目前只是**可复现的诊断性观察**，不是被接受的科学结论；在它通过独立审计前，不能写入论文正文。此外，本机macOS 无 CUDA、5090 为远端实例且已出现过 `BLOCKED_REMOTE_INSTANCE_CLOSED`、项目无 8×A100，这三条资源约束是本次筛选最硬的依据，一旦硬件条件变化（尤其拿到真实 8 卡），第二档中的 return-path existence Gate 优先级需要重新评估。所有 jury 与审计都标注 `same-family / provisional`，独立性不足，评分不应当作绝对排序。

## 参考

内部权威文档：

1. [当前研究状态与唯一执行线](README.md)
2. [MoE 推理毕业论文方向统一稿](MoE_推理毕业论文方向统一稿_2026-07-25.md)
3. [Gate 0 审计账本 2026-08-02](gate0_audit_2026-08-02.md)
4. [Gate 0-B 冻结输入完整性审计](../../EXPERIMENT_AUDIT.md)
5. [BCRD Gate-0 router-boundary root-cause delivery 2026-08-03](../../artifacts/bcrd_gate0/root_cause_delivery_20260803.md)
6. [RouteSlack 最终裁决](routeslack_final_verdict.md)
7. [SpectatorRoute N05 状态与冻结协议](../ideas/spectatorroute/README.md)
8. [Receiver-aware / EP return-path 分支状态](../ideas/receiver_aware/README.md)
9. [5090 多 MoE 层 inference-time 表征](../ideas/receiver_aware/inference_time_5090/README.md)
10. [Route-row FP8 快速 GPU 探索结论](../ideas/energy_slo/route_row_fp8/ROUTE_ROW_QUICK_RESULT_2026-07-23.md)
11. [C10 终审 KILL](../../refine-logs/C10_KILL_VERDICT.md)
12. [RouteSieve KILL 裁决](../../refine-logs/ROUTESIEVE_KILL_VERDICT_20260729.md)
13. [RouteGuard-KV CCF-B 审计](../../refine-logs/ROUTEGUARD_CCFB_AUDIT_20260729.md)
14. [Idea 候选报告与排序](../../idea-stage/IDEA_REPORT.md)
15. [已归档 / 非主线 idea 思路总览](../archive/killed_ideas/思路总览.md)

外部碰撞与反例文献：

16. [RaMP: runtime expert histogram 驱动的 kernel tactic 选择](https://arxiv.org/abs/2604.26039)
17. [Batch-conditioned refusal protocol（batch-invariant kernel 下 0/55 复现）](https://arxiv.org/abs/2605.27763)
18. [MarginGate: margin-triggered verification/fallback](https://arxiv.org/abs/2605.30218)
19. [vLLM batch invariance 特性文档](https://docs.vllm.ai/en/stable/features/batch_invariance/)
20. [METRO: memory-bound decode 下activated experts 优于 token count](https://arxiv.org/abs/2512.09277)
21. [Gimbal: 联合frontend/backend pressure 与 placement](https://arxiv.org/abs/2606.15177)
22. [AMoE: 异步 expert parallelism 与动态 re-batching](https://arxiv.org/abs/2505.08944)
23. [UltraEP: 逐 microbatch/逐层 exact-load balance](https://arxiv.org/abs/2606.04101)
