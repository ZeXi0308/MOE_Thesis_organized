# MoE 研究问题重构：严格审计与唯一主线

> 状态：**当前权威裁决文档**  
> 证据截止：2026-08-11
> 作用：统一 7 月 25 日多个并行进程留下的候选生成、代码框架和局部裁决；含独有证据的原文转入方向目录或历史汇总，已被完整吸收的中间“最终稿”不再单独保留。
>
> 长篇研究问题、候选边界和实验蓝图见[《MoE 推理毕业论文方向统一稿》](MoE_推理毕业论文方向统一稿_2026-07-25.md)；本页继续负责当前证据账本和唯一执行顺序。

## 0. 2026-08-11 当前结论

**JoinStream 当前 formulation 已正式封存。** 三阶段证据链必须同时保留：CPU exact Oracle 为 `SUPPORT_ACTION_SPACE / CPU_EXPLORATORY_SIGNAL`；synthetic single-GPU pilot 为 `WEAKEN_TAX_DOMINATES / SINGLE_GPU_EXPLORATORY_MICROBENCHMARK`；realistic MoE-tail pilot 最终为 `WEAKEN_UPPER_BOUND_TOO_SMALL / GATING_INSUFFICIENT / WEAKENS / SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK`。最终四个 cells 有 `3/4` natural windows、最大 `161.664 us`，但安全收益为 `0/4`；审计 `PASS / P0=0 / P1=0`。因此保留 action validity、memory legality、single-GPU schedulability、natural tail headroom 和 producer safety，削弱 critical-path utility，paper viability 为 `FREEZE`。核心边界是 `overlap opportunity != critical-path leverage`，并已标记 `NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`；不得继续优化 gate/priority/polling/stream/notification 或用多 GPU 假设抢救。完整证据与禁止外推边界见 [JOINSTREAM_FINAL_FREEZE](JOINSTREAM_FINAL_FREEZE_2026-08-10.md)。

**下一 Primary 仅是 Oracle-first 研究问题，不是已验证方法。** 对现有候选池的 fresh same-family provisional jury 选择 `Route-Conditioned Barrier Amplification Boundary`：先在 near-real、identity-complete full request DAG 上证明自然 MoE barrier 带来 charged end-to-end critical-path Oracle headroom，再讨论任何 action/runtime。2026-08-10 的 protocol-first preparation 裁决为 `BLOCKED_PROTOCOL_AMBIGUITY / FORMAL_GATE_NOT_RUN`：common natural regime、removable barrier/dependency semantics 与 resource capacity 尚未唯一冻结，且两个模型的 identity-complete trace 与 complete measured surface 均不存在；因此 evaluator 未实现，Primary 保持 `UNVALIDATED`。正结果之前不实现 scheduler/controller；若共同自然 cells 全部 `<5%`、actionable mass `<20%`，或 max-load/CV 在 holdout 上完全解释边界，立即冻结。候选 Top 3 与未决项见 [Next-Idea Jury](../../idea-stage/NEXT_IDEA_JURY.md)，preparation 边界见 [RCBA README](../ideas/rcba/README.md)。

**D10 Stability-Aware Expert Shape Lanes 已降级。** fixed-C8 correctness 证据仍成立，但冻结 continuous-decode cost Gate 裁决为 `NO_GO_D10_HEADLINE_COST`：fixed-C8 / serial-M1 expert GPU-time ratio 为 0.8491（要求 <=0.8），fixed-C8 / native token-step p99 ratio 为 1.4694（要求 <=1.05），padding fraction 为 77.4056%。因此 universal fixed-C8 不再是 headline method candidate，不得通过修改 `C`、阈值、workload 或指标救活。

同日完成的 single-contribution hindsight Oracle sweep 给出互补正结论：240 cells × 8 actions 中，abstaining Oracle 用 33 个正动作恢复 37/43（86.0465%）unprotected downstream route distance，覆盖 8 victims；相对 budget-matched conditional/global random 的优势分别为 17.5/36.690625，四个 STRONG 门全过。最终审计为 WARN，P0 0 / P1 1；局部 outcome 知识限制 preregistration 措辞，但不改变重算指标。它只证明冻结 enriched proxy 上的 action value existence；MaxGate-v1 仍为 -3，不能据此得到 online selector。

**StableBatch 已完成最终 fresh Selectability Gate，并按冻结规则停止。** 在 16 个全新 document-disjoint requests、240 cells、1,920 actions、exact B=33 上，Outcome Oracle reward 为 `57`，matched shuffle 为 `-4`，恢复 `57/84=67.86%` unprotected route distance并覆盖 12 个正收益 requests；因此 fresh intervention opportunity 成立。Static Compatibility Map 与 Online Observable Ridge 均为 `-7`，低于 shuffle，Recovered Oracle Gap 均为 `-0.04918`，LODO 均为 `0/16`。formal manifest、独立 aggregation 与从 240 个 U/1,920 个 A raw routes 重建的 verifier 全部零 mismatch；审计为 `WARN / P0=0 / P1=0`。因此停止 compatibility-aware coalescing 与 row-conditioned pre-action scheduling，不做第三个 selector、partition planner 或 controller；现象证据保留，但不构成系统论文机制。

Cheap selector 已单独被证伪：C09-v1 的 8-fold document-disjoint input-only linear model为 0 TP / 14 FP、零 admission，裁决 `KILL_C09_V1_ZERO_ERROR_ADMISSION`。新的 cross-companion 证据将 exact 机制裁决为 `MIXED`：fixed `M=2`/small-M ABI 内 2,048/2,048 focal labels 不随 companion 翻转，而跨 `M` safe-set 嵌套且 M32/M64 完全失效。因此 H1 row-intrinsic + H3 shape/cardinality-conditioned 是当前最窄解释，H2 pairwise 主因不受支持。

SemanticFence 分支的 Exact `M=2` Oracle maximum matching 覆盖 2,264/32,234 rows（7.0236%），同表 expert-stage projection 只节省 3.4034%。因此 exact-only RowFence 只保留为 baseline/fallback。同表 Semantic Oracle shadow 在刻意富集的 64 条 raw-unsafe edges 上得到 41 条 route/top-k-safe edges，maximum matching 覆盖 52/96 rows（54.1667%），additive expert-stage projection 节省 26.2038%。这是 reused-calibration shadow upper bound，不是自然 prevalence、fresh generalization 或 serving speedup；它曾授权在 safe-packing 机制族内进一步资格化 **SemanticFence-v2 / Semantic Stability Budget**，但 2026-08-10 Oracle-first 重筛后只列第二候选，不再是下一 Primary。

**SFV2-O1 已完成 fresh online-observability Gate，但不授权在线执行。** 在 4 个 document-disjoint test documents 的 128 条自然 edges 上，Semantic Oracle 得到 77 条 downstream ordered-top-k-stable proxy edges，maximum matching 覆盖 154/255 rows（60.3922%），4/4 documents 有正动作，additive expert-stage projection 为 29.2714%；这把 enriched shadow upper bound 提升为当前单栈 fresh route/top-k-stability proxy action-space 证据。可是 frozen witness-v1 只执行 5 pairs、其中 4 unsafe，覆盖 3.9216%，计入 measured prototype overhead 后 net projection 为 -183.3504%，故机械裁决为 `PIVOT_TO_SHADOW_VERIFY`。在 SemanticFence 分支内保留的是该 proxy action-space / observability headline，不是 witness-v1 方法；下一步只允许 shadow verifier / selective repair。该分支结果不取代当前 RCBA Oracle-first Primary，也不构成 task semantics、M2 runtime、serving 或 controller 证据。

此前 SemanticFence coarse contract 的 `WEAKEN` 与 backup CriticalSplit 的 `WEAKEN_ACTION_SPACE` 都保持有效，分别见 [SemanticFence README](../ideas/semanticfence/README.md) 与 [CriticalSplit 跟踪器](../../refine-logs/EXPERIMENT_TRACKER_20260810_173700.md)。上述 StableBatch/SemanticFence 结果只覆盖 pretrained OLMoE / BF16 / single RTX 5090 / eager、prompt-forward 或 decode-style 局部栈；JoinStream 则是独立 FP16 standalone grouped-expert microbenchmark。两类证据都不是 vLLM serving、EP/NCCL、多卡或论文结果。

## 0A. 2026-08-02 前序结论（仅作历史上下文，不授权当前执行）

当前**没有已经被正式实验证实的 MoE 系统主机制**。

截至 2026-08-02，当时只保留一条研究主线：

> 在自然连续到达、有限 admission/batching 能力和 exact model semantics 下，先验证 expert-side fragmentation、route-conditioned straggler 与 HoL 是否构成足够大的暴露关键路径；只有现象和 Oracle 空间跨模型成立，才研究 expert-pressure-aware 的在线决策。

当时的唯一动作不是同时实现 BCRD、DEPA 和 CPR，而是完成一份可同时约束 BCRD/DEPA 的**共同现象 Gate**。2026-08-10 的 Oracle-first reset 已取代这条执行顺序；以下列表只保留历史状态：

- BCRD 是有完整三门协议和代码 harness 的**候选 formulation**，不是主结果；
- DEPA 是更宽 action space 的**开发态 CPU 回放框架**，不是正式候选结论；
- CPR/RankLane 的 fixed actuator 已在冻结域内 NO-GO，真实 8×A100 return-path existence 仍未测试，但不占用当前单卡优先级；
- 5090 已补齐多 MoE 层的完整 KV-decode inference-time 表征，但单卡无 return all-to-all，不构成 Receiver congestion 或 `p_return` 证据；
- RouteSlack 最终审计为 `MEASUREMENT_ONLY / Gate 0 FAIL`，formal physical strategy latency/energy sample 均为 0；紧凑证据入口由 [`artifacts/index.json`](../../artifacts/index.json) 路由，不构成 controller 或系统 GO；
- PhaseMap、FJRC、ConfidenceGuard v3 等已停止的 formulation 不得换名字复活。

## 1. 为什么此前会出现多个“唯一主攻”

7 月 25 日的文档来自不同时间点和不同审计层次：先生成候选，再做 reviewer/red-team 复审，然后出现新的 RankLane 上界证据，最后又分别实现了 BCRD 和 DEPA 的验证框架。它们不是同一时刻的并列最终结论。下表记录归并来源；被本文完整吸收的中间排名稿已移出当前工作区，不能再作为独立裁决入口。

| 文档/产物 | 实际角色 | 当前使用方式 |
|---|---|---|
| `MoE_严格Idea探索与Top3` | 第一轮候选生成 | 来源快照，只保留问题背景和早期实验想法 |
| `MoE_12Ideas_Reviewer级严格复审` | prior-art / Oracle / reviewer 复审 | 来源快照，不继承早期排名 |
| `MoE_10Idea_RedTeam严格评审` | 第二轮去留与攻击面 | 来源快照，不继承“RankLane 首选” |
| `MoE_研究方向统一梳理_17项…` | 三份候选文档的去重稿 | 候选注册表；执行顺序已被后续证据取代 |
| `MoE_现象优先Gate审计…`、`P1_单5090…` | 证明 return-path 核心问题单卡不可测 | 保留为 8×A100 existence Gate 设计 |
| `CPR-MoE_5090快速验证…` | 单卡必要条件 harness 与 Code Review | 当前结果是 `INCOMPLETE_NECESSARY_GATES`，不授权 controller |
| `CPR_MoE_RankLane_5090快验协议与裁决` | fixed RankLane 的代数上界 | 权威局部 NO-GO 证据 |
| `MoE_大瓶颈优先_14Ideas…` | BCRD 研究设计和三门协议 | BCRD 候选规格，不再单独宣称“唯一主攻” |
| `docs/ideas/bcrd/experiments/` | BCRD 三门证伪 harness | 测试/烟测资产；尚无正式科学输出 |
| `docs/ideas/depa_moe/experiments/` | DEPA CPU 回放与三门 runner | 开发资产；正式能力开关全部为 false |

因此，文档中的“唯一”统一解释为：**同一时间只推进一条证伪链**，而不是多个进程各自宣布一个机制获胜。

## 2. 当前证据账本

### 2.1 已观察或已完成的窄结论

| 对象 | 当前状态 | 允许写入论文/汇报的结论 | 不允许外推 |
|---|---|---|---|
| Rank-tail matched-byte 质量结构 | `Observed / structural only` | 两模型存在 gate-rank 相关的质量敏感度差异，可作 motivation | 不能写成 62.5% 端到端通信下降或 TPOT/P99 改善 |
| RTX 5090 LUT / codec 微基准 | `Observed / single GPU` | 单卡串行 primitive 或 connected codec 的局部时延 | 不能写成 NCCL、RDMA、incast、多卡 EP 或 serving 结果 |
| RTX 5090 multi-MoE inference time | `SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE` | 16 层 LLM-jp 的本地 MoE block 成本在完整 KV decode 中近似均匀累积；profiled 占比约 82.8%–90.2%；粗分解显示 expert loop 已占约 74.9%–85.7%；context 与自然/合成配对 A/B 未出现新的单卡拥塞信号 | 不能将本地 router/expert/combine 占比写成 return-path fraction、Receiver congestion 或 TPOT/P99 serving 结果；vLLM/OLMoE 项因远端实例关闭未完成；通信 headroom 必须另行证明 |
| ConfidenceGuard v3 | `NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION` | 当前 prefill risk ranking 未产生跨模型 audit allocation 增量 | engineering pass 不能替代 scientific pass |
| FJRC corrected replay | `NO-GO` | keyed Join-Deficit bitmap 在冻结 replay 上未过跨模型门 | 不否定所有 receiver-aware scheduling |
| PhaseMap closed-pair | `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`；`holdout_opened=false` | 当前闭合两请求、work-conserving reorder formulation 无 action headroom | 不得放宽 κ、改模型阈值或查看 holdout 抢救 |
| fixed RankLane | `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20` | 在 `p_return≤20%`、codec 等税为零的冻结域内，最乐观相对 uniform FP8 E2E 改善仅 4.1667% | 不等于真实 8×A100 return path 不存在 |

### 2.2 尚未形成科学结论的资产

| 对象 | 已有内容 | 缺口 | 当前标签 |
|---|---|---|---|
| CPR 5090 quick validate | runner、kernel、provenance 校验、CPU 测试；旧质量数值重分析为 PASS | 同次 provenance 的正式质量重跑、5090 connected INT4 正式运行；核心仍需 8×A100 | `INCOMPLETE_NECESSARY_GATES` |
| BCRD | route-v3 契约、共享因果事件引擎、continuous-prefix baseline、document-disjoint split、symmetry-reduced exact local Oracle；Gate-0 A 物理 continuous-decode producer 已实现并通过 tiny OLMoE 开发资格测试；BCRD 69/69 CPU 测试通过 | 冻结且授权的 formal workload manifest、两模型 CUDA producer 实跑、expert/dtype 完整 service surface、full-path denominator、跨 layer/step counterfactual request-DAG、正式 Gate 结果 | [`GATE0_A_PARTIAL_IMPLEMENTED / FORMAL_NOT_RUN / REQUEST_DAG_OPEN`](gate0_audit_2026-08-02.md) |
| DEPA | 因果 CPU replay、request ledger、exact small oracle、三门 runner、10 个测试 | 四项 formal capability 均为 false；无正式 breakdown/episodes/surface；无完整 prior-art 边界文档 | `DEVELOPMENT_ONLY_NOT_SCIENTIFIC` |
| RouteGuard-KV R0-A | 冻结协议/数据/实现、25项 CPU 测试；RTX 5090 smoke v2 50/50、calibration 200/200 trajectory 且完整性/负控 PASS | calibration 仅为工程完整性；32文档 formal 未运行且未批准；最新查新已 KILL CCF-B 主候选身份；无跨模型、native INT4、serving 或多卡证据 | `GPU_CALIBRATION_INTEGRITY_PASS / FORMAL_NOT_RUN / CCF_B_ROUTE_KILLED / NOT_CURRENT_MAINLINE` |
| RouteSlack | 124/124 protocol-critical tests、双模型 development cached-decode exactness、单卡 isolated-expert characterization、compact audit capsule | 9 个开放 P0；无真实 continuous serving/EP actuator；formal physical strategy latency/energy N=0 | [`MEASUREMENT_ONLY / GATE0_FAIL`](routeslack_final_verdict.md) |
| optimized EP return-path existence | 8×A100 Gate 设计 | 真实 8×A100、optimized backend、timeline/transport/identity 闭合 | `NOT_TESTED_REQUIRES_8XA100` |

## 3. 新 Primary 的历史问题底座

> 当前执行权以第 0 节和 [Next-Idea Jury](../../idea-stage/NEXT_IDEA_JURY.md) 为准。本节的现象定义、route/service/denominator 要求可作为 `Route-Conditioned Barrier Amplification Boundary` 的 Oracle Gate 输入约束；BCRD/DEPA action 分支不再构成并行当前主线。

### 3.1 问题定义

研究对象不再是“再造一个 receiver/precision controller”，而是：

> 对真实 MoE route 和自然连续请求，在不改变 top-k、expert identity、权重、输出语义的前提下，当前 expert work 的分散、排队和合批损失是否占据足够大的暴露路径；如果存在，哪一个最小 action space 能在 deadline/SLO 下捕获 Oracle 空间，并显著超过最强简单策略？

这个定义将科学问题、机制和实现分开：

1. **现象：** fragmentation / HoL / route-conditioned straggler 是否真实、自然、跨模型且处于关键路径；
2. **上界：** future-known exact Oracle 是否存在足够空间；
3. **机制：** 只有上界成立后，才在 BCRD 或 DEPA 中选择一个最小 action space；
4. **系统证明：** 单卡只给候选资格，多卡 serving 才能给 EP/TPOT/P99 结论。

### 3.2 为什么不直接宣布 BCRD 或 DEPA 为主线

- BCRD 边界更窄、协议更完整，但其核心 natural replica fragmentation 尚未测；没有 Gate 1 就没有问题实例。
- DEPA 能覆盖 admission、batch composition 和 release，当前却过宽；开发夹具的 Gate 1/2 PASS、Gate 3 FAIL 均明确 `scientific_result_eligible=false`，不能用于方向排序。
- 两者共享 route、service surface、continuous-arrival episode 和 full-path breakdown。先各跑一套会重复生产证据，并制造选择性解释空间。

所以历史方案先冻结共同输入和现象门，再依据结果选择一个 formulation。2026-08-10 reset 后，这段只解释已有资产来源，不授权运行 BCRD/DEPA Gate 或并行调参。

## 4. Oracle Gate 准备约束（历史共同 Gate 收敛后）

本节 Gate 0/1 只作为新 Primary 的数据与会计规格。`PREPARE_ORACLE_GATE` 已 fail closed 为 `BLOCKED_PROTOCOL_AMBIGUITY`，不授权执行 Gate；Gate 2/3 的 action/policy 分支保持历史冻结，只有协议、双模型 trace/surface 和 charged full-request Oracle headroom 依次通过后才可能重新裁决。

### Gate 0：正式资格预检

以下四项缺一项即停止在 `BLOCKED_MISSING_FORMAL_EVIDENCE`，不能用开发夹具代替：

1. native continuous-decode route producer；
2. RTX 5090 实测、按 expert/dtype 绑定的 service surface，且仅在测量区间内保守插值；
3. identity-complete、exact-output replay；
4. 冻结的 natural workload / model / revision / seed manifest。

还需一个与目标 accounting boundary 一致的完整路径 denominator。不能从 proposed saving 反推 denominator。

截至 2026-08-02，第 1 项已有唯一开发入口和 fail-closed artifact
contract，但未完成正式资格。第 4 项输入已冻结：两个 canonical manifest
绑定 exact model/tokenizer revision、WikiText 样本与 prompt/token hashes、
BurstGPT 真实到达 trace 和 RTX 5090 软件环境；当前 dirty checkout 未提交，
本机也无 CUDA，两个正式 cell 均未运行。详见
[Gate-0 审计账本](gate0_audit_2026-08-02.md)。这不授权 Gate 1。

### Gate 1：共同现象测量

同一份冻结证据同时报告两组互不替代的量：

- **BCRD-specific：** replica fragmentation penalty。进入其 Oracle 门要求两模型共同自然 cell point estimate ≥10%、95% LCB >5%，至少一个 common cell 两模型均 ≥15%，actionable waves ≥20%。
- **broader expert-pressure：** fragmentation + route-conditioned straggler + HoL 的去重 exposed share。进入 DEPA 方向复审要求 point estimate ≥20%、95% LCB ≥10%。

统一停止规则：

- 所有共同自然 cell `<5%` 或 actionability `<20%`：停止 expert-pressure/BCRD 机制线；
- 只有 5%–10%：降级为 measurement/kernel 问题，不进入复杂 controller；
- 只在 synthetic skew、单模型、单层、极端 replica 数成立：不构成论文主线；
- identity、route、rows、request ledger 或 denominator 不闭合：`INVALID`，只修测量，不解释收益。

### Gate 2：历史 action-space 分支（当前未授权）

当前 builder/Oracle 只做 `single_layer_window` 回放，不会把 assignment/hold 改变的完成时间传播到后续 layer 和 decode step。因此 formal Gate 2 已硬编码返回 `INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`；即使 Gate 1 将来通过，也必须先实现并验证 full request-DAG，才能运行 Oracle 或 Gate 3。

Gate 1 后按现象来源分叉，但一次只允许一个分支：

- 如果增量主要来自 fixed replica set 内的 rows fragmentation，选择 **BCRD** 的 `assignment + bounded seal time` exact Oracle；
- 如果 BCRD-specific 不成立，但 broader HoL/admission exposed share 独立成立，先完成 DEPA 的 prior-art/action-space 收缩审计，再选择 **DEPA** 的小规模 exact SLO-goodput Oracle；
- 如果两者都成立，优先选择动作更少、因果链更短、最强简单 baseline 更难覆盖的 formulation；该选择必须写回本文后才能进入 Gate 2。

Oracle `<10%`、动作等价或只在 zero-cost / future-unbounded 条件下成立，立即停止。

### Gate 3：历史 policy 分支（当前未授权）

统一比较 current/default、deterministic random、threshold、EDF/least-laxity、greedy 和 chosen proposed policy；参数只能在 calibration split 选择。

- 简单策略捕获 ≥90% Oracle：取消复杂 controller；
- proposed 必须在冻结 evaluation 上同时满足净收益、Oracle 捕获率、相对简单策略差距、P99/公平性和决策开销门；
- 单卡通过只授予 `8×A100 Candidate`，不是系统 GO。

## 5. 与 8×A100 return-path Gate 的关系

optimized EP return-path existence 是另一条**硬件条件分支**，不是当前 5090 主线的并行任务。

- fixed RankLane 在 `p_return≤20%` 域内已经 NO-GO；
- 5090 的 16 层数据证明未来会计必须覆盖所有 MoE 层，不能只 profile selected layer；但不能直接将各层本地 MoE span 相加作为 return Oracle；
- 只有真实 8×A100 证明优化后 exposed return fraction 达到记录的 reopen condition，才允许重新评估 return representation；
- 在此之前不实现 RankLane codec/controller，不把 CPR quick gate 写成 EP 结果。

## 6. 明确停止和保留清单

### 停止

- PhaseMap 当前 closed-pair formulation；
- FJRC keyed Join-Deficit bitmap；
- ConfidenceGuard v3 当前 prefill-risk audit allocation；
- fixed RankLane actuator 在 `p_return≤20%` 冻结域内；
- StableBatch compatibility-aware / row-conditioned pre-action selector；
- 已归档的 CreditReduce、MassCover、TokenRace、QuotaEP、PLTB additive、RouteFidelity、Prefetch 等 formulation。

### 仅保留为证据或工具

- Rank-tail 结构证据：motivation；
- 5090 LUT/codec：局部测量工具；
- shared route/modeling/metric 代码：公共实验底座；
- BCRD/DEPA runner：Gate 实现，不是实验结论。

## 7. 当前最短执行清单

1. 不重跑或调参抢救 SemanticFence coarse contract、C09-v1、MaxGate-v1 或 CriticalSplit。
2. D10 cost Gate 与 StableBatch Selectability Gate 均已关闭；保留原始证据，不改 C/B、特征、阈值、workload 或 metric 救活。
3. 不实现 StableBatch selector、partition planner、controller 或 vLLM integration；Oracle upper bound 不能充当 online policy。
4. 下一 Primary `Route-Conditioned Barrier Amplification Boundary` 的 preparation 已因协议歧义 fail closed；不得运行 Oracle Gate或设计 action/runtime。若不另行完成 protocol-definition qualification，则返回 candidate discovery。

## 8. 文档裁决顺序

发生冲突时按以下顺序解释：

1. 机器可读 sealed/formal decision；
2. 本文的当前状态和执行线；
3. 专项实验裁决/协议；
4. 候选生成、Top 3、reviewer/red-team 长文；
5. 历史 meeting note 或探索性输出。

新的正式结果必须同时更新本文和对应 idea README。只新增一份“唯一主攻”文档而不更新本页，视为来源快照，不改变执行顺序。

### Artifact 路由与发布边界

- 科学裁决仍以 `docs/current/` 为准；[`artifacts/index.json`](../../artifacts/index.json) 只提供 canonical artifact 路由，bundle 本身不能升级 verdict。
- RouteSlack 的 canonical compact capsule 是 `20260728_160000_final_audit`；此前文档称 `20260728_115300` 为 canonical 的记录保留为历史原始证据，不删除、不失效，二者关系由索引显式说明。
- 新生成的 `artifacts/` run bundle 默认不进入 Git；完整规则见 [`artifacts/README.md`](../../artifacts/README.md)。现有已跟踪 bundle 的取消跟踪或历史重写不属于本次变更。

## 9. 一句话状态

> JoinStream 当前 formulation 已 `FREEZE / NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`；D10 headline 已因成本 Gate 降级，StableBatch pre-action 机制已 STOP。当前仍没有被正式实验证实的 MoE 系统主机制；下一 Primary 只是 `Route-Conditioned Barrier Amplification Boundary` 的 Oracle-first 研究问题，其 preparation 为 `BLOCKED_PROTOCOL_AMBIGUITY`，尚无 evaluator 或 headroom 证据。
