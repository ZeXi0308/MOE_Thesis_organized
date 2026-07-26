# MoE 研究问题重构：严格审计与唯一主线

> 状态：**当前权威裁决文档**  
> 证据截止：2026-07-25  
> 作用：统一 7 月 25 日多个并行进程留下的候选生成、代码框架和局部裁决；含独有证据的原文转入方向目录或历史汇总，已被完整吸收的中间“最终稿”不再单独保留。
>
> 长篇研究问题、候选边界和实验蓝图见[《MoE 推理毕业论文方向统一稿》](MoE_推理毕业论文方向统一稿_2026-07-25.md)；本页继续负责当前证据账本和唯一执行顺序。

## 0. 直接结论

当前**没有已经被正式实验证实的 MoE 系统主机制**。

现阶段只保留一条研究主线：

> 在自然连续到达、有限 admission/batching 能力和 exact model semantics 下，先验证 expert-side fragmentation、route-conditioned straggler 与 HoL 是否构成足够大的暴露关键路径；只有现象和 Oracle 空间跨模型成立，才研究 expert-pressure-aware 的在线决策。

当前唯一动作不是同时实现 BCRD、DEPA 和 CPR，而是完成一份可同时约束 BCRD/DEPA 的**共同现象 Gate**。在该 Gate 之前：

- BCRD 是有完整三门协议和代码 harness 的**候选 formulation**，不是主结果；
- DEPA 是更宽 action space 的**开发态 CPU 回放框架**，不是正式候选结论；
- CPR/RankLane 的 fixed actuator 已在冻结域内 NO-GO，真实 8×A100 return-path existence 仍未测试，但不占用当前单卡优先级；
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
| ConfidenceGuard v3 | `NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION` | 当前 prefill risk ranking 未产生跨模型 audit allocation 增量 | engineering pass 不能替代 scientific pass |
| FJRC corrected replay | `NO-GO` | keyed Join-Deficit bitmap 在冻结 replay 上未过跨模型门 | 不否定所有 receiver-aware scheduling |
| PhaseMap closed-pair | `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`；`holdout_opened=false` | 当前闭合两请求、work-conserving reorder formulation 无 action headroom | 不得放宽 κ、改模型阈值或查看 holdout 抢救 |
| fixed RankLane | `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20` | 在 `p_return≤20%`、codec 等税为零的冻结域内，最乐观相对 uniform FP8 E2E 改善仅 4.1667% | 不等于真实 8×A100 return path 不存在 |

### 2.2 尚未形成科学结论的资产

| 对象 | 已有内容 | 缺口 | 当前标签 |
|---|---|---|---|
| CPR 5090 quick validate | runner、kernel、provenance 校验、CPU 测试；旧质量数值重分析为 PASS | 同次 provenance 的正式质量重跑、5090 connected INT4 正式运行；核心仍需 8×A100 | `INCOMPLETE_NECESSARY_GATES` |
| BCRD | 三道串行 Gate、identity/oracle/causality 测试和 smoke harness | 自然 continuous-decode routes、完整路径 denominator、正式 5090 service curve、正式 Gate 1 结果 | `DESIGNED_AND_IMPLEMENTED / NOT_FORMALLY_RUN` |
| DEPA | 因果 CPU replay、request ledger、exact small oracle、三门 runner、10 个测试 | 四项 formal capability 均为 false；无正式 breakdown/episodes/surface；无完整 prior-art 边界文档 | `DEVELOPMENT_ONLY_NOT_SCIENTIFIC` |
| optimized EP return-path existence | 8×A100 Gate 设计 | 真实 8×A100、optimized backend、timeline/transport/identity 闭合 | `NOT_TESTED_REQUIRES_8XA100` |

## 3. 唯一研究问题

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

所以当前先冻结共同输入和现象门，再依据结果选择一个 formulation；不得并行调参竞争。

## 4. 唯一执行线

### Gate 0：正式资格预检

以下四项缺一项即停止在 `BLOCKED_MISSING_FORMAL_EVIDENCE`，不能用开发夹具代替：

1. native continuous-decode route producer；
2. RTX 5090 实测 expert service surface，且仅在测量区间内保守插值；
3. identity-complete、exact-output replay；
4. 冻结的 natural workload / model / revision / seed manifest。

还需一个与目标 accounting boundary 一致的完整路径 denominator。不能从 proposed saving 反推 denominator。

### Gate 1：共同现象测量

同一份冻结证据同时报告两组互不替代的量：

- **BCRD-specific：** replica fragmentation penalty。进入其 Oracle 门要求两模型共同自然 cell point estimate ≥10%、95% LCB >5%，至少一个 common cell 两模型均 ≥15%，actionable waves ≥20%。
- **broader expert-pressure：** fragmentation + route-conditioned straggler + HoL 的去重 exposed share。进入 DEPA 方向复审要求 point estimate ≥20%、95% LCB ≥10%。

统一停止规则：

- 所有共同自然 cell `<5%` 或 actionability `<20%`：停止 expert-pressure/BCRD 机制线；
- 只有 5%–10%：降级为 measurement/kernel 问题，不进入复杂 controller；
- 只在 synthetic skew、单模型、单层、极端 replica 数成立：不构成论文主线；
- identity、route、rows、request ledger 或 denominator 不闭合：`INVALID`，只修测量，不解释收益。

### Gate 2：只选择一个 exact Oracle

Gate 1 后按现象来源分叉，但一次只允许一个分支：

- 如果增量主要来自 fixed replica set 内的 rows fragmentation，选择 **BCRD** 的 `assignment + bounded seal time` exact Oracle；
- 如果 BCRD-specific 不成立，但 broader HoL/admission exposed share 独立成立，先完成 DEPA 的 prior-art/action-space 收缩审计，再选择 **DEPA** 的小规模 exact SLO-goodput Oracle；
- 如果两者都成立，优先选择动作更少、因果链更短、最强简单 baseline 更难覆盖的 formulation；该选择必须写回本文后才能进入 Gate 2。

Oracle `<10%`、动作等价或只在 zero-cost / future-unbounded 条件下成立，立即停止。

### Gate 3：简单策略差距

统一比较 current/default、deterministic random、threshold、EDF/least-laxity、greedy 和 chosen proposed policy；参数只能在 calibration split 选择。

- 简单策略捕获 ≥90% Oracle：取消复杂 controller；
- proposed 必须在冻结 evaluation 上同时满足净收益、Oracle 捕获率、相对简单策略差距、P99/公平性和决策开销门；
- 单卡通过只授予 `8×A100 Candidate`，不是系统 GO。

## 5. 与 8×A100 return-path Gate 的关系

optimized EP return-path existence 是另一条**硬件条件分支**，不是当前 5090 主线的并行任务。

- fixed RankLane 在 `p_return≤20%` 域内已经 NO-GO；
- 只有真实 8×A100 证明优化后 exposed return fraction 达到记录的 reopen condition，才允许重新评估 return representation；
- 在此之前不实现 RankLane codec/controller，不把 CPR quick gate 写成 EP 结果。

## 6. 明确停止和保留清单

### 停止

- PhaseMap 当前 closed-pair formulation；
- FJRC keyed Join-Deficit bitmap；
- ConfidenceGuard v3 当前 prefill-risk audit allocation；
- fixed RankLane actuator 在 `p_return≤20%` 冻结域内；
- 已归档的 CreditReduce、MassCover、TokenRace、QuotaEP、PLTB additive、RouteFidelity、Prefetch 等 formulation。

### 仅保留为证据或工具

- Rank-tail 结构证据：motivation；
- 5090 LUT/codec：局部测量工具；
- shared route/modeling/metric 代码：公共实验底座；
- BCRD/DEPA runner：Gate 实现，不是实验结论。

## 7. 当前最短执行清单

1. 不再生成新 idea 文档，也不再改 Top 3 排名。
2. 复核 BCRD 与 DEPA 的输入 schema，冻结一份共同 manifest 和 accounting boundary。
3. 补齐 Gate 0 的四项正式能力及 full-path denominator。
4. 只运行共同 Gate 1；先看科学 verdict，再决定是否进入一个 Oracle 分支。
5. Gate 失败就记录窄负结果并停止，不调 workload、阈值或指标救活。

## 8. 文档裁决顺序

发生冲突时按以下顺序解释：

1. 机器可读 sealed/formal decision；
2. 本文的当前状态和执行线；
3. 专项实验裁决/协议；
4. 候选生成、Top 3、reviewer/red-team 长文；
5. 历史 meeting note 或探索性输出。

新的正式结果必须同时更新本文和对应 idea README。只新增一份“唯一主攻”文档而不更新本页，视为来源快照，不改变执行顺序。

## 9. 一句话状态

> 目前已完成的是候选清理、多个 formulation 的窄负结果和两套严格 Gate harness；尚未完成的是能让 BCRD 或 DEPA 成为论文主机制的正式自然 workload 现象验证。当前唯一工作是共同 Gate 1，而不是继续扩展 controller。
