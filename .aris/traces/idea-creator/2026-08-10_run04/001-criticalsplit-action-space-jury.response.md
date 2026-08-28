# Fresh Research Jury 原始评审

- `review_independence: same-family`
- `acceptance_status: provisional`
- 证据范围：仅使用指定的 6 个文件。
- 本评审不是跨模型独立裁决，也不是最终选题结论。

## 1. CriticalSplit 封存负结果的 scope-of-failure

封存结果否定的是一个很窄的动作空间：

> 在冻结的 8 个 synthetic cells、固定完整请求 DAG、固定服务曲线和原子 batch completion 语义下，向 `WHOLE + bounded HOLD` 增加 revealed join-closing `CRITICAL/BULK` proper-subset launch，没有改善 exact Oracle。

具体证据边界：

- 8/8 cells 中，expanded split exact Oracle 与 whole-ready exact Oracle 的 flow 完全相同。
- `eligible_cells=0`，最优轨迹中 `critical_launches=0`、`bulk_launches=0`。
- 4 个 cell 没有 immediate→whole Oracle headroom；另 4 个 cell 的 whole Oracle 已捕获全部 headroom。
- loose/tight deadline 成对得到完全相同的 flow。因为 Oracle 的词典序目标首先最小化 flow，deadline 只在 flow 相同时参与后续比较，所以这 8 个标签实际上只有 4 种不同的调度动力学。
- `RunningBatch` 只有一个 `finish_us`；一个 batch 中所有 rows 在同一时刻完成。
- top-k 全部完成后才产生固定 `combine_us` release。提前完成非 join-closing row 没有可见的下游价值。
- 当前曲线为 M=1/2/4 对应 10/14/20 μs。典型二行 batch 拆成两个 singleton 后，总 service 从 14 増至 20 μs；它只是重新分配尾延迟，没有创造新资源或 producer–consumer overlap。
- whole-ready Oracle 已枚举所有 idle executor 上的全部 eligible queues 和 bounded HOLD，因此普通 EDF、queue score、cross-request priority 不属于缺失动作。

它不能否定：

- batch 内 row/token milestone；
- incremental exact reducer/consumer；
- running batch 的 preempt/resume；
- replica mobility；
- open/append lifecycle；
- 冗余 physical copy；
- link、DMA、reducer 等多资源状态；
- kernel tactic/residency；
- dispatch–compute–return 的内部 pipeline。

也不能证明或否定自然负载收益、GPU kernel 可行性、EP/NCCL/RDMA、在线策略或论文贡献。

因此，不应修复、调整或继续变化 Critical/Bulk；下一步必须换物理动作或 completion semantics。

## 2. 全部 14 项评分与排序

五维等权；总分仅用于排序。CCF C/B 分数只依据当前直接碰撞摘要，不能替代完整查新。

| 排名 | 候选 | 动作新颖性 | Exact Oracle upside | 系统深度 | CCF C/B | CPU Oracle 信息/成本 | 总分 | 最强 objection |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | C01 JoinStream | 9 | 9 | 8 | 8 | 9 | **43** | 如果 milestone curve 和 row order 可由 Oracle 任意塑造，正结果可能只是把答案编码进完成时间；必须冻结与请求结果无关的物理曲线。 |
| 2 | C07 QuantumYield | 9 | 8 | 9 | 8 | 7 | **41** | CPU 中保存 `remaining_service` 很容易，但真实 GPU expert batch 未必存在低成本、精确、可恢复的 cooperative preemption boundary。 |
| 3 | C12 ShapeLease | 6 | 8 | 9 | 7 | 8 | **38** | 只要人为设置交叉 service table，就能制造 adaptive Oracle 胜利；必须预先冻结 table，并击败每个 fixed-tactic exact Oracle。 |
| 4 | C11 HedgeJoin | 8 | 7 | 8 | 6 | 7 | **36** | 冻结 simulator 没有天然 stochastic straggler；若人为注入局部慢路径，很容易机械地制造 hedge 收益，且取消成本可能被低估。 |
| 5 | C09 MultiExpertFuse | 6 | 8 | 9 | 7 | 6 | **36** | mixed-expert curve 本身可以编码优势；若不能证明 join-specific contract，它更像 grouped/fused launch 的 synthetic 重述。 |
| 6 | C02 AccuJoin | 7 | 6 | 7 | 7 | 8 | **35** | 当前 `combine_us=1`，可移动工作很小；把 movable fraction 调大后得到正结果，只能说明人工增加的 reducer tail 可被覆盖。 |
| 7 | C04 TileConsume | 7 | 7 | 8 | 7 | 5 | **34** | 可 exact-safe 前移的 suffix 可能在第一个 normalization、attention 或其他 non-separable operator 前就结束，真实可覆盖工作可能过小。 |
| 8 | C06 ReplicaSteal | 4 | 8 | 8 | 5 | 8 | **33** | UltraEP/Gimbal 已形成动态 replica/load balance 与 placement 强近邻；若 residency 和迁移成本真实收费，剩余 novelty 很窄。 |
| 9 | C08 OpenBatch | 5 | 7 | 7 | 5 | 7 | **31** | AMoE 的 adaptive re-batching 是强近邻；如果 setup 不能与 future arrival 真正重叠，OPEN 退化为旧 Oracle 已有的 HOLD。 |
| 10 | C05 RouterWave | 8 | 5 | 7 | 6 | 4 | **30** | 当前 DAG 中下一 layer 只在 top-k join 和 combine 后 reveal；没有证据表明 next-router 输入能绕过中间 non-separable operators 提前合法产生。 |
| 11 | C03 SegmentReturn | 4 | 7 | 8 | 5 | 5 | 与 FinDEP 的细粒度 communication/compute pipeline 高度碰撞；CPU 正结果本身无法建立 MoE join-specific contribution。 |
| 12 | C10 DuplexWave | 3 | 7 | 8 | 4 | 6 | 显式 link/compute overlap 已被 FinDEP、ExpertPlex、Aurora、Lina 等强邻域覆盖；仅新增资源模型不足以形成论文贡献。 |
| 13 | C13 CutThroughEP | 2 | 8 | 9 | 3 | 3 | 它几乎就是 FinDEP 式 mandatory-stage chunk pipeline；状态空间很贵，而普通 positive result 仍不具备 join-specific novelty。 |
| 14 | C14 RouteAhead | 1 | 6 | 7 | 2 | 5 | SpecPrefetch、MoE-Infinity、fMoE 已直接覆盖 next-expert prefetch/cache；即使 exact Oracle 为正，也更适合作为低 novelty control。 |

## 3. 机制不重复的 Top 3

1. **C01 JoinStream**：completion semantics。
2. **C07 QuantumYield**：running-state transition。
3. **C12 ShapeLease**：physical execution tactic/residency。

三者分别来自 H1、H2、H4，不共享核心 actuator。

## 4. Top 3 最小 CPU exact Oracle

### Top 1：C01 JoinStream，8 cells

目的：只回答“原子 whole-batch finish 是否隐藏 row-prefix headroom”。

固定结构：

- 2 requests、2 replicas、top-k=2、2 layers。
- 原有 service curve保持 M=1/2/4 → 10/14/20 μs。
- executor 在最后 milestone 前始终 busy。
- 每个 row 只完成一次；join 和 `combine_us=1` 规则保持不变。
- `LAUNCH_WHOLE_STREAM` 不改变 batch membership，只能选择当前已 reveal rows 的合法顺序。

8-cell 因子：

- rows：`M∈{2,4}`
- milestone curve：
  - uniform：`t_i=S·i/M`
  - tail-heavy：`t_i=S·sqrt(i/M)`
- emission tax：`h∈{0,2}` μs

所有 milestone 为 `h+t_i`，最终 executor finish 为 `h+S`。其中 `h=0` 严格测试“相同最终 finish”；`h=2` 测试有限成本下是否仍有 headroom。

Exact comparator：

- Baseline：原有 `WHOLE + HOLD` exact Oracle。
- Expanded：baseline actions 加 `LAUNCH_WHOLE_STREAM(queue,row_order,frozen_curve)`。
- 两边使用相同未来可见性、DAG、combine、目标函数和 replay conservation。
- milestone curve 由 cell 固定；Oracle 只能选择合法 row permutation，不能选择 milestone 数值。

判读：

- 严格 flow improvement 且最优轨迹实际使用 stream action：证明所测 milestone contract 扩大了 action space。
- `h=2` 仍有 improvement：比仅 `h=0` 的 existence signal 更强。
- 全负只削弱这两条冻结曲线，不否定 transport streaming、consumer streaming 或其他 milestone。

### Top 2：C07 QuantumYield，8 cells

目的：判断 non-preemptive running batch 是否是 whole/HOLD Oracle 无法跨越的阻塞。

固定结构：

- 两个 top-k 请求；urgent request 的另一个 sibling 在 replica 1 上先完成，使 replica 0 上的新 row 成为 join-closing blocker。
- preemption quantum `q=2 μs`。
- 每个 running batch 最多 preempt 一次。
- 已执行 work 精确保留；resume 执行全部 remaining service，并支付冻结 penalty。
- suspended batch 不能迁移、拆分或丢弃。

8-cell 因子：

- incumbent rows：`M∈{2,4}`
- urgent reveal offset：`δ∈{3,7}` μs
- resume penalty：`p∈{0,2}` μs

deadline 固定 loose，不把 loose/tight 作为主因子，因为当前 objective 首先优化 flow，封存结果已表明 deadline 标签没有改变调度动力学。

Exact comparator：

- Baseline：`WHOLE + HOLD` exact Oracle；它可以预知 urgent arrival，因此必须允许其选择提前 HOLD，而不能强迫它先启动 incumbent。
- Expanded：baseline 加 `PREEMPT`、urgent whole launch 和 `RESUME`。
- 收费为已执行 work + urgent service + remaining work + penalty；禁止遗失或重复 work。
- 同样要求 action replay、node conservation 和 terminal-state closure。

判读：

- 只有 expanded Oracle 严格击败可预知 future 的 baseline，才说明 preemption 提供了 HOLD 无法表达的新能力。
- 收益仅存在于 `p=0`：只能算脆弱 existence。
- `p=2` 仍有收益：更有资格进入后续实现评估。
- 负结果只否定冻结 quantum/offset/penalty，不支持返回 Critical/Bulk。

### Top 3：C12 ShapeLease，16 cells

目的：判断 tactic/residency 状态是否产生任何固定 tactic Oracle 都无法覆盖的收益。

冻结 synthetic service table：

| Tactic | M=1 | M=2 | M=4 |
|---|---:|---:|---:|
| native | 8 | 14 | 24 |
| padded | 12 | 13 | 17 |
| persistent | 10 | 11 | 18 |

该表只能用于 CPU action-space existence，不能声称代表 GPU。

每个 episode 使用五个顺序 release waves；每个 wave 的 row count 由冻结 DAG arrival 生成，而不是由求解器直接注入。

16-cell 因子：

- mixture：`sparse-major / dense-major`
- reuse order：`blocked / alternating`
- initial residency：`cold-none / native-warm`
- reconfiguration cost：`c∈{1,4}` μs

建议冻结序列：

- sparse-major blocked：`[1,1,1,2,4]`
- sparse-major alternating：`[1,4,1,2,1]`
- dense-major blocked：`[4,4,4,2,1]`
- dense-major alternating：`[4,1,4,2,4]`

动作与状态：

- `RECONFIGURE(replica,tactic)`：支付 `c` 并更新 resident tactic。
- `LAUNCH_SHAPE(queue,tactic)`：只处理 whole-ready membership。
- 不允许 row split、Critical/Bulk 或同时采用多个 tactics。
- 同一初始 residency 对所有 comparator 生效。

Exact comparator：

- 分别求三个 fixed-tactic exact Oracles：整集 episode 只能用 native、padded 或 persistent。
- Baseline 取三者最优值，而不是任选一个弱 comparator。
- Expanded Oracle 可按 wave 切换 tactic，并支付全部 reconfiguration cost。
- 正结果必须同时满足：
  - 严格击败最佳 fixed-tactic exact Oracle；
  - 最优轨迹实际使用至少两个 tactics；
  - 所有 init/reconfiguration/service work 在 replay 中闭合。

否则只能说明某个固定 tactic 更优，不能支持 ShapeLease。

## 5. 唯一首跑

**唯一首跑：C01 JoinStream 的 8-cell exact Oracle。**

原因：

- 它最直接针对封存结果暴露出的原子 completion 假设。
- 不需要增加 replica、link、cache、tactic 或 suspended-batch 等新资源。
- 只需把 `RunningBatch` 的单一 finish 扩展为冻结 row milestones，状态增量最小。
- baseline 可完全复用现有 whole-ready exact Oracle。
- 无论正负，信息都清晰：它回答 completion semantics，而不是再次测试 queue membership slicing。
- 它的实现与求解成本显著低于 C07/C12，同时比 C02/C03 更少依赖人为放大 combine/communication work。

本首跑不包含、修复、调参或变化 Critical/Bulk。

## 6. 会改变结论的 P0/P1

1. **P0 — C05 causal legality**  
   当前代码只在 top-k join 加 combine 后 reveal 下一 layer。若真实 operator DAG 不能证明 tile 到 next-router logits 的 exact、无 non-separable barrier 路径，C05 不是低收益机制，而是非法动作，应直接出局。

2. **P1 — C01 milestone contract**  
   milestone curve 必须独立于请求结果预先冻结；Oracle 只能选择物理可表达的 row order。若 curve 和 order 都可任意选择，C01 的 positive result 会被高估，可能失去第一名。

3. **P1 — C12 comparator/table freeze**  
   tactic table、初始 residency、reconfiguration cost 必须在搜索前封存，并与三个 fixed-tactic exact Oracles比较。否则 ShapeLease 的优势可以被 service table 人工制造，不能保留 Top 3。

无其他会改变本轮排序或首跑选择的 P0/P1。

## 最终 provisional verdict

在当前证据范围内，最合理的路线不是继续挽救 Critical/Bulk，而是依次检验：

`row milestone completion → running-batch preemption → tactic/residency scheduling`

首跑仅授权 C01 的 CPU exact action-space existence 检验。任何正结果仍然不是 GPU、EP、NCCL/RDMA、自然负载或论文结果。

本结论为 **same-family / provisional**，不是独立最终裁决。
