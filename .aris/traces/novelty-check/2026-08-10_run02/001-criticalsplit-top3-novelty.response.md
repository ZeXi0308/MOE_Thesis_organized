# Novelty Check Report

**评审分类：`same-family / provisional`。**  
这是同模型家族的查新评审，不是跨模型独立裁决，也不声称绝对 novelty。检索覆盖截至 **2026-08-10** 的原始论文、会议页面与官方技术文档。

## 总体结论

| 候选 | Method novelty | Finding novelty | 决定 |
|---|---:|---:|---|
| C01 JoinStream | **5/10** | **6/10** | **CAUTION** |
| C07 QuantumYield | **2/10** | **4/10** | **ABANDON** |
| C12 ShapeLease | **3/10** | **5/10** | **ABANDON 作为方法；仅保留 benchmark finding** |

**首跑结论：C01 CPU exact Oracle 仍值得作为唯一首跑。**  
原因不是它已被证明新颖，而是：

1. C07 的核心动作已被 QLLM 与 ExpertPlex 高度覆盖；
2. C12 的动态 tactic selection 已被 EPS-MoE、RaMP、DA-MoE直接覆盖；
3. C01 尚余一个较窄但清晰的交叉点：**固定 whole-batch launch 与总 service，只改变真实 row completion 的可见性和顺序，并让跨 executor 的 top-k join 立即释放下游**。

该 CPU 首跑只能回答 **action-space existence**。正结果不授权 GPU、EP/NCCL、自然负载收益或论文 novelty。

---

## C01 — JoinStream

### Core technical claims

| Claim | Closest prior work 与具体重合 | 最小可保留 delta |
|---|---|---|
| 1. 单次 whole expert batch launch 内暴露 row milestone，而不改变成员、busy interval、最终 finish 或总 service | [FlashMoE](https://arxiv.org/html/2506.04667) 已用单个 persistent kernel 把 dispatch、expert compute、combine 分成 tile tasks；[Event Tensor](https://openreview.net/forum?id=PJqFhAbUHa) 已把 tile task completion 和数据相关依赖建模为一等事件 | **不是一般 tile completion**，而是一个可验证的 runtime completion contract：同一 expert batch 仍只 launch 一次，外部消费者能在 kernel 最终完成前安全观察特定 row 的最终 expert output |
| 2. 两个 top-k expert rows 到齐即可关闭该 token 的 join，并立即释放下一层 | [AMoE/AEP](https://arxiv.org/html/2505.08944) 已维护 token-level top-k dependency，但其 executor 在整个已选 layer batch 完成后才 dispatch outputs；[QLLM](https://arxiv.org/html/2503.09304) 也区分 partial/complete top-k tokens；FlashMoE 已在 tile 级调度 combine | 必须证明 **per-token join close → exact downstream release**，且同 batch 其他 rows 继续运行；不能只是 kernel 内部提前计算 combine、最终仍等待整个 operator |
| 3. 两个 expert executors 联合优先完成同一 token 的两条 rows，使下一层与尾 rows 重叠 | [ExpertPlex](https://arxiv.org/html/2607.18002) 已在 persistent kernel 内做 tile-level scheduling 和 bounded urgent-work preemption，但其优先级是 prefill/decode phase；AMoE/Gimbal 做 queue/request pressure scheduling | 剩余 delta 是 **跨 executor、以 top-k join closure 为唯一依赖信号的 row-order coordination**，不是 phase priority、request priority 或 expert queue scoring |
| 4. 收益不依赖 input split、re-batching 或新 launch | [FinDEP](https://arxiv.org/html/2512.21487) 明确切分 tensor 和 computation/communication tasks；[ExpertFlow](https://arxiv.org/html/2410.17954) 跨 batch 重排 tokens；[Expert Streaming](https://arxiv.org/html/2603.27624) 流式处理 expert micro-slices | 必须保持原 batch membership、一次 launch 和总 service。否则会退化成已经拥挤的 chunk pipeline、re-batching 或 persistent tile scheduling |

### Method novelty

**5/10，CAUTION。**

文献已经覆盖了三个基础构件：

- tile-level completion/event；
- top-k partial dependency tracking；
- persistent-kernel 内的动态 task scheduling。

目前未在核对的原文中发现对以下**完整组合**的直接声明：

> 保持 whole expert batch 的成员、一次 launch、总 service 与最终 finish 不变，仅让真实 row completion 提前成为跨 expert 的 token-join event，并据此启动下一层。

但这只是“未发现直接覆盖”，不是绝对 novelty。FlashMoE 与 Event Tensor 已使审稿人很容易质疑：JoinStream 是否只是已有 fine-grained task graph 的一个 MoE-specific scheduling policy。

### Finding novelty

**6/10。**

可能保留的新 finding 是：

> 在不靠拆 batch、re-batching 或额外 service 的情况下，join-aware row order 是否仍能产生严格的 request-DAG flow improvement。

这个 finding 比“能够做 tile completion”本身更有价值，但必须击败 generic tile/event scheduling，而不能只击败 atomic `WHOLE + HOLD`。

### 未验证项

- **未验证**真实 grouped-GEMM/persistent kernel 能否改变 row physical order，同时严格保持相同总 service 和最终 finish。
- **未验证**row result 在整个 kernel 完成前是否具有可消费的 memory visibility、数值完整性与低成本通知机制。
- **未验证**下一层可否在这些局部结果上合法启动而不遇到其他 non-separable operator 或 collective barrier。

---

## C07 — QuantumYield

### Core technical claims

| Claim | Closest prior work 与具体重合 | 最小可保留 delta |
|---|---|---|
| 1. running expert work 在固定 quantum 上 cooperative preempt/resume，已完成 work 不丢失 | [ExpertPlex](https://arxiv.org/html/2607.18002) 已在 persistent kernel 的 tile commit boundary 实现 bounded preemption；未领取的 tiles 保留，切换不需要 checkpoint、recompute 或 kernel relaunch | 只能保留“join-closing queue 是 preemption trigger”，不能再把 running-state preemption 本身作为贡献 |
| 2. MoE expert/layer partial state 可保存并恢复 | [QLLM](https://arxiv.org/html/2503.09304) 已提供 per-expert queues、partial top-k state、expert/layer-level preemption 与后续恢复；它并非精确的 mid-GEMM checkpoint，但已覆盖 MoE-specific state-preserving preemption contract | 必须限定为**同一 running expert kernel 内的 physical remaining-work conservation**，而非 layer/expert boundary context switch |
| 3. urgent join-closing work 优先于普通 request priority | QLLM 优化 LS/BE priority；ExpertPlex 优化 decode 对 prefill 的抢占；[FastServe](https://arxiv.org/html/2305.05920) 是 output-token/iteration boundary 的 MLFQ preemption | 仅剩“join closure 比 request/phase priority 更好”的 finding；这是调度信号 novelty，不是 preemption mechanism novelty |
| 4. future-aware HOLD 仍无法表达 running-state transition | 该区别在调度模型中成立，但 ExpertPlex 已提供真实 running-state action | Exact Oracle comparator 是严谨实验设计，不构成独立方法贡献 |

### Method novelty

**2/10，ABANDON。**

ExpertPlex 对核心机制的碰撞非常直接：persistent MoE kernel、tile boundary、bounded urgent-work preemption、无 checkpoint/recompute。QLLM 又覆盖了 MoE expert-level partial-state preemption 与恢复。

把 priority 从 LS/BE 改成 top-k join closure，不足以重新形成高 novelty 的系统方法。

### Finding novelty

**4/10。**

仍可能有一个窄 finding：

> 在未来到达已知且 HOLD 已最优的条件下，join-aware preemption 是否仍改善 request-DAG flow，以及该收益能否覆盖真实 tile-boundary penalty。

这可以是调度 characterization，但不宜继续包装成 QuantumYield 新机制。

---

## C12 — ShapeLease

### Core technical claims

| Claim | Closest prior work 与具体重合 | 最小可保留 delta |
|---|---|---|
| 1. 根据 runtime shape/load 在多个 MoE kernel tactics 间选择 | [EPS-MoE](https://arxiv.org/html/2410.12247) 已根据 load 动态选择 GroupGemm/DenseGemm；[RaMP](https://arxiv.org/html/2604.26039) 从 runtime expert histogram 在 134–268 个 kernel configurations 中选择；[DA-MoE](https://arxiv.org/html/2607.23099) 按每次 invocation 的 live histogram 在 GPU 上选择 fused-MoE kernel | 动态 tactic selection 本身不再可声明为新 |
| 2. tactic 是跨 waves 的 resident state，切换支付显式 reconfiguration cost | RaMP/DA-MoE 主要是 stateless/per-step dispatch；[ExpertPlex](https://arxiv.org/html/2607.18002) 有 persistent runtime state；补充近邻 [Moebius](https://arxiv.org/html/2606.26607) 已把 EP/TP 作为 resident runtime states，并显式支付 live switch cost | 只剩 **同一 MoE operator 的 native/padded/persistent tactics 之间，带 residency、hysteresis 和切换成本的跨-wave决策** |
| 3. expanded Oracle 必须严格击败最佳 fixed tactic 且实际使用至少两种 tactics | 这是比多数论文更严格的 action-space判据，但属于实验规约 | 可保留为 benchmark/负结果 protocol，不能单独支撑 method novelty |
| 4. logical whole queue 保持不变，变化仅来自 physical tactic | [vLLM modular fused-MoE](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) 已提供可替换 prepare/finalize 与 experts implementations，但其公开文档中的 `select_gemm_impl` 主要发生在 kernel object 初始化；[UltraEP](https://arxiv.org/html/2606.04101) 则按每个 microbatch/layer 动态规划并运行 persistent tile stream | 需要用真实 kernel state 证明 reconfiguration 不是普通 dispatcher 的附加 switching penalty |

### Method novelty

**3/10，ABANDON 作为论文方法。**

EPS-MoE、RaMP、DA-MoE 已直接占据“MoE runtime adaptive kernel selection”主张；Moebius 又表明“resident runtime state + paid transition + beat static layout”并非新的系统模式。

精确的 `native/padded/persistent` 三 tactic 组合尚未在这些原文中核实到，但更换 tactic 名称不足以形成 method novelty。

### Finding novelty

**5/10。**

可以保留为 benchmark finding：

> 在真实测量的 service table、真实 reconfiguration cost 和连续 workload waves 下，带状态的 tactic switching 是否严格击败所有 fixed tactics；若不能，则证明 stateless per-invocation dispatch 已足够。

当前 jury 中的 synthetic crossover table 不足以形成该 finding；它容易人为编码 adaptive Oracle 优势。

---

## 指定近邻逐一核对结果

| 来源 | 原文核对后的边界 |
|---|---|
| [AMoE/AEP](https://arxiv.org/html/2505.08944) | layer µ-queue、token metadata、top-k pool、adaptive re-batching；不提供同一次 whole expert launch 内的 row milestone |
| [ExpertPlex](https://arxiv.org/html/2607.18002) | adaptive persistent kernel、tile scheduling、bounded tile-level preemption；是 C07 的直接强碰撞，也是 C01/C12 的强基础近邻 |
| [ExpertFlow](https://arxiv.org/html/2410.17954) | 跨两个 batches 的 token rearrangement、routing prediction 与 expert cache；不是固定 membership 的 completion contract |
| [FinDEP](https://arxiv.org/html/2512.21487) | 切分 AG/EG computation 与 communication tensors，细粒度流水线；不是一次 whole launch 下的 row exposure |
| [Expert Streaming](https://arxiv.org/html/2603.27624) | chiplet expert micro-slice trajectories、expert completion events、token buffering；硬件/dataflow 相邻，但未覆盖 C01 的跨 executor token join contract |
| [Gimbal](https://arxiv.org/html/2606.15177) | request/DP pressure、queue ordering、source-aware placement；不进入 running expert batch 的 row/tile completion |
| [QLLM](https://arxiv.org/html/2503.09304) | MoE expert-level preemption、partial top-k state、batch/sequence恢复；直接削弱 C07 |
| [FastServe](https://arxiv.org/html/2305.05920) | output-token/iteration boundary preemption；是通用 preemption 基线，但弱于 C07 所设的 running expert transition |
| [DA-MoE](https://arxiv.org/html/2607.23099) | live routing histogram 驱动 GPU-resident fused-kernel dispatch；直接削弱 C12 |
| [EPS-MoE](https://arxiv.org/html/2410.12247) | load-aware GroupGemm/DenseGemm switching；直接覆盖 C12 的基本动机 |
| [RaMP](https://arxiv.org/html/2604.26039) | runtime histogram、wave cost model、大规模 polymorphic configurations；是 C12 最接近的 kernel-selection prior work |
| [UltraEP](https://arxiv.org/html/2606.04101) | 每 microbatch/layer 的 exact-load runtime planning 与 persistent tile streaming；主要是 load balancing，不直接覆盖 ShapeLease tactics |
| [vLLM modular fused-MoE](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) | 官方模块化 kernel contract、可替换 GEMM/prepare/finalize；公开文档未显示跨 waves 的 adaptive tactic residency |

补充发现中，真正改变判断的是 [FlashMoE](https://arxiv.org/html/2506.04667)、[Event Tensor](https://openreview.net/forum?id=PJqFhAbUHa) 和 [Moebius](https://arxiv.org/html/2606.26607)。

---

## C01 CPU exact Oracle：最终首跑判断

**是，仍值得首跑，但必须标为 `ACTION_SPACE_FALSIFICATION_ONLY`。**

允许解释：

- 负结果：削弱所冻结的 uniform/tail-heavy milestone curves 与 emission-tax cells；
- 正结果：说明 atomic batch completion 隐藏了一个可利用的调度自由度；
- 不允许解释：GPU kernel 可实现、自然负载有效、优于 FlashMoE/Event Tensor、EP/NCCL 收益或论文方法新颖。

C07 不应先跑，因为其 action 已被强 prior art 覆盖。C12 也不应使用 synthetic table 先跑；没有真实 tactic latency、transition cost 和 residency state 时，Oracle 胜利很容易由输入表机械制造。

## 真正会改变首跑判断的 P0/P1

1. **P0 — C01 physical legality**：必须给出可执行的 producer/consumer contract，说明 row output 如何在 whole kernel 结束前变为数值完整、memory-visible、可通知且可被下一层消费。若做不到，取消 CPU 首跑，因为该动作是虚构的。

2. **P1 — C01 prior-art comparator closure**：必须确认 JoinStream 的关键 schedule 不能直接由 FlashMoE/Event Tensor 的普通 tile task graph 表达。若只需换一个 ready-task priority，方法主张应放弃；CPU pilot最多保留为 join-aware finding。

3. **P1 — frozen physical curve**：milestone curve、允许的 row orders、notification/emission tax 必须在求 Oracle 前冻结，并独立于 request outcome。若 Oracle 能选择或间接塑造 milestone 时间，正结果无效。

## 最终 provisional verdict

`C01 JoinStream → CAUTION / 首跑保留`  
`C07 QuantumYield → ABANDON`  
`C12 ShapeLease → ABANDON method，finding-only`

当前唯一值得执行的是 C01 的 CPU exact action-space kill probe；其后仍需真实 GPU completion legality 和 strongest-prior-art baseline，不能从 CPU 正结果直接升级为系统机制。
