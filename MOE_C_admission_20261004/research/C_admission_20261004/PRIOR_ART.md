# C 准入研究：相关工作与动作边界

首轮检索日期：2026-10-04；定向更新：2026-10-07。首轮范围为 CONCUR、内存约束下的 LLM 准入控制及原生运行时配置；本次重开补核 CacheOPT、Scorpio 与持续服务近邻 JITServe。只核查原始论文、正式出版页面与作者项目；不是穷尽性综述，不据此声称首次提出恢复感知准入。

## 问题和可能残差

C 的待验证问题是：在实际 KV 可用量、活动请求数相近时，**真实等待恢复的请求及其消退变化**，能否提供额外信息，使只控制新请求首次 prefill 的机制改善全部外部到达请求的完整服务结果。

动态限流、缓存反馈、阈值迟滞以及防止 KV thrashing 本身均不是新贡献。可能的研究残差是恢复队列状态的增量决策价值及其适用边界；这必须由实际准入干预、完整等待计时、强简单基线和去信号消融支持。以下差异只是待验证的定位，不是已有创新性结论。

## 首轮背景来源

| 工作 | 信号与动作 | 目标和运行域 | 与 C 的边界 |
|---|---|---|---|
| CONCUR | KV 使用率和命中率反馈；AIMD 调整 agent 并发窗口，在 generation/tool 边界 admit、pause、resume | 离线多轮 agent batch，缓解前缀逐渐增长及异步工具调用引起的缓存 thrashing；主要报告整批完成时间 | 已覆盖“缓存拥塞反馈控制并发”。C 不能将此作为新颖性。C 首轮只延迟调度器内等待的新请求首次 prefill，保持已有请求调度与恢复不变；额外信号是实际等待恢复状态。 |
| WAIT / Nested WAIT | 根据 prefill/decode 阶段组成设置阈值；控制准入及已有请求在阶段之间推进；未知长度版本只用逐步显露的继续生成信息 | 单 GPU 内存约束、KV 随生成增长、溢出驱逐重启；分析稳定性，评估完成吞吐、到达至完成时间和 TTFT | 已覆盖过度准入、内生 KV 增长与驱逐重启。其动作可改变已有 decode 推进，超过 C 首轮动作范围；模型不单独刻画 CPU/SSD 换出换入时延。 |
| vLLM 原生调度配置 | 固定 RUNNING 上限、空闲 KV 水位、完整输入长度预留 | 通用请求调度与 KV 压力管理 | 说明固定并发和 KV 水位是必要的强简单基线；当前在线文档不代表本库实际安装版本具有同名字段，必须以运行代码为准。 |

### 1. CONCUR

- 原始论文：[CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control](https://arxiv.org/abs/2601.22705)，arXiv:2601.22705，v1 提交于 2026-01-30。2026-10-07 已核实 [ICML 2026 正式出版页](https://proceedings.mlr.press/v306/chen26fs.html)，PMLR 306:17881–17891；[该卷](https://proceedings.mlr.press/v306/)发布于 2026-09-29。arXiv HTML 仍显示旧标题 *Concur: Proactive Agent-Level Admission Control for Efficient Agentic Batch Inference*。
- 本次下载并核对 [PMLR 正式 PDF §4.2–4.3](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chen26fs/chen26fs.pdf)：使用率低于低阈值时窗口加 α；使用率高于高阈值且命中率低于阈值时窗口乘 β；其余保持。它允许健康命中率下的高使用率继续运行；admit/pause/resume 位于 agent generation/tool 边界，不中断正在执行的 generation。此动作及控制律与首轮所读 [arXiv §4.3](https://arxiv.org/html/2601.22705v1#S4.SS3) 一致。
- 原文实验参数为 α=2、β=0.5、U_low=0.2、U_high=0.5、H_thresh=0.2。这些是论文设置，不应直接当作 C 的合理阈值。
- 该论文没有在所核查的控制规则中使用“实际待恢复请求数及其消退速度”。这个观察不等于证明所有相关工作均未使用该信息。

### 2. WAIT / Nested WAIT

- 原始论文：[Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)，Ruicheng Ao、Gan Luo、David Simchi-Levi、Xinshang Wang；v1 为 2025-04-15，当前摘要页标记最新版 v4 为 2026-06-13。
- [原文 v4](https://arxiv.org/html/2504.11320v4) 建模外部随机到达、阶段式服务和内生 KV 增长。WAIT 在设置阈值时假设输出长度及到达率已知；Nested WAIT 通过 decode 段边界逐步显露存活请求，不要求在单个请求准入时读取其未来真实输出长度。
- 其“preemption”指保留 GPU KV 的暂停；eviction 丢弃 KV 并返回 prefill，未单独建模 CPU/SSD 换入时延。算法也控制已有请求的阶段推进，不能直接作为 C 首轮同动作空间的替换实现。
- 适合作为问题建模和阈值准入的强近邻；首轮不必等待完整复现。后续若比较，应明确哪些动作或信息条件被限制，避免将简化移植称为完整复现。

### 3. vLLM 原生基线

- [官方 SchedulerConfig 参数文档](https://docs.vllm.ai/en/latest/cli/bench/throughput/#schedulerconfig) 当前列出：`max_num_active_seqs` 只限制进入 RUNNING 的数量；`watermark` 在接纳 waiting 或 preempted 请求时保留一部分空闲 KV；`scheduler_reserve_full_isl` 在接纳时检查完整 prompt 的空间。
- [官方优化指南](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption) 明确指出频繁抢占和重算会恶化端到端时延，建议降低 `max_num_seqs` 或 `max_num_batched_tokens` 等配置。
- C 应记录本库实际版本和参数语义。特别是，通用 watermark 可能同时阻止 preempted 请求进入 RUNNING；C 的实验约束要求新请求门控不能阻止已有请求恢复，因此不能不加区分地照搬。

## 对首轮实验的直接要求

1. 保留最佳固定配置、简单 KV 水位、恢复积压规则三臂；固定配置只做有限开发集选择，不将明显过载默认值作为唯一对手。
2. 记录恢复信号具体改变了哪些准入决策，以及当时 KV、活动请求数、实际待恢复状态和最老新请求等待时间。状态相关性不能替代干预收益。
3. 出现正信号后，先移除恢复信号但保留相同迟滞、限额与等待处理结构。效果不变则删除恢复信号，不继续包装。
4. 再补 CONCUR 风格双信号 AIMD 近邻：只使用本运行时具有实际语义的缓存拥塞/失败反馈，注明是移植基线。没有相应命中率语义时，不用任意代理伪称完整 CONCUR。
5. 计时从共同外部到达开始，包含准入前等待；完整报告超时、失败、未完成、排空和自然输出长度。收益必须在这个口径下成立，不能只展示已有请求 decode 停顿减少。

查新小结：相关工作已经覆盖普通反馈限流和 KV thrashing 管理。恢复状态是否具有新增价值，需要原生准入干预与完整请求结果回答，不能从机制名称或状态相关性推出。

## 首轮 formulation 的负结果与研究定位（2026-10-07）

已有动作并不空白：[CONCUR](https://arxiv.org/html/2601.22705v1#S4.SS3) 通过 KV 使用率/命中率反馈调整 agent 并发窗口，[WAIT / Nested WAIT](https://arxiv.org/html/2504.11320v4) 通过阈值控制准入及阶段推进，[vLLM](https://docs.vllm.ai/en/latest/cli/bench/throughput/#schedulerconfig) 提供并发上限、KV 水位与完整输入预留。因此 C 的问题是恢复状态是否能在这些简单资源约束之外，改变一个原本允许的新 prefill 决定，并改善全人口结果；“动态接纳”“多一个反馈信号”本身不足以构成贡献。

已完成的未优化版、优化版稳态及一次固定burst诊断中，每组的两遍 recovery `changed_by_recovery` 均为0；时延不跨代码版本或到达轨迹混合。现有证据能说明：当前阈值/迟滞、10 s 年龄豁免、KV 水位和原生 waiting 路径共同作用下，尚未观察到恢复信号独有的门控动作。它不能证明在相近 KV/并发下恢复信息对未来服务结果没有额外价值，也不能证明所有负载下恢复信息冗余；同状态 gate 判定相同更不等于不同运行的全轨迹相同。此时 latency/goodput 的波动、对 fixed 的 decode 间隔改善，均不能归因于恢复信号；仍须检查 KV 对照、控制开销、运行级差异和自然输出量。

优化版反序重复及[一次 burst 诊断](BURST_PROBE_PLAN.md)均为0独有动作，因此将**当前 formulation 在已测单卡、OLMoE、64 GiB KV/16 GiB host、固定运行/恢复策略及这两类到达序列中的增量动作价值未成立**作为收束结论，停止将该候选推进为已证实的性能方法，不继续用零动作下的结果差异包装收益或无界搜索阈值。burst 是看过稳态后设计的动作窗口诊断，不是独立确认；它也没有形成“满足当前触发规则的真实恢复积压、未年龄豁免的新请求、KV/cap 允许且原生路径实际考察该请求”的交集，失败定位是已测运行域没有暴露该规则的动作机会，而不是整个恢复感知准入问题被证伪。只有新的真实轨迹或运行域证明这一机会存在，才值得重开同一 formulation；出现动作后仍须重新检验完整联合结果与去恢复信号消融，不能把动作存在当成收益。

## 重开后的定向查新：有界延后与持续服务（2026-10-07）

此次按用户目标准备最小有界延后探针，是对动作窗口的进一步研究，不撤销上述零动作结论，也尚未证明新的性能方法。以下区分出版状态、实际控制范围和 C 仍欠缺的增量证据。

### 原始版本与正式发表状态

| 工作 | 截至检索日核实的版本 |
|---|---|
| CONCUR | 已有 ICML 2026 / PMLR 正式版，见上节；本次机制核对使用正式 PDF，而非只沿用 arXiv 元数据。 |
| CacheOPT | [Mitigating KV Cache Competition to Enhance User Experience in LLM Inference](https://arxiv.org/abs/2503.13773)，最新为 v2，2025-03-24；[作者出版列表](https://www.cs.virginia.edu/~hs6ms/publications-selected.htm)仍将其列为 arXiv 预印本。本次未核实独立会议/期刊正式版，不将“未找到”写成确定未发表。 |
| Scorpio | [arXiv:2505.23022 v2](https://arxiv.org/abs/2505.23022)，2026-08-26，明确标注 WISE 2026 接收；[大会接收名单](https://conferences.sigappfr.org/wise2026/accepted_papers/)列出论文与作者。该版本声明不是 Version of Record，后者拟刊 Springer LNCS；本次未核实已发布的 LNCS 正式正文。[作者代码](https://github.com/MisterBrookT/Scorpio)亦标注 WISE 2026。 |
| JITServe（仅新增此一项近邻） | [JITServe: SLO-aware LLM Serving with Imprecise Request Information](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)，已正式发表于 NSDI 2026，pp. 825–848；机制核对使用 [USENIX 正式 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)。 |

### 信号、动作和目标的精确边界

| 工作 | 使用的信息与实际控制动作 | 原始目标 / 超出 C 的动作范围 | C 仍须证明什么 |
|---|---|---|---|
| CONCUR | KV 使用率 + 命中率驱动 AIMD agent 并发窗口；对 agent 下一次 generation admit/pause/resume。 | 离线、多轮、有工具调用的 agent batch；减少长期状态缓存抖动并缩短整批完成时间。其 agent 连续性不等于在线单次请求首次 prefill 的有界等待。 | 相同 KV/活动数下，实际恢复状态是否使原本允许的新 prefill 值得延后；缓存反馈动态并发本身已被覆盖。 |
| CacheOPT | 预测输出长度及置信补量，结合到达率、TTFT/TBT 剩余裕量和已分配 KV；为新请求及 returned 请求选批/分配 KV，提前补量并设置全局 KV 预留；还改变 victim 和 swap/recompute 选择。 | 降低 TTFT/TBT 尾部、提高相应 SLO 达标率和可支持到达率；联合处理排队与抢占。它已覆盖保护运行/返回请求、为继续生成预留容量，远超过只门控新 prefill。 | 不能把“保留继续执行余量”或“减少抢占”本身作为新颖性；必须证明无需输出长度预测、保持原恢复/victim 时，已观察恢复状态仍带来廉价而有效的额外决策。 |
| Scorpio | 预测长度与执行时间，使用异构 TTFT/TPOT SLO 和剩余裕量；TTFT Guard 按最早期限重排并拒绝不可达请求，TPOT Guard 以 virtual batch size 判定准入，并用 credit 改变逐轮 batch 选择。 | 联合 TTFT/TPOT SLO 的 goodput 与达标率；允许拒绝且会改变已运行请求的生成服务份额。v2 明确是预测误差下的 best-effort，而非确定性硬保证。 | 新规则若只是 slack/容量门槛，应承认与该类 guard 的关系。C 不拒绝、固定 decode，需按全部外部到达计等待和失败，不能用它较宽的动作空间掩盖本线因果比较。 |
| JITServe | 持续修正输出长度上界/依赖估计，以剩余工作和时间预算算所需服务带宽；GMAX 选 batch、准入和抢占。§4.2 估计 KV reload/recompute 停顿成本，预计抢占收益超过成本才执行。 | 面向逐 token、截止期及复合请求的应用 SLO goodput，按时间帧分配服务并修正调度；其 goodput 可依应用定义，不能直接等同 C 的联合达标完成请求数。 | “继续执行承诺”和恢复成本权衡已有直接近邻。仅给新请求设置最长延后时间不构成后续 decode 服务保证；须证明固定内部调度下的局部新动作仍有完整请求价值。 |

CacheOPT 的具体依据为 [v2 §3.2–3.4](https://arxiv.org/html/2503.13773v2)：其等待队列涉及新请求 TTFT 裕量和被抢占请求 TBT 裕量，returned 请求的 KV 补量也参与调度。因此不能将它概括为只有静态内存预测。Scorpio 的依据为 [v2 §3 及 Algorithm 1](https://arxiv.org/html/2505.23022v2)：这里只将明确的 TTFT Guard 失败处理称为拒绝，不把 TPOT Guard 暂不准入一概描述成丢弃。JITServe 的恢复成本来自 [正式版 §4.2](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)，是按序列长度及带宽/算力估算受影响停顿；它不支持将并行传输耗时简单相加为真实排空时间。

### 本线可检验的残差与比较口径

本次核查的这些控制规则没有直接给出 C 所需的答案：在当前原生恢复路径里，实际待恢复请求及消退是否能在 KV 和活动数之外指导**有上限、仅针对首次 prefill 的延后**。这是尚待实验证明的狭窄残差，不是“此前无人考虑恢复”的结论。CacheOPT 的容量预留和 JITServe 的恢复成本已要求本线更精确地说明与它们的区别。

最小探针应先回答延后是否真实发生、是否改善全人口结果。其后的归因至少需要：竞争力固定配置、相同控制结构的 KV/并发版本，以及相同最大延后规则但移除恢复信号的对照。若普通有界延后已经解释收益，贡献不能归于恢复信号；若只是把等待从运行中移到新请求队列，须由外部到达起算的 TTFT、完成时间和固定联合指标判断净收益。控制器解除门控的时间上限也不等于请求实际开始/完成时间上限，必须分别记录。

完整 CacheOPT、Scorpio、JITServe 都使用比 C 首阶段更宽的动作或额外预测信息。首个动作探针不以完整复现为门禁；后续最近邻比较应明确是完整系统对比还是限制到新 prefill 的移植规则，不能混称。只有出现稳定净收益，才值得进一步检验自然 EOS/质量、独立到达轨迹、第二模型及开销；在同人口、同轨迹上探索过的规则不能直接升级为独立确认。

## 未来同时驻留峰值：直接近邻（2026-10-08）

[Online Scheduling for LLM Inference with KV Cache Constraints，v5 §4式(5)及附录C Algorithm2](https://arxiv.org/html/2502.07115v5)已经提出未来轮次KV峰值可行性检查，并允许使用真实输出长度的上界。其MC-Benchmark按FIFO考察新请求，检查失败即停止本轮新增；不能将“FIFO＋声明max_tokens上界＋未来峰值”归为本线原创，也不能继承MC-SF在额外假设下的竞争比。原模型的连续单token推进和完成释放需要在本机实际原生路径中核实。

本线将其作为保守适配对照：旧请求全部处于可逐轮推进的resident decode时才允许放宽全额预算；新请求因分块prefill尚不能保证完成轮次，其完整占用平台不会在假定时刻消失。实际页粒度、当前步已分配页和释放边界计入条件。这是已有算法的运行时适配，尚无有效性或原创贡献结论；CacheOPT此前也已覆盖时序空间复用，不能忽略这一更强简单解释。

### v20之后的时间轴与主张边界澄清（2026-10-08）

[MC v5 §2 Batch Processing、Evaluation Metrics与§4式(5)](https://arxiv.org/html/2502.07115v5)的理论模型把一批处理定义为一个单位，启动后的请求连续处理；未来占用随执行轮次/token进度增长、到上界时释放。外部到达到完成的`c_i-a_i`本来就是其目标，不能称它忽略准入前排队。理论的单位轮时不意味着本线可以把某个真实秒数当作其承诺释放期限。

更关键的是，[§5.2 Simulation Setup与§5.2.1首段](https://arxiv.org/html/2502.07115v5#S5.SS2)已经采用连续Poisson到达与Vidur估计的可变batch处理时间，并评价平均端到端时延；文中明确讨论了batch时长变化时未来内存检查仍防溢出。因此不能将“真实prefill改变batch时长”本身写成MC未覆盖的模型失效。准确的剩余边界是：式(5)的准入检查没有显式比较新请求的边际墙钟服务代价；这不是证明新增此比较就能超过强简单策略。

设真实第k轮耗时为τ_k，r轮后释放对应累计Στ_k的墙钟时间。若旧请求每轮进度、物理释放边界与容量计费仍成立，只改变τ_k不会使轮次峰值错误。C的decode优先、cap256<量子1024与事件级阶段资格给出了有限的实现解释；异常返回、rollback、抢占、async eligibility等仍是边界。v20的TTFT/flow交换不是已观测的释放承诺失信。

墙钟服务代价也不是空白方向：[JITServe正式版§3，印刷页828](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf#page=5)按累计token交付期限描述流式goodput并跟踪生成速度；[作者v3 §4.2](https://arxiv.org/html/2504.20068v3#S4.SS2)以剩余生成时间和剩余期限构造最低服务带宽，并处理batch组成对速度的影响。这里§4.2细节引用作者v3；本次正式PDF重新抓取超时，未冒充重新逐字核查正式版该节。本线仍须证明在自身动作/目标下存在这些已有原则没有解决的重要损失。

v20已把旧prefill“本步排完剩余prompt”的状态纳入有依据的资格，获得真实156/152次额外许可与首prefill，观察到P95 TTFT改善而mean flow恶化；完整数据见RESULTS。可归类为运行时适配和开发取舍，不能升级为新的未来峰值算法或MC反例。

## 首 token 前再次抢占与短期 KV 预留：两项定向核对（2026-10-08）

[CacheOPT v2 §3.3.2、Table 2](https://arxiv.org/html/2503.13773v2) 已将 critical 新等待请求的最低 KV 分配写为 `prompt + B`，critical returned 请求为 `B`；这里 returned 是一次迭代返回的请求，并非专指等待恢复的请求。§3.3.1 的空间复用条件还计入正在运行请求在预测输出区间内的 KV 增长，§3.3.3 则提前补量并设置共享预留。故“完整 prompt 之外留固定余量”“考虑继续生成的空间”已有直接先例。它会实际分配容量、修改选批和抢占，不能等同 C 仅延后新 prefill 的门控；正文也不能作为 C 所见“完整 prompt 当时可放下、首 output 前再次抢占”这一精确事件的测量证据。

[vLLM v0.31.0 的 SchedulerConfig](https://docs.vllm.ai/en/v0.31.0/api/vllm/config/scheduler/) 已有完整 ISL 检查及 waiting/preempted 准入水位。[`KVCacheManager.allocate_slots`](https://docs.vllm.ai/en/v0.31.0/api/vllm/v1/core/kv_cache_manager/) 先检查完整已知序列，实际分配仍按本步 token 加 lookahead；据此推断，检查通过本身不构成直到首 output 的持续容量承诺。更接近已知短期增长的是 [`Scheduler.schedule`、`_request_remaining_blocks`、`_inflight_prefill_reserved_blocks`、`_spec_decode_step_blocks`](https://docs.vllm.ai/en/v0.31.0/api/vllm/v1/core/sched/scheduler/)：在异步 KV load 准入时，扣除其他尚在 prefill/异步加载请求的剩余完整序列需求，并计入 speculative decode 步所需额外块；代码明确将避免可预测抢占作为目的。这个额外扣除受 `load_kv_async` 分支限制，非 speculative 时该步补量为零；不能扩写成所有普通 decode 增长均受保护，也不能将这个官方版本等同本线部署版本。

本线固定运行栈为 vLLM 0.26.0 加仓库 pinned vendor scheduler；本线此前已定位同类 async load reserved helper，local new 路径的 `reserved_blocks=0`。上述官方 0.31.0 仅作近邻源码依据，不意味着本线缺失已有异步加载保护，也不构成换栈建议。因此固定 32 页、最多延后 250 ms 的探针只能检验普通 headroom 在当前路径是否有净收益，不能主张预留思想创新或首 token 保证。若以后从固定余量转向已知短期需求，增量仍须具体落在：哪些当前调度状态可算出的需求被现有规则遗漏、如何在不改变恢复/victim 的条件下产生不同动作、以及相对同结构固定余量是否稳定改善从外部到达计时的完整人口结果。首 output 前抢占是本线诊断事件；把它归因于其他 decode 增长或恢复竞争，需要本线分配轨迹证据，不能由上述文献替代。
