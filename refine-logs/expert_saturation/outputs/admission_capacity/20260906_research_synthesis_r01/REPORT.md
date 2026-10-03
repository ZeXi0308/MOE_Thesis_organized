# MoE 并发主线：证据、文献碰撞与继续方案

2026-09-06；`agent/publish-current-moe-code@2a37765`。检索截至本日。

> 后续更新见 [ADDENDUM.md](ADDENDUM.md)：共同引擎16个原生episode已完成，cap8在8/8配对中goodput占优；下文UNRUN和上传阻塞是历史状态。新增ELDR、WiSP、Service-Induced、CoRun等直接碰撞后，唯一下一步收窄为真正新输入的同配置复测，候选排序也已更新。

**判断：继续主线，但先研究原生运行时中真实的并发权衡，再研究专家信息的动作增量。当前不应实现新的 U/C predictor。** 最有希望的中心问题是：固定执行配置后，专家结构何时改变一次非抢占接纳的完整请求代价，以及这种信息是否来得及影响决策。非抢占、时延反馈、专家压力各自都不是新颖性；这一完整链仍待实证。

本轮完成相关论文正文和官方实现文档核验、已有 GPU 数据的结构重算、唯一实验链的收敛。**新增 GPU episode 为 0，拟合 predictor 为 0。** 没有把调研当作方法 GO，没有修改旧 raw、裁决或 `docs/current/README.md`。以下前景判断是基于已读文献的研究推断，不是全领域无遗漏查新或录用保证。

**首先需要纠正当前问题的重心。**

| 当前证据 | 能说明什么 | 对下一步的影响 |
|---|---|---|
| [固定 decode 前沿](../20260906_step_action_r01/REPORT.md)：32 cells，16/16 配对可见前缀与离散状态一致；升档均提高吞吐，goodput 仅4/16提高 | 接纳动作合法，custom 域有吞吐—TPOT 权衡；连续普通时延仍不同 | 无法据此建立 U/C 条件增量，也不能把变号解释为专家信号必然无用 |
| 同一 probe 的两个 cohort、同动作未来成员/输出/压力重现 | 文本和执行轨迹变化不是这些重复间全部时延差异的来源 | 先看普通运行速度、执行模式及计时域，不增加第三个 selector |
| [原生 vLLM](../20260906_native_transfer_r01/REPORT.md)：快速域cap8同时改善吞吐、TTFT和TPOT，全部旧SLO通过 | 有静态配置差异，未显示6/8间动态切换的必要性 | 收紧SLO不能替代证明真实的动作权衡；custom的6/8不自动是原生容量拐点 |
| [共同引擎准备](../20260906_native_fixed_engine_r01/REPORT.md)：固定引擎8，仅改空引擎接纳6/8；仍UNRUN | 已定位旧对照的具体混杂：FULL graph捕获覆盖不同 | 先运行已经准备好的16个episode；这是控制变量修复，不是新的论文贡献 |

共同引擎报告已从源码定位：原引擎6的FULL decode尺寸是1/2/4，width6补齐到8但缺FULL key；引擎8包含8。旧快速cap6中224/344个纯decode step处于width6。没有实际逐step graph-mode计时，故不能把旧16%–59%吞吐差全部归因于graph。这个缺口必须由真实共同配置执行关闭。

短episode的另一限制是：16请求在30ms内集中到达，随后主要是排空队列；custom step6时所有请求也已到达，且前六步从未受到cap6阻挡。它们是有效的便宜pilot，却没有覆盖持续到达、多个完成波次及在线降cap的恢复过程。当前step6升8还是静态cap8的后续执行，不能承担动态控制贡献。

**文献已占据的空间比“专家感知调度”这个名称更具体。**

下表按动作去重，列最相关11篇；阅读深度为方法/实验相关正文，未复现作者代码。除特别说明，出处写本次核验的arXiv版本，不据此猜测会议录用。来源均为论文原文；数值收益不用于估算本仓库收益。

| 工作、作者、年份、来源 | 信号/预测目标 | 实际动作 | 目标、运行域和保证边界 | 对本仓库的约束 |
|---|---|---|---|---|
| [Gimbal，Yifan Sun等，2026，2606.15177v1](https://arxiv.org/html/2606.15177v1)，§4/6/7.4 | KV、剩余prefill、等待tokens、MoE压力→引擎负载 | DP引擎选择、SJF、专家迁移 | 多GPU MoE，TTFT/TPOT；经验效果 | 已把专家压力接入请求调度；只加此类信号没有新颖性。固定单引擎B仍是不同实验问题 |
| [DA-MoE，En-Ming Huang等，2026，2607.23099v3](https://arxiv.org/html/2607.23099v3)，§IV–V | 当前层router后的histogram→最合适kernel/tile | GPU内选择fused-MoE kernel | B200/NVFP4/EP；完整MoE-layer graph，不是请求联合SLO | 已证明相同token bucket内分布会改变最佳kernel。仅做分布—kernel时延相图也不够；历史信号、真实接纳和请求后果才是剩余问题 |
| [Chiron，Archit Patke等，2025，2501.08090v1](https://arxiv.org/html/2501.08090v1)，§3–4/6.3 | ITL/SLO、前后吞吐→backpressure | 调batch上限，另调实例和混部路由 | 交互/离线混部；完整系统允许batch任务抢占，无硬保证 | 必须有其风格的普通反馈基线。限制到只延后接纳时，要注明是改编而非完整复现 |
| [SCORPIO，Yinghao Tang等，2025/2026v2，2505.23022](https://arxiv.org/html/2505.23022v2)，§2/3 | batch、context、输出长度预测、SLO→TTFT/TPOT | deadline排队、TTFT拒绝、VBS接纳、credit batching | 联合TTFT/平均TPOT goodput；arXiv注明WISE2026接收稿，非正式出版版本 | credit允许部分请求跳过decode，超出本任务动作。拒绝计失败，不能借不同分母做弱基线 |
| [SLOs-Serve，Siyuan Chen等，2025，2504.08784v1](https://arxiv.org/html/2504.08784v1)，§3–4 | profile成本、已接纳请求和SLO→未来token schedule | DP接纳、token分配、chunked prefill、可选speculation | 多阶段SLO容量；保护依赖成本预测，best-effort层可抢占 | 已规划接纳对既有请求的影响；“我们延后、别人拒绝”不是准确的新颖性论证 |
| [CONCUR，Qiaoling Chen等，2026，2601.22705v1](https://arxiv.org/html/2601.22705v1)，§4–5 | KV使用率/cache命中→拥塞 | AIMD agent并发、边界pause/resume | 离线多轮agent吞吐，含MoE；非联合TTFT/TPOT | 拥塞感知并发已有直接先例；generation/tool边界暂停不等于整个请求始终推进 |
| [TAPER，Swapnil Gandhi等，2026，2605.06914v1](https://arxiv.org/html/2605.06914v1)，§3、附录D | sequence数、context、deadline slack→分支外部性 | 每步调整额外branch宽度 | 分支并行，TPOT/token goodput；宽度可廉价撤回 | 不能把可撤回的一步slack规则直接移植到完整请求接纳；但“增加并发会拖慢同批请求”已被研究 |
| [AMoE/AEP，Shaoyu Wang等，2025，2505.08944v1](https://arxiv.org/html/2505.08944v1)，§2及设计 | 层队列与ready状态→可执行工作 | 异步EP、跨层重新组batch | 多GPU decode；改变内部调度 | 冷专家权重摊销与热专家拖尾存在竞争；不能预设集中度越高越坏，也不能把EP机制直接外推单卡 |
| [LLEP，Xuan-Phi Nguyen等，2026，2601.17111v1](https://arxiv.org/html/2601.17111v1)，§3 | expert/rank负载→不均衡成本 | 超额tokens及专家参数向空闲GPU搬运 | EP层级与整模型吞吐；非arrival-aware联合SLO | rank最大负载和单卡C是不同量；当前硬件不能承诺EP收益 |
| [XShare，Daniil Vankov等，2026，2602.07265v1](https://arxiv.org/html/2602.07265v1) | 当前router scores→共享专家集合 | 限制候选集合并重新top-k | decode及speculation扩展 | 专家工作集随batch增长已有研究；它改变routing，不是当前合法动作 |
| [FreeBalance，Pengfei Chen等，2026，2608.14205v1](https://arxiv.org/html/2608.14205v1) | 前层residual→下一层负载 | attention窗口内交换专家 | 多GPU，主要为长序列prefill | 跨层可预测性不能证明跨decode-step、接纳生效期间的信号持续性 |

两个邻近证据进一步限制备选方向。[Thinking Machines，Horace He等，2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)已经定位batch-dependent数值路径并提供batch-invariant实现；[vLLM官方文档](https://docs.vllm.ai/en/stable/features/batch_invariance/)已经列出多个MoE模型的验证支持。因此“batch改变输出”本身不足以复活一篇conformance论文，需要新的失败边界及具体系统后果。[Doubleword，Josh Cowan，2026](https://blog.doubleword.ai/moe-expert-coactivations)则用prompt相似性组织离线batch，报告专家复用与墙钟变化；它是第一方技术博客，不能把其trace统计当本仓库实测HBM或在线SLO证明。

物理机制也不能只用逻辑专家数代替。[NVIDIA TensorRT-LLM官方实现说明](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md)展示特定B200/NVFP4/TP8模块区间内，增加算术但改善矩阵形状反而更快；[SonicMoE，Wentao Guo等，2025，2512.14080](https://arxiv.org/html/2512.14080v1)讨论tile padding与IO成本，主要实验目标为训练。两者支持关注实际执行形状，均不能移植性能区间到5090/BF16/OLMoE。

检索还核验了[EasyBalance，Yize Wu等，2026，2608.07964](https://arxiv.org/html/2608.07964v1)：跨层选择/推迟微批、8×A800长序列域。其动作离当前接纳控制较远，不再扩展成候选。其它模型结构改变、近存硬件和offload文献只作排除线索，没有形成并行实验计划。

**本轮原始数据重算揭示了一个需要保留的细节。**

设当前层有N个tokens，每token选择k个不同专家，总专家E，N>0，则定义直接给出：

\[
\sum_e n_e=Nk,\quad U=\#\{e:n_e>0\}/E,\quad C=E\max_e n_e/(Nk),\quad 1/U\le C\le E/k.
\]

所以U/C不是与batch无关的两个自由维度，C的分母也不能改成只含活跃专家。以原始动作前数据E64、k8、N6、16层为例：

| 汇总 | c0 | c32 | 如何解释 |
|---|---:|---:|---|
| 各层活跃专家数之和 | 516 | 498 | 层均U为0.503906/0.486328；净差18，逐层绝对差之和42 |
| 各层最大expert token数之和 | 62 | 61 | 层均C为5.166667/5.083333；净差1，但11层不同，绝对差之和17 |

单层C的量化格为4/3，16层均值的格为1/12。**均值接近有部分跨层抵消，并非逐层结构几乎一样。** 同时只有两个文本组，依然无法从这些层差异建立泛化selector。当前应保留既有逐层原始量，不以此为理由启动高维feature search。即使U/C相同，专家身份、完整histogram及跨步复用仍未由两标量确定；“工作集大小”不等于“工作集复用”。

重算脚本为 [analyze_signal_resolution.py](analyze_signal_resolution.py)，数据为 [signal_resolution.json](signal_resolution.json)。运行命令：`python3 refine-logs/expert_saturation/outputs/admission_capacity/20260906_research_synthesis_r01/analyze_signal_resolution.py`，目标已存在时拒绝覆盖。来源是保留的GPU `analysis.json`；CPU只执行算术核对，不产生CPU/GPU性能结论。

**主攻应怎样表达，才能让实验真的改变判断。**

研究对象暂定为“原生MoE推理中，专家结构对非抢占接纳选择的增量价值”。动作保持FCFS和只调B，router、top-k、precision、placement固定。资源机制是候选解释：专家权重摊销/实际kernel形状、KV/attention与prefill干扰；必须先测到对应请求后果，不预设哪项主导。

核心响应变量是同输入、同到达、共同前缀下真实独立执行的动作差值，例如升档相对保持的goodput、TTFT、TPOT变化。普通状态至少包含batch/token/KV/queue/prefill、近期模型/iteration/ITL，以及已知执行配置。研究假说是：加入动作前U/C后，对动作排序的判断有稳定改善。只降低下一step时延预测误差，或只证明文本组不同，均不满足它。

比较顺序应为当前native默认配置、最好已测静态cap、只用普通状态的迟滞反馈/成本guard，再加U/C。Chiron风格基线限制为相同B档位及相同接纳语义；不能拿它完整的抢占、路由功能与受限策略混比。普通控制的阈值和标定预算与 proposed 对齐，全部排队/失败请求保留在分母。真正有信号后再补最近邻完整实现，当前不实现两套controller。

非抢占动作有两个时间，不宜用一个“生效延迟”概括：一是相对hold首次少接纳/实际batch分叉，二是实际活跃数达到新目标。8→6时第一条完成后hold8会补位，而down可能已不补位，收益可先于第二条完成、达到6出现。也要区分新接纳导致prefill负载变化和decode batch达到目标。**不能机械要求信号相关性一直维持到完全drain，也不能认为写入目标即刻减负。** 测试应沿各动作的真实未来报告从首次分叉到请求完成的累计后果；动作会改变route，固定trace自相关只能是诊断。

最有价值的可能贡献是：发现一种普通状态控制系统性遗漏的MoE接纳代价；用低成本历史信号在不可立即撤回的接纳前识别它；在完整成本后超过强简单策略。这里每项都是尚待验证的条件。如果普通策略已经覆盖收益，就保留工程结果，停止专家predictor。

**执行链收敛为以下先后关系，只有第一步现在执行。**

1. **共同引擎校准，直接复用已准备方案。** 原生OLMoE、引擎max_num_seqs=8、接纳6/8，四新进程6/8/8/6，两种快速到达×反序重复，共16 episodes；16真实文本、128prompt/16output。warmup覆盖实际形状并标phase；主SLO0.20s/0.009s为前批全部快速请求p75校准，旧5s/0.2s同时保留。见[冻结设计和命令](../20260906_native_fixed_engine_r01/DECISIONS.md)。这回答“旧差异是否只是不同引擎配置”，不回答U/C、动态Oracle或稳态容量。
2. **第一步有效后，找原生容量权衡，而非继续固守6/8。** 最小后续是一个32条真实请求、较长固定生成、持续steady/bursty到达的少档位扫描。候选为128/256生成预算量级；档位、到达率、预算和阈值必须在新结果前一次声明，以原生可用容量和真实占用为依据。先只改变造成无action space的具体负载因素，不同时加模型、长KV、offload和新controller。若最大测试cap在预定域中持续支配其它cap，就停止该域动态控制，不能靠换SLO救出赢家。
3. **只有第二步出现稳定权衡，才做动作增量probe。** 预先选自然文本/arrival episodes，不按U/C或收益筛选；同共同策略前缀后真实hold/up，必要时另测自然down。每策略独立未来KV/route/queue/completion。先看普通状态反馈能否覆盖，再检验U/C是否改变选择。随机化或反序配对，不把相邻decode步随机拆成训练测试；cohort/episode是独立单位。若样本仍不足，只报descriptive，不拟合一个“显著”selector。
4. **仅正信号触发确认。** 按机制选择能区分k/E、专家粒度或KV域的第二模型/运行域；自然停止、不同输出长度、开销和失败边界，代表性backend的强基线。第二模型必须适合现有硬件，不能为放进显存随意量化后宣称只改routing regime。仅EP/NCCL claim要求真实多卡。

第2–4步是条件路径，不是同时开工的矩阵。第1步已完成可独立推进的代码与5项定向CPU检查，没必要为本轮调研再跑全仓测试或重建协议。其此前更新源码上传遭两次自动审批拒绝，仍为`UNRUN / MODIFIED_SOURCE_TRANSFER_APPROVAL_PENDING`；本轮没有重试传输或换路径绕过。权限问题不是科学NO-GO。

**只保留1+1+1，其余方向维持原裁决。**

| 位置 | 保留内容 | 触发条件与停止边界 |
|---|---|---|
| 主攻 | 上述原生非抢占并发链，U/C为待验证信号 | 先关闭共同引擎混杂；原生无权衡/普通反馈已覆盖/动作来不及影响请求时停止对应formulation |
| 条件备选 | 执行配置怎样改变MoE容量曲线与结论的测量研究 | 只有受控graph/实际shape干预稳定翻转原判断，且有请求后果、跨模型或backend边界，才升级；单个配置坑或多个失败拼接不够成论文 |
| 不同机制候选 | Verify Precision等预算数值选择资格化 | 主攻与备选均无高价值下一实验才切换；复用8篇输入、fixed/reactive/random独立KV。QDQ双轨不是nativeINT4加速，必须先有优于强简单周期的选择空间及可实现后端 |

Verify的实际准备和限制见[独立方向报告](../../../../independent_ideas_20260905/REPORT.md)。Receiver缺真实EP暴露路径；Offload应先有自然权重/KV容量竞争；QuotaEP真实fused系统gate未跑；SemanticFence同步full-reference固定税未消除。没有新证据时，它们不自动升格。已有execution conformance先保留为来源定位资产；官方batch-invariance已经存在，暂不另启并行机制。

**最后的研究判断。** 当前值得投的是下一次能排除具体混杂的原生实验，不是论文故事的包装。共同配置差异即使消失，也只会否定旧静态对照的归因；若差异保留，仍须证明真实动态权衡。只有“自然权衡→普通状态之外的可选动作增量→完整请求净收益”闭合，才有方法论文主链。若只有一条有解释力、可复现的执行边界，可考虑测量论文；二者目前都未成立。

| 结束字段 | 本轮结论 |
|---|---|
| Verdict / evidence | `CONTINUE_NATIVE_QUALIFICATION / NO_NEW_METHOD_GO`；文献核验和既有GPU结构重算 |
| Measured / unmeasured | 新增16层信号分辨率重算；新增GPU为0，native共同引擎、动态动作余量、U/C selector仍未测 |
| Strongest baseline / Oracle | custom hold6较稳；旧native静态cap8占优但有配置混杂；无动态Oracle |
| Claim ceiling / failure | 研究决策和结构诊断；不能写容量方法、生产SLO或家族NO-GO |
| Reopen | 真实运行域内稳定请求级权衡、简单策略后的残余，或可验证的执行路径规律 |
| One next experiment | 已冻结共同引擎原生cap6/8的16个episode，保持原始结果和全部重复 |
