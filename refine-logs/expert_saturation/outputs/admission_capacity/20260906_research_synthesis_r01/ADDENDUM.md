# 研究选择补充：最新原生证据与直接文献碰撞

2026-09-06；`agent/publish-current-moe-code@2a37765`。本记录追加更新，不覆盖前文或任何实验 raw。

**当前建议：保留原生非抢占并发为主攻，但只投资下一次跨输入证伪，不实现 U/C controller。条件备选改为原生 MoE 内部执行形状的 conformance 边界；不同运行域候选仅保留自然显存不足下的接纳问题。Verify 双轨数值选择不再优先投入性能实验。** 这是研究投资排序，不更改各方向 sealed verdict，也没有得到 CCF B 方法 GO。

## 1. 新数据已关闭上一份记录中的配置混杂

本轮调研期间，共享工作区新增了[共同引擎原生结果](../20260906_native_fixed_engine_r01/REPORT.md)。本轮只读核对这些结果，没有另启动 GPU 实验。

- **16/16 episodes COMPLETE，8/8 配对的 cap8 完成吞吐及 goodput 胜 cap6。** 两臂 engine args、graph capture sizes 和 FULL 图数量相同；只在完全排空时改变接纳上限。
- 主 SLO 为已预声明的 TTFT 200ms / mean TPOT 9ms，cap6 达标 **89/128**，cap8 达标 **117/128**；原 5s / 0.2s 仍全通过。这是同一旧输入上校准的探索阈值，不是业务承诺。
- cap8 的 goodput 优序不意味着全部延迟指标占优：TPOT 中位数 cap6 在5组更低、cap8在3组更低。最强简单基线是**已测集合 `{6,8}` 内的固定8**，不是全局最优，更不是动态 Oracle。
- 该结果说明旧 graph 配置差异不能单独解释 cap8 优势；没有逐step graph mode计时，仍不能分配旧收益的归因份额。
- raw 重算记录为16 cells、0 issues；无抢占、computed adjustment 或多token chunk。只有16条不同输入，尚无动态动作、U/C采集或跨输入泛化证据。

因此，原[REPORT](REPORT.md)中“共同引擎仍UNRUN、等待上传、先执行共同引擎”的段落已经过时。历史阻塞保留在该实验的 PREPARATION_RECORD，**不再是当前阻塞**。上一轮先延长输出的建议也后移：先利用已证实的静态基线检验输入变化，而不是重复已完成校准。

## 2. 最影响判断的文献补充

下列来源按信号—动作—目标区分。只核验论文和官方实现，不将作者实验视作本仓库复现；未查到同样实验不等于证明新颖性。除标注正式出版来源者，按本次读到的预印本处理。

| 原始来源、作者、年份 | 信号 / 动作 / 目标 / 运行域 | 对方案的实质限制 |
|---|---|---|
| [ELDR，Sangjin Choi等，2026 v2](https://arxiv.org/html/2607.00466v2)，§3–4/6 | prefill专家签名与decoder负载；PD交接时选择decoder；MoE TPOT、原生多GPU | 已研究相同batch下激活专家数与成本，并让动作前签名改变请求组成。“专家集合影响延迟”及“prefill可预测decode专家”不能再作独立贡献；其固定512输出也不是自然停止验证 |
| [METRO，Yanpeng Yu等，2025](https://arxiv.org/html/2512.09277v1)，§III/IV/VI | 当前top-k→激活副本；在同专家副本间派发token；memory-bound EP负载 | 已区分激活专家与token平衡，且不改变top-k选择。单卡C不等于rank负载；模拟Pareto不能写成真实请求收益 |
| [Service-Induced Congestion，Ruicheng Ao等，2026](https://arxiv.org/html/2606.15555v1)，§2/5/6 | admission、KV增长、完成/驱逐的动态系统；限制每轮接纳率；完成量及稳定性 | 已分析接纳改变后旧cohort继续增长的暂态。其理论包含驱逐及确定长度，不能直接当无抢占、自然输出、MoE墙钟SLO保证 |
| [WiSP，Jiamu Zhang等，2026 v2](https://arxiv.org/html/2606.21868v2)，§3.3/3.5、Appendix B/G | expert/KV边际价值、admission floor；专家分页和双池resize；固定VRAM的e2e时延 | “专家×KV联合分配”已直接覆盖。live resize在KV完全排空的barrier进行；公开controller代码暂缺是复用成本，不是可以忽略其贡献的理由 |
| [CoRun，Shiju Zhao等，2026](https://arxiv.org/html/2608.14376v1)，§IV/VI/VII | isolated prefill、固定decode形状、约束归约；确定性推理 | 已覆盖通过调度稳定执行形状，且已讨论MoE回滚负担。仅观察router flip、再加padding或generic verifier不足成新方法 |
| [LLM-42，Raja Gond等，2026](https://arxiv.org/html/2601.17768v2) | 动态fast path、固定形状验证、提交/rollback | post-action verification与回退已存在；需要MoE特有的新边界或成本优势 |
| [MarginGate，Kexin Chu等，2026](https://arxiv.org/html/2605.30218v1) | logit margin触发稀疏验证、当前K/V修复 | 不应换一个margin阈值继续做第三个事前/验证selector |
| [QSpec，EMNLP 2025正式论文](https://aclanthology.org/2025.emnlp-main.240/) | 互补量化起草与多token并行验证、覆盖accepted KV | 精度切换后验证/修复并非空白；其摊薄验证费用的路径，正是当前逐token H+L双轨缺少的东西 |
| [SpecMD，2026](https://arxiv.org/html/2602.03921v1)及[Apple官方页](https://machinelearning.apple.com/research/specmd-expert-prefetching) | forward-cycle/layer-position感知缓存替换及expert prefetch研究 | 后续offload不能只赢LRU；软件限制cache得到的结果不能证明本机自然缺显存 |
| [FluxMoE，2026](https://arxiv.org/html/2604.02715v1) | 内存压力驱动专家驻留、按层流入流出与overlap；主要目标aggregate throughput | 给KV腾空间的驻留机制已有覆盖；若重开，必须回答持续到达和请求SLO中未解决的后果 |

原记录中以下工作也需放在方法论证正面：

- [Gimbal](https://arxiv.org/html/2606.15177v1)已联合KV、prefill、queue和MoE压力；动作是DP引擎派发及专家迁移。
- [Scorpio v2](https://arxiv.org/html/2505.23022v2)已将长度预测用于SLO接纳与credit batching；credit跳过活跃请求某些decode步，超出本项目动作范围。不能把受限改编叫作完整Scorpio复现。
- [Chiron](https://arxiv.org/html/2501.08090v1)提供基于普通时延反馈的batch控制先例；本项目应比较同B档位、同非抢占语义的普通反馈基线。
- [CONCUR](https://arxiv.org/html/2601.22705v1)的KV反馈、AIMD、generation/tool边界pause/resume，已经覆盖保持in-flight执行连续的并发调节；其agent边界暂停也不等于请求一直decode到完成。
- [TAPER](https://arxiv.org/html/2605.06914v1)已预测增宽对同batch的外部代价，用context/slack调额外branch；额外分支可逐步撤回，不能直接搬成不可立即撤回的请求接纳保证。
- [DA-MoE](https://arxiv.org/html/2607.23099v3)已用当前router之后的histogram选择GPU kernel。因此“同token数、不同分布、不同最佳kernel”也不是待发现空白；它的当前层选择可用信息不同于本项目动作前历史信息。

**调研结论比最初的“研究动作时延即可”更严格。** 专家影响成本、接纳改变未来KV、动作有暂态，这三件事各自都已有工作。把它们组合不能自动形成贡献。仍可证伪的窄问题是：在共同原生配置、无驱逐、普通状态可比时，专家信息是否稳定改变一次真实接纳动作的请求级收益，并超过强简单策略。

## 3. 科学问题需要三个更精确的区分

**A. 信号预测时延，与信号决定动作，是两个目标。**

主响应应是同输入/到达的独立未来执行中，动作相对baseline的goodput、TTFT与TPOT差值。U/C能解释慢step，并不说明应该升还是降B。只有加入专家信息能在独立episode上改善动作选择，且兑现为完整请求净收益，才进入方法阶段。历史U/C不会因为“动作会改变未来route”而失去全部观察价值；但不能复用一个动作的未来route给另一动作打分。

**B. 非抢占效应至少分为首次分叉和完全达到目标。**

8→6时写入目标不立即缩小当前batch；第一条完成后，hold8补位而down可能不补位，两臂已开始不同；第二条完成后才可能达到6。因此要分别报告观测/决策时刻、首次不同接纳、首次不同decode及达到目标的时刻。完整收益可能在完全drain前出现，不能用“U/C相关时间小于完全drain时间”机械判死。仅测自相关既不是动作价值，也不是可控性的充分/必要定理。

mean TPOT是完成时的平均间隔，不是逐token deadline。已有慢间隔可以被后续更快间隔摊薄；没有已知剩余长度时，不应编造可证明的“TPOT剩余slack”。首版保留用户目标，同时报告ITL和TTFT，不用平均数掩盖暂停、积攒输出或请求饥饿。

**C. 逻辑路由结构不直接等于物理工作。**

对按expert以固定token tile宽度m补齐的一个实现，结构诊断量可以写作 `sum_e ceil(n_e/m)`。若所有非零n_e都不超过m，则它等于激活专家数：C可以改变而tile数不变。这是条件代数例子，不是本轮GPU解释。vLLM 0.26的[原生fused-MoE源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/fused_moe.py)存在按token数选配置及expert block alignment；实际配置还可能由tuned map覆盖。必须记录真实后端与选择配置后才能应用这个解释，不能据源码推定本机全部cell的tile或HBM流量。

原始U/C的逐层差别要保留：两个cohort的层均C差小，有跨层抵消，并不表示各层几乎相同。已有[分辨率重算](signal_resolution.json)足以说明这一点，不因此启动高维特征搜索。

## 4. 唯一下一实验：真正新输入下的共同引擎复测

最新固定引擎结果已完成；**不再执行同一组旧输入的16-cell校准，也不立即增加长输出、offload、模型或controller。** 下一步只改变请求文本，检查固定8的优序能否保持。

| 项目 | 预定内容 |
|---|---|
| 问题 | 在未用于本轮admission研究的新文本上，同一native引擎中固定8是否仍优于6？ |
| 输入 | 同一WikiText103 test来源，旧manifest覆盖位置之后，按dataset顺序选首16个足够128tokens的未用文本；不读取U/C或收益进行筛选 |
| 固定项 | OLMoE revision/BF16、128 prompt/16 output、FA2/Triton、共同engine max8、token budget1024、max model len256、到达scale.02、FCFS、线程配置与完整warmup |
| 主/参考SLO | 主200ms/9ms；参考5s/0.2s；两套保留，均不因新结果调整 |
| 执行 | 6/8/8/6四个新进程；每进程steady/bursty及两次反序重复，共16 episodes；独立KV和队列，全部保留 |
| 强简单基线 | 固定8；固定6作为较低并发对照，不称全局最优或动态Oracle |
| 允许结论 | 当前短请求域的跨输入静态响应；不主张U/C增量、自然EOS质量、稳态服务容量 |

输入有一个具体实现限制：旧formal manifest有128个非空原文，但只有**48条**满足128token门槛，offset0/16/32均已用于custom研究，故`offset48`为空。不得把既有cohort换名叫fresh。新输入应另存，保持旧formal文件不变；freshness只相对当前admission已用文本，不声称全仓或WikiText文章级独立。

**共享工作区协调与本轮实际准备。** 收尾时，另一会话已经冻结了[第一组新输入及执行规则](../20260906_native_fresh_cohort_r01/DECISIONS.md)，选择row205–256中的16条，沿用相同16-episode对照。它是上述唯一下一实验；本轮不重复启动，不用后来准备的文本替换其canonical。本文尚未看到这组新输入的完整GPU结果。

本轮另完成了[native_fresh_inputs](native_fresh_inputs/config.json)的离线准备，作为后续有证据需要时才使用的预留输入，**GPU UNRUN，不自动追加第二个campaign**。从同一冻结Arrow按顺序选择row `257,258,259,260,261,266,271,276,277,279,284,286,297,298,303,304`，每条128 prompt /16 output tokens；排除当前64个已用或已冻结prompt，原文和前128-token hash均无重叠。101行准备脚本与[CPU检查](native_fresh_inputs/checks.json)确认：冻结Arrow SHA及tokenizer文件一致，旧128行原文/token身份复现，新输入符合native runner读取合同，异目录重建的config/workload/provenance逐字节一致。实际`Dataset.from_file` fingerprint与旧builder记录不同，已在[provenance](native_fresh_inputs/provenance.json)保留两者，未把原因推定为已确认。到达仍是明确标注的合成规则，不冒用BurstGPT来源；主SLO需沿用native命令的200ms/9ms覆盖，prepared配置保留5s/0.2s参考值。

判读只保留三种路径：

1. **固定8仍稳定胜出**：停止这个短请求域的6/8自适应方法搜索；更复杂信号尚无动作动机。只有真实负载需求与已有成本数据支持另一个有权衡的运行域时，才单独声明一次域资格化，不逐个换seed/阈值救方案。
2. **固定6有重复一致的优势**：先按同配置重复和连续时延定位差异。跨文本排序不同仍不自动归因专家；随后才做普通状态可比的真实动作probe，先比较普通反馈，再测U/C增量。
3. **排名变号或只跨极窄门槛**：保留所有结果，做一次同配置受控复测；不先拟合selector，不按最有利repeat解释机制。

## 5. 主攻之外只保留两项，且都不同时执行

| 位置 | 候选与明确触发条件 | 最便宜的区分实验 / 停止条件 |
|---|---|---|
| 条件备选 | **固定全局shape是否仍不足以固定MoE内部数值执行。** 主线已无高价值动作不确定性时才切换 | 固定两个自然target的单层输入，原companions / 相同companions置换 / 同宽度替换三臂，加同臂重复。只观察原生fused MoE的target输出、expert内位置/实际形状。消失则保留custom backend边界；稳定且揭示新前提才查完整KV与用户可见后果 |
| 不同运行域候选 | **真实显存不足时，expert驻留与不可抢占请求接纳的SLO耦合。** 仅当模型、合法KV和workspace实际产生压力 | 复用pager，先比较静态内存分配与两个cap，记录自然miss到consumer等待。无暴露等待或静态策略已覆盖则停止；不限制一个tiny cache人为造机会 |

conformance的窄残差是推断，不能宣称CoRun错误。其论证还要求每请求计算不受其它请求数值影响；固定全局形状/位置不变性并不单独验证替换companions后的MoE内部路径。已有[vLLM batch-invariance官方支持](https://docs.vllm.ai/en/latest/features/batch_invariance/)必须作为竞争实现。仅重现float差异、router flip或已有verify/rollback故事，不够成为论文。

自然offload也不应现在启动。旧tensor-header预算为权重12.888GiB，cap8×4096的weights+KV下界16.888GiB；这没有证明约32GiB的5090发生压力。cap32×4096的28.888GiB也只是下界，workspace和实际可服务性尚缺。WiSP/FluxMoE已经占据一般联合分配机制，下一工作必须找到它们之后的真实请求代价。

**暂停优先投入Verify Precision性能链。** 当前每步同步双轨在同一资源顺序付费时 `T >= T_H + T_L + clone/verify/commit > T_H`。等served-high预算可以修正旧数值选择的公平性，无法消除全部physical-high调用；INT4 QDQ后BF16执行更不能给出原生低精加速。只有具体低成本证据或多token摊销后仍有MoE特有增量，才重开系统路线。旧数值资格化保留为UNRUN，不把成本结论扩写成整个Verify family NO-GO。

Receiver、QuotaEP及其多卡缺口维持既有状态；同步全参考SemanticFence不重开。无需为了填满候选表而再创造机制。

## 6. 怎样才可能形成完整论文

方法链必须是：**可复现的自然动作权衡 → 普通反馈之外的专家信息增量 → 动作实际来得及改变请求 → 完整成本后超过强简单策略**。还需要能区分机制的第二模型或routing regime及代表性负载；这些在正信号后补，不设为当前pilot前置条件。

测量链则必须识别一个具有适用条件和实际系统后果的新边界，例如全局shape保证与内部expert形状的差别。单个配置坑、同U/C下的运行噪声、多个失败实验的集合，都不自动构成可发表规律。

当前最合理的投资是有限的原生跨输入验证，**不是承诺已有好方法，再用实验补故事**。本轮完成文献原文核对、最新raw派生指标确认和16条真实文本的CPU准备；未训练predictor、未启动新GPU、未改封存原始结果或`docs/current/README.md`。动态Oracle和方法收益仍未验证。
