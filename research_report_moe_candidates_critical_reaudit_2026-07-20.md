# MoE 系统研究候选方向批判性重审计报告（2026-07-20）

## 执行摘要

对当前标记为 PASS/PROMISING 的候选（receiver-aware、Per-request Quality Isolation、Expert Prefetch）以及此前未被充分审视的存活候选 Energy-SLO Precision EP 进行了逐条重审。结论：receiver-aware 只有"结构性拥塞下静态画像有效"这一狭窄结论完全站得住，v3 自适应控制器的优势已被本轮因果修正证明统计上不显著；Quality Isolation 的"请求存在稳定内在脆弱度标签"这一核心假设已被同一天更晚做的第四轮 GPU 实验（prefill→decode 迁移）实质性证伪，比用户问题描述中反映的证据更弱；Expert Prefetch 的信号本身真实，但系统机会的 oracle 上限在正确评估后收缩到接近零，应停止投入；Energy-SLO Precision EP 是真实、尚未被同等强度攻击的存活候选，但缺一个关键质量轴。综合排序：Quality Isolation（重构为 predictor-free 质量债务公平机制）＞ Receiver-aware（重构为 direct-benefit + homogeneous lane + receiver credit 的系统特征化）＞ Energy-SLO（补质量轴后作第二贡献）＞ Expert Prefetch（停止）。推荐统一主线仍是 Receiver-aware + Quality Isolation 合并系统，但"预测脆弱度"应降级为辅助信号、predictor-free 公平兜底作为核心保证；该主线能否升级为完整系统论文的唯一决定性瓶颈是能否获得真实多 GPU/RDMA 环境。

## 一、候选全景与状态更新

项目历史上出现过约21条候选（完整清单见附录A），其中 PLTB、R-layout（降级为baseline）、Graceful-EP/QTree-EP、additive-KL MILP、Tiered-Fidelity Shadow Experts、Residual-EP、QuotaEP-H/PartialGuard、CreditReduce、RouteFidelity-EP、WaveCredit-EP、MassCover-EP、TokenRace-EP 已在多轮独立复核（含本轮"已死路线knob复查"，专门检验"能否靠调参/混合算法/更强启发式挽回"）中判定 KILLED，答案一致为不能，本报告不再重新提审，只保留在附录中。本报告聚焦四条仍存活或被误判为已充分验证的候选：receiver-aware、Quality Isolation、Expert Prefetch、Energy-SLO Precision EP。

一个容易被忽略的证据链：Quality Isolation 在同一天内经历三轮递进收紧的审计——P0（oracle上限+可迁移proxy，看起来很强）→ GPU第二轮（严格calibration/validation/test三段划分，proxy相关性大幅缩水，只在LLM-jp成立）→ GPU第四轮（prefill信号预测未来decode风险，全面NO-GO，且不同动作间脆弱度排序本身不一致）。用户问题描述引用的"model-specific、post-prefill、action-specific的弱正结果"正是第二轮的结论，而第四轮（时间上更晚）进一步下修了这个结论，本报告以第四轮为准。

## 二、Receiver-aware：核心假设拆分与重构可行性

### 2.1 假设成立 vs. 实现/评估失败的区分

Receiver-aware 实际包含四个可分离的子假设，命运完全不同。子假设一："receiver间存在结构性/持续负载偏斜，且能被静态画像捕获而无需在线信号"——在v2系统性重跑（跨2模型×2placement×2origin_mode×3budget、真实路由数据）中稳健成立，hotspot场景下oracle/causal/calib_static三者几乎重合(+5.6%~+11.1%)，这是**目前唯一可以放心写进论文的结论**。子假设二："同一时刻负载信息在真实在线系统中可得(oracle_same_step)"——这是一个**评估方法错误**而非系统结论，oracle_same_step本身是不现实的先知假设。子假设三："用因果合法的短窗口统计量(HHI)可自动分类regime并部署自适应控制器"——这是本轮**因果bug**（检测器观察前30%后把策略回溯应用到warm-up期）修复后被推翻的：修正后adaptive相对最强固定基线`always_causal_prev_step`的优势在全部detect_frac下95%CI都跨0（如detect_frac=0.3，OLMoE +0.286pp，CI[-0.400,+0.885]pp），且检测准确率越高（0.5窗口达100%）warm-up税反而让总收益更差。子假设四："progressive/残差式渐进精度编码在真实codec和链路速率下更优"——这是**表示学习假设本身的证伪**：INT4 progressive的KL是direct方案的3.53倍，真实Triton codec开销让break-even点仅0.9–59Gbps，远低于现代100–800Gbps EP网络。

结论：receiver-aware不是"一个假设"，而是"结构性画像成立+在线自适应分类器不成立+渐进编码不成立"的组合，前者是真实科学发现，后两者是被更严格实验证伪的具体机制选择，不应混为一谈评判整条线死活。

### 2.2 新机制能否重建价值

能。批判性设计文档已提出的"direct-benefit predictor"框架（直接预测下一microbatch的排队/关键路径收益，而非先分类regime再套策略）在方法论上是正确的修复方向，天然避开分类器的两个致命伤：阈值容易只识别生成器参数而非真实信号；分类正确率提高但warm-up税更贵的悖论。问题是这个方向此前只有公式推导，**没有任何一次真实实验**，是下一步最值得投入的方向（见五、实验4）。

### 2.3 四种重构路径逐项评估

queue/latency/NIC telemetry驱动控制：方向正确，但项目至今没有一次用过真实队列/ECN/QP信号，全部是"路由trace+线性带宽回放"。外部文献显示receiver-driven拥塞控制在数据中心网络有扎实先例（RCC, ICNP'21；RFCC, IEEE 2025；ICI, arXiv 2511.04639改进DCQCN拥塞隔离，尾延迟改善31%），但**没有一篇专门针对MoE all-to-all/combine的receiver-driven精度控制**，说明novelty风险中等而非高——机制不新，但应用场景基本没被占。

regret-minimizing/contextual bandit控制器：外部grounding明确**未找到微秒级、每-microbatch决策粒度的bandit落地先例**，多数bandit/RL拥塞控制在秒级或更粗粒度运作（视频码率、云资源调度）。若把bandit作为核心机制依赖风险很高。更稳妥做法是把regret分析仅作理论包装（delayed-feedback Hedge或shadow-replay做离线regret上界），核心在线执行用确定性queue-threshold+hysteresis+minimum dwell time规则——这恰好也是"无需准确分类regime的鲁棒策略"的答案。

receiver critical-path、incast、tail-latency感知：这是当前证据链**最大的空白**，三轮GPU实验没有一次测过真实排队/incast。本轮GPU审计明确写"远端只有一张RTX 5090，没有多GPU、NIC/RDMA topology"，无法合法验证。这是本报告的**头号风险项**：若两周内无法获得至少2-4卡NVLink/RDMA环境，receiver-aware的系统claim上限被锁死在characterization层面。

无需准确分类regime的鲁棒策略：direct-benefit predictor（不做二元分类，直接回归"这个动作能省多少排队时间"）加安全fallback（LCB低于quant开销时不动作）就是正确答案，公式已具备，只差真实实验验证。

## 三、Per-request Quality Isolation：核心假设的持续收缩

### 3.1 假设成立 vs. 实现/评估失败

需纠正一个误判：用户问题把现状概括为"仅在LLM-jp上得到弱正结果"，这在第二轮意义上成立，但**当天更晚的第四轮实验进一步显示，即便在LLM-jp上，把prefill信号迁移到预测未来decode风险也是NO-GO**：冻结的第二轮proxy迁移到decode任务后，四个动作相关性CI全部跨0（量化动作ρ=0.350但CI[-0.224,0.756]，drop25%仅ρ=0.029）。更关键的是**跨动作真实harm相关性本身低且不均匀**（量化vs drop50%仅0.129，drop25%vs drop50%却有0.865）——这直接证伪"文档/请求存在与降级机制无关的内在脆弱度标签"这一核心假设，不是proxy没选好的实现问题。这也暴露一个此前未被发现的confound：Part B的跨机制相关性(0.75-0.98)很可能只是"同族动作"（都是量化类）内部的相关性，被误读成"任意机制通用"。

因此当前唯一站得住的结论极窄：同一prompt、同一次forward内（非跨时序），LLM-jp上，post-prefill的router/NLL统计对**同一份**fixed-tail INT4降级的KL有中等强度、可复现的相关性（冻结确认ρ=0.517, CI[0.315,0.673]）。这对在线系统几乎没有直接可用性，因为在线系统需要预测未来，不是解释已发生的。

### 3.2 四种重构路径评估

action-conditioned harm estimation：**不是可选项，是第四轮证据要求的必要条件**。不同动作族脆弱度排序不一致，任何后续predictor必须按动作分组训练独立head。

uncertainty-aware allocation：所有predictor的worst-decile AUROC在0.286–0.75间剧烈波动，validation选出的最优配置常在sealed test上不复现（LLM-jp drop50% validation ρ=0.595但test仅0.044）。uncertainty-aware的意义不是锦上添花，而是**生存下限**：置信度低时必须自动退化为不依赖预测的公平机制。

streaming quality debt：**唯一完全不依赖predictor有效性、理论上最稳健**的方向——VTC-like debt只需知道请求最近被降级几次、多少，不需预测未来。这是本报告认为Quality Isolation两周内唯一值得投入的方向。

tenant-level fairness：项目从未构造真正多租户/多请求流workload（现有实验都是32-96篇独立文档的一次性静态配额），值得作为streaming debt实验的延伸，但不是第一优先级。

## 四、Expert Prefetch：信号真实，系统机会已被证明近乎为零

跨层路由可预测性这个**信号层面**的结论是所有候选里证据链最干净的：P0/P0-B/P0-C在严格calibration/test分割、真实路由数据、Holm校正显著性检验下，top-1 transition相对frequency baseline领先13-19pp，lookahead到8层仍有+8~14pp，是唯一被预注册门槛完全通过的正结果，不该因系统层面失败被一并否定。但本轮GPU审计把信号放进正确评估框架后（`(layer_id,expert_id)`cache key+真实full top-k+真实并发H2D/compute争用）发现了决定性负面事实：LLM-jp在batch=32下，91.2%的layer-batch已需全部expert（working set饱和），任何predictor相对frequency/random都没有选择空间；OLMoE在正确per-layer cache、safe overlap budget=1条件下，即便**理论上界oracle**收益也只有+0.256%，而按原方案固定预算=8，系统收益直接反转为-12.9%~-26.3%。这不是"实现细节没做对"，而是**oracle天花板本身在正确评估口径下已接近0**——与MassCover-EP被杀死的原因（简单信号已吃掉几乎全部可挖掘空间）是同一模式但方向相反：不是"简单baseline已足够好"，而是"硬件约束（overlap window极窄）本身不允许任何预取策略产生系统级差异"。叠加ExFlow/ProMoE/HOBBIT/Fate/LayerScope/PreScope/PROBE/ST-MoE已覆盖几乎全部"跨层预测+预取"机制空间，继续投入不会产生新的可发表系统结果。

能否重建价值：唯一值得一提的新问题是"当full top-k使working set饱和时，是否应完全放弃预测，改用确定性流水线加载或降低top-k"，但这已不再是"路由预测驱动预取"，而是完全不同的研究问题（更像MoE kernel/内存系统设计），不建议在当前时间预算内新开，只建议把现有发现（信号存在但系统机会为零+两次符号反转的教训）写成一段扎实的方法论负结果。

## 五、Energy-SLO Precision EP：被忽视的存活候选

这条候选未在用户问题中提及，但检索发现它是当前**唯一没有经历过与receiver-aware/Quality Isolation同等强度批判性复核**的存活候选。真实GPU功耗数据（`pynvml`/`nvidia-smi`实测，非仿真代理）显示两个独立杠杆都成立且量级可观：batch size从1到64，OLMoE能效提升17.4倍（235.9→13.6 mJ/token），TPOT只增47%（142.2ms→209.5ms）；真实FP8 tensor core（`torch._scaled_mm`）相对bf16 GEMM，吞吐提升2.03倍且净能耗降低34.3%/matmul。由此推出的"联合Pareto前沿"——只要SLO容忍度高于约210ms，batch=64在延迟和能耗两个维度上同时严格支配小batch，不存在"省电必须牺牲延迟"的权衡区间——是一条对controller设计直接有用的规则。

但存在一个**至今未被检验的关键质量confound**：FP8 tensor core能耗测试只在"形状匹配的独立matmul micro-benchmark"上做的，从未在真实模型的实际forward里把expert FFN计算换成FP8并测量下游KL。项目其他候选已反复证明"格式切换在微观指标上接近，但模型级质量可能完全不同"（receiver-aware第三轮教训："微观MSE接近并未转化为模型级KL等价"），因此"FP8计算是免费能耗收益"这个论断**还不能算成立的科学结论，只能算未被推翻的假设**。此外能耗感知的batch/精度联合调度在通用ML系统领域已有一定先例，正式投稿前需专门查重（本报告未安排检索，留作后续任务）。placement/EP mesh通信能耗维度完全未覆盖，与receiver-aware面临同样的资源瓶颈。

## 六、决定性实验设计（最小成本、按方向）

### Quality Isolation（最高优先级，可复用已收集数据或现有脚本，零/低新增GPU时间）

**实验1：predictor-free质量债务公平性。** 假设：即使不依赖任何预测信号，consecutive-K或VTC-like debt规则也能显著降低worst-request/CVaR，因为这是调度公平性质而非预测准确性质。真实可用信号：已收集的decode_fragility 48篇文档×4动作真实KL数据，用bootstrap构造合成多轮请求流即可，无需新GPU时间。对照基线：random、FIFO、静态一次性配额（已有Part A/C结果）。Go/No-Go：worst-request CVaR相对random改善≥20%，且吞吐/配额代价<3%（与批判性设计文档kill criteria一致）。最可能的失败confound：若虚拟请求身份的降级历史构造得太浅（轮数太少），debt机制来不及积累差异，需保证bootstrap有足够轮数T。

**实验2：one-step在线decode predictor。** 假设：用当前decode step的causally-available特征（router margin、logit entropy、hidden norm，均为任意forward的免费副产品）预测下一步同一动作的KL，比静态prefill-only proxy更有效，因为避免了"预测遥远未来"的时序漂移问题。真实信号：复用`run_decode_fragility_strict_gpu.py`的数据采集框架，只需把特征提取时点从prefill换成每个decode step。对照基线：静态prefill-only proxy（已证明失败）、random、arrival lexical。Go/No-Go：sealed test上95%CI不跨0且worst-decile recall至少是random的2倍，需在至少2个动作上复现。最危险的confound：若特征取自"已应用近似动作之后"的路径会引入动作后信息泄漏，必须严格保证特征来自full-precision参考路径。

**实验3：action-conditional predictor。** 假设：按动作族（量化类vs丢弃类）训练独立predictor head比单一共享模型更好，因为第四轮数据显示跨族相关性低至0.129。真实信号：同实验2特征，按动作分组重新拟合。对照基线：单一共享模型。Go/No-Go：分组模型的ρ相对共享模型有统计显著提升，且每族内recall有实质改善。风险confound：动作族样本量进一步减半会加重小样本CI过宽的问题（第四轮test只有16篇），需权衡是否先扩大采集规模。

### Receiver-aware（次优先级，部分依赖多卡环境）

**实验4：direct-benefit predictor替代regime分类器。** 假设：直接回归"下一microbatch的排队/关键路径收益"比"先分类regime再套策略"更稳健，天然避免v3因果bug。真实可用信号：复用已有真实路由trace，补一个用`torch.distributed`（gloo或NCCL多进程模拟多rank）产生的真实all-to-all完成时间，替代线性带宽模型——这是单卡也能做到的"更真实一步"信号源。对照基线：`always_causal_prev_step`、原HHI adaptive、静态profile。Go/No-Go：direct-benefit predictor相对最强固定基线的改善95%CI不跨0，且改善量超过量化/反量化开销估计。最危险confound：单卡多进程模拟的all-to-all时延仍不含真实网络incast，结果只能标注为"必要不充分"的中间验证。

**实验5：homogeneous one-shot lane + receiver credit。** 假设：整条peer/lane一次性选择FP8或4-bit（而非逐token-rank混合或progressive）配合队列阈值控制，能在真实codec开销下超过uniform FP8基线。真实信号：复用第三轮已实现的真实Triton codec（pack/unpack实测延迟）+ 已有路由trace构造的模拟队列深度。对照基线：uniform FP8、uniform INT4、静态profile、oracle。Go/No-Go：在matched quality budget下，模拟combine完成时间相对uniform FP8改善≥10%（对齐批判性设计文档kill criteria）。最危险confound：break-even点高度依赖tile size（第三轮数据显示32×512与512×2048的break-even相差60倍以上），若tile size选择不当会得到虚假的负结果或虚假的正结果。

### Energy-SLO（第二优先级，一次GPU实验即可补齐关键缺口）

**实验6：FP8计算的下游质量代价。** 假设：把OLMoE/LLM-jp的expert FFN计算换成真实FP8 tensor core路径后，下游KL相对bf16参考的增幅仍在项目既有可接受阈值（NLL_MARGIN≈0.005量级）之内，从而让"FP8计算是免费能耗收益"这个论断真正成立。真实信号：在已加载的模型上用`torch._scaled_mm`替换expert gate/up/down投影的实际前向计算（而非仅做matmul micro-benchmark），用与其他候选一致的paired bootstrap方法测KL。对照基线：bf16参考、以及项目里已知的uniform INT4通信降级KL（0.257，作为"明显不可接受"的参照上界）。Go/No-Go：mean KL增幅显著低于uniform INT4通信降级的量级，理想情况下落在0.005-0.05区间内视为可接受。最危险confound：若FP8计算KL接近或超过uniform INT4通信降级的量级，说明"计算精度"和"通信精度"的质量敏感度完全不同，不能类比迁移，Energy-SLO需要重新引入quality-aware fallback，整条Pareto front论断需要重新画。

**实验7：三维联合Pareto（TPOT×能耗×质量）。** 假设：在实验6确认质量代价可接受的前提下，batch与精度两个杠杆对能耗的贡献接近独立（乘性），使controller可以分别优化两个维度而不必联合搜索。真实信号：复用已有batch-能耗数据+FP8/bf16能耗数据+实验6的质量数据，做一次联合扫描（batch∈{1,4,16,64}×precision∈{bf16,fp8}）。对照基线：单独优化batch或单独优化precision的次优解。Go/No-Go：联合最优解相对"仅优化其中一维"的能耗改善≥10%，验证联合控制器确有必要而非过度设计。风险confound：small-batch下FP8的相对能耗优势可能被kernel launch固定开销抹平（类似expert prefetch遇到的"固定开销主导"教训），需要在batch=1这类边界点专门检查。

## 七、方向归类

| 候选（重构后表述） | 归类 |
|---|---|
| Quality Isolation → predictor-free质量债务公平调度 | 可作独立小论文主贡献（若做完真实multi-round流+多tenant），或作合并系统的质量控制面（第二贡献） |
| Quality Isolation → one-step/action-conditional predictor | 只能作辅助/第二贡献，不构成独立claim的核心 |
| Receiver-aware → direct-benefit+homogeneous lane+receiver credit | 条件性主贡献：拿到多卡/RDMA环境前只能算系统特征化（第二贡献/负结果丰富的章节） |
| Energy-SLO Precision EP | 第二贡献（能耗特征化独立小节），不建议作为唯一主贡献 |
| Expert Prefetch | 负结果/经验分析，应停止投入 |
| PLTB、Graceful/QTree、additive-KL MILP、Tiered-Fidelity、Residual-EP、TokenRace-EP、MassCover-EP、QuotaEP-H/PartialGuard、CreditReduce、RouteFidelity-EP、WaveCredit-EP | 已确认应停止投入，仅作方法论教训 |

## 八、候选排序与评分

评分采用0-10分制，"当前证据分"衡量现有数据在严格审计标准下能支撑的结论强度，"重构潜力分"衡量按第五节实验方案执行且假设最理想成立情况下的天花板。

| 排序 | 候选（重构表述） | 核心假设状态 | 当前证据分 | 重构潜力分 | 主要瓶颈 |
|---|---|---|---|---|---|
| 1 | Quality Isolation → predictor-free debt fairness | predictor假设已证伪，debt fairness假设未测但理论稳健 | 3 | 6 | 需真实多轮/多租户workload |
| 2 | Receiver-aware → direct-benefit+homogeneous lane | 结构性画像成立，adaptive分类器已证伪 | 4 | 7（有多卡）/3（无多卡） | 真实RDMA/incast环境可得性 |
| 3 | Energy-SLO Precision EP | 两杠杆独立成立，联合controller未验证 | 5 | 6 | FP8计算质量代价未测 |
| 4 | Expert Prefetch | 信号存在，系统机会oracle上限≈0 | 1.5（系统层面） | 2 | 硬件overlap window过窄+novelty冲突 |

Quality Isolation的predictor方向单独评分：当前证据分2，重构潜力分4（受限于第四轮已证伪的核心假设）。

## 九、推荐统一论文主线

维持批判性设计文档的核心判断：**不应把receiver-aware、Quality Isolation、Expert Prefetch三条硬塞进一篇系统论文**，Expert Prefetch应完全排除在主线之外。主线仍是Receiver-aware（拥塞感知何时/何处降级）+ Quality Isolation（降级伤害由谁承担）合并为一个完整控制回路，但相对此前文档需要两处修正：一是把Quality Isolation的核心机制从"预测脆弱度"降级为"predictor-free质量债务公平"，predictor仅作为置信度足够时的辅助信号，因为第四轮证据显示预测本身不可靠；二是**不建议**把Energy-SLO并入这条主线（会重现Prefetch被排除的"三套瓶颈、三套控制面"问题），应作为独立的第二贡献章节或附加小论文。

主线最小可信claim（在完成真实多卡原型前）：一个只在下一microbatch真实排队/关键路径收益足以覆盖量化开销时才降级、并用predictor-free质量债务限制重复伤害的近似通信控制器，在matched quality budget下于synthetic trace回放中改善模拟排队指标，且在telemetry不可靠或收益不足时自动回退到原生通信。完成真实多卡/RDMA原型后可升级为在真实EP incast下的P99 TPOT改善claim——这一步的达成完全取决于能否获得多GPU/RDMA资源，是本报告识别出的全项目最高杠杆的非研究性行动项。

## 十、接下来两周实验清单

第一周：（1-2天）实验1 predictor-free debt fairness——复用现有数据零成本，优先执行；（2-3天）实验6 FP8计算质量代价——补齐Energy-SLO最关键缺口；（3-5天）实验2 one-step在线decode predictor；（5-7天）实验5 homogeneous one-shot lane + receiver credit——复用第三轮已实现的Triton codec。

第二周：（8-10天）实验4 direct-benefit predictor替代regime分类器，用多进程模拟多rank获取真实all-to-all时延；（10-12天）实验3 action-conditional predictor；（12-13天）实验7 三维联合Pareto；（13-14天）汇总go/no-go决策，冻结进入合并系统的机制清单，同步向导师/实验室确认多GPU/RDMA资源可得性（这是决定主线能否升级为完整系统论文的最高优先级非实验行动项），更新Approach Registry并撰写统一主线的论文大纲/摘要。

两周内不建议为Expert Prefetch或已KILLED的候选分配任何新实验预算。

## 十一、局限

本报告完全基于项目现有文档与实验产物的重新解读与综合，没有执行任何新代码或新GPU实验，第五/六节的实验设计是提案而非已验证结果。外部文献grounding受限于检索深度，Energy-SLO方向的直接novelty查重未完成，需要在正式投稿前补做。所有"重构潜力分"都是条件性估计，其中receiver-aware和Energy-SLO的placement/通信维度都系统性受限于当前只有单张RTX 5090的硬件环境，这是贯穿多个候选的共同瓶颈，而非某一条候选独有的缺陷。

## 附录A：全部候选历史状态一览

| 候选 | 状态 |
|---|---|
| PLTB | KILLED |
| R-layout | 降级为baseline（非死） |
| Graceful-EP/QTree-EP | KILLED |
| additive-KL MILP | KILLED |
| Tiered-Fidelity Shadow Experts | KILLED（决定性） |
| Residual-EP | KILLED（决定性，必要条件证伪） |
| QuotaEP-H/PartialGuard | KILLED AS CORE |
| CreditReduce | KILLED |
| RouteFidelity-EP | KILLED |
| WaveCredit-EP | KILLED（文献碰撞） |
| MassCover-EP | KILLED（复查确认无空间） |
| TokenRace-EP | KILLED（GPU追加验证不可挽回） |
| Receiver-aware | CONDITIONAL_STRUCTURAL_ONLY（本报告：需重构） |
| Per-request Quality Isolation | 核心假设已被第四轮证伪（本报告：需重构为debt fairness） |
| MoE路由预取/Expert Prefetch | NO-GO作为独立系统论文；离线信号保留 |
| Energy-SLO Precision EP | 两杠杆真实验证成立，联合controller与质量轴未完成 |
| Route-keyed Temporal DPCM Combine | 零正证据，未执行，条件备份 |
| CodecMap/Uniform-FP8 Sufficiency Envelope | 评价骨架，非独立贡献 |
| CUDA Graph × MoE动态调度协同捕获 | 调研建议阶段，未判定 |
| 投机解码×MoE路由可预测性 | 调研建议阶段，未判定 |

## References

1. [MoE_Approach_Registry_2026-07-19.md（项目内部文档）](file:///Users/leandrozhao/Desktop/毕设论文资料/MoE_Approach_Registry_2026-07-19.md)
2. [ExFlow](https://arxiv.org/abs/2401.08383)
3. [ProMoE](https://arxiv.org/abs/2410.22134)
4. [HOBBIT](https://arxiv.org/abs/2411.01433)
5. [Fate](https://arxiv.org/abs/2502.12224)
6. [LayerScope/PreScope](https://arxiv.org/abs/2509.23638)
7. [PROBE](https://arxiv.org/abs/2602.00509)
8. [ST-MoE](https://arxiv.org/abs/2606.15453)
9. [Aurora](https://arxiv.org/abs/2410.17043)
10. [UCCL-EP](https://arxiv.org/abs/2512.19849)
11. [VTC, OSDI 2024](https://arxiv.org/abs/2401.00588)
12. [Proteus, ASPLOS 2024](https://doi.org/10.1145/3617232.3624849)
13. [Apparate, SOSP 2024](https://doi.org/10.1145/3694715.3695963)
14. [SuperServe, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/khare)
15. [Decentralized Contextual Bandits with Network Adaptivity](https://arxiv.org/abs/2508.13411)
16. [Improving dynamic congestion isolation in data-center networks (ICI)](https://arxiv.org/abs/2511.04639)
17. [Comet: Fine-grained Computation-communication Overlapping for Mixture of Experts](https://arxiv.org/abs/2502.19811)
18. [Ensuring Fair LLM Serving Amid Diverse Applications (FairServe)](https://arxiv.org/abs/2411.15997)
19. [Receiver-Driven RDMA Congestion Control by Differentiating Congestion (RCC), ICNP 2021](https://icnp21.cs.ucr.edu/papers/icnp21camera-paper45.pdf)
20. [Congestion Control for Large-Scale RDMA Deployments (DCQCN)](https://dl.acm.org/doi/abs/10.1145/2829988.2787484)

