# GPU 实验结果汇编（第五辑）：Idea A/B 复核、Placement 负结果与三个新方向探索（2026-07-20）

> 本文档承接 `docs/02_gpu_audits/` 已有的"GPU 第一轮至第四轮有效性实验结果"（receiver_causal+ExpertPrefetch → Quality Isolation → Receiver Progressive/Codec → Prefill→DecodeFragility），汇编 2026-07-20 当天在此之后产出的全部单次实验结果，按主题分六节呈现，不重复前四轮已有内容。原始 6 个根目录文件已在整理后删除，内容完整迁移于此，不做二次总结式改写，所有关键数字表格原样保留。

## 目录

1. Idea A：combine 轴根基性主张（Rank 长尾 + FP8-first Pareto）真实 GPU 严格再验证
2. Idea B：Expert 权重/计算精度轴（Precision-Sufficiency Shadow Verification）Day-1 决定性实验
3. Receiver-aware：Expert Placement 优化（拓扑+拥塞视角）真实验证 —— 负结果
4. 三条决定性实验：Energy-SLO / Quality Isolation debt / Receiver direct-benefit 真实 GPU 结果
5. 三个新创新方向的真实验证结果（AIMD 自适应周期 / 真实 decode-loop 运行时 / 双轴联合控制器 POC）—— 均为负结果
6. 附录：6 个新 idea 完整评估（本轮 Idea A/B 实验的设计动机来源）

---

## 一、Idea A：combine 轴根基性主张真实 GPU 严格再验证

硬件：远程 RTX 5090 32GB（单卡）｜ 脚本：`experiments/idea_a_mac/run_idea_a_rank_lut_gpu_rigorous_verify.py`

### 1.0 为什么验证这个，而不是最近的 receiver-aware v3/direct-benefit

"Idea A"真正的根基性主张不是最近失败的自适应控制器（v3因果bug/direct-benefit混合结果），而是更早、更基础的 **Rank 长尾结构**：在 top-k 路由内部，不同 rank 对 combine 输出的贡献存在稳定长尾，同等 byte budget 下压低"tail rank"远比压"head rank(rank1)"安全。这个结论从 2026-06-24 到 07-14 跨 3 个模型反复复现（OLMoE top-8: 58x，Mixtral-TinyMistral top-2，LLM-jp top-16: 107.8x），是整条 combine 谱系（FP8-first tail-INT4、R-layout、QuotaEP-H、receiver-aware）的共同根——如果它不成立，下游全部不成立；如果它稳固，就是全项目里最值得写进论文的静态结构性发现。

但所有这些数字都是在 Mac M5 Pro CPU、无 GPU、无 bootstrap CI（只有单点估计）的条件下得到的。这次验证的目标是：换成真实 RTX 5090 GPU、用一批从未被任何实验碰过的全新文档（offset=600）、补上文档级 paired bootstrap CI，把这条最强的证据从"令人印象深刻但统计上不严谨"升级为"真正站得住"。

### 1.1 两个预注册判据

**Claim 1（matched-byte-budget tail-vs-head，"决定性证据"）**：`rank1_int4`（只把 rank1 换成INT4）vs `rankk_int4`（只把 tail rank 换成INT4）——两者改变的 byte 量完全相同，任何 KL 差异只能来自"改的是哪个 rank"。GO 判据：文档级 paired bootstrap 95% CI 下界 > 0，且比值 > 5x。

**Claim 2（FP8-first tail-INT4 Pareto 前沿）**：从 `uniform_fp8` 开始逐步把更多 tail rank 换成 INT4，KL 应平滑单调上升，且在扫描网格里最差的点也要比 `uniform_int4` 有 ≥5x 的安全边际。

### 1.2 真实结果

| 模型 | Claim 1 head/tail比值 | Claim 1 diff 95% CI | Claim 1 判定 | Claim 2(网格最差点)安全边际 | Claim 2 判定 |
|---|---|---|---|---|---|
| OLMoE (top_k=8, N=128) | **48.11x** | [0.194, 0.214] | **GO** | 4.65x | NO-GO(差一点) |
| LLM-jp (top_k=16, N=128) | **117.35x** | [0.193, 0.208] | **GO** | 5.78x | **GO** |

Claim 1 是两个模型上都干净、稳固的 GO，且和原始 Mac CPU 报告的点估计（OLMoE 58x，LLM-jp 107.8x）几乎精确复现，现在第一次有了紧致的 bootstrap CI 背书（两个模型的 CI 宽度都只有约0.02，相对效应量而言极窄，说明信号非常干净，不是噪声）。

### 1.3 Claim 2 的诚实解读：不是复现失败，是判据设计过严

OLMoE 上 Claim 2 按代码里写的判据是 NO-GO（4.65x < 5x），但仔细核查后发现这是判据设计的问题，不是结论的问题：扫描网格一直延伸到 68.75% saving（`fp8top2_rest_int4`），比原始报告实际主张的 62.5% saving 点（`fp8top4_rest_int4`）更激进、更深。在原论文真正主张的62.5%这个具体点上重新核查：

| 模型 | 62.5%点位KL(95%CI上界) | uniform_int4 KL(95%CI下界) | 该点安全边际 |
|---|---|---|---|
| OLMoE (`fp8top4_rest_int4`) | 0.02424 | 0.24937 | **10.29x** |
| LLM-jp (`fp8top8_rest_int4`) | 0.00902 | 0.19935 | **22.1x** |

在原论文实际主张的那个点上，两个模型都远超5x门槛，甚至远超10x——Claim 2 在"论文真正声称的范围"内是干净的 GO。62.5%以上更激进的区域(65.6%~68.75%)确实呈现边际快速收窄的趋势，这是一个新发现（"再往深压 tail 会加速失效"），不是对原结论的否定。

### 1.4 结论

Idea A 的根基性主张（tail-vs-head 长尾结构，Claim 1）在真实 RTX 5090 GPU、128篇全新文档、文档级 bootstrap CI 的严格条件下**完全站得住**，效应量（48x-117x）和原始 Mac CPU 报告高度吻合。FP8-first tail-INT4 Pareto 前沿（Claim 2）在原论文实际主张的62.5%饱和点上同样干净通过（10x-22x安全边际）。

证据边界：这次测的是单次前向的 combine 输出 KL，不是 decode 循环、不含真实通信延迟——latency/throughput/P99层面的系统性 claim 仍需靠 receiver-aware/direct-benefit 那条独立的线支撑。INT4 用的是逐 token 对称 fake-quant 代理（和原论文一致），真实 block-scaled MXFP4/NVFP4 下绝对 KL 数值会更小；本次验证确认的是**相对**的 tail-vs-head 排序，这个排序已被格式审计证实为格式无关的稳健结论。

产物位置：`experiments/idea_a_mac/run_idea_a_rank_lut_gpu_rigorous_verify.py`；`outputs/idea_a_rank_lut_gpu_verify_2026-07-20_olmoe/`、`outputs/idea_a_rank_lut_gpu_verify_2026-07-20_llmjp/`（per_document.csv, policy_summary_with_ci.csv, metadata.json, report.md）。

---

## 二、Idea B：Expert 权重/计算精度轴 Day-1 决定性实验真实 GPU 结果

硬件：远程 RTX 5090 32GB（单卡）｜ 脚本：`experiments/idea_a_mac/run_expert_precision_persistence_shadow_verify_p0.py`

### 2.0 这条实验回答的问题

Idea B 的核心假设是：decode 内**短 horizon 量化风险信号存在持续性**，因此一个只依赖自身过去 realized 信号的因果影子验证控制器可以在不预测未来、不跨请求泛化的前提下，用较低的高精度开销把 INT4 权重带来的质量风险控制住。这条实验完全在新的轴上（expert 的 `gate_proj/up_proj/down_proj` 或 `w1/w2/w3` 计算精度），与 receiver-aware/旧版 Quality Isolation 所在的 combine/dispatch 通信轴是两个独立的张量、两套机制。

两个预注册假设：
- **H1（主判据，决定性）**：同一文档内，lag=1..8 步的 realized KL 存在持续性。GO 判据：document-level block bootstrap 95% CI 下界 > 0.2。
- **H2（次判据，操作性）**：一个因果式"周期验证+阈值升级"控制器，能否用较少的高精度开销拿到大部分质量保护。GO 判据：相对 always-low 降低累计 KL ≥50%，高精度步占比 ≤50%，且不超过 non-causal oracle 上界的 2 倍。

### 2.1 真实结果（OLMoE 与 LLM-jp，各32篇文档，48步decode，calib=12/test=20）

**H1：持续性确实存在，但强度低于预注册门槛**

| 模型 | lag=1 rho | lag=1 CI | lag=2 rho | lag=4 rho | lag=8 rho | GO(任意lag) |
|---|---|---|---|---|---|---|
| OLMoE | 0.180 | [0.122, 0.228] | 0.103 | 0.099 | 0.068 | NO-GO |
| LLM-jp | 0.203 | [0.125, 0.277] | 0.069 | 0.116 | 0.025 | NO-GO |

两个模型在每个 lag 上的 CI 几乎都不跨0，说明持续性是真实、统计显著、跨模型可复现的信号——不是随机噪声。但强度只有 0.10-0.20 量级，没有达到预设的 0.2 门槛，因此 H1 严格判 NO-GO。

**H2：控制器仿真——弱持续性已经足够撑起一个真实有效的控制器**

| 模型 | period | 相对always-low降幅 | 降幅CI | 高精度步占比 | vs oracle倍数 | GO/NO-GO |
|---|---|---|---|---|---|---|
| OLMoE | 4 | **50.1%** | [42.1%, 56.6%] | 43.4% | 1.66x | **GO** |
| OLMoE | 8 | 41.7% | [32.0%, 50.6%] | 35.8% | 1.94x | NO-GO |
| OLMoE | 16 | 38.9% | [22.5%, 52.9%] | 34.4% | 2.03x | NO-GO |
| LLM-jp | 4 | 47.0% | [40.2%, 53.4%] | 40.3% | 1.26x | NO-GO（差3pp未过线） |
| LLM-jp | 8 | 29.4% | [19.9%, 40.2%] | 31.5% | 1.68x | NO-GO |
| LLM-jp | 16 | 25.1% | [15.9%, 33.9%] | 28.1% | 1.78x | NO-GO |

关键发现：period=4 是两个模型上均明显最优的配置。OLMoE 在 period=4 上干净通过全部三条 GO 判据。LLM-jp 在 period=4 上只差 3 个百分点未达 50% 降幅门槛（点估计47.0%，CI上界53.4%已经越过50%），且相对 oracle 的倍数（1.26x）反而好于 OLMoE（1.66x）。

### 2.2 如何诚实解读这个结果

H1 用 0.2 作为门槛本身是偏保守的工程选择，H2 才是真正回答"这个信号对系统有没有用"的问题。真实情况是：即使 lag-1 持续性只有 0.18-0.20 这么弱，只要配合周期性重新验证和保守阈值，因果控制器依然能拿到 40-50% 的质量保护，且离非因果 oracle 上界只差 1.3-1.7 倍。这正是当初选择 Idea B（"验证而非预测"）而不是 Idea F（"预测未来"，已被 round2/4 证伪）的理论预期：**弱持续性对预测器判死刑，但对配备安全网的反应式控制器只是优雅退化，不是崩溃**。这是目前整个项目里为数不多的、当初设计动机被真实GPU数据直接验证的例子。

局限：H1 NO-GO / H2 一个模型GO一个模型接近GO，只能定性为"方向成立、量级中等、存在真实但不算戏剧性的模型依赖"。若把门槛放宽到45%，两个模型在period=4上都会GO。

### 2.3 已知confound核查

INT4 weight-only 是 quant-dequant 代理，不是真实低位内核，因此本结果只是质量信号，不含wall-clock证据。教师强制在真实语料续写上进行，不是模型自由生成。escalate阈值只在calibration上标定一次直接用于test，没有反向调参（两个模型的τ分别为 OLMoE 0.067，LLM-jp 0.079，彼此接近但不同，符合"模型有自己的风险尺度"的预期）。同一步的路由特征相关性诊断表里 `full_route_rank1_hhi_mean` 和 `full_route_active_expert_fraction_mean` 两列恒为0，这是因为单token单步下这两个统计量在数学上几乎不含信息，不是bug。

### 2.4 下一步建议（已部分执行，见第五节）

优先做的两件事（复用已收集的 `per_step_samples.csv`，零新增GPU时间）：一是把 verify period 网格加细到 {1,2,3,4,6} 并把 escalate-quantile 扫描到{0.6,0.7,0.75,0.8,0.9}；二是把GO判据的50%降幅门槛做敏感性分析（45%/50%/55%三档）。若确认period=4附近对两个模型都成立，下一步是把当前的"离线仿真"（`simulate_policies` 是对已收集trajectory的事后重放）升级为真正的decode loop运行时实现。

产物位置：`experiments/idea_a_mac/run_expert_precision_persistence_shadow_verify_p0.py`；`outputs/expert_precision_persistence_2026-07-20_olmoe/`、`outputs/expert_precision_persistence_2026-07-20_llmjp/`（per_step_samples.csv, lag_persistence_results.csv, controller_simulation_results.csv, same_step_diagnostic_correlations.csv, metadata.json, report.md）。

---

## 三、Receiver-aware：Expert Placement 优化真实验证 —— 负结果

方法：真实路由数据驱动的LPT贪心均衡分箱 ｜ 成本：零新增GPU时间（纯CPU/pandas分析）

### 3.0 出发点

Receiver-aware 这条线里，expert placement（专家分配到哪个物理节点）此前一直被当成固定输入。placement优化是一个经典的、有成熟解法的组合优化问题（多路数字分割/负载均衡分箱），完全不需要在线信号、不需要多GPU，只需要真实路由数据。具体做法：用calibration文档的真实路由数据统计每个expert的真实被选中频率，跑经典LPT贪心，与 `contiguous`、`round_robin` 两个完全不看数据的基线对比，复用 `run_receiver_aware_v2_systematic.py` 一模一样的场景构建和瓶颈字节-时间指标（`fp8_total`）。

### 3.1 真实结果：干净的负结果，不是bug

| 模型 | origin_mode | 相对最强现有基线的降幅 | 95%CI | GO/NO-GO |
|---|---|---|---|---|
| OLMoE | balanced | -0.25% | [-37.3, +15.1]us(跨0) | NO-GO |
| OLMoE | hotspot | -0.72% | [-110.3, -41.8]us(**显著为负**) | NO-GO |
| LLM-jp | balanced | +0.55% | [+5.5, +16.1]us(不跨0但太小) | NO-GO(未达10%门槛) |
| LLM-jp | hotspot | -1.60% | [-97.7, -72.0]us(**显著为负**) | NO-GO |

四个格子里没有一个GO，其中两个格子（OLMoE hotspot, LLM-jp hotspot）甚至统计显著地更差。

### 3.2 排查：不是没有真实skew可利用，是这个skew没用在正确的目标上

真实expert popularity的变异系数：

| 模型 | popularity均值 | 标准差 | CV | 最冷/最热expert |
|---|---|---|---|---|
| OLMoE(64专家) | 6144 | 1178.5 | **0.192** | 3372 vs 9262(2.75倍) |
| LLM-jp(32专家) | 24576 | 1746.6 | **0.071** | 20567 vs 27563(1.34倍) |

真实skew是存在的、有意义的，LPT算法也确实正确利用了这个skew（校准后每个rank的负载被压到只有~2.5%的差异），但"平均负载均衡"根本没有转化成瓶颈指标的改善。根本原因：评测复用的 `fp8_total` 指标，在每个全局时间步取的是跨rank的**max**（瓶颈/尾部驱动），不是跨rank的总和或均值。16个并发job、随机stagger叠加之后，任何单一时间步的瞬时egress峰值主要由"恰好在这一步同时活跃的那几个job，恰好选中了哪些专家"这种瞬时偶然性决定，而不是"这个专家平均被选中多少次"这种全局统计量决定。**平均负载均衡(mean-balancing)和尾部/瞬时突发均衡(tail-balancing)是排队论里两个不同的目标**，前者做得好不保证后者也好。

hotspot模式下变得更差的原因：hotspot把大部分job的receiver钉死在rank 0，这个ingress瓶颈是placement完全控制不了的常量项，任何sender侧的重新洗牌都只是在噪声里晃动。

### 3.3 结论

这条路径没有产出可以写进论文的正结果，但排查过程给出了一个有价值、可复现的诚实结论：MoE路由训练带来的expert popularity skew是真实存在的（CV 0.07-0.19），但在这个项目一直使用的、真正决定P99/TPOT的瞬时瓶颈型流量指标下，静态的平均负载均衡placement优化没有可利用的空间——这与本项目反复出现的元教训一致（MassCover-EP、Expert Prefetch都是"看似有信号，但目标指标下这个信号已经没有可挖掘空间"的模式），只是这次"没有空间"的原因是信号本身和优化目标的粒度不匹配（平均量级信号 vs 瞬时/尾部量级目标）。**没有继续换更复杂的图分割/协同激活算法去凑结果**：即使换成基于专家共激活图的分割算法，只要评测指标仍是多job并发聚合下的瞬时max，同样的结构性障碍大概率依然存在。

若还想在这条线上找机会：应调整目标指标本身（如"降低瞬时max的P99而非全部steps求和"），而不是换placement算法；或彻底放弃placement优化，承认固定period=4影子验证/direct-benefit controller是这条线上唯一真正有效的杠杆。**建议不再投入新的placement变体**。

产物位置：`experiments/idea_a_mac/analyze_expert_placement_optimization.py`；`outputs/expert_placement_optimization_2026-07-20/`（results.csv, metadata.json）。

---

## 四、三条决定性实验：真实 GPU 结果与结论更新

> 范围：对审计报告中提出的三个"最小成本决定性实验"（receiver-aware direct-benefit 控制器、Quality Isolation predictor-free 债务公平调度、Energy-SLO 的 FP8 计算质量 confound 门槛）进行脚本设计并在真实 RTX 5090 上实际执行。

### 4.1 结果总览

| 候选（重构后） | 预注册 Go/No-Go 结果 | 关键数字 | 相对审计报告的评分变化 |
|---|---|---|---|
| Energy-SLO：FP8 计算质量 confound | **GO**（两模型一致） | OLMoE KL=0.00675 CI[0.0061,0.0075]；LLM-jp KL=0.00905 CI[0.0082,0.0104]；均远低于0.05阈值，比已知的uniform INT4通信降级KL(0.257)低约30-40倍 | 当前证据分 5→**7**，最大缺口已补齐 |
| Quality Isolation：predictor-free 质量债务公平调度 | **NO-GO**（全部4个动作），但效应真实显著 | worst-tenant改善10.6%-12.5%，95%CI下界均>5%（不跨0），总系统harm变化≈0 | predictor-free方向证据分（此前未测）→**4**，方向被证实有效但未达20%门槛 |
| Receiver-aware：direct-benefit + hysteresis + receiver credit | **模型依赖：LLM-jp GO / OLMoE NO-GO** | LLM-jp: controller−causal_no_hysteresis = +0.91pp, CI[0.23pp,1.62pp]；OLMoE: +0.38pp, CI[-0.05pp,0.85pp] | 当前证据分 4→**5**，从纯理论提案升级为"至少一个模型上有真实统计显著证据"，但暴露了新的模型依赖性问题 |
| Expert Prefetch | 未执行（按计划不分配预算） | — | 不变，维持停止投入 |

### 4.2 Energy-SLO：本轮最干净的正结果，附三个真实 bug 的方法论教训

FP8 计算对下游质量的伤害在两个模型上都很小（KL 量级 0.007-0.009），且远低于"通信精度降级"的已知不可接受量级（0.257）。这意味着"精度切换是免费能耗收益"的论断，在补上这个此前缺失的质量confound之后**站住了**，可以正式升级为可信结论。

调试过程暴露了三个真实的实现 bug：（1）`torch._scaled_mm` 要求 a 矩阵行主序、b 矩阵列主序；对已经是 fp8 的权重张量做 `.t().contiguous()` 会把列主序视图重新拷贝回行主序，直接触发 cuBLASLt 报错，修复为转置后不再 `.contiguous()`；（2）真实 MoE 推理中存在路由到 0 个 token 的专家，`amax()` 对空张量维度归约会报错，修复为空批次直接返回零张量；（3）**最有价值的一个**：LLM-jp 实际是 Mixtral 架构（`layer.block_sparse_moe.experts`，子模块命名为 `w1/w2/w3`），脚本最初只识别 OLMoE/Qwen2-MoE 风格的 `gate_proj/up_proj/down_proj`，第一次跑 LLM-jp 时脚本"成功"运行且报出 `mean_token_kl=0.000000`——这是一次**静默的假阴性**：patch 完全没有生效，但脚本本身不报错，还给出了一个看起来"完美"的 GO 结果。这与本项目反复点名的"对象身份定义错误是最危险、最难被统计检验发现的 bug 类型"完全同构：**一个静默返回"零效应"的 patch，比一个报错的 patch 更危险**。修复后 LLM-jp 才出现真实的非零 KL。

### 4.3 Quality Isolation：真实但不够大的效应

debt-based 调度让 worst-tenant 累积 harm 比 random 降低约 11%-12.5%，四个动作全部方向一致、CI 不跨 0，且总系统 harm 几乎不变（代价为零）——这是一个真实的、predictor-free 就能拿到的公平性增益，但没有达到预注册的 20% 门槛，按冻结标准应判 NO-GO。这个结果说明 debt 调度机制本身有效，但当前 16 篇 sealed-test 文档 + 12 租户 + 200 轮的规模下，harm 分布本身的长尾程度还不足以让 debt 调度显著超车。

### 4.4 Receiver-aware：模型依赖性是新暴露的问题，不是噪声

Direct-benefit + hysteresis 控制器相对"无 hysteresis 的因果基线"在 LLM-jp（E32K16）上有小但统计显著的增益（+0.91pp），在 OLMoE（E64K8）上则没有（CI 跨 0）。两模型的拟合阈值差了约 2.6 倍（1110 vs 2179）。这暴露了一个新变量：**hysteresis 的价值可能依赖专家数/EP 拓扑规模**（64 专家的 OLMoE 里单个 receiver 的负载波动被更多专家"摊薄"，信号本身更平滑，用不用 hysteresis 差异不大；32 专家的 LLM-jp 里负载更集中，hysteresis 抑制震荡的价值才显现）。

产物位置：`run_energy_slo_fp8_compute_quality_gate.py`（`outputs/energy_slo_fp8_compute_quality_2026-07-20_olmoe/`、`_llmjp/`）；`run_quality_debt_fairness_p0.py`（`outputs/quality_debt_fairness_2026-07-20/`）；`run_receiver_direct_benefit_controller.py`（`outputs/receiver_direct_benefit_2026-07-20_full/`）。

---

## 五、三个新创新方向的真实验证结果（诚实汇报，均为负结果）

> 承接第二节 Idea B 固定 period 网格的结果，把当天验证过的最简单机制（固定period=4影子验证）拿去跟两个更复杂的候选方案正面比较。

### 5.1 Task 1：自适应验证周期控制器（AIMD风格）—— 负结果

用已采集的真实轨迹数据（零新增GPU成本），在calibration文档上网格搜索超参数，用鲁棒评分(均值-标准差)选参数，再在held-out test文档上评估：

| 模型 | 固定period=4(第二节结果) | 自适应(本次) |
|---|---|---|
| OLMoE | 降幅50.1% CI[42.1%,56.6%] 占比43.4% oracle比1.66x **GO** | 降幅55.9% CI[42.6%,67.8%] 占比50.6% oracle比**2.45x** NO-GO |
| LLM-jp | 降幅47.0% CI[40.2%,53.4%] 占比40.3% oracle比1.26x NO-GO | 降幅47.3% CI[37.3%,55.9%] 占比42.6% oracle比1.94x NO-GO |

自适应机制在OLMoE上把"降幅"这一指标推得更高了(55.9%>50.1%)，但代价是"效率"这个维度明显变差(2.45x vs 1.66x)——这是校准阶段只优化单一指标导致的过拟合式取舍，calibration样本量只有12篇也放大了这个风险。LLM-jp上自适应基本没有帮助。**结论：更复杂的AIMD机制没有换来真正的提升，固定period=4依然是目前证据最支持的方案。**

### 5.2 Task 2：真实decode-loop运行时实现 —— 环境受限，降级为分析型估算

尝试用 `torchao.quantization.Int4WeightOnlyConfig`（PyTorch原生 `_weight_int4pack_mm` 真实打包INT4内核）在两个模型的真实expert矩阵形状上做self-test，两次合理尝试都失败：默认路径缺少名为 `mslk` 的依赖库；改用旧版路径会触发内部形状assertion。这是真实的环境限制（RTX5090 + torch2.8+cu128的组合），不是思路错误。

退而测了一个更根本的问题：当前脚本里用的INT4 fake-quant代理，相对纯bf16矩乘，有没有真实计算时间差异——实测两者耗时比值0.80-0.81x，完全在噪声范围内（两条路径本质是同一个bf16矩乘kernel，只是权重数值不同，没有任何真正的低比特计算）。

诚实的补救：用已算出的high_frac直接分析性推算内存/带宽收益：

| 模型 | period=4的high_frac | 加权bytes/weight | vs always-bf16(2.0 bytes)降幅 |
|---|---|---|---|
| OLMoE | 43.4% | 0.434×2.0+0.566×0.5=1.151 | **42.5%** |
| LLM-jp | 40.3% | 0.403×2.0+0.597×0.5=1.105 | **44.8%** |

这是一个明确标注"待真实打包内核验证，目前是分析性估算"的、可以写进论文的数字。

### 5.3 Task 3：双轴联合控制器POC —— 两版尝试都是清晰的负结果

查了combine轴现有数据（`receiver_direct_benefit_2026-07-20_full`），发现combine轴的controller优化的是排队/字节saving比例，其metadata明确写"quality cost of the low state is NOT modeled here"——两个轴目前用的是**不可通约的单位**（combine轴没有KL，compute轴没有排队延迟）。因此只能做一个清楚标注为"概念验证、非决定性系统结论"的POC，用已知的combine轴参考常数构造近似的combine风险流，和真实的compute轴KL轨迹配对。

第一版（固定时间顺序贪心）和第二版（每轮内按代价升序贪心，仍保持因果合法）两次尝试，在两个模型、5个预算档位上**全部**显示joint分配器显著劣于独立分帽（CI完全为负，两版几乎一样差）。根本原因：构造的合成combine流每步代价（~0.0054）比真实compute流每步代价（~0.03-0.18）小10-30倍，这个巨大的尺度差异让"给combine单独一半预算"天然就绰绰有余，而joint的逐轮贪心会让combine那个虽然便宜但数量多的流持续"蚕食"共享预算，反而挤占了compute更有价值的名额。这不是调小样本就能修好的噪声问题，是这次POC构造方式本身的结构性产物。

**结论：这次POC不支持双轴联合控制器的假设，且印证了"这个方向最新颖但也最不确定"是对的**——在没有真实的、两轴共享的代价/收益计量单位之前，任何联合仿真都建立在近似常数上，容易产生尺度伪影。不建议继续调参数去凑正结果，应降级为"需要先做真实跨轴代价画像才能重新评估"的未来工作。

### 5.4 当天（Idea B 主线）整体结论

固定period=4的影子验证控制器（第二节结果）仍是唯一的正结果：OLMoE干净GO(降幅50.1%，效率优良oracle比1.66x)，补上了分析性内存收益估算(42.5%字节降幅)；LLM-jp技术上NO-GO但很接近(47.0%，CI上界53.4%已过线)。三个新尝试(自适应周期、真实运行时、双轴联合)都没有改善这个结果，但都提供了诚实、有信息量的边界确认——巩固了固定period=4控制器作为主机制的地位。

产物位置：`experiments/idea_a_mac/analyze_adaptive_shadow_verify_controller.py`（Task1）+ `outputs/adaptive_shadow_verify_controller_2026-07-20.json`；`experiments/idea_a_mac/analyze_dual_axis_joint_controller_poc.py`（Task3）+ `outputs/dual_axis_joint_controller_poc_2026-07-20/`；Task2无独立产物文件（self-test在远程交互式跑的，环境限制记录在本节）。

---

## 六、附录：6 个新 idea 完整评估（第二、五节实验的设计动机来源）

> 本节是 2026-07-20 当天较早时候产出的理论设计文档，先做外部文献 grounding，再提出6个idea并做10段结构化分析，最终推荐 Idea B 为主线——第二节和第五节的实验正是对这里 Idea B 核心假设的直接检验。为避免信息丢失，关键 grounding 结论和最终推荐摘录如下，完整10段分析（含每个idea的机制设计、可复用资产、falsification方案、审稿人攻击点、CCF-B评分）保留在原脚本文档历史中，此处摘录决策性结论。

### 6.1 关键外部文献 grounding

- **DynaExq**（arXiv 2511.15015）：MoE运行时专家级动态精度分配，信号=路由激活频率EMA热度，**明确没有基于KL/准确率的质量验证与回退机制**——这是差异化定位的关键："热度+quality信号融合"是它明确留白的空间。
- **MoPEQ**（ICCV2025 workshop, arXiv 2509.02512）：VLM-MoE 专家级Hessian敏感度+激活频率联合位宽分配，用Hessian trace（权重空间代理）而非真实端到端KL。
- **QuantSpec/EAGLE/Medusa/Draft&Verify** 系列：全部验证"draft token是否会被接受"（token正确性），从未做"验证一个已生成token的精度是否够用"（精度充分性）——这是 Idea B 的关键差异化落脚点，检索未发现直接撞车工作。
- **MoE-Lightning/MoE-Gen/PagedWeight/Diff-MoE**：均属"量化/offload释放显存→更大batch→吞吐提升"的成熟方向，说明这条因果链本身不是新闻，必须靠"融合质量信号"撑住novelty。

### 6.2 六个idea一句话定位

Idea A（Fragility×Hotness联合专家分配，补齐DynaExq缺失的质量闭环）；Idea B（Precision-Sufficiency Shadow Verification，decode阶段影子验证闭环控制器）；Idea C（Phase-Differential量化容忍度，prefill保护/decode放宽）；Idea D（CVaR约束的全局分配+Predictor-free质量债务安全网，融合A与已有debt弱正结果）；Idea E（Token级Gate-Mass集中度选择性保护）；Idea F（Prefill-Derived INT4步数预算回归，第一轮即淘汰——本质仍是"用prefill预测未来"，与round2/4已证伪假设同构）。

### 6.3 两轮筛选结果

第一轮淘汰 Idea F（吸收进B作为可选warm-start初始化）；Idea C/E降级为组件（供B/D复用，不单独参赛）；进入深度比较：**B（closed-loop shadow verification controller）、D（CVaR约束分配+债务安全网，内含A）、C（支撑角色）**。

第二轮深度比较认为 Idea B 在核心novelty（验证"精度充分性"而非"token正确性"，检索未发现直接撞车）、可复用资产比例、最可能的性能收益（真实wall-clock TPOT/throughput）、CCF-B级完整故事完整度上均优于 D 和 C。

### 6.4 最终输出

**首选方向**：Idea B（precision-sufficiency shadow verification闭环控制器），以D的静态分配作为初始化、以已验证的债务机制作为跨请求安全网，C的phase-differential结论作为参数先验来源——三者融合为一篇论文，B是核心机制与系统贡献，D和C是支撑贡献。

**核心假设与最小falsification**：B的核心假设是"decode内短horizon信号持续性"（lag-1~8的Spearman相关，PASS标准ρ的CI下界>0.2）——**此假设已在第二节的H1实验中检验，结果为 lag-1 rho 0.18-0.20，NO-GO（未达0.2门槛），但H2次判据（因果控制器）在OLMoE上GO**，与本节6.4"为什么它比当前Quality Isolation更强"一段预判的"优雅退化"模式完全吻合：即使假设部分失效，真实的周期性BF16 shadow verify仍能兜底质量。

**为什么它比当前Quality Isolation更强**：Quality Isolation的失败根源是"用prefill信号预测跨时间/跨模型的未来风险"，这个预测目标已被round2/4系统性证伪；Idea B完全不做这类预测——它只依赖同一请求内部、短horizon（数个decode step）的realized信号持续性，这是一个远弱、远更可能成立的假设，且即使该假设部分失效，真实的周期性BF16 shadow verify仍能兜底质量（predictor失效的代价是verify频率需要更高、收益变小，而不是质量失控），这是一个优雅退化而非"predictor错了就全盘崩溃"的脆弱结构。

**暂定论文标题**：*"Verify, Don't Predict: A Closed-Loop Precision-Sufficiency Controller for Quality-Bounded MoE Decode"*（中文暂定：《验证而非预测：面向质量受限MoE解码的闭环精度充分性控制器》）。

**已验证的弱正结果如何嵌入**：已验证的predictor-free质量债务结果（4个动作、worst-tenant改善10.6%-12.5%、CI不跨0、零效率代价，详见本文档第四节4.3）直接作为跨请求层的安全网机制嵌入本方向——不再是一个"未达20%门槛因而尴尬"的孤立结果，而是"即使within-request的shadow verify因某些原因失效，跨请求层仍能保证没有单个租户被反复伤害"的第二道防线。

---

## 七、后续状态说明

本文档汇编的实验结果已被吸收进 `docs/04_research_plans/` 目录下的审计报告演进链（见该目录的"版本演进索引"），并直接催生了第四代审计报告（`research_report_moe_next_direction_2026-07-21.md`）提出的 `Quality-Constrained Adaptivity Existence Test`。该必要性检验已实际执行（receiver task-quality GPU 实验），结果显示精细在线策略（controller）相对三种粗粒度策略（uniform_full/uniform_low/calib_static）没有稳定优势（winner probability 最高约46%，未达GO门槛），因此 receiver-aware 在线自适应部分被判定不建议继续投入；Idea B 固定period=4控制器（本文档第二节）仍是当前唯一"设计动机被真实数据直接验证"的正结果，是推荐的论文主线核心机制。
