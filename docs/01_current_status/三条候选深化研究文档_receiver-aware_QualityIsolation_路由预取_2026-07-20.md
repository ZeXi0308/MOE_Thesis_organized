> **【2026-07-20 事后更正】** 本文档记录的部分结论已被同日晚些时候的独立批判性审计（`GPU首轮有效性实验结果_receiver_causal_ExpertPrefetch_2026-07-20.md`、`三条MoE研究方向_批判性研究设计与投稿路线_2026-07-20.md`）推翻或大幅降级：①receiver-aware v3的"adaptive controller全面超过固定基线"结论有因果评估错误（把检测后才选定的策略回溯应用到warm-up期），修正后adaptive相对最强固定基线的优势在统计上不显著（CI跨0）；②MoE路由预取系统原型v1/v2的LRUCache只用`expert_id`做key（未含`layer_id`），产生大量不存在的跨层cache hit，此前报告的13-16%换页收益和2-3%/5.32%端到端收益均无效；且只测了top-1，补上真实full top-k后working set严重饱和（LLM-jp 91.2%的layer-batch需要全部专家），predictor相对frequency baseline的增量几乎为零。详细审计过程见上述两份文档，以及`MoE_Approach_Registry_2026-07-19.md`里已更新的对应条目。本文档保留作为"思维背景"和"方法演进过程"的记录，但其中标注"结论"的部分请以更正后的版本为准。

# 三条候选深化研究文档：receiver-aware、Per-request Quality Isolation、MoE路由预取

> 日期：2026-07-20。本文档整理三条目前处于PASS/PROMISING状态候选的完整脉络——每条包含"思维背景"（为什么这么想、系统性直觉从哪来）和"落地实验"（具体怎么设计、怎么跑、跑出什么结果），供后续写入论文或继续深化时直接引用。三条候选彼此独立但共享同一套方法论纪律：calibration/test分割、paired bootstrap置信区间、真实GPU硬件测量、跨模型(OLMoE+LLM-jp)交叉验证。

---

# 一、receiver-aware：从"单一假设的仿真"到"可部署的自适应控制器"

## 1.1 思维背景

**最初的问题**：MoE跨节点combine时，不同token对应的expert贡献大小不同（tail-rank贡献小），如果receiver当前正在拥塞，把这些低贡献的tail-rank通信降精度（省字节），能不能比"不管拥塞、随机降精度"更好？这是一个很自然的"感知系统状态再决策"的直觉——类似网络里的ECN/拥塞控制思想搬到MoE combine场景。

**v1(旧仿真)的问题**：旧的Mac仿真器把"倾向选择remote pair"这个偏置和"识别热点"这个信号混在一起，导致降级判决时怀疑收益全部来自这个confound，而不是真正的拥塞感知能力。

**关键的方法论转折点**：与其直接接受历史判决，先用"confound隔离"的思路——把候选池限定在"tail-rank AND remote"（排除偏置），单独看"hot(真实负载分数)"是否还能跑赢"random"。这一步（v1修复）已经证明：控制偏置后，负载感知信号依然独立存在。

**v2的深层直觉**：v1只用了一个模型、一个场景、且隐含了一个"温和的先知假设"——调度器在打分时已经知道**同一时刻**全部并发请求的真实负载分布。这是AI Infra里最容易犯的错误：仿真里默认"信息即时可得"，但真实在线系统只能看到**过去**已实现的状态。于是v2把"信息可用性"显式拆成三级(`oracle_same_step`同时刻先知 / `causal_prev_step`只用上一步真实负载 / `calib_static`纯离线画像)，同时把"拥塞"本身也拆成两种性质完全不同的regime：**结构性/持续拥塞**(hotspot，某几个receiver总是热)和**瞬时/随机拥塞**(balanced，热点位置随机切换)。

**v3的直觉跃升**：v2诊断出"两个regime需要两种不同策略，且'现实可行的在线信号'在瞬时regime下反而是负资产"，但v2从未回答"系统在运行时怎么知道自己在哪个regime"——这正是receiver-aware要真正部署所缺的最后一环。v3的关键洞察：不需要ground-truth标签，只需要一个**因果合法、可以从短暂观测窗口里量出来**的统计量，就能自动分类regime——这是经典的"在线变点检测/自适应控制"思路搬到这个具体场景。

## 1.2 落地实验

### 实验一：confound隔离重跑（v1修复版）

**脚本**：`run_receiver_isolation_experiment.py`（项目里早就写好但从未真正执行过）。

**方法**：候选池固定为"tail-rank AND remote"（排除"偏好远程"这个混淆），只比较`hot`(真实负载分数) vs `cold` vs `random`。

**结果**：在OLMoE真实路由数据(`olmoe_signal_comparison_n32`)上，budget_fraction=0.25/0.50/0.75的全部12个(origin_mode×num_jobs×fraction)组合下，`hot`均稳定跑赢`random` 4-12个百分点（如balanced/jobs=8/frac=0.5: hot=20.8% saving vs random=10.8%，random 95% CI完全不包含hot值）。仅budget=100%（无选择空间）时无差异，这是数学必然。

**结论**：直接反驳了当年"降级"的理由——不是仿真做得不好，是当年的判决本身就建立在一个错误的confound怀疑上。

### 实验二：系统化重跑v2（跨模型、跨文档、时序动态、信息陈旧度分层）

**脚本**：`run_receiver_aware_v2_systematic.py` + `capture_routes_v2.py`。

**方法**：
- 语料库换成WikiText-103（`prompts.py`新增`wikitext103_docs`选项），彻底摆脱WikiText-2只有约121篇文档的"sealed数据耗尽"问题。
- 跨模型：OLMoE(E64K8) + LLM-jp(E32K16)，各采集40篇全新文档的真实路由。
- 时序动态：job以staggered offset到达（模拟continuous batching下不同请求在不同层的真实并发状态），而不是单一静态snapshot复制。
- 信息陈旧度三级：`oracle_same_step` / `causal_prev_step` / `calib_static`。
- 24 scenario seeds × 2 placements(contiguous/round_robin) × 2 origin_modes(balanced/hotspot) × 3 budget_fractions，两个模型全跑。
- 性能优化：把逐次重扫全表的byte accounting改成"基准+增量修正"，实验从卡住数小时降到16.5分钟跑完。

**结果**（真实GPU路由数据，`receiver_aware_v2_raw.csv`）：

| origin_mode | 信息可用性 | hot−random (OLMoE) | hot−random (LLM-jp) |
|---|---|---|---|
| hotspot(结构性) | oracle/causal/calib_static | +5.6%~+11.1%（三者几乎重合） | 同上 |
| balanced(瞬时) | oracle_same_step | +2.3%~+6.1% | +1.0%~ |
| balanced(瞬时) | causal_prev_step(现实可行) | **-1.1%~-2.8%**（反转为负） | 同方向 |
| balanced(瞬时) | calib_static | -1.7%~-7.6%（更负） | 同方向 |

staleness cost（oracle−causal）在balanced下稳定在+4.2%~+8.0%，跨2模型2placement高度一致，24 seeds里几乎全部落在random的95% CI之外（不是噪音）。

**结论**：hotspot下静态离线画像已经拿到几乎全部收益，不需要在线信号；balanced下"同时刻先知"是不现实的假设，换成真实可行的信号后反而比随机更差。判定从"降级"改为`CONDITIONAL_STRUCTURAL_ONLY`。

### 实验三：v3因果合法regime检测器 + 自适应控制器

**脚本**：`run_receiver_aware_v3_adaptive.py`（复用v2的场景生成/打分函数）。

**方法**：
1. **检测器离线校准**（仅用calibration场景，不接触test set）：分别测量"hotspot"和"balanced"两种calibration场景下，receiver负载的Herfindahl集中度指数（load在各receiver间的集中程度：1/ep_size=完全均匀，1=全部集中在一个receiver），取两者中点作为判决门槛。（注：第一版尝试用"lag-1自相关"做检测统计量，两个regime的calibration自相关几乎重合(-0.43 vs -0.37)，检测完全失败；换成"负载集中度Herfindahl指数"后两个regime清晰分离(0.2965 vs 0.1417)，这本身是个有价值的方法论教训——检测统计量的选择要直接对应生成过程里被操纵的那个维度，不能想当然套用时间序列常用指标。）
2. **在线检测**（因果合法）：每个test场景只用前30%的global step（短暖启动窗口），计算同一统计量，与门槛比较。
3. **策略切换**：判定"结构性"→用`calib_static`；判定"瞬时"→退化为`random`。

**结果**（24 scenario seeds，50%hotspot+50%balanced混合测试集，控制器不知道当前regime）：

| model | 检测准确率(balanced/hotspot) | adaptive | always_calib_static | always_causal | always_random |
|---|---|---|---|---|---|
| olmoe | 100.0% / 87.5% | **0.1706** | 0.1521 | 0.1643 | 0.1232 |
| llmjp | 100.0% / 91.7% | **0.1729** | 0.1399 | 0.1594 | 0.1225 |

adaptive在两个模型上都超过全部三个固定策略基线（包括原来认为"整体最强"的causal_prev_step），且是在**不知道当前regime**的混合测试集上取得的。

**结论**：从"诊断出两个regime需要不同策略"升级为"部署时可自动执行的自适应控制器"，不依赖ground-truth标签，跨两个模型一致。

## 1.3 局限与尚未做的深化

- 仍是纯带宽分析回放，没有真实RDMA队列/incast测量。
- 检测准确率非100%（hotspot方向8-13%误判），检测窗口大小(30%)未做灵敏度扫描。
- 只支持二元regime分类和"检测一次、全程固定"，不支持regime中途漂移（真实系统里拥塞模式可能随时间切换）。
- **下一步设计**：①扫描detect_frac∈{0.1,0.2,0.3,0.5}找真正最优窗口；②滑动窗口周期性重检测，构造"前50%hotspot后50%balanced"的漂移场景测试跟随能力；③加入滞后/置信度机制避免震荡切换(flapping)。

---

# 二、Per-request Quality Isolation：从"依附于死机制"到"独立可行的信用分配"

## 2.1 思维背景

**最初的定位问题**：这条候选在选题地图里被设计成"Graceful-EP的第二贡献"——当系统整体做质量降级时，平均质量可能很好，但少数请求可能持续多层受损，需要类似网络公平队列的"降级credit"机制来保护worst-case。但它的宿主机制Graceful-EP（criticality-aware graceful degradation/QTree-EP）已经被判死（严格复核显示：温和降级多耗0.0098 KL只多省1.68个百分点字节，two-lane字节数反而多耗15.85%-45.47%）——所以Quality Isolation原本"没有独立存在的基础"。

**关键的重新设计直觉**：Quality Isolation的核心逻辑（"给配额，优先保护最脆弱的请求"）本身不依赖Graceful-EP具体是什么机制，只需要**任何一个"对不同请求做不均匀质量分配"的宿主策略**。项目里早就有一个活的、已验证的宿主——`fixed_tail`均匀降级基线（PLTB/layer_budget实验里的对照组：把某些层的tail-rank通信统一降到4bit/8bit精度）。把Quality Isolation重新挂载在这个基线上，逻辑完全成立，且不需要发明新机制。

**第二个关键直觉（信号从哪来）**：给"哪个请求该优先保护"打分，最朴素的想法需要一个"预测该请求会被降级得多惨"的信号。这个信号能不能从**另一个已经算过的、独立的降级机制**里白捡？如果文档级的"降级脆弱性"是文档的内在属性（和具体用什么降级机制无关，更多取决于文档本身的内容复杂度/长度/困惑度），那么任何一个已经跑过的机制的KL都可以当作**另一个**机制的免费诊断信号——这是一种"用已有计算的副产品做决策"的系统思维，类似缓存里"用一个信号服务多个决策"的复用思路。

## 2.2 落地实验

**脚本**：`run_per_request_quality_isolation_p0.py`（纯本地Mac分析，复用已有`layer_budget`实验的`sample_metrics.csv`，零新GPU时间）。

### Part A：oracle上限

给定固定的"全精度配额"M（占文档数10%-50%），比较"把配额给KL最差的M个文档"(oracle,用测试标签) vs "随机给M个"，在匹配总字节预算下测P95 KL降幅：

| model | quota=10% | quota=25% | quota=50% |
|---|---|---|---|
| olmoe (fixed_tail4) | oracle多降35.4pp | 多降32.6pp | 多降30.6pp |
| llmjp (fixed_tail8) | oracle多降20.7pp | 多降25.1pp | 多降38.6pp |

### Part B：信号是否可迁移（现实中能否拿到这个信号）

检验文档级"KL风险"能否用**另一个已经算过的机制**的KL来预测（Spearman相关）：

- OLMoE：与kl_profile_3_5/p95_profile_3_5/gate_mass_profile_3_5等的相关系数**0.75-0.85**（全部p<0.0001）。
- LLM-jp：相关系数**0.98左右**（全部p<0.0001）。

**结论**：文档的"难度"是跨降级机制稳定的内在属性，不是机制特异的噪声。

### Part C：现实可实现版本（非oracle）

用Part B里相关性最强的机制的KL作为**因果合法**的优先级信号（不用测试标签本身），重新分配同样的配额：

- LLM-jp：全部5个配额档位与oracle**完全一致**（proxy_realized_p95_reduction_pct精确等于oracle_p95_reduction_pct）。
- OLMoE：低配额档位几乎一致，高配额时略低于oracle但仍远超随机。

**结论**：Per-request Quality Isolation不需要新机制、不需要训练，只需要在已有的多个降级策略之间共享"哪个文档更脆弱"的诊断信号，就能把P95最坏情况的质量损失降低20-54个百分点（相对随机分配的均匀降级），且信号现实可得。

### 附加实验：与路由预取信号的协同性检验（回答"是否有重复工作"）

**脚本**：`run_quality_routing_synergy_check.py`。用与Routing-Predictability P0**完全相同**的45篇文档（wikitext2_docs:test offset16 n=45），额外补跑一次`fixed_tail4`的KL，检验"文档级路由可预测性"（专家预取要用的信号）能否兼职当Quality Isolation的信号：

```
Spearman(top1_hit_rate, mean_token_kl)        = -0.177  (p=0.245，不显著)
Spearman(mean_routing_entropy, mean_token_kl) = -0.021  (p=0.891，不显著)
```

**结论**：**两者没有重叠**，是正交维度——"容易预测专家路由"和"容易被降级破坏"不是同一件事，两个系统必须分别维护各自的诊断信号，不能偷懒合并。

## 2.3 局限与尚未做的深化

- 当前信号本身的计算成本还是"再跑一次完整降级机制的forward"，不是真正零成本。**深化方向A**：检验router的gate_mass分布统计量（唯一一次真实forward里router已经算过，真正零成本）能否达到接近的相关性。
- 当前验证是"静态一次性配额分配"，不是真实的流式多轮信用累积。**深化方向B**：用bootstrap构造合成多轮请求流，实现"连续被降级k次后强制给全精度"的动态credit规则，对比静态配额规则在长期P95上的差异。
- 只在OLMoE(fixed_tail4)和LLM-jp(fixed_tail8)各一个base_strategy上验证过。

---

# 三、MoE路由跨层可预测性驱动的专家预取：从"离线统计信号"到"暴露真实kernel瓶颈"

## 3.1 思维背景

**触发问题**：一次系统性调研(4个并行子agent + 微信文章检索)在"MoE推理系统还有哪些深层优化空间"这个问题上，发现"投机解码/专家预取利用路由可预测性"是一个明确、具体、没被充分覆盖的空白——已有工作(MoESD/SP-MoE)证明了投机解码对MoE有效，但都没有显式利用路由本身的时序可预测性设计draft机制。而这个方向的验证成本几乎为零：项目已经积累了大量真实OLMoE/LLM-jp路由trace，可以纯离线验证。

**核心直觉**：如果给定token在第L层选择的专家，能比"全局频率baseline"更好地预测它在第L+1层的专家选择，那这个信号就可以用来做HBM专家预取（提前把可能用到的专家权重从CPU/慢速内存搬到GPU HBM）或投机解码的draft设计。这本质上是把"路由的确定性结构"当成一种可开采的因果信号，而不是把路由当成纯随机过程。

**验证的层层递进逻辑**（避免像TokenRace-EP/receiver-aware v1一样掉进"看起来有信号但站不住"的陷阱）：
1. 先证明信号本身存在（P0：能不能预测下一层？）
2. 再证明信号能维持多远（P0-B：预测8层之后还有效吗？如果只能提前1层，任何预取机制都来不及做有用的动作）
3. 再把"存在信号"转化为"系统能拿到多少实际收益"（P0-C：给定真实的预取带宽/候选数限制，命中率能到多少？）
4. 最后必须落到**真实硬件测量的系统原型**，而不是停留在离线统计——这正是TokenRace-EP教训的正面应用：不能假设"仿真里算出来的收益"能直接照搬到真实系统，必须用真实GPU的H2D带宽、真实compute时间去验证。

**系统原型阶段的关键教训**：做出原型后发现换页收益本身是真实的（13-16%），但被一个粗糙的compute代理（逐专家Python循环，10ms/层）稀释成了几乎看不见的端到端收益（2-3%）。换成更真实的融合compute kernel后，意外发现原来的"固定预取预算=8"反而让系统变差——这暴露了一个此前完全没考虑到的新维度：**预取预算必须和compute-communication overlap窗口的实际大小联动**，不能只按"统计学命中率最优"来定。这是这条候选系统性思考走得最深的地方：从"证明信号存在"，到"证明信号能转化为收益"，到"发现收益依赖一个此前未建模的系统约束"，再到"给出具体的、可执行的设计规则"。

## 3.2 落地实验

### P0：信号是否存在

**脚本**：`run_routing_predictability_p0.py`。复用真实sealed数据（OLMoE 45篇+LLM-jp 60篇正式test文档，calibration/test分割，与项目一贯方法论一致）。

**方法**：`top1_transition`（用当前层top-1专家预测下一层top-1专家，用calibration学到的转移表）vs `freq_baseline`（下一层全局最高频专家，零per-token信号）；负对照`topk_transition`（用完整top-k集合做状态，测试是否会因状态空间太稀疏而过拟合）。

**结果**：`top1_transition`比`freq_baseline`高**+19.02pp(OLMoE) / +13.85pp(LLM-jp)**，两模型早期层/晚期层全部通过5pp门槛，CI远离0，Holm校正后p<0.0001。`topk_transition`在部分层未通过门槛，证明实验设计对虚假信号敏感（不是随便什么predictor都能"碰巧"过线）。

### P0-B：能维持多远

**脚本**：`run_routing_predictability_p0b_lookahead.py`。扫描lookahead=1,2,3,4,6,8层。

**结果**：lookahead=8层（半个模型深度）优势仍有**+14.11pp(OLMoE)/+8.14pp(LLM-jp)**，衰减是渐进的而非断崖式，说明预取机制有相当大的调度灵活性。

### P0-C：能转化成多大实际收益

**脚本**：`run_routing_predictability_p0c_prefetch_budget.py`。把问题换成更贴近真实系统的"预算约束"问题：只预取N个候选专家时，命中率比同预算的频率baseline高多少。

**结果**：预算=8个专家（OLMoE专家总数的12.5%）时，命中率**74.3% vs 频率baseline 40.1%**，领先34.1个百分点；LLM-jp在6个候选（18.75%）时领先幅度最大，24.5pp。

### 系统原型v1：真实GPU硬件测量，暴露compute代理瓶颈

**脚本**：`run_expert_prefetch_system_prototype.py`。

**方法**：真实测量H2D拷贝延迟（pinned memory→GPU，真实expert权重大小）和MoE层compute延迟（逐专家Python循环，用真实加载的expert模块）；模拟GPU常驻专家LRU缓存(容量8/6)+预测性预取（用calibration学到的top1转移表，在当前层compute窗口内异步预取下一层候选），对比reactive(纯需求驱动)。

**结果**：

| model | 换页延迟节省(排除compute) | 端到端总延迟节省 |
|---|---|---|
| olmoe | **16.0%**(CI 15.1-16.9%) | 3.08%(CI 2.92-3.22%) |
| llmjp | **13.3%**(CI 12.5-14.0%) | 1.89%(CI 1.79-1.97%) |

**诊断**：换页收益本身真实、显著、跨模型一致，但被compute代理(OLMoE 10.09ms/层)稀释——H2D单次成本(248.6us)只占compute窗口的2.5%，换页收益占比自然被压得很小。

### 系统原型v2：融合kernel，发现"过度预取"新问题

**脚本**：`run_expert_prefetch_system_prototype_v2_fused.py`。

**方法**：把compute代理从"逐专家Python循环(64次kernel launch)"换成"融合`torch.bmm`(3次kernel launch，堆叠所有expert权重批量计算)"——这更接近生产级grouped-GEMM MoE kernel(DeepEP、Triton fused MoE)的实现思路。

**结果（第一步，固定预算=8不变）**：
- compute时间：OLMoE 10088us→543us(18.6x)，LLM-jp 5115us→99us(51.6x)。
- 端到端延迟节省**从正转负**：OLMoE +3.08%→**-32.97%**，LLM-jp +1.89%→**-26.45%**——因为预取8个候选所需H2D时间(8×249us=1992us)远超新的compute窗口(543us)，大部分"预取"根本来不及被掩盖，暴露成额外延迟。

**修复（overlap-capped动态预算）**：把预取预算从固定值8收紧为"compute窗口能免费掩盖多少个"(`floor(compute_time / h2d_time)`，OLMoE=2、LLM-jp=1)：
- 端到端延迟节省**重新转正**：OLMoE **+5.32%**(CI 5.00-5.65%)，LLM-jp **+3.07%**(CI 2.87-3.26%)，跨两模型一致。

**结论**：给出可直接落地的系统设计规则——`prefetch_budget = min(统计学最优候选数, compute_time / H2D单专家耗时)`。预取预算不能只按P0-C的"命中率最优"来定，必须动态匹配当前kernel实现能提供的免费overlap窗口大小；用了真实融合kernel之后，正确的预取预算比P0-C建议的小得多，但只要按窗口收紧，收益依然真实显著。

## 3.3 局限与尚未做的深化

- 只验证了top-1专家（真实combine涉及完整top-k）。
- 还没设计具体系统机制的完整实现（真正的HBM预取 vs 投机解码draft，目前是"如果这样做会怎样"的模拟，不是端到端跑通的serving系统）。
- 数据都来自WikiText同一语料家族，需要分布外语料验证信号的泛化性。
- 融合compute代理本身仍是"全experts全tokens"的dense计算（浪费FLOPs），不是真正的稀疏grouped-GEMM，真实kernel的compute时间可能比543us更短，overlap-capped budget可能进一步收紧。

---

# 四、三条候选的共同方法论脉络

三条候选走的是同一条从"发现信号"到"验证信号真实可用"再到"暴露被信号掩盖的系统约束"的路径，且互相印证了同一个核心教训：**仿真/离线代理与真实硬件/真实部署约束之间的差距，可能不只是幅度上的衰减，而是符号上的反转**——receiver-aware v2发现"同时刻先知"假设反转为负；专家预取v2发现"compute代理变精确后固定预算反转为负"。这提示任何后续候选，只要涉及"用某个信号做在线决策"，都应该假设"这个信号在更真实的执行环境下可能表现完全相反"，并把"用真实约束重新验证一次"作为标准流程的最后一步，而不是可选项。
