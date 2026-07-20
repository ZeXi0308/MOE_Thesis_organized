# MoE推理系统深层优化方向调研报告（2026-07-20）

## 执行摘要

本报告调研了四个系统侧深层优化方向（专家权重内存管理、PD分离×CUDA Graph协同、算法层动态计算量、多租户SLO/能耗调度），并用微信公众号搜索做了中文社区交叉验证。核心结论：**四个方向都存在真实的技术空白，但空白的"性质"差异很大**——有的空白因为需要硬件资源（多卡/CXL/功耗测量）而对本项目不可行，有的空白虽然硬件可行但已被2025-2026涌现的大量工作快速填满，只有极少数交叉地带（CUDA Graph与MoE动态调度的协同、MoE路由可预测性用于draft机制）同时满足"单卡可验证"和"仍有明确空白"两个条件。中文社区交叉验证确认：专家权重offloading、AFD(Attn-FFN分离)是当前最热的方向，印证了英文调研的判断，且AFD是2026-07-15刚出现的最新演进，说明PD分离这条线还在快速迭代。

## 一、专家权重内存层级管理（子任务A）

**已高度覆盖**：fMoE(EuroSys'26)、ExpertFlow(DAC'26)、OD-MoE(2025-12)、PiKV(KV+专家权重联合)已经把hot/cold分层缓存、预取预测(准确率95%-99.94%)、KV-Expert联合调度基本占满。异构GPU放置(HeterMoE)、CXL-NDP(2025-12刚出现)存在空白但需要企业级CXL硬件或多卡异构集群，本项目完全不可行。

**中文验证**：colibri(2026-07-16)、vLLM Weight Offloading v2、GTC2026的KVBM多层存储卸载方案，确认这是当前中文社区最热的MoE优化话题之一，进一步印证"高度饱和"的判断。

**唯一现实可行的子方向**：纯理论/模拟方向——构建MoE内存层级模拟器验证不同卸载策略，或在Mac/单卡上复现OD-MoE式的"影子模型预测"思想做轻量化改进。但创新度有限，容易被判定为对已有工作的小修小补。

**结论：不建议投入**，硬件门槛和拥挤程度都不利。

## 二、PD分离 × CUDA Graph协同设计（子任务B）——四个方向里潜力最高

**PD分离本身已被覆盖**：DuoServe-MoE(arXiv:2509.07379)、MegaScale-Infer(SIGCOMM'25)已做MoE专属PD分离，**且中文社区确认2026-07-15/16刚出现更新一代方案AFD(Attn-FFN分离)**，让吞吐再提升1.45倍——这条线还在快速演进，如果现在切入"PD/AFD分离"本身，很可能几个月内又被下一代方案超越。

**但CUDA Graph与MoE动态路由的协同优化是真正的空白**：vLLM文档明确用FULL_AND_PIECEWISE这种"打补丁"式方案应对MoE的动态shape问题，说明官方也没有令人满意的通用解法；FlashMoE(NeurIPS'25)、Mirage MPK(2025-12)提出persistent kernel作为CUDA Graph的替代范式，但集成度不如Graph方案成熟。子agent识别出的具体空白：

1. **按专家激活模式预捕获多套图**（离线Profile高频shape组合，运行时选择最匹配的预捕获图）——这个可以直接复用本项目已有的真实route trace数据，不需要新硬件设计，只需要在已租用的GPU上验证。
2. **Router阶段用CUDA Graph稳定路径，Expert阶段用轻量persistent kernel**的混合方案——填补FlashMoE(全persistent、丧失灵活性)和vLLM CUDA Graph(对MoE动态性支持不足)之间的空白。

**关键优势**：这个方向的核心矛盾（graph capture固定shape vs MoE动态调度）**已经在本项目TokenRace-EP的GPU P1实验里被实测证实**（eager 59-67us vs graph replay 18-30us），意味着本项目已经有了验证这个矛盾存在的第一手真实数据，比大多数候选更容易起步——不需要从零开始建立实验基础设施。

**结论：这是四个方向里最值得优先考虑的**，单卡RTX 5090可验证核心矛盾，且本项目已有相关实测数据可以直接复用/扩展。

## 三、算法层动态计算量与投机解码结合（子任务C）

**Dynamic top-k已被BEAM(2026-05)、ZEDA(清华,2026-05)、SMIDT(AAAI2026)高度覆盖**，收益量级2-2.5倍已被吃掉，继续做这个方向创新度低。

**投机解码+MoE路由结合仍有明确空白**：MoESD(NeurIPS'25 Spotlight)、SP-MoE(2025-10)证明了投机解码对MoE有效(甚至比dense模型受益更多)，但**都没有显式利用MoE路由本身的token级可预测性去设计draft机制**——这是一个具体、清晰、别人还没做的空白。子agent识别的机制：用前几层的真实routing pattern预测后续层的routing，辅助draft模型设计，或反过来用投机解码的draft结果提前预热/预取专家。

**这个方向与本项目现有资产高度契合**：本项目已经积累了OLMoE和LLM-jp的大量真实路由trace数据，做"路由可预测性"分析完全可以在Mac本地完成（纯离线统计分析，不需要训练新模型），是四个方向中**验证成本最低**的。

**MoE层级早停(Dr.LLM框架迁移)** 成熟度最低，需要额外设计MoE特定的cost model，工作量较大，创新度中等，作为备选。

**结论：值得投入，且和本项目现有数据资产的匹配度最高**。

## 四、多租户SLO调度与能耗感知（子任务D）

**能耗/功耗方向（PALS、GreenLLM、Festina）需要真实功耗测量硬件+多卡长期占用，完全不可行**。多租户SLO方向（FaaSMoE、Laser、SOLA）虽然理论上可以用仿真验证，但子agent明确指出"MoE专家竞争对SLO的影响"这个细分空白目前只能停留在仿真+理论层面，缺乏真实系统验证会削弱说服力，且这类工作通常需要真实多租户负载trace做支撑，单卡短租很难产出有竞争力的实证数据。

**结论：不建议投入**，硬件门槛最高，且即使做仿真也难以在有限时间内产出有分量的结果。

## 五、综合排序与建议

| 方向 | 空白确定性 | 硬件可行性 | 与本项目资产契合度 | 综合建议 |
|---|---|---|---|---|
| CUDA Graph × MoE动态调度协同 | 高（vLLM官方文档承认是打补丁） | 高（单卡RTX 5090可验证核心矛盾，已有TokenRace-EP实测数据） | 高（直接复用GPU P1实验基础设施） | **优先** |
| 投机解码 × MoE路由可预测性 | 高（MoESD/SP-MoE均未做） | 极高（可用现有route trace纯离线分析） | 高（现成大量真实路由数据） | **优先** |
| 专家权重内存管理/offloading | 低（已被5+工作占满） | 低（多需要CXL/多卡异构） | 低 | 不建议 |
| 多租户SLO/能耗调度 | 中 | 极低（需要真实功耗硬件+多卡长期占用） | 低 | 不建议 |

**具体建议的下一步（如果继续推进）**：
1. 先做**MoE路由可预测性分析**（零成本，用现有OLMoE/LLM-jp route trace，纯离线统计），验证"用前k层真实routing预测后续层"这个假设是否有足够高的准确率支撑一个draft/预取机制——这一步几乎不消耗新资源，可以立刻在Mac本地开始。
2. 如果第1步显示路由确实有可预测结构，再设计一个轻量级的"路由感知投机解码/专家预热"机制原型，在已有的GPU实例上做真实模型验证。
3. 平行/备选地，可以用已租用的RTX 5090复现"CUDA Graph固定shape vs MoE动态子集"这个矛盾的更细粒度版本（比如具体测量"分桶shape+多graph切换"这个具体机制能挽回多少graph replay的性能优势），这一步可以直接复用GPU P0/P1的实验代码和方法论。

两个优先方向都规避了本项目已经踩过的坑（uniform FP8天花板、简单静态信号逼近oracle上限、combine/layer-advance同步屏障的开销陷阱），因为它们的因果层完全不同——一个是"graph capture策略"，一个是"路由的时序可预测性用于draft"，都不是"给定路由后重新分配资源"这类已经被反复验证走到头的范式。

## 参考文献汇总

### 专家权重内存管理
1. [fMoE: Fine-Grained Expert Offloading (EuroSys'26)](https://ar5iv.labs.arxiv.org/html/2502.05370)
2. [ExpertFlow (DAC'26)](https://marsggbo.github.io/blog/2026/expertflow/)
3. [OD-MoE: On-Demand Expert Loading for Cacheless Edge Inference](https://arxiv.org/html/2512.03927v1)
4. [PiKV: KV Cache Management for MoE](https://arxiv.org/html/2508.06526v2)
5. [HeterMoE](https://arxiv.org/html/2504.03871v1)
6. [Context-Aware MoE on CXL-Enabled GPU-NDP Systems](https://arxiv.org/abs/2512.04476)

### PD分离与CUDA Graph
7. [DuoServe-MoE](https://arxiv.org/abs/2509.07379)
8. [MegaScale-Infer (SIGCOMM'25)](https://arxiv.org/abs/2504.02263)
9. [From Tokens to Layers (MLSys'26)](https://arxiv.org/abs/2510.08055)
10. [FlashMoE (NeurIPS'25)](https://arxiv.org/abs/2506.04667)
11. [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219)
12. [vLLM CUDA Graphs Design Doc](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
13. [AEP/AMoE](https://arxiv.org/abs/2505.08944)

### 算法层动态计算量
14. [SMIDT (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39403)
15. [MoESD (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2505.19645)
16. [BEAM](https://arxiv.org/abs/2605.14438)
17. [ZEDA (清华)](https://github.com/TsinghuaC3I/ZEDA)
18. [SP-MoE](https://arxiv.org/abs/2510.10302)
19. [Dr.LLM (ICLR 2026)](https://arxiv.org/abs/2510.12773)
20. [Task-Aware Expert Merging](https://arxiv.org/abs/2509.19781)

### 多租户/能耗调度
21. [PALS](https://arxiv.org/abs/2605.21427)
22. [GreenLLM](https://arxiv.org/abs/2508.16449)
23. [FaaSMoE](https://arxiv.org/abs/2604.26881)
24. [EcoServe](https://arxiv.org/abs/2502.05043)
25. [SOLA (MLSys 2025)](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html)

### 中文社区交叉验证（微信公众号，2026年）
26. 继PD分离之后Attn-FFN分离: AFD让MoE推理吞吐提升1.45倍（AI圈的9527，2026-07-15）
27. 长上下文饿死MoE?Attn-FFN分离部署让GB200吞吐提升1.45倍（HyperAI，2026-07-16）
28. GitHub精读:JustVugg/colibri（AI.Florx，2026-07-16）
