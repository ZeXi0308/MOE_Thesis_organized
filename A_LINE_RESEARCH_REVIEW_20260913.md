# A 线研究评审 — 2026-09-13

结论：固定 KV 压力下主动恢复/轮转，是有真实完整请求证据、值得继续的研究问题；“三个已闭合关系足以支持在线预测模型”目前不成立。保留轮转测得的停顿—完成时间权衡，撤回错误恒等式与纯 decode 步数的物理浪费解释。状态仍为 **NATIVE_SERVING / MEASUREMENT_ONLY，在线模型 OPEN / UNRUN**。

## 范围与证据

本次只读审查并新增本文件，没有启动 GPU、SSH、修改 runner、冻结预测、原始数据或科学权威入口。检查时 HEAD 为 `7059fc98e0d98a127ff4edda86d44f7f634d26f4`，工作区有 146 条 dirty/untracked 项，既有变更保留。读取 `docs/current/README.md`、`docs/ideas/README.md` 后，以本方向最新 9 月报告判断已测状态。

路径简写：`O = refine-logs/expert_saturation/outputs/admission_capacity/`，`E = refine-logs/expert_saturation/experiments/admission_capacity/`。

- H：`O/20260913_rotation_holdout_r01/REPORT.md`，20 格、640 次请求执行、64 篇新文档。独立统计单位不是 640 个请求。
- P：`O/20260913_pressure_sweep_r01/`，ABSENCE_DECOMPOSITION、PREDICTIONS、DECISIONS、INTERIM、STEP_FLOOR_FINDING；d2 完成，d0/d4/d6 未取得有效扫描结果。
- V：`O/20260913_rotation_victim_order_r01/REPORT.md`；F：`O/20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md`。两项已执行，不再列为未做的下一步。
- 本次直接读取 H 的 `execution03/gpu_results/cohort0-block0-{native,rotate}/raw.json`，定向复核计数与返回 token；没有重跑全部实验审计。d2 数字来自报告，未另称本次完成原始轨迹复核。

## 研究判断

| 层次 | 判断 | 依据与边界 |
|---|---|---|
| 问题真实性 | 成立于当前运行域 | native 最大 ITL 4.447–4.480 s，轮转 1.004–1.014 s；单卡 OLMoE/vLLM 0.26、固定 32 请求及长度/到达配置 |
| 动作有效性 | 已有证据 | 真实释放 KV、恢复请求并独立推进；并非离线轨迹掩码 |
| 完整请求收益 | 有明确权衡 | 对 native 吞吐 −0.619% 至 +0.475%，平均完成慢 1.109%–2.284%；不能声称吞吐非劣 |
| 强简单基线 | 已取得有价值对照 | 对 headroom 吞吐 +1.943%–2.781%，平均完成 −4.257% 至 −5.067%，最大 ITL 也更低；尚非统计确认 |
| 新模型 | 未成立 | 错误会计等式、策略依赖的分母、事后时长回填，不能推出动作条件预测 |
| 方法/论文 GO | 尚未达到 | Oracle residual、最近邻动作比较、异构终止长度及代表性泛化尚缺；也没有证明机制是 MoE 特有 |

平均完成变慢不意味着没有用途：改善的是长停顿，代价由其他请求承担。若目标是平均完成最小化，当前轮转没有胜过 native；若目标是约束长停顿同时保住服务量，则有继续研究的理由。需要事先明确目标和可接受代价，不能事后切换胜出指标。

## 必须修正的六点

### 1. 总缺席不是全局额外迭代的恒等式

P/ABSENCE_DECOMPOSITION.md:14 的关系数值上即不成立：native `318/1024=0.3105`，而 `(1241-1024)/1024=0.2119`；轮转分别为 `0.2607` 与 `0.0918`。request-step 缺席与全局 step 不能直接相减或相除后等同。

正确的槽位会计须先统一时间域。对固定请求集合 N、T 个全局迭代，以每次调度前状态定义：

`N*T = 已调度请求槽位 + 已到达未完成但未调度槽位 + 未到达或已完成槽位`。

已调度槽位还要区分有用的新 token 进展和 prefill/recompute 等无新输出进展。将返回 token 与迭代正确对齐、确认每请求每迭代至多一个新 token 后，才能进一步写输出槽位守恒。总缺席本身无法消去完成后空槽位，也无法决定全局迭代数。平均完成时间则对应未完成请求数的时间积分，不是总缺席或最后完成时间。

### 2. 纯 decode 步数不是可以直接除以输出长度的物理浪费

E/analyze_pressure_sweep.py:67–82 排除全部含重算/prefill 的调用，也排除最后一个调用。混合调用仍可产出新 token，且排除比例随策略变化。

本次原始数据复核：

| cohort0/block0 | native | rotate |
|---|---:|---:|
| 全部 scheduler 调用 | 1348 | 1257 |
| 纯 decode 调用，包含最后一步 | 1242 | 1119 |
| 同时含重算与 decode 的调用 | 8 | 40 |
| 上述混合调用实际返回新 token | 233 | 1185 |
| 全部返回 token | 32768 | 32768 |

`engine_steps.scheduler_step_start/end` 一对一映射上述混合调用，返回 token 数与计数一致。纯 decode 少 123 步，其中混合调用类别增加 32 步；总调用实际少 91 步。后者是实际轨迹差异，前者不能全算成消除的浪费。

非 speculative、每请求每迭代至多输出一 token 时，`max(L_i)` 可以给全部进展迭代一个弱下界，不能直接约束已经剔除部分有效输出的 pure 子集。所有请求在首个完成前入场也不能修复这个问题。即使使用有效下界，距下界的差也不一定能在固定 KV 约束下被调度回收。原“回收 56% 步浪费”应撤回，纯步数可保留为阶段计数。

### 3. onset 同值只证明首次干预之前的共同轨迹

P/ABSENCE_DECOMPOSITION.md:53–74 将其扩为“策略无关”过宽。如果两个策略在首次压力发生之前动作相同，onset 相同是合理的；改变准入、prefill 顺序、预留或到达过程就可能改变 onset。K=1.013 是 d2 校准因子，须称校准模型，不能同时称无自由参数。d4/d6 预测通过只能支持这个压力族的迁移精度。

### 4. max-ITL 闭合是解释证据，尚非预测证据

真实最长缺席段乘该段实际平均步时，闭合到 1%，支持当前轨迹中停顿由缺席主导。它没有独立预测未来段长或动作之后的步时。恢复已调度但尚未返回 token、调度边界到交付边界的时间，也须保留在残差中。P/ABSENCE_DECOMPOSITION.md:45 的“256 块连续可用”不是一般 paged-KV 可行性条件；应使用当前请求实际所需块及可释放块，不能假定物理连续或只能等完成释放。

### 5. 剩余压力扫描需先写执行前更正，不覆盖冻结预测

P/PREDICTIONS.md:73 预测 d6 为 2.48，:94 又写 2.87；这是具体冲突。保留旧稿，在新结果到来前注明采用哪一项及原因。保留原 pure 步数预测作探索性数值检验，但不能据此验证已否定的物理解释。

P/DECISIONS.md:83–90 将样本最大重复差作为“可分辨”门槛：最多只能说超出已观察差值，不能保证真实效应可识别、非劣或未来噪声上界；跨运行顺序差还混入顺序效应。P4 应统一吞吐/时间单位，并报告效应、不确定性及预先确定的实际代价容忍度。不要由旧 max 差乘三构造新门槛。d0 无干预是负控用途，与内存配置不可执行须分别记载。

### 6. 在线贡献的最弱链路还未测试

d4/d6 两个固定策略的完整运行，只能测压力外推与权衡边界，不能证明“状态+候选动作”能预测或选优。未来实际 EOS、到达与动作后的批次/步时不可直接从现有轨迹复用。当前近完成保护使用配置长度，在固定输出实验合法，但不是自然终止时间预测。

先明确短期、可观测的目标：例如当前可用块和合法 victim 能否让最久等待请求恢复、预计恢复前要等多少步，以及会给 victim 增加多少停顿。要证明选优，后续需从同一动作前状态分别执行候选动作，并在未见 episode 上评价排序与完整成本。不要立即拟合完整未来缺席序列。持续 most_output 已是需要考虑的强简单策略；first-swap 单次改动已测且不复现其全部收益，不能当成未做的新方案。

## 新颖性边界

[FastServe 官方页面](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang) 已包含 token 粒度抢占与反馈队列；[VTC 官方页面](https://www.usenix.org/conference/osdi24/presentation/sheng) 已研究连续 batching 的服务量公平。因此“抢占+等待优先”本身不足以主张新颖。本次仅作官方摘要级碰撞检查，没有完成算法/代码等价性比较，不能据此判死 A。潜在 residual 应落在固定 KV 下恢复可行性、重算与有效输出交织的完整成本，以及约束长停顿的可执行保证；这些仍待验证。

## 唯一下一实验与停止条件

先以小型 addendum 修正上述会计、预测冲突与判据，再执行原 d0/d4/d6 的 native/rotate × 两顺序区组扫描。保留已有运行和原预测，另报全调用、有效输出、混合重算、完整完成时间及最大 ITL。资源隔离未满足则 BLOCKED；不为得到可跑 d0 而临时更改请求数或内存利用率。

纯分析修正不占 GPU，预计半小时到一小时；原 12 格执行预算沿用已有约 15 分钟估计，但取决于加载和资源可用性，本次没有实测保证。此实验是现有压力假说的低成本边界测试，不是在线 selector 验证。

| 新结果 | 允许结论 | 不允许结论 |
|---|---|---|
| onset 预测通过 | d2 校准模型在所测固定负载压力点有效 | onset 普遍策略无关 |
| 两策略停顿优势保持 | 固定轮转的所测压力域权衡稳定 | 在线动作预测正确、吞吐非劣 |
| pure 预测通过、完整成本不改善 | 阶段计数可预测，代理与目标分离 | 回收可用容量或浪费 |
| 压力升高导致代价或符号改变 | 适用边界/权衡拐点 | 整个恢复调度 family 死亡 |
| d0 配置不能启动 | 当前内存配置不可执行 | 零压力下策略失效 |

最终回答：A 的问题和真实轮转动作合理；目前从缺席会计直接升级在线预测的论证不合理。继续的依据是完整请求的长停顿收益，而不是有缺陷的“步效率恒等式”。Oracle material headroom 与方法新颖性仍未验证。失败类别是分析口径/推理链缺陷，并非真实动作实验被整体否定。

## 评审过程

使用 `research-review` 技能；fresh secondary reviewer 为 `/root/a_line_research_review`，模型 `gpt-5.6-sol`，reasoning `ultra`，`review_independence=same-family`、`acceptance_status=provisional`。主 agent 同时独立核验上述两份原始轨迹。依仓库要求未建立 `.aris`。原始证据未修改；本文件为 review addendum，不替代 sealed verdict。

### 初始评审提示全文

```text
请作为资深系统研究审稿人，对A线做一次有界、对抗但不压制探索的review。用户明确请求“你review一下A是否合理”。适用research-review技能指定你为fresh gpt-5.6-sol/ultra，review_independence=same-family，acceptance_status=provisional。只读文件，不GPU、不SSH、不改原始产物、不另开子agent。请在一轮给出可执行意见，不无限审计。共享cwd=/Users/leandrozhao/Desktop/、++++++++，HEAD7059fc98e0d98a127ff4edda86d44f7f634d26f4，有146条dirty/untracked，保留。当前9月主线以refine-logs/expert_saturation最新报告为事实，docs/current旧8月入口不自动改。

A问题：单卡固定KV下用准入/恢复轮转减少已有decode长停顿、保住服务量。原生OLMoE BF16/vLLM0.26/RTX5090，N32，prompt3072/output1024，50ms到达，KV usable7671。策略为最早缺席恢复、合法victim交换、30等待/20冷却/30恢复驻留、近完成保护；不是只重排队列。已有4臂native/safe29/headroom/rotate8格256请求；独立64新文档2cohort×2block含A/A共20格640请求，rotate maxITL1.004–1.014 vsnative4.447–4.480s，吞吐Δnative−0.619%–+0.475%，平均完成+1.109–2.284%；对headroom吞吐+1.943–2.781%、平均完成−4.257–5.067%。每格多重算30876位置，含重算调用也产其他请求新decode，总调用1348→1257。两条新victim新增约0.95s停顿。报告MEASUREMENT_ONLY，不能称吞吐非劣/显著/新颖性已立。

A新工作：d2压力四格完成 native纯decode1241/rotate1118、真缺席318/267、最长连续缺席220/41步；用20.07/24.23ms乘后maxITL约4.414/.993s，声称闭合1%。onset两臂709，解析16*(usable/(32*K)-192)，K1.013，预测714。d4/d6未执行，用d2校准冻结native步/下界1.69/2.48。用户拟升级“state+candidate action→缺席序列→maxITL/步效率→在线选动作”，优先剩12格压力扫描，再异构长度域。C Qwen失联，不要恢复GPU。

请读取并独立核实以下文件（都小文本；不要重复重算巨大raw）：
1 refine-logs/expert_saturation/outputs/admission_capacity/20260913_pressure_sweep_r01/{ABSENCE_DECOMPOSITION.md,PREDICTIONS.md,DECISIONS.md,INTERIM.md,STEP_FLOOR_FINDING.md}
2 refine-logs/expert_saturation/experiments/admission_capacity/{analyze_pressure_sweep.py,verify_step_floor.py,absence_rotation.py,rotation_native.py} 定向看计数/合法动作
3 outputs相同前缀下20260913_rotation_holdout_r01/REPORT.md，20260913_rotation_victim_order_r01/REPORT.md，20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md（后两都已完成，别把旧next当未执行）。

需要重点验证：
A “步效率超出=总缺席/下界”是否恒等？文件给318/1024=.3105 vs(1241−1024)/1024=.2119，rotate .2607 vs.0918。不要只重复根agent意见，给正确可用的守恒定义。
B 纯decode_steps/max(outputlength)是不是物理下界/可回收浪费？源码排除全部prefill/recompute混合调用（其中其他请求可产生有效decode），还排除最后step。是否边界误差可远超1–2步？所有请求在首完成前admit对全局迭代下界是充分/必要？step少如何与meancompletion变慢并存？
C onset策略无关是否只对首次干预前同轨迹成立？K以乘性修正初始相位偏移有何泛化边界？
D maxITL用真实缺席段内均时闭合是否独立预测证据？恢复已被schedule却未产token、step外交付如何算？
E 冻结压力协议P1–P4把跨序最大差当“可分辨”，P4用时间差约束吞吐，是否需pre-execution addendum；数据不足不判family死。
F 在线状态是否足够？未来终止长度、arrival不可见；优先短期合法恢复时刻/资源可行性+候选动作排序，而非直接预测完整未来序列，哪些最小必要测试？d4/d6两固定策略结果能支持何种而不能支持何种claim？
G 论文可行性：区分问题真实性、动作已验证、模型正确性、净收益、novelty未核验；不要凭knowngenericaging判死。给一句总体verdict，最多6个按严重度排序的发现（含文件行号），最小修正和唯一next实验、结果→允许claim矩阵。总回复约1500–2500中文字符，直接返回，不写文件。引用新的prior art若未核验标未知，不要编造。

主agent同时本地核验计数代数与代码实现，最终会将你的全文/提示和协调意见保存在root review文档；用户AGENTS要求未显式索要independent trace时不创建.aris，本次不写.aris。
```

### 后续消息全文

```text
根agent已用H/execution03 cohort0-block0两份raw定向核验：native/rotate总scheduler调用1348/1257，pure含最后步1242/1119；同时recompute+decode的步8/40，内含decode位置233/1185。说明纯decode减少123步中32步是混合步类别增量，总调用确实减少91，不能抹掉真实收益；也不能将纯步数解释成可回收浪费。正在核验engine_step返回token映射。请返回审阅结论（若已完成即可），不需额外展开。

本次定向核验已确认engine_steps一对一映射混合调用且确实返回233/1185新token。已写根目录A_LINE_RESEARCH_REVIEW_20260913.md准备汇总，请现在返回当前有界评审，不再扩大查阅。最需要你独立指出主结论有无反例或遗漏；完整后可直接final。
```
