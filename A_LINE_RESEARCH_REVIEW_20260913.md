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

### 独立评审结论与协调

Reviewer 返回 `PROVISIONAL CONTINUE / MEASUREMENT_ONLY`。六项判断与主评审一致：错误槽位恒等式；完整未来序列目标过远；max-ITL 事后闭合不能当预测；onset 限首次共同轨迹；执行前更正判据与冲突；论文贡献未闭合但不能判死。补充指出恢复被调度到首个新输出还需四个调用，须与未调度缺席分开；校准系数 0.777/1.107 冲突也须记录。唯一下一实验同为更正后剩余压力扫描。

主 agent 采纳上述意见。对 reviewer 提议的“已观察可分辨尺度”采用更窄措辞“已观察重复差”，避免将有限样本最大差变为可靠可分辨界。P4 采用配对 wall 差与其精确吞吐换算作为描述性结果，不宣布非劣通过。没有要求第二轮审计，也没有将同家族评审标为独立统计确认。

## 执行与成本模型更正续记

用户随后授权继续并提供 westc:53036。已登录验证 GPU `GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9`、RTX 5090、vLLM 0.26.0、torch 2.11.0、transformers 5.15.1。新包 `P/execution_review_westc_r02` 保留旧源副本，仅将 pool=7671 断言改为四个冻结内存字节数的精确块数映射，执行前 addendum 已冻结。计划在新 GPU 执行 16 格，包含本机 d2 参照。

初始检查无 GPU 计算进程；准备期间另一会话启动 cohort2 强基线，PID1770 占约29430MiB。本包启动检查随即退出93，记录 `BLOCKED_BEFORE_GPU_INITIALIZATION`，**本包零 GPU 初始化、零新增测量**。对方工作目录 `/root/autodl-tmp/moe-rotation-strong-baseline-20260913-r01`，属于需要保留的同问题强基线验证；没有干扰。不可在对方每格之间的短暂空闲抢占资源，须待其整批执行结束后再核查。

新增 E/analyze_call_progress.py，读取已有 H 的8份 native/rotate raw，验证 scheduler↔engine 一对一、engine 调用不重叠、全部返回 token=请求保留 token。结果为 P/call_progress_review_r01.json。四个配对计数一致：native/rotate 全调用1348/1257、pure含末步1242/1119、prefill调用98/98、recompute调用8/40；每格输出32768，prefill调用产生1547个新输出，recompute调用产生233/1185个新输出。

cohort0/block0 的互斥 engine 调用时间：native prefill/pure/recompute 为2.273/20.063/0.228秒；rotate为2.254/19.081/1.149秒。纯阶段少0.982秒，同时重算类别多0.921秒。各类别都包含实际并发有效输出，不能把整个重算类别时间称为重算税；这只是策略各自轨迹的阶段分解，不是保持未来轨迹不变的因果扣减。该分解解释了“总调用减少但吞吐接近”这一成本关系，是本轮新增实质分析。

## FastServe 动作级核对补记

已读 [NSDI 2026 原文](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf) §4.1、算法1、§4.2相关描述及§6.3缓存消融。FastServe 不仅做 token 粒度抢占：它会按等待时间提升饥饿任务优先级，并按队列优先级主动换出/换入 KV；也已有重算与响应式换页基线。因此“按等待时间恢复”和“主动解决 KV 压力”均不是足够的新颖性依据。

| Work | Signal/state | Prediction target | Action | Objective | Regime | Guarantee |
|---|---|---|---|---|---|---|
| FastServe | 输入长度、已用时间片、饥饿等待、队列优先级 | profile估计下一迭代时间 | skip-join/降级/饥饿提升，主动KV换页 | 请求延迟与服务吞吐 | 可用主机KV存储与传输 | 所读部分是机制/实测，未据此声称严格max-ITL界 |
| 当前A | 当前块、等待年龄、已生成进度、冷却/保护 | 尚无通过验证的候选预测 | 真实驱逐重算，资助最久等待者恢复 | 最大停顿与完整服务量权衡 | 固定GPU KV，当前未用主机KV交换 | 当前仅合法性检查与测量；无一般停顿上界 |

上述差别不是贡献证明。A 的下一模型应检验：同样可执行的恢复优先动作下，块可行性和“重算同时推进其他请求”的完整成本，能否产生超过普通饥饿提升/简单victim策略的可重复决策收益。主机KV交换未比较，不能称其收益已被A覆盖；同底座移植的队列规则也只能称 FastServe-style，不是完整 FastServe 复现。原文未完成公开代码对照，论文级新颖性仍未确认。该补记缩小贡献范围，不新增GPU队列项，也不替代压力扫描。

## 2026-09-14 压力扫描已执行

此前资源阻塞已解除，按协调交接完成16格，全部原始数据回读校验。完整报告见 [压力扫描结果](refine-logs/expert_saturation/outputs/admission_capacity/20260913_pressure_sweep_r01/execution_review_westc_r02/REPORT.md)。512请求/524288新输出对齐；两个d0轮转零动作标记保留。d0可行，旧显存门槛估计被否定；d4/d6 onset误差约1.7%/5.2%，但d6 native pure实测1748对预测2543，触发原M2。

恢复动作的长停顿收益跨压力保持，d6 native约14.2–14.4秒降至2.81秒，同时平均完成慢4.49%–8.90%，吞吐随block变号。原评审关于成本模型与目标权衡的担忧得到新数据支持，不能据此判死恢复问题。下一步先在d6加入headroom/most_output直接强简单基线，不先拟合完整未来缺席预测。新结果fresh同族审阅已返回WARN，无影响主表/冻结证伪的P0/P1；范围限原生vLLM进程内测量，高压强基线仍缺。GPU已交接，无额外占用。

## d6强简单基线接续（2026-09-14）

完整结果见[八格报告](refine-logs/expert_saturation/outputs/admission_capacity/20260914_d6_strong_baselines_r01/REPORT.md)。8/8 COMPLETE，256请求，262144输出，仍是旧32输入、两反序block、APC关闭。least/native平均完成+7.47%/+9.32%，最长ITL约14秒→2.80/2.94秒；most/least整批吞吐+2.75%/+9.58%，平均完成却+8.03%/+0.60%，最长ITL接近。headroom/least平均完成+50.33%/+40.91%，最长ITL也更差。此处仅描述观测，不称差异显著或吞吐非劣。

新边界：更少调用及更早最后完成，仍可伴随更晚的多数请求完成。block0 most/least第16个完成24.612/22.027秒，最后25.764/26.474秒；未完成请求数积分准确还原各格全部请求延迟。A的最弱链路应收窄为动作条件下的恢复/退出与完整请求代价，而非继续拟合总步数。后续候选状态分支尚未实现或运行；不声称已获得在线选择器或Oracle。GPU已交下一会话，本轮无追加。

## 动作条件分支接续（2026-09-14）

真实KV资格与三分支已完成：[前态资格](refine-logs/expert_saturation/outputs/admission_capacity/20260914_d6_action_state_r01/REPORT.md)、[三分支](refine-logs/expert_saturation/outputs/admission_capacity/20260914_d6_action_branches_r01/REPORT.md)。每臂校验请求/有效KV后独立推进，六格192请求完成；立即least/首次most/延迟一步的4/4/5恢复调用预测均命中。但首次most的平均剩余+0.04%/−1.84%，延迟−0.51%/−0.64%且最后完成更晚，不支持方法GO。仅当前状态短期资源模型得到验证，完整成本排序未成立。首次动作确实改变1240步后续请求调度，不是无动作负控。下一仅CPU补请求完成/释放的资源进展模型，以三条真实未来校准，不把真实未来复用于候选预测。

## 2026-09-14 资源进展模型补充

[本轮报告](refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_progress_model_r01/REPORT.md)：26条既有轨迹的调度、块计数和完成步匹配；12条未拟合轨迹条件mean/last成本最大误差2.10%/2.60%。这支持固定长度域的模型保真性，尚未证明在线选择收益。31个首次victim没有预测平均完成收益，停止该截点的GPU扫描；下一仅CPU查看后续决策状态是否有完整请求收益空间。无新GPU执行，无新增独立审计。

## 2026-09-14 后续单事件空间

[结果](refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_later_victim_r01/REPORT.md)：802个合法victim替换最佳预测mean−0.074%；取消一次轮转最佳−0.373%并使gap+11.22%。当前只支持单事件机制空间不足的模型诊断，不是整个问题或联合动作NO-GO。下一CPU定位持续轮转对完成释放的影响；无新GPU执行。

## 2026-09-14 持续轮转与恢复成本

[报告](refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_stop_r01/REPORT.md)：三档62个停止点没有同时改善平均完成和最大沉默。12条真实轨迹完成面积闭合，净损失集中于含重算的混合调用区间（不可当纯重算税）。下一检验同状态KV暂存往返相对重算的物理成本空间；没有新GPU或方法GO。

## 2026-09-14 KV往返实测

[补充](refine-logs/expert_saturation/outputs/admission_capacity/20260914_kv_roundtrip_feasibility_r01/GPU_ADDENDUM.md)：三尺寸含打包往返16.50–20.54ms，24次内容正确。局部物理成本支持下一资格化，尚无请求收益。安装vLLM已有native OffloadingConnector；下一优先测纯native offload基线，现轮转connector guard不变，不重写pager。

## 2026-09-14 默认native offload基线

[结果](refine-logs/expert_saturation/outputs/admission_capacity/20260914_native_offload_baseline_r01/REPORT.md)：4格128请求结束，默认prompt-only实际减少86.03%重算，但平均完成+23.83%/+6.73%。第一on尖峰完整保留，第二block亦净负；局部传输空间不等于请求收益。下一保持问题/配置，定位后端调度和执行税；full-decode及rotation兼容未测。

## 2026-09-14 默认offload有限开销观测

[报告](refine-logs/expert_saturation/outputs/admission_capacity/20260914_native_offload_cost_r01/REPORT.md)：4格128请求、执行/输出/传输一致，connector互斥开销1.200/1.112s，大部分时间仍在engine未细分部分。profile开关配对wall+2.217%/+4.968%，不能当免费观测或据CPU时间直接定位Python。下一限定共同decode片段的CPU/CUDA活动定位，不增加全程埋点；无新方法GO。

### 2026-09-14 默认offload共同decode活动定位

`20260914_offload_decode_trace_r01` 两臂COMPLETE；相同32调用/32完整输出一致，各10274 GPU事件可匹配API correlation。窗口648.874→868.054ms，GPU活动并集511.701→511.551ms，额外时间主要在记录GPU活动外；不能归因特定函数或宣称可移除收益。单次off/on且带profiler，顺序/主机漂移未隔离；唯一下一步同窗口反序复测并记录CPU环境。证据上限NATIVE_PROFILER_DIAGNOSTIC，非方法GO，原件及分析见该目录REPORT.md。

### 2026-09-14 默认offload反序诊断完成

`20260914_offload_decode_trace_reverse_r01` 两格COMPLETE；32调用调度/32完整输出一致，GPU事件10274/臂且全部API correlation匹配。off/on窗口640.596/844.566ms，GPU并集511.943/511.607ms，主机侧差异反序复现。CPU阶段占比不稳，on第527调用同步后66.556ms尖峰；包围窗口cgroup节流增量0，不能排除频率/主机漂移。下一只定位Python调用/GC事件，不猜测性删逻辑；证据上限NATIVE_PROFILER_DIAGNOSTIC。

### 2026-09-14 offload主机profile发现观测器成本

`20260914_offload_python_cost_r01` COMPLETE，32调用签名/完整输出匹配；一次gen2 GC68.947ms/collected0。memory_telemetry.state64次累计107.4ms，request_state2050次/get_blocks2050次；嵌套含GC、不可相加或直接归为offload增量。下一只做等价低分配块数观察器消融并无profiler复测，不禁用GC。NATIVE_HOST_PROFILE_DIAGNOSTIC，非方法净收益。

### 2026-09-14 KV观察器八格完成，微优化停止

`20260914_kv_observer_cost_r01` 256请求完成，direct平均完成对original：off−2.81/+2.94%，on+5.06/−2.77%，均翻转，无稳定收益。逐步schedule/输出/传输量一致；规范化ID后memory差异仅前69步的到达集合，共享请求状态0差异，严格全状态等价不通过。此实现停止，不归为KV问题NO-GO；下一CPU核对native connector保存/驱逐/恢复契约，禁止直接移除不兼容保护。

### 2026-09-14 Native保存/驱逐契约CPU定位

`20260914_native_store_contract_r01`：store完成用completed_jobs，finished_sending始终空；驱逐只flush已有任务，不自动为未保存victim造任务。封存worker原方法3CPU case通过，fake传输/无GPU。候选须保存登记→原生提交/等待→驱逐两阶段，下一只用真实首次前态核对提前边界及KV保持可行性，不删connector保护。

### 2026-09-14 选择性保存提前边界CPU资格

`20260914_native_store_contract_r01/selective_boundary.json`：两block step328 visible选择3571命中原329victim，31decode/free149→148，释放207块后目标205块静态可容纳；native store-builder原方法+fake host/key/block生成任务，decode允许时3296token保存/10已计算尾部未保存，默认prompt-only3072/234。CPU条件可行性，非GPU/真实hash/净收益；下一单victim真实KV保存恢复接口资格，保留原生保护和成本。

### 2026-09-14 单次选择性保存真实接口观察

`20260914_selective_store_once_r01` 两臂64请求完成；328store/329flush/330load/332恢复首新token，保存3296token，实际store432013312/load864026624bytes（两load），victim gap170→104ms；mean+2.37%/wall+1.66%，n=1不支持净收益。25/32完整输出一致、victim一致，7请求动作后分叉；KV保真未测，下一定向保存前/加载后相同逻辑前缀指纹，非方法GO。

### 2026-09-14 单次保存的完整分母解释修正

[时间分解补充](refine-logs/expert_saturation/outputs/admission_capacity/20260914_selective_store_once_r01/TIME_PARTITION_ADDENDUM.md)：同到达/同前328步调度，保存动作开始前on已慢0.927040s，完整平均完成慢0.610306s；动作后平均剩余时间差−0.316734s。该恒等分解不能用作漂移校正，也不能证明净加速；原+2.37%仅为单次观察，不归因于保存机制。局部恢复170→104ms已观测，稳定完整请求收益仍未验证。KV保真单格已远端18文件校验、GPU_UNRUN，结果分析器的3个CPU替身检查不构成实测。当前唯一下一实验仍该单格，排streaming整组之后；通过后才进入性能重复。

### 2026-09-14 首次恢复的保存前缀真实保真

`20260914_selective_kv_fidelity_r01` 单格32请求/32768输出完成；328保存前与331首load完成后的3296token×16层SHA一致，432013312bytes、206逻辑块，物理映射不同。331 computed3296/output235，332才执行11位置恢复；一次store/两次load，本次只检查首load。哈希新增0.419/0.373s全属诊断，非性能证据。排除该前缀首次搬运损坏，不覆盖其他请求/第二load/质量；净收益仍未验证。下一同非指纹底座完整交错重复，不作漂移扣除。

### 2026-09-14 单次保存交错重复与混合服务会计

`20260914_selective_store_repeat_r01` 四格128请求完成，均完成on/off +1.713%/−18.779%翻转；block1动作前已快1.671s，同臂全调度/输出相同但off wall27.142→33.637s，净效应UNRESOLVED。目标gap两对−33.60/−50.22%，重算少6591tokens；含重算调用26→21却纯decode1745→1748，135新输出转移，总调用只少2。混合恢复仍服务其他请求，重算量不能直接当可删除串行税。按冻结规则停止追加同域重复，下一CPU把混合正常decode服务纳入动作状态/成本模型。GPU已释放；fresh审阅因模型capacity未执行，非PASS。

### 2026-09-14 保存前缀状态模型的ready边界

`20260914_saved_prefix_model_r01`：原模型已含重算混合decode进度；补原生指定抢占后，无保存两重复329–1868逐步完全匹配。补加载先占块/留waiting/ready后恢复后，保存两重复329–1040匹配，1041均失败：首次load等待2步，第二次实际1步，固定2步近似不成立。末步仍同1866不能覆盖轨迹失败。原least/most/defer预测回归相同。下一CPU定位native提交/完成查询/ready可见时序，不喂未来ready标签，不追加GPU。

### 2026-09-14 Native异步完成通知状态回放闭合

`20260914_load_ready_contract_r01`：worker执行结束查询finished_recving，scheduler接收后下一步才可提升WAITING_FOR_REMOTE_KVS；computed/占块不等于ready。真实worker事件按perf_counter定位首次331→332、第二次1040→1041；事件回放下两保存重复各1538步调度/空闲块/输出数全匹配。明确使用实际未来完成通知，仅状态机验证，非预测或反事实收益。默认旧三策略回归不变。下一CPU比较原缺席请求优先与资格实现刚抢占victim排队首的恢复排序，保持异步通知未知边界。

### 2026-09-14 单次恢复排序的贡献边界

`20260914_saved_recovery_order_r01`：实测旧资格优先恢复刚抢占3571（332/333），原缺席3640仍1045/1047恢复。CPU原缺席优先且不保存令首服务交换714步，调用1869→1836，但仅两请求完成+75/−76步，均完成步号只降1/32；同排序保存增量−2/−1/+1调用随假设load1/2/4步翻转。不是GPU/墙钟预测，不把排序收益归保存。停止单事件排序GPU扩展，下一盘点既有完整轮转的重复恢复成本空间，保留混合服务与反事实边界。

### 2026-09-14 完整轮转保存量与混合输出比较

`20260914_rotation_save_volume_r01` 复用旧8格；least39事件预算直接复用核验。most44抢占/163764重算tokens却总1315调用，优于least39/137272/1581，不能用重算量评价策略。least/most含重算调用生成4226/4739新输出；累计4.2–4.9s不能全当可删税。按完整块每次全存least17.954GB、most21.418GB单向，最长逻辑前缀合计4.059/12.053GiB仅条件容量盘点。多次保存未被成本空间排除，也未有净收益；下一仅most两阶段native保存/抢占合同与真实前态合法性，保留最强同底座基线，不删旧保护。


### 2026-09-14 更新：真实两阶段单事件的收益边界

见`refine-logs/expert_saturation/outputs/admission_capacity/20260914_staged_store_probe_r01/REPORT.md`。两臂64请求完成，原缺席优先/实际store-flush-load链跑通；保存少3392重算tokens却总调用不降(1794)，原缺席首新输出同333步。on均完成+12.56%且动作前漂移+1.14s，只作资格结果，不作稳定性能结论。当前最弱链路是重复保存是否改变总服务进度，不能继续用重算tokens减少代替完整请求收益。


### 2026-09-15 重复保存从单次正信号进入同域交错重复

复用服务窗口原方`20260915_repeated_kv_service_r01`统一分析（当前临时analysis在/private/tmp/moe-repeated-kv-service-20260915-analysis.json，正式归档交接中）。192请求六格全COMPLETE，其中diag两格不混入主ABBA四格。主两对保存相对不保存：均完成−5.755%/−2.128%，输出吞吐+5.864%/+2.033%，max引擎返回gap−18.367%/−14.928%；TTFT均值+2.18ms/+65.83ms，完成更慢请求0/32与1/32。相同staged most_output、6656 GPU块、实际16GiB host KV allocation；不是增加host预算后的不公平对比。

当前证据NATIVE_INPROCESS_FIXED_LENGTH：同一32请求闭合cohort、两对执行顺序平衡，幅度仍有波动，质量/自然EOS/新请求域未验证。可把保存most作为本资源域待确认的强基线；不能把既有native pager的收益全算新调度贡献，更不能据此宣布完整论文方法成立。下一研究不再以重算tokens为收益代理，应优先在新请求集合保持机制和预算不变验证保存后的完整服务边界；不启动新的独立Controller。现GPU归双实例已登记窗口，本文不新增GPU组。


### 2026-09-15 TTFT归因边界补充

正式`20260915_repeated_kv_service_r01/analysis/REPORT.md`已核对：全部请求首输出在call97之前，首prepare/store在329；此前8909输出事件及32请求前缀一致。故原主表TTFT+2.182/+65.829ms只是两次运行的观测差异，不得称保存动作造成的代价；on/off动作前偏移−13.644/+214.286ms亦保留，不据此扣除或校正后续收益。两对完成收益仍为小样本同域观察，未声称统计显著。下一文档迁移四格不改参数，恢复启动机制由原测量/资源执行方先定位E到实际可执行边界，本方不并行创建Controller。
