# 从请求账本到 paging 成本：资产核查、最小模型与执行建议

2026-09-12。探索交付；不修改 `docs/current/README.md` 或历史 sealed verdict。主结论：**四项原生KV对照已完成并独立复算：增配约1.47–1.57GiB KV后，4.56–4.59秒的最长停顿降到83–99毫秒，吞吐提升3.99%–4.70%，平均请求完成延迟则增加1.02%/0.15%。当前resident域先由普通配置解释；下一步是真实pager的最小兼容与成本实验，尚无MoE方法GO。**

## 1. 仓库现实与已有资产

指定分支 `agent/publish-current-moe-code`；本地 HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`。本次 `git ls-remote` 核实公开分支仍为 `d65416c5cd09eb30aa6e46875339b16ebf7f5901`；本地领先三个提交。开始时已有 `expert_union_tracker.py`、`run_expert_union.py` 修改及相关新增文件，未覆盖。执行中另一流程继续新增同任务代码及 KV 运行；本报告明确区分继承资产、本次复算和同步取得的新结果。

已读权威：`docs/current/README.md`、`docs/ideas/README.md`；目标资产入口：`EXPERIMENT_TRACKER.md`、`DECODE_CAP_BRANCH_GATE.md`、`gpu_pressure_sketch/README.md`、`route_pressure_sketch.py`、admission 源码、9月最新结果和纠正记录。前两个权威入口主要记载8月历史裁决；本次用户明确选择普通KV及paging问题，不据旧执行顺序退回数值溯源，也不升级旧方向状态。

下表路径以 `refine-logs/expert_saturation/` 为根。`E/` 表示 `experiments/admission_capacity/`，`O/` 表示 `outputs/admission_capacity/`。

| 实际文件／入口 | 实现与执行状态 | 可复用部分；缺口 |
|---|---|---|
| `experiments/gpu_pressure_sketch/route_pressure_sketch.py`；对应 README 的测试入口 | NumPy参考与可选GPU实现；源码/README明确尚未接入vLLM服务路径 | U/C、expert histogram；不是物理传输或方法收益 |
| `E/run_native_capacity.py` → `native_capture.py` → `admission_feedback.py`；`--help` 可查入口 | 多个9月原生vLLM实测campaign | request/step/receipt关联、非抢占降准入；降cap不释放既有KV，waiting=0时没有新请求可影响 |
| `E/runtime.py`、`metrics.py`、`run_capacity.py` | custom-runtime和CPU测试资产 | 到达/失败/完成分母；不能代替native GPU |
| `O/20260908_capture_ladder_paired_r01/`、`...paired_repeat_r01/` | 64个原生episode；CPU复算可运行 | 同引擎静态/反馈对照、成本阶梯；没有专家信息增量 |
| `O/20260908_native_preemption_r01/{run_probe.py,native_capture.py,memory_telemetry.py,analyze_native_preemption.py}` | 4个完整native cell，128/128请求，raw在 `gpu_results/*/raw.json.gz`（部分同时保留plain） | KV块、原生抢占/重算、完整token时间；本次再核算保存指标与身份 |
| `E/kv_feasibility_admission.py` → `run_kv_feasibility.py`、`gated_capture.py` | CPU逻辑/测试与runner存在；该策略自身的GPU状态不能借用safe29结果 | 最终长度保留基线；token级保留须换成逐请求block取整，参数上界和真实输出分开 |
| `O/20260908_kv_safe_static_r02/` | 原生保守静态测量 | `usable_blocks / ceil(max_length/block_size)`的强静态对照；零抢占不是整体性能代理 |
| `O/20260910_kv_budget_r01/`：`run_one_cell.py`、`run_campaign.py`、`execution.tar.gz` | 原冻结 .95/.90/.90/.95 包；旧attempt无请求raw | 科学代码和输入可直接复用；旧端口和后台WAITING不是新机器执行证据 |
| `O/20260912_kv_budget_r01/`：`execution.json`、`gpu_results/` | 另一流程已完成上述四cell；本次独立复算通过，最终数值见第8节 | 同硬件普通KV干预；不再重复启动 |
| `../../independent_ideas_20260911/per_request_prefill_share_r01/run_prefill_share.py` | 两个fresh block原生实测 | global512/per_request512/native1024的不同动作；per-request512完整请求指标未胜native1024 |
| `E/run_expert_union.py`、`expert_union_tracker.py`、`adjudicate_expert_union.py` | 本轮开始时已有dirty源码、CPU测试；原始GPU结构数据未取得 | batch union/跨step复用；compiled引擎加载后才装Python hook，有graph replay不执行hook的实际风险；零采样不是负结果 |
| `E/run_session.sh` | shell实现，静态扫描部分明确 `SKIPPED` | 注释说“同次引擎两测量”不等于代码已实现；不能以脚本存在宣称P1–P3已跑 |
| `O/20260912_feasibility_envelope_r01/analyze_envelope.py`、`E/check_regime_discriminability.py` | 基于64旧episode的CPU分析 | 复用decode成本阶梯和prefill税；不是已部署的在线候选动作预测器 |
| `E/paging_cost_model.py`、`analyze_pause_ledger.py`及测试 | 本轮期间另一流程新增；本次先实跑4+5项；修复状态机/FCFS后6+5项通过 | 一套CPU状态/字节原语和暂停分解；无pager、无策略接入、无GPU因果预测 |
| `E/recompute_native_resources.py`、本目录 `native_resources.json` | 本次新增分析脚本并实际运行；4cell/128请求逐项核算成功 | 严格身份/配置/保存指标核对、完整吞吐和重算阶段；拒绝覆盖，不产生反事实结果 |

**9月资产不存在整体缺失。**普通cap、prefill和长暂停均有本地原始数据；尚缺真实pager运行、H2D成本、在线两动作集成、独立请求集方法结果。最新专家结构采集仍不能借CPU测试写成已测。

本次旧数据重算：native32对safe29吞吐提升 **16.9853% / 17.0795%**，4cell均32/32。repeat0最大ITL **4.473448403 s**，首重算调用之前 **4.357883330 s**，重算调用跨度 **0.115565073 s**。后两者是互斥host区间；前段不全等于纯scheduler等待，后段不等于纯GPU kernel，不对重叠victim求和。

可行性包络的3.25%误差、60/64分类一致可以保留为**跨campaign事后重建**。代码使用被解释episode的实际宽度、完整期间prefill步率和平均prefill量；相同32篇文本重复，不是独立文档预测。它不能证明“所有策略无收益”“任何预测器都不够精确”，也不能从静态零负载桶成本推断低到达率时cap32必然跑width32。上限10.15req/s只在其稳态、固定成本与平均量假设下成立。

## 2. 最近邻工作与真正剩余的问题

核查当前论文版本和实际源码；作者性能数字没有在本项目复现。不能再用“前人只放置给定资源”概括本领域。

| 工作／版本 | 控制对象、资源、可见信息和目标 | 公开实现／实测边界；对本题的重合 |
|---|---|---|
| [WiSP v2](https://arxiv.org/html/2606.21868v2)、[release v0.2](https://github.com/nokia-applied-research/WiSP/releases/tag/v0.2) | expert/KV共同预算、工作集边际价值、LRU pager；保留KV admission floor | **2026-08-30已发布在线双池controller**，README落后；[main源码](https://github.com/nokia-applied-research/WiSP/tree/86f69720f0de0647c51728ea2e27e4289bf3dea2/src/wisp/dynamic)含serve hook，动作等scheduler排空。论文0.11.2/3090，动态主测in-process；0.26/5090及新serve路径性能未复核。共同预算/resize不是本题新意 |
| [FluxMoE v3，9月10日](https://arxiv.org/html/2604.02715v3) | GPU压缩/CPU backing、expert驻留和KV竞争；活动batch与materialization比率 | 有真实在线到达评测，原型vLLM0.23；动态KV的41次launch模拟排除重启、重建prefill和未建模resize成本，是乐观估计。此次未核实官方开源。驻留自适应不同于直接选新请求及prefill量 |
| [MoE-Gen v1](https://arxiv.org/html/2503.09716v1) | 硬件成本驱动模块微批/缓冲/CPU attention规划，优化吞吐 | 论文主讨论排除continuous batching；[原仓库已转BatchGen](https://github.com/batchgen-project/batchgen)，当前planner/sequence scheduler不能归回旧论文。batch共享搬运和模型搜索已有 |
| [MoE-Lightning](https://arxiv.org/html/2411.11217v1) | 分层Roofline、CPU/GPU/IO pipeline、权重/KV/微批 | 离线batch为主；本次未核实正式公开实现版本。性能模型、重叠和带宽分层不是新意 |
| [Layered Prefill v2](https://arxiv.org/html/2510.08055v2) | layer-group prefill，维持decode，优化TTFT/TBT/SLO | [实际scheduler](https://github.com/scale-snu/layered-prefill/blob/main/nanovllm/engine/scheduler.py)已存在。直接控制新prefill对在途decode的影响；核心重复权重读为HBM/GDDR到计算层次，不能写成PCIe H2D |
| [DuoServe-MoE v2](https://arxiv.org/html/2509.07379v2) | 两阶段预取/驻留、预测与补载，QoS | 最新v2于2026-04-09；主实验单请求，batch1–12吞吐是扩展；官方实现范围未验证。区分prefill/decode不构成独立贡献 |
| [FreeToken v1](https://arxiv.org/html/2608.16157v1) | 依实测CPU/PCIe带宽分配missing expert到CPU计算或GPU cache fill，弹性KV/expert池 | [main](https://github.com/FlashML-org/FreeToken/tree/953565667f3141c90d0f0eb469bb2655d2407140)有完整runtime；rebuild为idle-only，含同步及graph重捕。[open PR300](https://github.com/FlashML-org/FreeToken/pull/300)已提请求边界KV/MoE ladder，未合入。不能主张“请求触发预算”无人做过 |
| [Gimbal](https://arxiv.org/html/2606.15177v1) | prefill队列/KV/expert pressure联合选择DP engine、局部短prompt+aging和placement | 多卡DP/TP/EP，官方发布代码未核实；请求压力联合决策已有，单卡H2D成本域不同 |
| [QLLM](https://arxiv.org/html/2503.09304v1) | 高/低优请求的layer/expert级抢占，保留combine依赖 | A100/4bit Mixtral到达评测；开源更新未核实。直接改变新请求与已有请求进度；不能以expert级调度为创新 |
| [Diff-MoE，SC25](https://doi.org/10.1145/3712285.3759903) | 差异缓存、batch、优先级细节仍未验证 | ACM全文403；[作者官方目录](https://cgi.cse.unsw.edu.au/~jingling/bycat.html)确认条目但无可得全文。保留查新未决，不补造机制 |
| [SLOs-Serve](https://arxiv.org/html/2504.08784v1) | 性能模型、soft admission、DP token分配和动态chunk/batch，保护已有多阶段SLO | **覆盖本题通用算法骨架**；没有据此确认其expert paging状态。必须作为近邻baseline，不声称MPC/准入模型本身新颖 |
| [QoServe / Niyama](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/QoServe_CC_final_v2.pdf) | 已有请求deadline slack→prefill chunk，hybrid priority | [官方artifact代码](https://github.com/microsoft/sarathi-serve/tree/niyama_asplos2026)可读；旧依赖不直接移植。应实现同动作的slack规则作强基线 |
| [ExpertFlow v2](https://arxiv.org/html/2410.17954v2) | 路由预测、相邻decode batch重组、减expert union与预测cache | 单GPU；作者GitHub链接本次404，代码未验证。与locality batch选择强碰撞；尚未确认其在线入场/max-ITL目标 |

候选核心主张只有两个：C1 同预算下，入场/prefill改变专家缓存与搬运，会给既有请求造成可复现的额外停顿；C2 这份动作前可见的信息能改变强普通基线的选择并改善完整请求。**C1、C2均未建立。**框架新颖性低；潜在新颖性在经过同资源因果验证的作用机制或失效边界。Diff-MoE全文仍是未关闭的碰撞，不能写first。

## 3. 一个主问题与一个备选

**首选问题：**在固定expert/KV划分、LRU、dtype和backend的单GPU pager内，候选新请求及其prefill量引起的共享加载和后续驱逐，是否会改变长度/队列/KV/decode-slack基线给出的动作排序，并改善相同到达流的请求goodput与最大ITL？

选择原因：已有账本和准入/prefill资产，新增工作只需补真实pager成本和两动作接口；避开WiSP共同预算、FreeToken CPU执行和ExpertFlow重排既有decode的动作。反证风险：专家并集迅速饱和、cache状态几乎不变、计算/attention占主导、更多batch摊薄加载、普通slack已覆盖增量、预测需要未来route、pager本身不支持所需工作集。任一结果只缩小实测域；不以固定百分比阈值判死offload家族。

**不同的备选：**保留resident原生vLLM，以请求KV增长和原生恢复等待解释“提前排队/中途停顿/吞吐”边界，做通用serving测量与模型。只控制新准入与prefill，不改恢复优先级。已有完整raw和成本表支持，实施成本更低；与SLOs-Serve/QoServe重合更大，只有新的可复现边界才可能成为论文贡献，不能自动称MoE专属方法。

两条线不同时实现controller。当前KV配置对照已关闭这段resident长暂停的必要性解释；主问题随后只做真实paging成本存在性。若paging没有决策增量，转备选的测量结论，不换第三个预测器。

## 4. 完整但最小的系统模型

### 状态、可见信息与动作

在决策时刻t记录已到达未执行集合W、已入场prefill P、decode D、被原生抢占待恢复R、完成F。每请求有arrival、prompt长度、已计算token c_i、已返回token g_i、max_output上界、实际KV blocks、上一token时刻。引擎含真实token budget、实际running/waiting、KV pool；专家键为(layer,expert)，记录驻留C_l、实际miss/load/evict、传输队列与完成历史；硬件含实测服务曲线、有效H2D带宽、启动税和暴露等待。

在线只见t之前的信息。未到达请求、未来EOS、候选动作执行后的route、整段episode平均宽度不得当输入。训练/校准参数固定后，评价仅更新已完成历史状态。

动作a=(A,q)：A是W中从未开始的请求子集，q是本步prefill量；已有P可继续按FCFS/aging处理，D正常推进。候选先缩为FCFS前缀数量 `{0,1,2}` × prefill `{0,256,512,1024}` 内合法点，实际总budget留出decode需求；零prefill不得伪装成已经进入计算的入场。先不优化请求身份；只有历史工作集有增量时，再在已到达的极小候选窗口比较身份。迟到/长请求计最长等待，不能饿死。

**现有接入边界：**非抢占准入已经实现；global token budget/long-prefill threshold目前只在引擎排空时切换。未来每step的q需要在native scheduler“已有decode选择后、新prefill分配前”增加独立额度接口，不能直接把drained API改成运行中可用，也不能削减总budget后跳过已有decode。两动作模型现在只有CPU原语，不是已经接入vLLM的策略。

### 资源与状态转移

GPU分配守恒：`M_nonexpert + M_expert + M_KV_pool + M_workspace <= M_GPU`。这里KV pool为分配的物理storage；另跟踪池内live blocks，不能把两者再相加。`torch.reserved`包含allocation，不另列为新增资源。主机权重、CPU副本、pinned buffers、临时loader内存也必须在**容器**限制内；共享/别名storage计一次。

无prefix-sharing、单full-attention组时，b_i=`ceil(c_i/block_size)`；总live blocks≤实际可用池。模型输出的最后一个token未必已经forward，因此KV不是prompt+returned tokens的简单和。原生preempt释放对应KV、保留输出，恢复的已知计算缺口为 `max(0,prompt+max(g_i-1,0)-c_i)`。完成按实际EOS/max_tokens事件释放；预测不能提前使用实际未来完成时刻。降低目标cap只减少新入场，不回收P/D/R资源。下一步可能增长的block在候选比较中逐请求取整，不能依赖同一步尚未完成的释放而掩盖峰值。

对于已经过router的某层batch：

`D_l(B,C) = Σ_{e∈U_l(B)\C_l} physical_weight_bytes(l,e)`。

这是每所需专家最多加载一次时的基础bytes。驱逐引发的再加载、padding、传输粒度、其他复制另计；跨请求共享专家不能按请求重复收费。改变batch既可能增加union和驱逐，也可能摊薄权重成本。真实pager若要求本次union同时装入scratch，必须满足其容量约束；若按expert波次执行则还需统计波次/launch/activation成本。**union饱和不证明所有专家必须全层永久同时驻留，更不否定被迫offload时的调度价值。**

零模型 `E[U]=E*(1-(1-k/E)^B)`只假设跨token独立均匀路由。对未来动作可用已完成历史的边际probability建立近似 `Σ_e w_e*(1-Π_j(1-p_{j,e}))`；跨token相关性、候选导致的route变化均需评价，不能把精确后验U塞回在线模型。初始成本实验可直接测动作曲面，不要求先训练predictor或完整exact Oracle。

时间模型是互斥的 `T_step = T_nontransfer_compute + T_exposed_transfer + T_host_schedule`，计算曲线按decode宽度、KV长度、prefill量和执行模式分桶。总copy span不等于暴露等待；用时间线依赖或开关传输的配对测量校准。没有数据时只报告带宽/overlap敏感性，不生成“实测时间”。请求arrival→completion账本已经包含排队、恢复与调度，不把这些span再加一次。

### 目标和求解

预先定义到达窗口 `[t0,t1)`、所有到达请求的终止追踪窗口T、TTFT上限、每请求平均TPOT和max-ITL条件。goodput=`满足全部条件并在T内完成的到达请求数/(t1-t0)`；拒绝、超时、失败、未完成保留且不计good，另报总完成率/最长等待。有限batch则单列 `completed/episode_wall`，不要混称稳态容量。32请求的分位数是样本描述，不是生产p99。

CPU先枚举少量合法a，用校准曲线预测一个短时域的KV峰值、已有请求下一token间隔和新请求首token进度，只执行第一步，再读真实状态。可采用词典序：减少预测SLO违反→增加完整请求进度→减少等待/搬运。假设误差和目标权重在calibration确定，评价固定。H=1–4步只约束近端风险，不能假装完整请求goodput的exact Oracle；最终收益来自各策略独立真实执行。无material residual时删去专家信息模块。

## 5. 第一批执行顺序及每种结果的用途

| 顺序 | 配置与对照 | 采集、结果解释、下一步 |
|---|---|---|
| R0：已完成的普通KV清算 | 同RTX5090、OLMoE BF16、32×3072/1024、50ms、cap32、budget1024；四新引擎 `.95→.90→.90→.95`，均native preemption；每cell三次旧暖机并保留raw | 读实际KV池，不把util当字节；全部请求/失败、TTFT/TPOT/maxITL/完成/重算/恢复。95首先资格。若更大池解决暂停且净改善，停止以这段历史损害推销专家机制；若只去抢占无净收益，零抢占不是代理；若90本轮不复现，记录环境域变更；符号翻转不选有利repeat。新结果见末节 |
| R1：pager兼容性与最小加载检查 | 独立checkout/venv；优先WiSP已发布源码，固定替换/dtype/预算，关闭dynamic resize。先无模型toy pager，再确认OLMoE后端支持；不能默认0.11.2兼容5090/0.26 | expert→slot一致、miss/hit/evict、真实H2D、工作集overflow、同tokens计算正确。CPU逻辑通过不当GPU兼容。旧环境若无法支持Blackwell，停止该移植点并评估公开FreeToken backend；不在GPU租期临时重写整套pager |
| R2：动作成本曲面，唯一机制前探针 | OLMoE人为限制expert预算，仅标受控验证；同pager/KV池/替换/dtype/kernel。16–32请求，先一个近容量域、短长混合；同pre-action状态：A hold-new、B admit+prefill256、C admit+prefill1024；全resident为负控 | action各自生成后续route/KV/完成；同block实验重复/反序，计observer税。记录既有请求ITL、new TTFT、唯一H2D/reload/evict、暴露等待。若更多batch摊薄搬运就保留此方向；若ordinary状态解释排序，停专家预测；仅局部cache收益不进入completion则停method |
| R3：有残差后的最小policy | 同load/resource对比：B0调好静态cap/prefill；B1 token/KV/slack规则（SLOs-Serve/QoServe同动作思想）；B2 B1+实测搬运成本；B3再加专家历史。只先实现B2，B3须B2残差支持 | 独立文档calibration/evaluation，低负载/近容量/一次bursty；比较request goodput、maxITL、TTFT、公平性和决策开销。不得proposed同时获得新量化、CPU计算或更多显存；事件数据仅可作诊断，无离线同trace反事实收益 |
| R4：自然超显存与论文验证，条件后续 | 单卡Qwen3-30B-A3B BF16之类真实超24–32GiB模型；新增快照之前核对精确大小、容器RAM和存储 | 当前盘剩约29.4GiB，不能容纳约60GB BF16权重及现有环境；需要先解决存储/内存，未授权采购。代表性runtime、连续到达、独立repeat、第二模型、free generation/质量检查。没有前期残差不进入 |

前三项关注一次最弱链路，不做普通cap全扫描。已存在专家union是可复用的结构诊断，不构成R0先决条件。若读union，先确保16层、实际行数、纯decode身份与graph采集有效；mixed-prefill被排除只能说明decode统计，不能据它判定prefill缓存污染。既有冻结阈值只报告原定义，不能扩大成所有paging的必要条件。

预算不捏造分钟数：R0复用现成四cell；R1先完成CPU依赖和接口检查再短租/用已有设备；R2每arm初始16–32请求、两次交错重复，先测实际engine启动与单episode时长，再给总GPU小时。模型下载和环境适配在计费前完成。一次pager不兼容是implementation/runtime结果，不是研究假说NO-GO。

## 6. 实际运行与复现

本目录 `native_resources.json` 为本次从历史raw重算，`new_gpu_executions=0`仅指该CPU重算。共享工作区中同期KV原生执行另见 `../20260912_kv_budget_r01/`，不要用这个零值覆盖其新数据。

```bash
# 在仓库根目录；输出必须是不存在的新路径。
python3 refine-logs/expert_saturation/experiments/admission_capacity/recompute_native_resources.py \
  --campaign refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01 \
  --output /tmp/moe-native-resources-review.json
python3 -m unittest discover \
  -s refine-logs/expert_saturation/experiments/admission_capacity \
  -p 'test_paging_cost_model.py' -v
python3 -m unittest discover \
  -s refine-logs/expert_saturation/experiments/admission_capacity \
  -p 'test_pause_ledger.py' -v
```

新KV结果的完整复算入口如下；首次未设置PYTHONPATH时因历史helper的 `metrics` import路径退出，补充冻结metrics模块路径后正常完成，未更改原始数据或科学分析逻辑：

```bash
PYTHONPATH=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/frozen \
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/analyze_kv_budget.py \
  --output-dir /tmp/moe-kv-budget-reanalysis
```

本次四cell分析严格校验raw/config/status、输入身份、safe29块算术、preemption模式、保存metrics逐字段一致，128/128完成。六项cost-model、五项pause-ledger测试通过；这是CPU工程结果。源码少量复用，无新增第三套scheduler。

既有GPU四cell真正入口（**当前流程在跑时不能重启**）是 `E/run_frozen_kv_remote.py`，端点/新output/remote-dir为CLI参数；科学输入仍为9月10日的execution.tar.gz。新实例不得直接运行写死11155的旧driver。认证通过已配置SSH会话，不在代码或命令文档存口令。

已有pager最小GPU命令来自已核官方源码，**本次未执行、须先有可用独立环境**：

```bash
# 位于已固定commit的WiSP checkout；PYTHONPATH只影响此条命令。
PYTHONPATH=src /path/to/isolated-pager-venv/bin/python -m pytest tests/test_wisp_state.py -q
# 若CPU-only被skip，结果是SKIPPED，不是pager通过。
```

该命令测试toy CUDA pager，不运行OLMoE，也不证明端到端兼容。WiSP的 `ensure_resident` 在单组working-set overflow时抛错，但其公开forward已在L1082起实现按当前route贪心分组、逐组加载和执行。因此不能把union超过scratch等同于整个pager不可执行。R2需确认本机确实走到该分组路径并计packing/多次launch/重载成本；不减少top-k来跑通。

## 7. 可能的论文结构

1. **现象与模型**：KV增长、恢复等待、prefill干扰；分清普通容量配置与真实专家搬运外部性，给可重算账本和适用域。
2. **最小机制**：相同cache/pager下的准入和prefill两动作；先展示强通用基线漏掉了什么，再介绍新增成本项，删去无增量的路由信息。
3. **实际runtime**：单一主runtime/请求账本，给合法动作位置、状态更新、成本及fallback；线上版本不借drained API宣称逐步控制。
4. **同预算实验**：主表goodput/TTFT/TPOT/maxITL/失败，消融B0–B3，连续负载与自然超显存模型，报告全部repeat和失败边界。

若只有稳态/恢复测量规律，则论文收敛为measurement+model，不虚构最小机制章节的成功。单卡offload搬专家权重；EP搬token激活并有rank/collective/topology约束，本轮没有多卡结论。

## 8. 最终裁决更新

四项新KV运行已完成并全量回传。本次只读接收了共享工作区另一执行流程的原始结果，再独立调用冻结分析器重算，未启动重复GPU任务。完整数值见 [重算报告](kv_reanalysis/report.md)、[机器结果](RESULTS.json)、[原执行记录](../20260912_kv_budget_r01/execution.json)。原包SHA为 `e9649d8314d593d9c0b5a879483e471d8ddda7bd57482aa3acf9ead6d9c2880b`；四cell运行源码一致，除util外engine参数相同，12次warmup raw保留，128/128请求和池会计检查闭合。

| repeat | util | 实际KV GiB / usable blocks | 完成 | wall s | 抢占 / 重算token | 最大ITL s |
|---|---:|---|---:|---:|---|---:|
| 0 | .95 | 16.45703 / 8425 | 32/32 | 22.55161 | 0 / 0 | 0.08315 |
| 0 | .90 | 14.98438 / 7671 | 32/32 | 23.45206 | 2 / 7685 | 4.59064 |
| 1 | .90 | 14.98438 / 7671 | 32/32 | 23.33146 | 2 / 7685 | 4.55972 |
| 1 | .95 | 16.55078 / 8473 | 32/32 | 22.28443 | 0 / 0 | 0.09888 |

95相对90，吞吐 **+3.9928% / +4.6985%**，wall **−3.8395% / −4.4876%**。增配 **1.47265625 / 1.56640625 GiB**，而非从util直接换算KV。同一util的实际池随新引擎profile不同，所以逐cell报告。平均请求arrival→completion分别由21.03519→21.24891s、20.93874→20.97066s，增加 **1.0160% / 0.1525%**；不能用整批wall改善替代平均请求改善。TPOT中位数也增加0.2304/0.0878ms；TTFT与尾部改善不完全相同。结论是本域普通KV增配消除了观察到的秒级停顿并改善批完成，不能写成所有指标单调改善。

SSH只读查询核实该实例cgroup上限 **92GiB**、模型三个分片齐全、盘剩约29.4GiB。两次GPU快照显示100% utilization而没有可见compute PID；不能把空进程列表升级为全程物理隔离证明。新结果只作有限paired observation；环境局限由完整性复核进一步记录。

一次fresh same-family模型/查新复核结论为 `PROCEED_WITH_CAUTION / MEASUREMENT_FIRST / NO_METHOD_NOVELTY_YET`。其指出的未执行请求可被抢占/零工作改变phase、FCFS依赖输入顺序已作最小修复，新增两项回归通过；原“eviction test”改名为caller-supplied resident-set计费测试。**实际替换/重载模拟、在线action-specific forecast/cutoff接线仍未实现**，以缺口保留，不把CPU原语当完整优化器。时戳字段本身也不能证明forecast没有用未来信息。

| 结束字段 | 本轮结论 |
|---|---|
| Verdict | 普通KV问题已得到有限实测回答；paging主问题仍 `OPEN / UNRUN`，无method GO |
| Evidence type | 4cell `NATIVE_SERVING` 的同步in-process请求测量；历史账本CPU复算；来源核查；CPU模型原语 |
| What was measured | 相同cap/负载下真实KV池增配、128请求、抢占重算、完整完成和token间隔；旧128请求复算 |
| What was not measured | 真实expert pager/H2D、在线两动作策略、独立文本泛化、质量、第二模型、EP、生产SLO |
| Strongest baseline | 本次是native cap32/util90；未来必须同pager的token/KV/slack和locality-only基线，不能只比较弱默认 |
| Oracle/headroom | 同预算动作Oracle未运行；已有批吞吐改善来自更大KV预算，不是可部署策略headroom |
| Claim ceiling | 当前单模型/单GPU/两repeat的配置干预与描述性请求结果 |
| Failure category | 原秒级暂停支持普通KV容量/恢复路径解释；paging是未运行，不是NO-GO；通用算法骨架有强prior-art碰撞 |
| Resurrection condition | 真实超显存或固定缓存预算下观察到动作相关paging成本，且强普通状态基线漏掉请求级影响 |
| One next smallest experiment | 独立环境跑WiSP toy pager，再做1–2请求的OLMoE固定预算全驻留/受限池正确性与真实H2D成本对照；先不做controller |

**直接回答：下一笔GPU时间花在真实pager是否可用、搬运是否暴露的最小检查；下一批代码花在唯一请求账本内的load/evict/reload/transfer-wait事件和两动作接口。当前resident OLMoE的这段长暂停不再足以支持专家感知机制必要性；专家状态能否提供额外决策价值，仍需在同预算真实paging下回答。**

独立GPU结果复核：[`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md) 为 **WARN / same-family provisional**。来源、分母、归档/源码/输入/指标以及独立CPU重算通过；唯一未关闭项是连续GPU隔离未验证，不影响保留这次有限paired测量，但不支持生产泛化。四cell输出token逐请求一致也只适用于这组固定输出长度，不是质量、route/KV状态一致或同state反事实证明。
