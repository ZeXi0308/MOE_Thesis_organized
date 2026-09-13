# 请求准入、prefill 与专家 paging：资产、模型和下一次实验

2026-09-12启动，更新至2026-09-13。当前结论：**普通 KV 增配已消除已测resident负载的抢占长停顿；真实WiSP paging与0.26原生请求路径均已跑通。小prefill改变新旧请求等待，但0.26重复仍有wall排序翻转；旧token分组同层重复搬运约占实际payload的60%；已实测的expert互斥分组在同cap/KV下使四请求wall从3.20/3.37s降至1.48/1.23s，并消除该执行路径的同层重载。它是已有思路的强执行基线，尚有晚到请求max-ITL代价。强底座上的六组注入确认了小chunk的即时停顿与完整工作量代价；跨进程同动作完整耗时漂移47%，随后同实例四次重复差异2.94%。旧请求完成后解除chunk8的普通阶段规则已实测完整wall改善约9%，伴随后续decode搬运及max-ITL小幅增加；随后同release8底座四机制八次对照中，decode保留使搬运+0.41%、分组+30%，未改善完整请求；专家信息的决策增量仍未成立。** 本报告不更改 `docs/current/README.md` 的历史科学结论。

本轮读取了附件全文；本地 `agent/publish-current-moe-code` HEAD 为 `7059fc98e0d98a127ff4edda86d44f7f634d26f4`。实时 `git ls-remote` 显示公开分支仍为指定基准 `d65416c5cd09eb30aa6e46875339b16ebf7f5901`。9 月源代码与原始输出已在工作区，不应退回 8 月状态。开始时已有 expert-union 未提交改动；执行中另一个工作区任务增加了 paging/pause 模块和 KV runner，均保留。

已读权威及目标入口：`docs/current/README.md`、`docs/ideas/README.md`、`expert_saturation/EXPERIMENT_TRACKER.md`、`experiments/DECODE_CAP_BRANCH_GATE.md`、`experiments/gpu_pressure_sketch/README.md`、`experiments/admission_capacity/README.md`、原 KV `DECISIONS.md`、最新 feasibility report 和相关纠错记录。旧 gate 保留各自适用域；本次用户明确选择的普通 KV 与 paging 问题，不以完成无关的全部数值溯源为前提。

初始KV实验合同（已完成）：最弱因果环节是“新增实际 KV blocks 能否减少恢复等待并改善完整请求”；实验为同 cap32 的 `.95/.90/.90/.95`。允许结论上限为单模型单卡 native in-process 请求测量。首个 .95 初始化或充分容量资格失败则停止；完整指标混合则保留权衡；只有普通配置后仍有损害且真实 paging 动作存在，才进入固定 pager 动作面。禁止将旧局部 formulation 的失败扩写为整个调度家族失败。

## 1. 已有资产及实际调用路径

以下相对路径以仓库根为准：`E=refine-logs/expert_saturation/experiments/admission_capacity`，`O=refine-logs/expert_saturation/outputs/admission_capacity`，`I=refine-logs/independent_ideas_20260908/host_timing_boundary_r01`，`J=refine-logs/independent_ideas_20260911`。完整成功命令见 [CPU 复算记录](cpu_reanalysis/COMMANDS.md)。

| 实际文件／入口命令 | 状态 | 可复用内容 | 尚缺什么 |
|---|---|---|---|
| `E/run_native_capacity.py --help` → `native_capture.capture_episode` → 包装 `scheduler.schedule`，调用 `engine.add_request/step` | 已原生 GPU 实测 | 动态到达、每策略独立请求/KV/输出、实际 scheduled 内容 | 真实 paging、搬运成本与联合动作模块 |
| `E/admission_feedback.py` → `apply_nonpreemptive_limit` | CPU 测试＋原生实测 | 不截断已有 running；改变新请求准入 | 目标改变至实际 binding 的预测 |
| `E/metrics.py`、`E/analyze_capacity.py` | CPU 测试＋实测使用 | 所有请求、失败、host token receipts、TTFT/TPOT/ITL、完整墙钟 | 每请求 max-ITL 的显式新 SLO 需单独报告 |
| `O/20260908_native_preemption_r01/{native_capture,memory_telemetry,run_probe,analyze_native_preemption}.py`；`python3 …/analyze_native_preemption.py --output-dir NEW` | 4 个完整 GPU cells，128 次测量请求；本轮复算通过 | 首选账本：实际 blocks、allocation、preemption、成功无输出调用、重算与恢复等待 | H2D/expert events 关联字段 |
| `O/20260908_kv_safe_static_r02/safe_static.py` → `qualify_safe_cap` | 已 GPU 实测 | full-attention 单组、null block、真实可用块、逐请求取整 | prefix sharing/speculation 等其他布局不外推 |
| `E/kv_feasibility_admission.py`、`run_kv_feasibility.py` | 实现＋本轮 14 项 CPU 测试通过 | 最大 footprint 预留基线 | 通用 token helper 未完整覆盖 null/group/逐请求取整；safe29 实测不能冒充此动态模块的实测 |
| `I/prefill_budget_midpoint_r01/run_prefill_budget.py` | 512/1024 GPU 对照；4 测量＋4 warmups；本轮复算通过 | 引擎排空后改实际 token budget，共同编译能力 | 不等于在运行 episode 内逐步联合优化 |
| `J/per_request_prefill_share_r01/run_prefill_share.py`、`runtime/native_capture.py` → `set_empty_policy` | 12 测量＋12 warmups；本轮复算通过 | 排空后改 `max_num_scheduled_tokens`、`long_prefill_token_threshold` | 下一 step 逐请求 chunk actuator 未完成 |
| `I/waiting_order_r01/run_waiting_order.py`、`J/per_request_prefill_share_r01/runtime/waiting_order.py` | 初始两块及 repeat forward 原始数据在本地；repeat reverse 仅远端完成观察 | 只重排从未 scheduled 的 WAITING | reverse raw 未取得；不能控制 preempted/recovering 优先级 |
| `experiments/gpu_pressure_sketch/route_pressure_sketch.py` | GPU 类有实现；本轮 5 项 CPU 测试通过；非测试调用者为 0 | compact per-layer max/active、last-token signatures | 原生路径未接入；无 GPU 开销或收益证明 |
| `experiments/run_vllm_decode_cap_branch.py`；`… --help` | 8 月初始 cohort 分支实现 | OFF/ON、完整请求身份与初始 cap | 不是 9 月持续到达、实时 cap 结果的执行入口 |
| `O/20260912_feasibility_envelope_r01/analyze_envelope.py` | 64 个旧 episode 的 CPU 条件重建 | decode 成本桶、mixed/pure 成本关联 | 前瞻动作预测、独立文档、TTFT/max-ITL 验证 |
| `E/expert_union_tracker.py`、`run_expert_union.py` | 既有在改实现；本轮开始时 GPU 采集 UNRUN | 按 layer/step 的结构并集 | 不能充当真实 misses/bytes/暴露时间；已有截断采集不用于吞吐 |
| `J/natural_prefill_tails_r01` | CPU 准备，GPU UNRUN | 完整自然文章、固定 token 输入 | 自然尾部的实际 native 观测 |

9 月代码不是“未取得”；确实缺的是 waiting-order reverse 的本地 raw、自然尾部 GPU 测量，以及真实专家 pager 与请求账本的联合路径。StableBatch 选择器、JoinStream 当前重叠机制、固定 RankLane 的原结论原样继承。

本轮独立复算的历史结果：

| 对照 | 两次重复的结果 | 允许解释 |
|---|---|---|
| native32 相对 safe29，3072/1024 | 吞吐 +16.99%/+17.08%；native wall 23.161/23.311 s，safe29 27.095/27.293 s | 保守预留牺牲准入与总体吞吐；不等于所有抢占都好 |
| 同上长停顿 | 4.473448 = 4.357883 + 0.115565 s；repeat 为 4.512153 = 4.396435 + 0.115718 s | 分别是首重算调用前等待、重算调用跨度；后者不是纯 GPU 时间 |
| token budget512 相对1024 | pooled ITL p99 −29.90%/−29.48%；平均 TTFT +6.36%/+7.80%；wall +2.12%/+2.38% | 明确的指标权衡，不能宣称总体加速 |
| per_request512 相对 native1024 | 平均完成延迟 +1.57%/+0.51%；wall +1.13%/+0.64% | 已测动作未改善完整请求；不再扫该阈值 |

### 最新可行性模型的解释修正

本轮复现 `τ=c(w)/(1−λ·tax)` 的 3.25% 中位相对误差、6.09% p90、60/64 判决一致。但验证输入含**事后实际宽度中位数、完成 episode 内的 prefill step 频率、实际平均 prefill 量**；跨 campaign 校准仍复用同 32 篇文档。

因此这是“给定已实现工作量后的平均成本重建”。不能称动作前预测，更不能从稳态必要条件推出所有策略无 goodput 空间、任何预测器都不够准、或 cap24 在低负载必定失败。低负载实际宽度可能远小于 cap。原 report 不原地修改；本节是独立解释补充。10.15 req/s 和 98.5 ms 只能作为待检验模型推论，不能替代本轮长上下文 KV 对照。

## 2. 最接近的相关工作与实际开源范围

核查截止 2026-09-12。论文性能均为作者报告，本轮没有重新执行其实验。**已有工作不只处理给定负载的放置：Gimbal、Layered Prefill、QLLM 和 SLOWeave 已直接改变新旧请求的执行关系。**

| 工作／版本 | 控制对象、可见信息 | 资源约束、动作与目标 | 已发表实现和与本题的差异 |
|---|---|---|---|
| [WiSP v2](https://arxiv.org/html/2606.21868v2)，8月30日 | 当前路由、历史复用、KV peak/floor | 专家 LRU paging、expert/KV 池划分；边际延迟/byte | **公开 v0.2 已发布 controller**，README 落后于代码；当前 serve hook 在 drain 后调池。论文低并发3090；不是固定池下候选 admission/prefill 对在服请求影响的验证 |
| [FluxMoE v3](https://arxiv.org/html/2604.02715v3)，9月10日 | active/unfinished batch、materialization profiles | GPU未压缩/无损压缩/host分层与驻留规划，vLLM0.23 | §6.5 已有真实 Poisson 在线服务；§6.4 动态扩 KV 以41次引擎启动模拟，排除重建 prefill/重启开销。公开实现链接未核实 |
| [MoE-Gen v1](https://arxiv.org/html/2503.09716v1) | 硬件曲线、模块 ready 工作 | weights/KV/激活与 PCIe；模块合批和配置搜索，离线吞吐 | [原仓库](https://github.com/EfficientMoE/MoE-Gen)已重定向 [BatchGen](https://github.com/batchgen-project/batchgen)；当前main不可自动视为原论文复现。搬运摊薄和模型搜索不新 |
| [MoE-Lightning](https://arxiv.org/html/2411.11217v1) | HRM、硬件/内存/带宽 | CPU/GPU/IO流水、placement、batch配置，主要离线吞吐 | [作者 artifact](https://github.com/caoshiyi/artifacts/tree/asplos25)有实现；性能模型本身不足以构成贡献 |
| [Layered Prefill v2](https://arxiv.org/html/2510.08055v2)，MLSys2026 | decode/prefill队列、token/长度与KV | layer groups交错prefill和decode；TTFT/TBT | [公开 nanovllm scheduler](https://github.com/scale-snu/layered-prefill/blob/main/nanovllm/engine/scheduler.py)为decode-first；主要减少HBM读取，不能写成PCIe paging；非0.26直接patch |
| [DuoServe-MoE v2](https://arxiv.org/html/2509.07379v2)，4月9日 | 专家历史、popularity/affinity | 两阶段加载、缓存与预取，吞吐/延迟 | 主要single-request，另有batch扩展；公开代码未核实。“区分prefill/decode”没有独立新颖性 |
| [FreeToken v1](https://arxiv.org/html/2608.16157v1) | 当前miss、带宽、cache、agent状态 | CPU执行/GPU fill划分、LRU、双buffer和弹性池 | [公开runtime](https://github.com/FlashML-org/FreeToken)，含5090实测；当前prefill优先、resize仅idle。加入这些能力必须分别消融 |
| [Gimbal v1](https://arxiv.org/html/2606.15177v1) | running/waiting prefill、KV、近期EP压力 | DP请求分派、SJF+aging、expert placement/migration | 4×H100 DP2/TP2/EP4；明确pressure-aware admission。分布式激活/热点成本不同于单卡权重paging |
| [QLLM](https://arxiv.org/html/2503.09304v1) | LS/BE与两阶段优先级 | 专家级抢占BE以保护LS | A100/HF原型，开源列未来计划；用户最小动作不主动暂停已有decode |
| [Diff-MoE](https://doi.org/10.1145/3712285.3759903) | 已确认高/中/低优先级专家缓存和predictor | 详细admission/batch/priority定义未验证 | ACM全文本轮仍不可达；[作者实验室公告](https://grid.hust.edu.cn/info/1053/2965.htm)不足以排除或确认完整动作重合 |
| [SLOWeave v1](https://arxiv.org/html/2609.07883v1)，**9月7日新发现** | 在服请求last-token/deadline、pending prefill、单调cost table | 选择满足最早decode deadline的最大chunk | simulator＋论文声称iteration GPU验证；代码/raw未取得。直接碰撞普通deadline-aware chunk方法，尚未包括固定pager的共享miss/驱逐成本 |

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)已建立chunked prefill与decode共置基线；[2602.02987](https://arxiv.org/html/2602.02987v1)也建模prefill admission对decode减速。不能把“准入会干扰旧请求”当作新现象。可检验剩余项是**相同普通状态下，pager引入的动作依赖成本是否改变决策并带来净请求收益**，目前不能给“first/novel”结论。

### Pager 复用决策

**当前主底座为vLLM0.26.0原生Triton＋复用WiSP固定cap/LRU**。先在隔离0.11.2跑通WiSP作为兼容校准；随后复用另一工作区任务已实测的0.26桥接，避免重复移植。0.11与0.26时间不合并。最小策略仍输出admission/prefill动作；当前先清算同层重复加载这一已测实现税，不同时引入预取、CPU执行、动态分池或新Controller。

WiSP 作为 pager substrate 的首个候选：[v0.2](https://github.com/nokia-applied-research/WiSP/tree/v0.2)指向 `8ab27cc39dd7ad5be2cfc95a9cc24f4de5d16340`；当前HEAD `86f69720f0de0647c51728ea2e27e4289bf3dea2`。已核对 `plugin.py → install_wisp_serve_controller → serve_hook.py/controller.py`，源码确实发布，不能再说缺 controller。`controller.py`是peak/floor/headroom简化规则，不等同于完整边际估计器。

直接迁移有明确接口差异：[WiSP paged forward](https://github.com/nokia-applied-research/WiSP/blob/86f69720f0de0647c51728ea2e27e4289bf3dea2/src/wisp/integrations/vllm/fused_moe.py#L990)预期router logits/top-k配置；[vLLM0.26 unquantized forward](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py#L337)接收已算出的topk IDs/weights。WiSP仍pin0.11.2。该差异通过独立0.11.2接入和随后0.26桥接实测解决；保留版本边界，不将新的源码字段追认为旧运行已有采集。

FreeToken 当前HEAD `953565667f3141c90d0f0eb469bb2655d2407140`；其[调度主循环](https://github.com/FlashML-org/FreeToken/blob/953565667f3141c90d0f0eb469bb2655d2407140/python/freetoken/scheduler/scheduler.py#L831)先prefill，缺prefill才decode；[prefill admission](https://github.com/FlashML-org/FreeToken/blob/953565667f3141c90d0f0eb469bb2655d2407140/python/freetoken/scheduler/prefill.py#L50)预留input+output KV，未见按H2D/deadline选请求。Qwen3已注册，OLMoE未找到。它有5090路径但意味着换runtime/账本，暂不同时移植。若WiSP适配成本过高，再明确切换并重建所有baseline。

## 3. 固定研究问题与当前机制假说

**主问题：在单卡、固定专家缓存与KV预算下，怎样安排新请求准入和prefill工作量，减少新增工作对已有生成请求的等待损害并提高有效服务量？** 当前机制假说是共享加载、驱逐和重载可能改变普通KV/deadline模型给出的动作排序；这是待验证的机制，不与整个问题等同。

| Idea Card字段 | 本轮定义 |
|---|---|
| Problem / regime | 单GPU，模型或受控专家池不能全驻留；实际CPU→GPU搬运暴露，在线到达 |
| Scarce resource / signal | 固定KV blocks、专家池、有效H2D带宽；现在的residency、transfer queue及已完成历史 |
| Action | 少量合法候选请求入场＋下一step各prefill token量 |
| What changes | 后续batch、KV增长、请求首token、专家工作集和缓存轨迹；更大batch也可能摊薄成本 |
| Invariant | 权重、dtype、top-k、GPU计算、cache容量/LRU、硬件；已有decode不由policy主动停顿 |
| Target / denominator | 同到达cohort的request SLO-goodput、TTFT/max-ITL/完成权衡；arrival至完整观察终点 |
| Strong simple / closest prior | tuned static与普通KV/length/deadline查表；SLOWeave、WiSP、Gimbal |
| Potential residual | 资源固定时需求由准入内生产生；共享miss与驱逐损害能否影响完整请求决策 |
| Cheapest falsifier | 固定pager，少量真实动作面；普通状态matched后检查exposed搬运差异与动作收益排序 |
| Positive / negative | 正：有可复现且可提前利用的动作残差；负：仅成本表征或通用serving问题，不加predictor |
| Reopen | 新硬件/模型带来真实paging暴露，或不同可执行动作消除已测固定税；不靠换seed/阈值复活 |

用户后续要求持续推进同一问题，因此不再维护独立备选选题。resident下的KV增长与恢复等待是已测运行域的边界证据；若专家历史没有增量，可以修正成本模型或执行机制。只有充分覆盖目标问题的证据才允许换题，运行环境或单个selector失败不足以判死。

选择首选的原因不是预期必胜，而是它改变了真实稀缺资源和动作成本，区别于旧cap/阈值扫描。主要反证风险：并集快速饱和、普通KV/长度已解释所有差异、H2D被隐藏、host同步/图捕获税吃掉收益、路由历史没有提前可选性、最近工作已覆盖动作。以上都可产生窄负结果。

## 4. 最小系统和优化模型

### 状态、可见信息与合法动作

决策点固定在本step尚未schedule之前。`S_t`包含：

- 每请求arrival、阶段、prefill完成数、输出数、实际cached token/blocks、last-token时间；`W_new`、prefilling、decoding、recovering、completed分开。
- 实际token budget、engine上限、准入目标、scheduler running/waiting、usable/free KV blocks及block size；共享prefix和speculative reservations初版关闭。
- `(layer,expert)`驻留集合、固定LRU顺序/容量、正在传输的权重与queue；已完成step的统计。当前已router的ready任务身份可用，未来层/未来动作route未知。
- 在独立校准集测得的GPU服务曲线、有效H2D bytes/s、copy启动时间、pager host开销及有依赖证据的hidden overlap。未测参数保持UNMEASURED，不以假数填满控制器。

动作 `a_t=(A_t,p_t)`：`A_t⊆W_new`、只含已到达且从未运行的请求；`p_t`为新/已有prefill的处理量。小实例先枚举0或1个新请求和离散chunk；已有decode各保留一个合法推进位置。合计scheduled tokens不得超过实际token budget，KV不得超池。降低目标只影响后续入场，不删除running/KV。native缺KV导致的preemption由引擎处理；no feasible action不是许可伪造恢复或丢请求。

### 容量守恒与请求转移

`M_nonexpert + M_expert_pool + M_KV_pool + M_workspace ≤ M_GPU`。

这些是互斥storage分类；Torch allocated/reserved、NVML与已分类对象不能再叠加。逻辑KV占用位于物理KV池内。主机另核算canonical weights、pinned buffers、CPU workspace与cgroup限制；同一pinned权重buffer如果就是canonical存储，只算一次，存在副本则真实计入。

单组、无共享的初版：`K_t = Σ_i ceil(cached_tokens_i / block_size)`，要求 `K_t≤usable_blocks`。每step按真实prefill/decode/recompute allocation增长；完成或真实驱逐后释放。首个输出常由最后prefill产生，输出token数不能直接当作新增KV数。

短期需求预测使用已知prompt剩余、已执行输出、当前KV和一个**标明来源**的剩余长度估计；固定1024输出实验可用其公开上限，free generation不可偷看实际结束位置。保守最大长度预留是baseline而非唯一真模型。未来完成释放是估计，下一真实step重新校正。

preemption后保留逻辑历史但释放实际KV；recovering期间等待时间累积，重算只恢复历史，不凭空产生第二份输出。账本区分新入场、preempt、首重算调用、恢复后首新token。模型内若为recovering保留逻辑准入额度，这是保守policy约束，不等同于native `len(running)`。

### 共享搬运与时间

已确定的一层执行组：

`D_l(B,C)=Σ_{e∈U_l(B)\C_l} weight_bytes(l,e)`。

每请求独立加同一专家成本会重复计费。跨layer或不同执行组不能盲目dedup。`D_l`只是每个miss最多加载一次的下界；实际bytes还需记录容量不足的分组重载、先驱逐后再需要、padding/粒度、prefetch和其他复制。纯LRU dry-run只能验证这套会计，不能当真实backend行为。

均匀独立top-k分析基线 `E[U]=E·[1−(1−k/E)^B]`。OLMoE的E64/k8下，B1/4/8/16/32期望为8/26.48/42.01/56.44/63.11。**这是公式值，不是实测。** 即使并集饱和，batch仍可能摊薄全层搬运，层间流水仍可能存在；饱和只削弱某些expert-identity区分度，不否定所有paging/合批。

先用同配置mixed-iteration查表建立 `T_base(n_decode, context, p, shape)`，再对实际依赖路径上的额外暴露搬运和pager开销建模。可写：

`T_hat = T_base + transfer_exposed + pager_host + policy_overhead`，

但只在`T_base`未包含这些paging项时使用。若直接测joint pager iteration，则用joint LUT；不再次加load span。`transfer_exposed`需要copy依赖与GPU等待的交集，不能简单把profiler span总和相加。microbenchmark可估`n_launch*alpha + bytes/beta`，在线还需transfer queue、竞争与overlap校正。

完整请求会计：`completion−arrival = first-token前等待/执行 + first-token至completion`；内部可互斥分成queue、execution、恢复等待等bucket，总和不可重复。长ITL内的重算call跨度已在请求墙钟中，禁止再加一次。

### 有限动作滚动选择

先做1步或数步小枚举，不训练复杂predictor。各动作从当前状态独立预测，保留KV增长、完成释放和缓存状态；执行仅第一步，随后用真实状态更新。CPU模块只做转移与合法动作验证，**本轮未实现这个在线求解器**。

设计目标可写为有限观察窗口内完成且通过SLO的请求数减去未达标/拒绝/超时成本，subject to容量、decode正常推进和等待上限。短视实现优先排除会使已有decode剩余slack不足的动作，再在可行集合中比较prefill/新入场进度和未来KV风险；无法满足所有SLO时显式记录violations/等待，不能静默丢请求。误差margin来自校准，不提供无条件deadline保证。

SLO分别声明 `TTFT_i≤T_i`、`mean_TPOT_i≤P_i`、研究暂停时再加 `max_ITL_i≤G_i`；mean TPOT不等于每token硬截止。主goodput为同一到达cohort中completed且全部目标达标的请求数除以固定窗口长度；有限cohort另报arrival至最后完成/超时的drain吞吐，不冒充稳态容量。失败和未完成全部保留，未来未到达请求在abort时另列。请求级小样本p99只作描述统计。

### 本轮可运行代码

[resource_transition_model.py](../../../experiments/admission_capacity/resource_transition_model.py)是dependency-free immutable observed-event reducer和有限动作枚举器。已有观测驱动admit/prefill/decode/preempt/recompute/complete；不会生成假latency、route、token或性能曲线。另提供唯一加载下界、LRU重载会计、uniform基线、estimate可见时间cutoff和暴露时间核算。

对应[测试](../../../experiments/admission_capacity/test_resource_transition_model.py)仅覆盖状态独立性、身份/时间、KV/token预算、恢复与共享加载会计。它不接入vLLM；不支持prefix共享、投机预留和native恢复调度。时间cutoff检查不能证明调用者没有伪造预测来源；真正online验证仍需运行时输入日志。

## 5. 第一批实验与每种结果的含义

| 顺序 | 配置与对照 | 必须采集 | 如何影响下一步 |
|---|---|---|---|
| **K0 已完成** | cap32、3072/1024、50ms、budget1024、原生preemption；0.95/0.90/0.90/0.95；每cell新engine，3次相同warmup | actual池/物理storage、所有请求/step/KV/preempt/recompute、TTFT、TPOT、max-ITL、完成、失败，环境/进程 | .95改善全部→当前痛点优先普通配置；只消除暂停→记录tradeoff；仍暂停→定位非KV等待；.90没压力→本机无相同反事实；资格失败→不改阈值救结果 |
| **P0 已完成极小完整模型接入** | 独立vLLM0.11.2＋WiSP固定HEAD；两请求×8输出，vanilla/paged/paged-traced；同512MiB KV | 16层真实forward、完整token、miss/eviction、tensor加载bytes、KV/scratch/主机权重 | 三次measurement输出相同；支持该短任务接入，不是一般质量或方法收益 |
| **P1 实际执行与当前后续** | 先完成3请求事件注入及强执行基线；0.26 expert/chunk8、128与hold六组已完成。当前只做同实例chunk128四次重复，清空专家缓存并共同暖机 | 完整请求、实际copy、完整cache/LRU、逻辑KV和逐step CPU/runqueue等待 | 先定位同动作漂移；动作排序与开销可校准后，才扩大独立文档和普通状态对照，不继续盲扫chunk |
| **P2 必要时** | B0 tunedstatic；B1普通length/queue/KV/deadline；B2=B1+搬运成本；B3=B2+expert history，同pager同资源 | 每请求goodput、等待/超时/拒绝；模型误差、动作不同率、全部成本 | B2赢B1但B3无增量→贡献不依赖路由预测；B3独立holdout仍增量→才讨论专家信息；无净收益→停止该机制 |
| **P3 确认阶段** | 真正超出显存的Qwen3-30B-A3B BF16、持续到达；低负载/近容量/突发＋少量短长混合；独立文档/episode | 同上，补free generation与必要质量；2–3交错重复呈现波动 | 只在人工OLMoE池有效→限实现/模型验证；自然offload仍成立才有代表性结论 |

P1的动作面是初始条件匹配后各自实际执行；如果只在一次长trajectory中交错动作，历史缓存不相同，就按观察性pilot报告，不能称同状态counterfactual。未来route与KV不得从一条baseline重用。未来known Oracle不是起步必要条件；有限实测action surface足够判断是否继续。

第一个paging probe先关闭动态分池、预取和CPU执行。受控OLMoE预算须明确为人为限制，不能证明真实部署必须offload。随后[Qwen3 BF16](https://huggingface.co/Qwen/Qwen3-30B-A3B)约30B参数本身就要求远大于32GB的权重空间，但具体磁盘/主机开销需按snapshot与实现清点，不能套用WiSP约80GiB参考值当保证。

时间安排是上限估计，不是跑数：K0使用现成环境，4个fresh engines；后续P0先限定约30分钟兼容性排障，失败收窄问题；P1先1个低压力与1个暴露压力域、两个交错blocks，典型短租预留约1–2 GPU小时，实际以load/compile时长修正。确认前不展开所有模型×arrival×SLO组合。当前实例cgroup内存上限92GiB，数据盘初始剩余约29.4GiB，后续实测剩余19,635,216,384bytes、root overlay剩余27,749,675,008bytes，**不能直接在此盘再下载Qwen3 BF16**；先准备模型存储/主机容量方案，不自动购买扩容。

## 6. 本轮实际执行与复核

四个历史数据分析器退出0，结果保存在 [cpu_reanalysis](cpu_reanalysis/COMMANDS.md)。既有compact sketch 5项、KV-feasibility 14项CPU测试通过。新结构模块测试结果在 [CPU_MODEL_CHECKS.txt](CPU_MODEL_CHECKS.txt)。新结构模块13项测试通过；这些是CPU合同与会计证据，不是新GPU收益。

远端提供的5090、版本、3个模型分片内容哈希和空闲状态已实时核实，记录在 [readiness.json](kv_attempt/readiness.json)。旧9月10日runner的真实状态已为STOPPED_FOR_DIAGNOSIS，不再沿用旧WAITING文案。

首次上传被自动审批拒绝；用户随后明确授权上传代码、配置与请求输入，传输成功。解包时发现另一工作区任务已启动**同一个哈希的四cell KV实验**，故未重复启动。共享原始执行与回传入口是 [20260912_kv_budget_r01](../20260912_kv_budget_r01/execution.json)。本报告使用其已完整落盘、核对身份后的数据；不把另一任务启动的过程写成本代理独立复现。

### 完成后的KV结果

四个cell均COMPLETE，128/128次测量请求完成，12个warmup raw保留；是同32篇文章的重复执行。四个归档本地SHA256与执行记录一致，源码/软件核对通过，仅engine参数gpu_memory_utilization不同。本报告独立调用原分析器复算新raw，见 [完整KV表](kv_analysis/report.md)、[机器结果](kv_analysis/analysis.json)、[归档核对](kv_attempt/readback_verification.json)。

| 执行顺序 | actual usable blocks | KV物理GiB | wall s | req/s | 抢占 / 重算token位置 | 最大ITL s |
|---|---:|---:|---:|---:|---:|---:|
| r0 .95 | 8425 | 16.45703 | 22.55161 | 1.41897 | 0 / 0 | 0.08315 |
| r0 .90 | 7671 | 14.98438 | 23.45206 | 1.36449 | 2 / 7685 | 4.59064 |
| r1 .90 | 7671 | 14.98438 | 23.33146 | 1.37154 | 2 / 7685 | 4.55972 |
| r1 .95 | 8473 | 16.55078 | 22.28443 | 1.43598 | 0 / 0 | 0.09888 |

两次within-repeat吞吐增量分别**+3.9928%/+4.6985%**，wall **−3.8395%/−4.4876%**。新增物理KV分别1.47266/1.56641 GiB；.95两次实际分配不完全相同，按实测报告，不用配置比例推算固定bytes。完整请求的p99完成延迟分别减少0.399/0.561s。

这不是所有延迟指标都变好：mean-TPOT的请求中位数分别增加0.230/0.088ms；TTFT p99第一组近乎不变、第二组减少约91ms。原参考TTFT5s/meanTPOT200ms不能刻画4.5s单次暂停，不根据新结果事后选择一个max-ITL阈值然后声称预注册goodput胜利。

**直接回答K0：在本机、该负载与两次重复内，普通KV增配已消除原生抢占及多秒恢复停顿，并改善完整批次吞吐。** 它不需要专家回收、路由预测或新scheduler；由于实际给了更多KV内存，也不构成同预算方法收益。现在没有证据以这组resident痛点为理由启动专家Controller。paging研究只在真实offload的新运行域，以固定资源baseline重新证明成本增量。

复算命令（仓库根执行；输出目录必须新建）：

```bash
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260910_kv_budget_r01/analyze_kv_budget.py \
  --results-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/gpu_results \
  --execution-archive refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/execution.tar.gz \
  --output-dir /tmp/moe-kv-budget-independent-recompute
```


### WiSP最小GPU primitive检查：6/6通过

KV四项结束后GPU空闲，本轮在独立`/root/autodl-tmp/wisp-pager-smoke-20260912/`目录运行**未修改的upstream测试**。只向该目录的`deps/`安装pytest8.3.5，以PYTHONPATH导入固定commit源码；借用已有Torch2.11/CUDA13解释器，没有将WiSP注册到现有vLLM环境，也没有加载模型或调用其插件patch。

固定缓存相关6项通过，3项resize测试明确deselected。验证pinned CPU专家内容加载至GPU scratch、miss/hit、LRU victim、当前needed专家保护、capacity overflow、随机序列map一致性及resident identity。测试使用8个极小FP32人工专家，不是OLMoE/BF16服务；**不能推出有效H2D带宽、完整模型质量、吞吐或vLLM0.26接入兼容**。证据为GPU synthetic primitive correctness，原始日志见 [tests.log](pager_smoke/readback/tests.log)、[执行记录](pager_smoke/readback/execution.json)、[源文件身份](pager_smoke/source.json)。测试前后GPU compute进程均为空，回传归档SHA256已核对。

可重跑的实际命令（该主机已存在的隔离目录；重复时另设junit输出文件）：

```bash
PYTHONPATH=/root/autodl-tmp/wisp-pager-smoke-20260912/deps:/root/autodl-tmp/wisp-pager-smoke-20260912/source/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WISP_PLUGIN_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -m pytest \
  /root/autodl-tmp/wisp-pager-smoke-20260912/source/tests/test_wisp_state.py -v -k 'not resize'
```

现在P0剩余的是**完整模型与runtime接口**，不是重新测试LRU。下一笔GPU时间应在隔离的受支持runtime中完成极小OLMoE/BF16 pager off/on，再决定是否移植0.26，不能把本次primitive PASS升级为已经跑通serving。

### 后续执行更新：完整OLMoE/BF16已接入，真实加载已测

独立安装vLLM0.11.2、Torch2.9.0/cu128、Transformers4.57.1；已有0.26环境保留。复用WiSP HEAD `86f69720f0de0647c51728ea2e27e4289bf3dea2`，固定cap16/LRU、关闭prefetch/dynamic/prefix cache。完整原始结果与解析见 [integration_analysis.json](../20260912_wisp_olmoe_r01/integration_analysis.json) 和 [原始归档](../20260912_wisp_olmoe_r01/integration-readback.tar.gz)，归档SHA256为 `7e4689fecc22a825a137121daebff99dd0c70329f93c65a7051de40ebf3e0c92`。

三次新进程均成功，各保留2个warmup与2个measurement请求，每请求输出8token。vanilla、paged、paged-traced的两个测量输出序列完全相同。pager的16层都真实执行，测量共128层调用，1750次miss与1750次eviction。实际KV物理池均536,870,912bytes、256blocks；paged GPU scratch为3GiB，pinned CPU expert master为12GiB。OLMoE本可常驻，限制专家池是人为成本诊断域。

| 测量执行 | 两请求wall | 最大ITL | 解释 |
|---|---:|---:|---|
| vanilla | 0.065251s | 0.009208s | 权重全部常驻，不能用此比值声称调度方法收益 |
| paged cap16 | 0.581240s | 0.051787s | 实际专家加载运行域 |
| 相同paged＋trace | 0.597461s | 0.053081s | 一次on/off差值约2.8%，尚不是稳定观测开销估计 |

首次paged warmup为70.614s，全部保留；其后measurement为0.581s，不能将首次warmup长暂停当成稳态paging代价。带trace测量的实际权重copy payload为22,020,096,000bytes，逐层入口缓存下界15,992,881,152bytes，额外6,027,214,848bytes。**全部额外加载集中在13个物理token的prefill调用**：11,953,766,400bytes实际加载，对应5,926,551,552bytes唯一缺页下界；后续2-row decode调用的额外项为0。这定位的是固定cap下的分组/驱逐加载成本，不是每个请求的独立加载费。

CUDA load-section合计421.386ms；其中含expert-map更新及可能的host launch空隙，不能称纯PCIe耗时或独立相加的总成本。host route-read spans与GPU工作可能重叠。trace目前只关联engine call，request到physical-row映射仍`UNAVAILABLE`；因此可解释layer-batch成本，不能把共享加载归给某个request。

接下来已启动固定专家/KV资源的三请求受控注入：两旧请求各输出4token后加入新prefill，比较不新增、short8、long128/chunk8、long128/chunk32，正反顺序各一遍。各进程独立演进并保存共同前态及完整LRU；这是调度动作诊断，尚未产生方法结论。最新命令和状态写在 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

### 三请求动作结果与运行漂移清算

上述8个traced cells全部完成。共同前态逐字段相同：两旧请求各输出4token、computed35/prompt32、完整slot/map/LRU ticks/clock、KV blocks与free249/256均一致。实际注入步为2/10/34token；每条已有decode始终各推进1token，无抢占/重算。short8与long8首步的专家集合与实际加载量完全相同，负控吻合；short8该步还完成prefill并sampling，所以只作即时shape校准。

long8首步实际加载6,706,692,096bytes，long32为19,478,347,776bytes。但整段分别为163.090GiB与159.012GiB，**小chunk没有节省总加载量**。各自重复的所有调用shape、逐层/分组专家集合及加载bytes一致。long32旧请求的后续输出在第10/11token后与long8发生差异；各arm内部重复一致，使用的是各策略独立生成轨迹，不是固定future replay，也未形成任务质量结论。

traced long8的wall为6.049/4.492s，long32为4.229/4.300s。随后4个无trace cells按32/8/8/32执行：long8为5.703/4.363s，long32为5.323/5.350s，完整墙钟排序发生翻转。原始不利结果全部保留。测量期GPU memory clock为13801MHz、SM主要2872MHz、CPU cgroup无新增throttling；这些新采样不能追溯解释旧run，也没有证明漂移原因。

主机为双NUMA Xeon Gold6459C，原进程可用CPU0–127，GPU邻近NUMA0。下一组仅将两臂实验进程都绑定到CPU0–7，保持OMP8、GPU/模型/专家/KV不变，仍使用原WiSP无trace。两次重复得到一致权衡：

| 固定CPU后的cell | 3请求wall s | 旧请求max-ITL s | 旧请求完成 s | 新请求TTFT s | 新请求完成延迟 s |
|---|---:|---:|---:|---:|---:|
| r4 chunk32 | 5.3961 | 0.8229 | 5.3961 | 3.0674 | 3.8737 |
| r4 chunk8 | 5.8161 | 0.3087 | 4.5593 | 4.0064 | 4.4023 |
| r5 chunk8 | 5.7843 | 0.3073 | 4.5310 | 4.0401 | 4.3721 |
| r5 chunk32 | 5.2683 | 0.8101 | 5.2683 | 3.0296 | 3.7932 |

完成延迟均从各自arrival扣除；两旧请求对称，表中列第一条。CPU allowlist确为0–7，hardware采样窗口只出现预期model worker的GPU PID。记录到的anonymous mappings大多在N0，但只有约1.4GiB，**不足以定位12GiB pinned expert master**；Mems_allowed仍为0–1。本轮是CPU放置控制，不是已验证的membind，不将稳定性反推为“NUMA是唯一原因”，也不把绑定带来的变化当scheduler收益。

累计16个注入episode、46次measurement请求执行全部完成，复用3篇文章，另保留32条warmup请求；这不是46条独立样本。原始数据分别在 [injection](../20260912_wisp_olmoe_r01/injection/)、[plain](../20260912_wisp_olmoe_r01/plain/)、[affinity](../20260912_wisp_olmoe_r01/affinity/)。三份归档SHA256分别为 `053b33de8c2409584a1c57856287220faa265937767ff883363cff46d7da7675`、`4b20e673b314b952a6220ee2b47d5a86dae056255ff343328123e256f209d3a9`、`f43695da1bd5d7d51401935786bbb1c5a92dde828c2cd4e5c9e9dacd73ed7384`。fresh targeted复核四项核心完整性检查P0=0/P1=0；WARN是漂移和证据范围，不追加审计流程。

**本轮回答：新prefill真实改变专家加载与已有请求等待，小chunk能保护旧请求生成间隔，却增加新请求等待及总搬运；不是无代价优化。** 模型必须保留batch共享/重复加载和运行时成本，不能用一次测量直接拟合可部署policy。当前只比较两个静态chunk，还没有最强静态点、普通反馈与新增机制的完整对照，更没有统一SLO-goodput结论。

完成一次设计peer review及一次**fresh same-family GPT-5.6-Sol ultra**复核。后者结果为 **K0 integrity PASS；总体WARN仅针对范围；P0=0/P1=0，acceptance=provisional**。核实四cell顺序、128请求、12warmups、归档/源码/模型身份、互斥重算会计、指标实际调用和全部数值；独立identity/output哈希spot-check也一致。论文查新边界被接受为谨慎表述，未给予method novelty或paging GO。

复核限制：使用现有完整validator加raw spot-check，而非另逐token重写分析器；GPU进程隔离只在边界检查；同32篇文档两次重复；该次K0审计发生在完整runtime集成前，只覆盖当时6项primitive；后续完整runtime证据单独列在上文。摘要与输入哈希见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)。按用户要求，review结论合并在本报告，不生成额外`.aris`或另一套审计文档。

### 0.26固定资源prefill复核与强执行基线

复用已验证的0.26适配入口，完成default/prefill32/prefill32/default四个cell，每cell4请求、64输入/8输出、50ms名义到达，共16次测量请求执行；CPU0–7、OMP8、cap24专家scratch4.5GiB、实际KV1GiB。原始结果见 [native_transfer](../20260912_wisp_olmoe_r01/native_transfer/results/)，归档SHA256 `321788ea162c24b28d80c2b39ced07d2e2e24f116b95a3c62a9f30c688420a21`。

| 同一策略两个repeat | 完整wall s | 实际专家copy bytes | 同层调用中可识别重复copy bytes |
|---|---:|---:|---:|
| default | 3.6359 / 3.3590 | 124,897,984,512 | 74,956,406,784 |
| prefill32 | 3.3573 / 3.6634 | 126,156,275,712 | 76,919,341,056 |

两策略各自的group/byte计数一致，但wall排序翻转；CPU绑定未消除所有漂移。每次都执行相同配置的完整暖机，但该归档版本未记录measurement入口cache/LRU和每组required experts，不能从更新后的源码补认“前态逐字段一致”。重复copy由同一layer-call内`loaded_experts`多次出现直接数得；这不是可从缺失入口缓存追认的完整unique下界。该字段足以证明当前执行组织存在大量实际重载。

本轮唯一问题：**同样专家/KV资源下，按专家互斥分组能否把这些已测重载消掉，并在额外归约和launch之后仍改善完整请求？** 首组先服务active resident，再填missing，剩余missing互斥分组；只使用本层router后已知信息，保留所有top-k权重并分别调用原生partial kernel。每组专属map排除非本组resident，防止重复贡献。该做法是MoE-Gen/FreeToken相关思路下必须比较的强执行基线，不作新颖性包装；BF16 partial相加改变浮点归约顺序，需要同输入reference和完整生成检查。

配置预写在 [expert baseline protocol](../20260912_wisp_olmoe_r01/expert_baseline_frozen/protocol.json)。两臂以相同顺序暖机两种执行，再核对入口完整缓存；validation独立于性能。实际copy、额外buffer、映射/归约/launch均计费。语义或身份错误则定位修复；若净正，才在此底座继续旧decode与新prefill的剩余权衡。没有用任意百分比阈值判死主问题。

### Expert互斥分组：实际执行结果

新增149行模块复用WiSP状态迁移与vLLM0.26原生Triton，未新增expert GEMM kernel。4项CPU分组检查通过；一次定向源码复核未见P0/P1。先独立跑4请求资格，再按token/expert/expert/token顺序完成4个性能cell，全部原始结果见 [expert_baseline](../20260912_wisp_olmoe_r01/expert_baseline/)。

资格覆盖16层首个测量调用全部64行，16/16 all-finite、诊断allclose(rtol/atol均0.01)，0/16逐位一致；max-abs跨层最大0.03125、relative-L2范围0.0001584–0.0047597。此检查不是通用质量保证，资格wall和临时fullweight峰值不计性能。所有性能cell的4个请求共32个输出token完全相同，两种共同暖机输出亦一致。

| 性能cell | 4请求wall s | 实际copy bytes | group数量 | CUDA allocator测量峰值 bytes |
|---|---:|---:|---:|---:|
| token r0 | 3.204255 | 124,910,567,424 | 994 | 6,889,734,144 |
| expert r0 | 1.482506 | 37,685,821,440 | 303 | 6,893,975,552 |
| expert r1 | 1.232186 | 37,685,821,440 | 303 | 6,893,975,552 |
| token r1 | 3.366169 | 124,910,567,424 | 994 | 6,889,734,144 |

实际KV去重tensor storage均1,073,741,824bytes、16块storage；512个物理block、排空free511。专家scratch均4,831,838,208bytes、host pinned master12,884,901,888bytes。expert模式测量allocator峰值多4,241,408bytes，已经计入；这是固定权重/KV资源，不能表述成额外workspace为0。expert重复仍有20%左右wall波动，全部保留，未选最好一次。

四个cell入口完整slots/LRU ticks/clock相同，warmup结构与输出相同，所有request-position行相同；每个策略内部重复的分组/加载/输出一致。跨执行方式34/160个layer-call的required union不同，首次为step0/layer4：分组归约改变了后续数值/路由。因此是每策略独立真实执行的完整请求对照，**不是固定route的纯复制反事实**。每次expert call实际loaded恰为自身入口missing集合、同层重载为0。

四请求TTFT与完成时间均改善；第一条已有请求max-ITL由1.489/1.478s降至0.251/0.234s。后两请求的max-ITL却由82.685/110.306ms升至138.767/130.465ms，不能写成所有延迟指标都更好。前三个prefill/mixed steps的实际copy从108.33GB降至20.98GB，后续decode仍有约16.7GB加载与额外partial/map成本。这说明prefill执行税是主要收益来源，decode成本和最终缓存状态仍有剩余问题；phase-aware执行属于随后应考虑的简单基线，不把本轮包装为新调度贡献。

随后完成的实验检验强底座上的真实新旧干扰：两旧请求P32/O16均输出4token后释放第三P128/O8，固定expert模式/cap24/KV1GiB，比较chunk8、chunk128与no-new负控；首步应分别10、130、2tokens，enginebudget160。两个chunk共同暖机，记录事件处完整缓存/KV与准确step起止，snapshot成本计入ITL/TTFT。该接口补充到现有native请求账本，不新增Controller或另一套runtime。

### 强底座上的事件注入：完整权衡与模型误差

六组按hold/8/128/128/8/hold完成，共16次measurement请求执行、224个输出token；仍是重复3篇文档。原始归档SHA256 `82c83284ab453f3cce544f10ec6a0070d783673301a8fd4a9b5fe1cd2e348960`，详见 [native_injection/analysis.json](../20260912_wisp_olmoe_r01/native_injection/analysis.json)。分析核验六组身份、事件前完整slots/map/绝对LRU、旧请求前4token、逻辑KV及实际1GiB storage一致，首动作实际为2/10/130行；逻辑KV一致不证明KV tensor逐位相同。

| Cell | 首边界ITL ms | 旧请求事件后max-ITL ms | 完整wall s | 新TTFT s | 新完成延迟 s | 全程copy GB |
|---|---:|---:|---:|---:|---:|---:|
| hold r0 | 68.093 | 76.783 | 1.301841 | — | — | 20.849885 |
| chunk8 r0 | 93.662 | 179.129 | 3.075969 | 2.301879 | 2.764289 | 78.630617 |
| chunk128 r0 | 234.996 | 234.996 | 1.587038 | 0.234996 | 0.851990 | 36.339450 |
| chunk128 r1 | 181.630 | 181.630 | 1.077920 | 0.181630 | 0.593138 | 36.339450 |
| chunk8 r1 | 120.911 | 173.941 | 2.932569 | 2.215774 | 2.460872 | 78.630617 |
| hold r1 | 70.716 | 77.398 | 1.275771 | — | — | 20.849885 |

两旧请求本轮时间对称；事件后max-ITL包含首边界；新请求指标均减实际事件arrival，GB为十进制。hold只有两请求，不能拿它的wall计算同任务吞吐。每个策略两次记录的required union/groups/bytes和输出完全一致；跨chunk在step4/layer0并集即不同，旧请求0的输出从token12起不同。这是各策略独立状态演进的真实执行，未采每行top-k，不能称逐token路由或跨策略输出完全等价。

大chunk全程copy少53.78%，wall少48.41%/63.24%，但首边界与事件后max-ITL更长。小chunk并未消除干扰：它反复支付加载，且旧请求完成后新prefill仍未结束。没有统一业务SLO，本轮支持成本权衡，不能宣布goodput或Pareto收益。

运行前固定的均匀独立模型只使用E64/k8/C24与候选token数，预测首步copy分别1.887437/5.934499/8.053063GB；实测为0.893387/2.692743/7.449084GB，actual/predicted−1分别−52.67%/−54.63%/−7.50%。小步中的缓存复用与路由相关性不能忽略；该实验否定的是均匀独立近似在当前状态下的数值精度，尚未证明专家历史比普通近期时延有决策增量，也不是独立文档验证。

chunk128的慢/快wall相差47.23%，漂移在动作前已出现：首个相同64行/6.254GB prefill的engine时间206.43/157.37ms，CUDA load-section136.74/123.07ms；相同2行/0.793GB的前态decode为62.61/34.84ms，load-section19.88/16.80ms。漂移包含load-section之外的大量时间，不能只归因于PCIe或把这些重叠span相加。事件snapshot仅0.077–0.218ms，不足以解释该差异。

进一步对均匀miss误差作事后分解：令p=C/E=24/64，实际逐层并集U和entry hits H，则预测高估为`(1−p)(U_hat−U)+(H−pU)`，先替换并集，再替换命中假设。hold/8/128的并集项分别−0.062915/1.923696/0.401080GB，缓存项1.056965/1.318060/0.202899GB。该分解使用动作后的真实路由，只解释误差来源，不是前瞻预测。

另一个普通基线缺口已由账本定位：chunk8两次均在call15完成旧请求，新prompt只计算96/128；随后calls16–19在没有old decode时仍各执行8-token prefill，共搬运14,180,941,824bytes，engine时间0.549211/0.386878s。待同实例计时边界明确后，先实测“old全部完成即解除chunk cap、合并剩余32token”，与固定8比较；这是普通阶段规则，不构成专家历史贡献。不能把四步现有成本全当成潜在节省，因为合并步仍须真实加载和执行。

随后完成同一engine、同一chunk128四次重复：保留tensor分配，每次排空后将专家metadata/map清空并按相同顺序暖机；保存每段初态、完整结果和逐step CPU/runqueue计数。KV free-list不作内部手术，物理block差异如实记录。它区分进程初始化/内存放置和进程内运行漂移，不作为新调度策略，不通过更换chunk、seed或筛选repeat制造稳定收益。

### 同实例重复：计时边界收窄

固定chunk128的四个episode全部完成，共12次请求执行、160输出；每次排空并清空专家metadata/map，再走同一短warmup与8/128注入暖机。12GiB host master、4.5GiB专家scratch、1GiB KV的tensor/storage pointers保持不变，实际前态cache/LRU/device map与逻辑请求状态、后续记录的rows/union/groups及输出均相同。KV free-list与物理block IDs从repeat1起不同，未比较KV内容逐位等价。

| Repeat | 完整wall s | 首边界ITL ms | CUDA load-section ms | CPU观察包络 ms |
|---|---:|---:|---:|---:|
| 0 | 1.089326 | 184.348 | 735.274 | 2.856 |
| 1 | 1.058170 | 181.164 | 731.066 | 2.699 |
| 2 | 1.070879 | 182.823 | 730.953 | 2.794 |
| 3 | 1.068858 | 181.744 | 731.444 | 2.746 |

各轮16个engine steps、256个layer calls、实际copy36,352,032,768bytes且等于自己的unique下界。慢/快wall差2.944%；这是同一进程内的本次稳定性结果，不能与前一campaign直接当纯环境反事实：新的cache重置/暖机改变了入口，copy总量也与前组六组略有不同。8个每轮GPU边界加最终状态只见本次workerPID30210，初始化为空；未知PID30615仅在本组COMPLETE之后被观察到。

每轮cgroup的nr_throttled/throttled_usec增量均0，进程major faults为0；记录的主线程runqueue增量低于0.6ms，但`sched_schedstats=0`，只保留其观测值。thread CPU约等于wall，不能将其视为纯CPU计算或与GPU时间相加，CUDA等待可能占用CPU。观察包络约2.7–2.9ms，全部在episode wall中。该诊断没有确定旧47%漂移的唯一原因，也不证明NUMA、GPU clock或整个主机永久稳定；它支持继续在同实例、重建相同缓存的交错对照中验证动作。

完整结果见 [same_engine/analysis.json](../20260912_wisp_olmoe_r01/same_engine/analysis.json)，归档SHA256 `ec46b79048ede7da477bbfedc3c6288e34fefd442e8674034d07b356c57c9539`。全局pager summary为四轮合计，initial cache仅最后一轮；前态核验使用每个repeat自己保存的快照，未回填。

随后执行的普通强基线：同实例fixed8/release8/release8/fixed8，release8仅在两旧请求全部完成后将prefill threshold置0，native总budget160保持固定。每次共同暖机fixed8和release8，记录解除时刻及新prompt剩余量，完整计入动作开销。研究目标是减少新请求等待且核对旧请求完整轨迹不受影响，不把通用阶段反馈归为专家历史新贡献；结果如下。

### 普通阶段基线：old完成后解除chunk8

同一engine按fixed8/release8/release8/fixed8完成4个episode，另保留每轮short32→fixed8→release8共同暖机。每个episode3请求/40输出，共12次测量请求执行、160输出。release仅在两old均完成后执行一次threshold8→0，native总budget160、专家cap24、KV1GiB不变。

| Run | Old TTFT / max-ITL / completion ms | New TTFT / max-ITL / completion ms | 完整wall s | old完成后的wall s |
|---|---|---|---:|---:|
| fixed8 r0 | 160.972 / 119.365 / 1560.571 | 1652.400 / 40.043 / 1885.828 | 2.175743 | 0.615172 |
| release8 r1 | 162.564 / 121.750 / 1577.612 | 1439.437 / 41.853 / 1687.325 | 1.979471 | 0.401859 |
| release8 r2 | 162.498 / 122.683 / 1604.874 | 1445.414 / 43.365 / 1691.913 | 2.006894 | 0.402020 |
| fixed8 r3 | 167.858 / 121.308 / 1584.938 | 1664.615 / 40.573 / 1901.694 | 2.206423 | 0.621485 |

两old时间对称；请求指标均扣各自arrival。实际解除发生在call16：old各输出16且completed，新prompt computed96/128，动作7.6–7.8微秒。剩余prefill从4×8变成1×32。配对完整wall改善9.02%/9.04%，但新请求max-ITL略升；完整旧请求时间受repeat波动影响，不能声称所有原始时间戳完全相同。

| 实际copy bytes | fixed8 | release8 |
|---|---:|---:|
| 全episode | 77,674,315,776 | 70,111,985,664 |
| old完成之后 | 19,088,277,504 | 11,525,947,392 |
| 剩余prefill | 14,092,861,440 | 6,165,626,880 |
| 随后decode | 4,995,416,064 | 5,360,320,512 |

prefill省7,927,234,560bytes，后续decode反增364,904,448bytes，净省7,562,330,112bytes（9.74%）。每轮actual=unique、同层extra=0。**逐层不重载并不消除缓存终态对后续请求的影响。** 相同新decode position对齐后的112个layer calls中，4个required union、97个entry resident集合不同；不能用原decode trace替换新策略后续成本。

所有arm的call0–15已记录row/group/entry-resident集合/bytes和old完整输出一致，prefix copy58,586,038,272bytes；四轮最终输出也相同。入口cache/device map、tensor分配、KV/scratch和allocator峰值一致，物理KV blocks不同。call16没有直接dump完整slot/map/LRU，所以不追加未经采集的“全量前态实测相同”主张。CPU观察开销约0.21%，运行边界只见PID33029。

结果见 [release_baseline/analysis.json](../20260912_wisp_olmoe_r01/release_baseline/analysis.json)，复算脚本 [analyze_native_phases.py](../../../experiments/admission_capacity/analyze_native_phases.py)，归档SHA256 `5861bbca22b0fc938836f4bf863c3dfe167c7d386d8a0bf385c09baf92e81934`。该规则成为后续普通强基线；只在旧请求已完成的尾段改变动作，尚不是持续在线负载的SLO-goodput证明，也不构成专家历史贡献。

当前唯一机制假说：**同release8基线上，在mixed层保留当前decode expert集合到该层结束，能否用额外分组开销换取后续miss减少并净改善完整请求？** 在线仅使用本层已经产生的route和真实row phase；先执行入口resident active，之后通过WiSP needed集合保留已加载的目标专家，而每个partial map只计算尚未执行的本组，维持本层每个入口miss加载一次。对照为resident-first、当前路由最高频专家保留、decode专家保留、同保护数量/入口resident交集的hash选择；频率基线不约束resident交集，所有后续轨迹独立真实推进。CPU helper与原生接入已完成，12项针对性CPU检查通过，GPU随后完成，见下节；不以未来route选缓存，不新增pager或扩大资源。

查新边界更新：[DuoServe-MoE v2](https://arxiv.org/html/2509.07379v2)已覆盖分阶段prefetch和batch扩展；[FreeToken §3.1/4.1](https://arxiv.org/html/2608.16157v1)已用global slot pool、prefill双buffer、存活cache种子及GPU侧LRU，不能将“缓存影响下一阶段”本身称新颖。当前候选要测的是小固定层池中mixed row角色、保持当前copy最小与后续缓存的具体权衡；全局池/双buffer/CPU执行与其并不等价，代表性baseline仍需后续补足。Diff-MoE全文仍未核实，暂不下action-level novelty结论。

共享工作协调（2026-09-13）：已登记[结论台账与长上下文四臂修正](../../../experiments/admission_capacity/RESULT_LEDGER.md)，不再为同一输入重复生成无新问题的停顿推导。长上下文轮转必须同轮实跑native32/safe29/headroom32-fast/rotate32-c20；本短mixed paging对照沿用自己的同资源强基线。数值资格限独立运行中首个实际跨组保护mixed层调用，不写成16层覆盖。GPU检查推广到两个活动入口，查询失败或其它进程均退出并保留快照；边界检查不等于全程独占。

### Mixed 层缓存保留：四机制八次完整请求对照（2026-09-13）

**实测未出现 decode 身份保留的完整请求收益。** 同一原生0.26 engine、expert cap24、KV1GiB、BF16/Triton，按 none/frequency/decode/hash/hash/decode/frequency/none 完成8次；全部继承release8，每次重置pager元数据后共同预热short32及四种release8模式。24次测量请求执行、320输出，来自同3篇文章；40份性能预热episode单独保留。

独立资格为decode/hash各一轮、各3请求40输出。两者均在layer0/step4的10-row mixed调用实际保留14个专家并执行3个partial；与完整权重参考相比均finite/allclose(rtol=atol=.01)、非bitwise，maxabs0.0009765625，relative-L2为0.002739/0.003070。各仅1个层调用，不能写成逐层或模型质量验证。

| Run | 完整 wall s | 旧 max-ITL ms | 旧完成 s | 新 TTFT s | 新 max-ITL ms | 实际 payload GB | groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 none | 2.008549 | 123.795 | 1.603736 | 1.450629 | 41.948 | 70.363644 | 635 |
| 1 frequency | 2.035761 | 126.829 | 1.638435 | 1.493702 | 41.771 | 69.168267 | 810 |
| 2 decode | 2.054369 | 129.275 | 1.647802 | 1.510570 | 42.036 | 70.653051 | 826 |
| 3 matched_hash | 2.142740 | 128.700 | 1.651643 | 1.514162 | 127.868 | 70.627885 | 806 |
| 4 matched_hash | 2.162248 | 234.602 | 1.759825 | 1.623168 | 44.512 | 70.627885 | 806 |
| 5 decode | 2.204787 | 130.226 | 1.654559 | 1.518779 | 182.174 | 70.653051 | 826 |
| 6 frequency | 2.028209 | 126.983 | 1.628862 | 1.486687 | 40.914 | 69.168267 | 810 |
| 7 none | 2.023289 | 124.945 | 1.615606 | 1.464435 | 43.279 | 70.363644 | 635 |

对各自正序/反序的none作描述性比较：decode完整wall增加2.28%/8.97%，旧请求max-ITL增加4.43%/4.23%；payload增加289,406,976bytes（0.41%），groups635→826（+30.08%）。最高频基线payload减少1,195,376,640bytes（1.70%），但groups增至810，wall仍增加1.35%/0.24%。hash的payload也增加264,241,152bytes；未用它的较差时间替代更强的none/frequency基线。

decode在first-action后到old完成前仅少搬25,165,824bytes；old完成后的prefill多75,497,472bytes、随后decode多239,075,328bytes，完整成本反而增加。所有层均actual=自身entry-miss唯一下界、extra=0，保护动作合法并不等于缓存选择有用。每种非none模式均有192个eligible mixed layer calls，其中189/189/190个真正发生跨组保留；反复ensure会刷新LRU，未计成独立命中收益。

所有8轮的入口完整cache/map、分配指针、逻辑动作前状态、call0–3记录前缀与最终3请求输出相同；物理KV块和free queue顺序不同，未比较KV内容。同模式两轮的记录route/cache/group/bytes/输出轨迹相同；跨模式后续轨迹不同并独立执行。实际KV1,073,741,824bytes、expert scratch4,831,838,208bytes、pinned12,884,901,888bytes、测量allocated峰值6,889,629,696bytes各臂相同。每次old完成后均在call16将新prompt余下32tokens一次处理，未改变请求数量/输出长度。

19条GPU检查全部PASS：加载前为空，重复边界及完成只见worker36942。检查不证明边界之间绝对独占。hash第二轮call9及decode第二轮call18出现额外时间尖峰，同模式bytes/groups/输出不变，仍不把尖峰归因于保护机制或假定可扣除。两次none wall差约0.73%；不以挑选最有利repeat宣布稳定性能幅度。该差只是两次已观测差值，不是噪声底或置信界；本轮未估计稳定效应量，也未证明统计意义上的无损或普遍性能退化。

证据：[资格](../20260912_wisp_olmoe_r01/retention_qualification/analysis.json)、[完整请求与互斥阶段分析](../20260912_wisp_olmoe_r01/retention_performance/analysis.json)、[预写协议](../20260912_wisp_olmoe_r01/retention_frozen/protocol.json)。原始归档SHA：qualification `52b58a5652e44e0e3616da88776f6da9017150b4701b6d53fd707b037b581040`；performance `a167eb2bce1e5392793bb4e3d2c56d1c6577a0b7a73a9d3d0370edf82e26445d`。首次launcher语法错误发生在GPU启动前，单独保留；随后执行同一冻结source包，没有改策略挑选结果。

本轮停止当前“完整保留decode集合”的formulation，不扫保护阈值或再换选择器。失败归类为选择未减少完整搬运，且增加分组执行成本；不是缺少合法动作，也不是整个cache/admission问题NO-GO。现有轨迹未提供per-row route及frequency直方图，不能进一步独立归因到哪一类请求造成新增miss。唯一下一步先用现有同层相邻调用，核对受保护集合在下一真实调用的复用及未保护槽位损失；这是新的诊断问题，不能把未来集合用作在线选择器或counterfactual Oracle。

新增相邻调用诊断见 [protection_reuse.json](../20260912_wisp_olmoe_r01/retention_performance/protection_reuse.json)。每轮192对上次eligible mixed的同层相邻实际调用：176对旧请求仍在生成、16对跨旧完成边界，全部1536对满足下一entry cache等于上一final cache、无缺失next。旧仍活跃时，decode保护/未保护集合下一次复用率为79.33%/70.15%，frequency为83.65%/68.16%；跨旧完成边界decode变为87.39%/95.68%。对应下一调用缺失次数none4153、frequency4058、decode4157、hash4177，两repeat结构相同并分别保留。保护集合比未保护集合更易复用，不自动减少整层miss；完成边界还可能改变其相对复用价值。未保护部分是策略选择后留下的集合，各策略未来route不同，因此这只是与搬运结果相容的观察，不能当跨策略因果解释或Oracle。

当前停止结论只针对这个小cohort的完整decode集合保留动作，不再增加第三个选择器。若重开，先要求新的实际mixed工作域或一个能改变已测分组成本的动作；在新域先做同baseline A/A与共享负控的变异检查，再预写一个独立cohort对照，而不是把本轮0.73%设为所有指标的噪声阈值。

fresh GPT-5.6-Sol同族独立复核为WARN、无P0：实际运行、数值分母、当前信息使用、请求/row对齐、互斥会计与各策略独立后续执行通过。运行environment中的六份源文件SHA与预写协议全部一致。P1证据边界是缺少per-row route及frequency直方图，无法从raw独立重建保护集合的完整选择过程；不得将源码核对升级为原始路由重建。审查另指出分析helper未随结果封存，已将两份分析源保留到[analysis_source](../20260912_wisp_olmoe_r01/retention_performance/analysis_source/provenance.json)，在临时目录复算与原analysis.json逐字节一致，原件未改；此项由执行者补证关闭，未扩展下一轮审计。复核未包含随后新增的相邻调用诊断。结果范围为单卡人工小专家池的探索性native请求测量，无质量、持续负载SLO、跨模型或新颖性结论。

### 固定保护集合下的执行顺序修正（2026-09-13，16次性能测量完成）

本轮重开依据是具体可消除的实现税，保护selector保持原样。令当层active集合为A、入口cache为C、固定保护集合为P、容量K，a=|P∩C|。在每个入口miss只加载一次、入口active hit不重载、最终P驻留的约束下，入口保护的a个槽位不能中途释放；g组最多覆盖a+g(K−a)个不同专家。非空A的最少组数为max(1,ceil((|A|−a)/(K−a)))。可达构造是先处理unprotected入口resident和missing，保留入口P，最后才加载缺失P并执行全部P。此保证只针对当前已知路由的执行组数，不是未来缓存或完整请求时间的下界。

对旧3072个测量层调用逐个构造，none每repeat635→635、frequency810→713、decode826→728；覆盖、容量、entry-miss-only和最终P全部通过。结果见[当前调用结构账](../20260912_wisp_olmoe_r01/retention_order_frozen/planner_bounds.json)。实现新增late顺序，默认early保留；所有新臂均记录已转CPU的row top-k，不另加路由同步。10项针对性CPU检查通过；更新分析器对旧结果复算逐字节一致。

[预写协议](../20260912_wisp_olmoe_r01/retention_order_frozen/protocol.json)冻结none/early-frequency/late-frequency/late-decode四臂，两个fresh engine各正反序8次，第二engine旋转两个位置；同资源、同3篇文章、共同预热，逐策略生成后续状态。每臂4次用于描述，不声称跨文档确认、稳定小幅时间效应、统计无损或通用噪声阈值。先完成两项独立单调用数值资格：frequency/decode均layer0 step4、10rows、3/2groups、finite/allclose(.01)、maxabs0.0009765625、relativeL2 0.002865/0.002424，非bitwise；两份各384层的row来源与加载不变量复核通过。

随后16次性能测量全部完成：48次请求执行、640输出，另保留80份预热episode。每臂四次的全部wall如下，按两个engine内正/反序block排列，不挑选repeat：

| Arm | groups/episode | payload bytes/episode | wall s，四个block |
|---|---:|---:|---|
| none | 636 | 70,388,809,728 | 1.954664 / 1.940662 / 1.953142 / 1.930604 |
| frequency early | 810 | 69,306,679,296 | 2.017835 / 1.988447 / 2.087704 / 2.086428 |
| frequency late | 714 | 68,702,699,520 | 1.950922 / 1.932188 / 1.956808 / 2.926094 |
| decode late | 731 | 70,011,322,368 | 2.072546 / 2.092408 / 1.970012 / 2.814700 |

必要的请求代价也保留。以下为四次已观测范围，并非统计误差界；逐块、逐请求数值在完整比较中：

| Arm | 旧请求max-ITL范围 ms | 新请求TTFT范围 s |
|---|---:|---:|
| none | 118.346–119.008 | 1.409761–1.424835 |
| frequency early | 124.562–223.391 | 1.459527–1.559593 |
| frequency late | 120.794–173.082 | 1.407341–2.046144 |
| decode late | 123.716–231.995 | 1.439804–2.006044 |

每种策略四次的搬运、分组、已记录执行轨迹相同，16次最终输出均相同；跨策略未来路由/cache重新执行，未被固定。全部6144测量层调用保持actual=entry-miss-only、无重载；late逐调用均达到自身当前A/C/P的最少组数。实际frequency early的当前轨迹下界是711，late实际714，两者未来状态不同；不能把相差3组视为planner未达下界。所有策略KV1GiB、scratch4.5GiB、pinned12GiB、allocated峰值6,889,629,696bytes一致；物理KV block/free-queue不同，未比较KV内容。38条GPU边界检查全部通过，六份运行源SHA与协议一致。

late-frequency比early-frequency少96组（11.85%）、少603,979,776bytes（0.87%）；相对none少搬1,686,110,208bytes（2.40%），但仍多78组（12.26%）。其相对none的四块完整wall为−0.19%/−0.44%/+0.19%/+51.56%；相对early-frequency为−3.32%/−2.83%/−6.27%/+40.24%。因此**实现分组税已减少，仍未确认超过none的稳定请求收益**。late-decode相对none的四块wall为+6.03%/+7.82%/+0.86%/+45.79%，本轮没有观测到完整wall改善。

四次none wall范围1.930604–1.954664s只是本轮观测范围，不覆盖其它臂或未来运行的误差界。第二engine末两次late策略明显变慢，在保护尚未执行的calls0..3就已出现：frequency为0.299761→0.433742s，decode为0.286948→0.371887s。完整load CUDA span也上升，但不足以解释全部wall变化；它与host span重叠且含调度/同步，不相加或当成纯PCIe时间。thread CPU随wall增加，但包含CUDA轮询；cgroup无新增throttle，运行边界GPU频率约2872–2880MHz。现有数据不能定位根因，也不能扣除该块来宣布收益。见[时间定位补充](../20260912_wisp_olmoe_r01/retention_order_performance/timing_addendum.json)。

证据：[完整比较](../20260912_wisp_olmoe_r01/retention_order_performance/analysis.json)、[资格结果](../20260912_wisp_olmoe_r01/retention_order_qualification/analysis.json)。资格归档SHA `5fd991f917937d5cb45912ef654433d0cbe0e6078cb0093743d106b38ec64e8d`，性能归档SHA `65df53d24ed1880b15c05959c6987a90db434883be00734c35b6ad3122dabf6a`，来源与分析脚本随bundle保留。首次远端PATH错误发生在launcher/GPU执行前，attempt01保留。fresh GPT-5.6-Sol限定复核完成：same-family/provisional WARN，P0=0、P1=0；未来信息、请求/step/row对齐、互斥会计、强基线与独立策略状态四项通过。A/B/C/D/F通过，E因单模型、同3篇文档、每臂4次和明显动作前时间漂移保持WARN；只支持结构/搬运改进，不支持稳定请求收益。

唯一下一实验：同程序无保护A/A，保留相同共同预热与记录生命周期，增加GC耗时及CPU执行核心/频率观测，区分观察/分配开销与运行环境变化；不改变缓存selector，也不以关闭GC后未计入的回收成本冒充请求收益。

### 同程序 A/A：GC 与两个请求尖峰对齐（2026-09-13）

**在本轮none基线中定位到GC时间与两个mixed步骤尖峰对齐；此前约0.9s的持续变慢未复现，根因仍未全部关闭。** [预写协议](../20260912_wisp_olmoe_r01/runtime_variance_frozen/protocol.json)固定同一engine、8次none/early/release8、相同3篇文章P32/P32/P128与16/16/8输出、原五段共同预热和全程内存保留trace。仅增加GC callbacks、CPU核心/驱动频率、RSS观测；未改变GC启用/阈值、selector、expert cap24、KV1GiB或计算后端。步骤观测开销保留在episode wall，并按发生位置进入请求时间；episode外围快照、文件写出和最终归档不属于该wall。本轮未报告完整运行周期收益，初版协议/analysis的宽泛措辞由[计时边界补充](../20260912_wisp_olmoe_r01/runtime_variance_performance/ADDENDUM.md)限定。

8测量全部完成，24次请求执行、320输出；40份预热为104次请求执行、1296输出，单独留存。3072测量层的逐row top-k、执行分组/加载/淘汰和最终输出在8次间相同；每次636组、70,388,809,728 payload bytes，无同层重载。逻辑动作前状态、初始cache、tensor allocations一致，物理KV block/free queue不同，KV内容未比较。19条GPU边界检查通过；7份运行源码SHA匹配冻结包；实际allocated峰值6,889,629,696bytes。完整[phase分析](../20260912_wisp_olmoe_r01/runtime_variance_performance/phase_analysis.json)与[GC/CPU交集分析](../20260912_wisp_olmoe_r01/runtime_variance_performance/analysis.json)保留全部数据。

| Repeat（0起） | 完整wall s | 旧请求max-ITL ms | 新TTFT s | call11 wall ms | call11内gen2 GC包络 ms |
|---|---:|---:|---:|---:|---:|
| 0 | 1.994746 | 119.763 | 1.439157 | 119.492 | 0 |
| 1 | 1.975943 | 120.272 | 1.436399 | 120.018 | 0 |
| 2 | 1.993048 | 132.759 | 1.456262 | 120.594 | 0 |
| 3 | 2.046696 | 198.957 | 1.512577 | 198.723 | 77.824 |
| 4 | 1.957226 | 121.329 | 1.426823 | 121.095 | 0 |
| 5 | 2.095948 | 256.549 | 1.564177 | 256.328 | 135.313 |
| 6 | 1.953381 | 119.341 | 1.423119 | 119.066 | 0 |
| 7 | 1.950321 | 119.821 | 1.420326 | 119.612 | 0 |

两次较长call11分别与77.824/135.313ms的gen2回收区间对齐，同一调用的其它6次wall为119.066–121.095ms。这把本轮两个请求尾部尖峰定位到含GC的host执行区间；尚未做GC/记录生命周期干预，不能直接将这段时间相减生成“消除GC后的收益”。GC区间按绝对perf_counter与engine/episode区间相交并取并集，不与thread CPU、host apply、CUDA load span重复相加。回调包络含其自身工作，CUDA可能同时推进，不能命名为独占纯GC成本。

GC保持启用，全部样本阈值为默认(700,10,10)。CPU端点频率样本约3.9GHz，观察到不同核心及一次运行内核心变化，不能据此推出整步实际频率恒定；cgroup新增throttle为0。逐步观测包络合计5.047–24.694ms，repeat2最高，说明新增观测自身也有波动，仍全部计费。动作前calls0..3总wall为0.287953–0.309799s，未复现上一轮late策略0.37–0.43s的动作前耗时；本轮完整wall范围也不构成其它臂/未来运行的噪声上界。

全生命周期保留的层记录从首次测量前1600条增至末次测量后15648条，测量前RSS从15,242,317,824增至15,406,313,472bytes。包括预热/归档阶段在内共记录2563次GC，其中11次gen2；完整事件见runtime_events.json。这只提供“长寿命trace是否增加回收扫描成本”的具体待测假说，不把记录数随时间增长当成已证实因果。此前的大幅持续变慢仍未解释，也不据此恢复late-frequency的完整请求收益主张。

远端driver退出0，全部归档回传SHA256为`5bd2bf1fc00f879f9ab16cfc0c3b68d5178df6c4a374e8a3e7490e6ea5e67d80`；[运行结果](../20260912_wisp_olmoe_r01/runtime_variance_performance/results/execution.json)保留命令、环境、全部失败字段，远端原件保留。新增观测的CPU检查为4个event tests（含计时边界和相同调度分配）及2个observer tests；这里只证明CPU接口与会计边界，不代替上述GPU实测。

复核已完成：新建fresh审查线程因数量上限未成功，改由未参与本轮实现的既有只读审查者重建，标记为reused reviewer / same-family / provisional；不称fresh GPT-5.6-Sol。结论WARN，P0=0、P1=0；A/B/C/D/F通过，E因单engine/同3篇文章/人工cap与物理KV边界保持WARN。独立按整数纳秒复算192个调用的GC交集，与派生值差小于5ns；3072层row身份/top-k/分组、所有测量与预热数量、19GPU检查、源码/分析/归档hash均匹配。计时范围措辞由上述ADDENDUM关闭；本轮仅支持GC与两个尖峰的时间定位。

唯一下一实验：固定默认GC和同一none/release8策略，仅比较“全程内存保留trace”与“完成episode后结算CUDA events、写出并释放记录及event引用”。全部原始记录必须保留，写出/回收/最终归档及整个执行周期一起计费，不能把成本移出测量窗口冒充系统加速。先回答长寿命测量记录是否造成额外回收成本，再恢复机制时间比较。

### Trace 生命周期干预：记录积累有成本，完整周期改善有限（2026-09-13）

**按排空的 episode 写出并释放层记录及 CUDA-event 引用，降低了本轮观测到的 GC 包络和请求尖峰；完整周期只观察到小幅改善，不能解释为调度方法收益，也未关闭此前持续漂移的全部根因。** [冻结协议](../20260912_wisp_olmoe_r01/trace_lifecycle_frozen/protocol.json)预先固定四个新 engine，顺序 memory/episode/episode/memory。每格8次相同 none/early/release8 测量，保持3篇文章、共同五段预热、默认 GC(700,10,10)、expert cap24、KV1GiB、budget160、CPU0–7/OMP8。两臂使用同一重构 recorder，只改 flush 位置；不手动回收、不关闭 GC、不改 selector 或计算后端。

四格均完成并退出0，共32测量 episode、96次请求执行、1280输出；160份预热为416次请求执行、5184输出，全部保留。每格15648条层记录的全局 call ID 连续无重复，其中3072条为测量。全部32次的请求/位置/top-k、分组、加载/淘汰、终态 cache 签名和输出相同；每次636组、70,388,809,728 payload bytes。运行源码7份与冻结哈希一致；76条GPU边界检查通过；每次allocated峰值均6,889,629,696bytes。它们证明观测工作量一致，不等于未采集的KV内容或整机状态相同。

| 新 engine | 进程总 wall s | 初始化后 cycle s | 8次 capture 合计 s | 已观测 GC 与 cycle 交集 s | 其中gen2次数 / 包络 s | 所有flush包络 s |
|---|---:|---:|---:|---:|---:|---:|
| 0 memory | 133.982307 | 87.965790 | 16.112728 | 1.715812 | 11 / 0.867808 | 0.961933 |
| 1 episode | 114.098724 | 86.837590 | 15.987079 | 1.044026 | 19 / 0.174456 | 1.046891 |
| 2 episode | 113.771235 | 86.720028 | 15.930931 | 1.049464 | 19 / 0.175192 | 1.107782 |
| 3 memory | 114.262250 | 86.894231 | 16.018575 | 1.694744 | 11 / 0.871466 | 0.834548 |

相邻反向配对 episode 相对 memory：进程总时间为−14.840%/−0.430%，cycle为−1.283%/−0.200%，capture合计为−0.780%/−0.547%。第一对进程相差19.884s，其中18.755s发生在cycle之外；不将−14.840%归因于trace释放。cycle从首轮reset前至shutdown后，包含全部预热、外围快照、同步、写出、引用释放和最终收尾；自身最后marker的写出由进程wall计费。四格结束后的统一tar压缩/回传不属于此执行周期。GC observer先于事件导出/shutdown关闭，表中GC仅是已观测部分；详见[计时与协议补充](../20260912_wisp_olmoe_r01/trace_lifecycle_performance/ADDENDUM.md)。

| 新 engine | 每episode旧请求max-ITL范围 ms | 其8次均值 ms | 新请求TTFT的8次均值 ms | 首次至末次测量前RSS增量 bytes |
|---|---:|---:|---:|---:|
| 0 memory | 119.499–248.801 | 148.998 | 1469.543 | 164,126,720 |
| 1 episode | 120.624–132.992 | 124.243 | 1453.115 | 6,402,048 |
| 2 episode | 120.021–136.243 | 123.902 | 1449.983 | 6,406,144 |
| 3 memory | 119.537–245.720 | 148.245 | 1462.955 | 164,073,472 |

memory两格各自第3/4次测量出现218.517/248.801ms和218.071/245.720ms的旧请求尖峰；episode两格没有同量级尖峰。memory的gen2包络随运行从约6–7ms增长至234–240ms；episode更频繁，但单次为约1.4–17.0ms。测量前持有记录数从1600增长至15264，改为episode后保持在1600/随后各1568；这约束了本实现的长寿命记录积累，不能把RSS差当全部host memory节省。

已观测GC交集两对减少0.672/0.645s，同时episode写出包络增加0.085/0.273s。完整cycle已包含这些成本，未将GC直接扣除生成反事实。干预同时改变Python记录和CUDA-event引用生命周期，尚不能拆分各自贡献，也不证明已消除所有运行变异。每模式只有两个engine，同三篇文章的连续episode不是独立请求样本；不设新的噪声倍数阈值、不报告显著性或生产尾延迟。允许的结论是这个测量实现存在可消除的记录保留成本，以及本次完整周期收益有限。

新增3项针对性trace测试和已有10项expert分组测试通过：分批/最终写出记录与汇总一致、call ID连续、空flush不覆盖文件、partial write保留文件和引用并禁止自动重试。结果见[analysis.json](../20260912_wisp_olmoe_r01/trace_lifecycle_performance/analysis.json)和[execution.json](../20260912_wisp_olmoe_r01/trace_lifecycle_performance/results/execution.json)；完整结果归档SHA256 `356dce274e7e811c368e3ac9d1582e2a44a97ff378ad56ddd8b4d4b944d2b805`，代码归档SHA256 `837e4df1c8845c04bbe26e73a9d8605a07640e075e27d2182c418bc1c821446c`。原始远端与本地归档均保留。

限定复核已完成：由未参与本轮实现的既有只读审查者执行，reused reviewer / same-family / provisional WARN，P0=0、P1=0。独立从raw重建全部请求、层记录和动作时点，核对flush实物字节、来源哈希、GPU边界及整数纳秒GC交集；报告与ADDENDUM已明确cycle外差异、GC观测尾部和reset继承措辞。停止扩展本轮审计。

唯一下一实验：将episode写出固定为接下来机制对照的共同测量路径，复测已有none/early与frequency/late两臂，保留同域none重复、反向顺序、GC和完整周期会计，回答减少搬运后的剩余组数代价是否仍抵消请求收益。不再为测量优化扩展新方向，不增加第三个selector；此次改动不追认旧late-frequency时间结果。四格均已退出；归档回传后的GPU检查见其它PID47546，下一次执行须重新通过空闲检查，该结束后快照不证明与本轮运行重叠。

### 同一记录路径上的机制复测：少搬运仍未带来稳定净收益（2026-09-13）

**固定episode写出后，完整频率集合的晚加载保护仍未稳定超过none/early普通基线。** [协议](../20260912_wisp_olmoe_r01/retention_lifecycle_frozen/protocol.json)在运行前固定两个新engine，各8次none/early与frequency/late，顺序ABBA BAAB及交换。共同release8、五段预热、默认GC、cap24/KV1GiB/budget160与计算后端不变；两臂分别实际执行其后续KV、路由和cache。新增runner仅接入这个矩阵和已有观察/写出路径，未修改保护身份或分组算法；继承既有单层数值资格，不追认为全模型质量验证。

两格均退出0，16测量episode、48次请求执行、640输出；80份预热为208次请求执行、2592输出，全部完成并保留。每格15648条层call ID连续，其中3072条为测量；8次none彼此、8次frequency彼此的完整row/top-k/分组/cache签名一致，全部16次输出相同。共同逻辑动作前态、初始cache和分配检查通过；38条GPU边界PASS，7运行源码/4分析源码绑定冻结包，allocated峰值均6,889,629,696bytes。物理KV内容未比较，边界检查不保证连续独占。

两种策略每次分别为636/714组、70,388,809,728/68,702,699,520 payload bytes，即frequency少搬2.395%、多12.264%分组；各层入口miss只加载一次。以下按预定相邻配对列出frequency相对none的全部百分比变化，负数为时间较短：

| engine / 相邻位置 | capture wall | 每轮cycle | 旧请求max-ITL | 旧请求完成 | 新请求TTFT | 新请求完成 |
|---|---:|---:|---:|---:|---:|---:|
| forward 0/1 | +0.413% | −2.294% | +1.818% | −0.731% | −0.179% | +1.223% |
| forward 2/3 | +0.910% | +0.695% | +10.880% | +1.082% | +1.106% | +1.056% |
| forward 4/5 | −0.318% | +0.140% | −9.570% | −1.550% | −0.599% | −0.385% |
| forward 6/7 | +0.670% | +1.817% | +1.051% | +0.943% | +0.926% | +0.820% |
| reverse 0/1 | +1.345% | +1.484% | +7.523% | +1.780% | +0.793% | +0.792% |
| reverse 2/3 | −1.793% | +0.083% | −7.642% | −1.736% | −2.270% | −2.167% |
| reverse 4/5 | −0.886% | +1.048% | +1.214% | +0.066% | −1.381% | −1.102% |
| reverse 6/7 | −0.903% | −0.635% | +1.075% | −1.076% | −0.158% | +0.009% |

capture八对四升四降；每个engine内四次均值比较为+0.417%/−0.564%。每轮cycle包含reset、预热、请求、外围文件写出及trace同步/释放，两个engine内均值分别+0.069%/+0.494%；自身cycle_progress写出及共享尾部由整个engine周期计费，不分摊成臂收益。两个engine的进程总时间119.083/115.737s，初始化后cycle86.253/86.290s；每个engine含两种策略，不能拿这两个总数比较策略快慢。GC只分析observer已覆盖的区间，不扣除或与CUDA/CPU包络重复相加。

| 当前策略自身实际执行阶段 | none组数 | frequency组数 | none payload GB | frequency payload GB |
|---|---:|---:|---:|---:|
| 注入前4次调用 | 96 | 96 | 10.091495 | 10.091495 |
| 首次注入调用 | 32 | 33 | 2.743075 | 2.743075 |
| 后续至old完成 | 351 | 428 | 46.003126 | 44.442845 |
| old完成后的prefill | 45 | 45 | 6.253707 | 6.115295 |
| 随后decode | 112 | 112 | 5.297406 | 5.309989 |

主要代价位于旧请求仍活跃的后续mixed阶段：多77组，少搬1.560GB。该段CUDA load包络的模式中位数在两个engine分别约916.0→897.5ms、916.5→897.2ms；实际engine调用wall中位数却只从1.196584→1.192860s、1.198073→1.196276s。这些是重叠计时的分别呈现，不能相减得到纯分组税；它们说明只按搬运字节排序不足以预测完整请求时间。frequency每轮flush均值也从none的125.2/135.9ms增至150.3/146.6ms，额外记录的写出成本已进入cycle。

旧请求注入后的首个ITL八对均增加0.099%–20.275%，其中forward最后一对为81.254→97.728ms。保留的[逐对计时定位](../20260912_wisp_olmoe_r01/retention_lifecycle_performance/timing_addendum.json)显示该frequency调用内GC包络约16.1ms，不能把整个20.3%归因保护算法或将其删除。其余七对首ITL增加0.099%–2.119%，当前证据仍不支持无损保证。全部16次旧max-ITL最高135.478ms，动作前4调用为0.290454–0.313031s；此前的大幅持续漂移未复现，但这里的范围也不是未来噪声上界。

首两次尝试因外部GPU进程在初始化前ABORT/91，无请求结果；完整原件保留在attempt01/02。随后确认共享rotation holdout全部20格完成、占用进程退出、GPU重新为空后，在新r03目录用同包执行，没有改变矩阵或挑选结果。完整[分析](../20260912_wisp_olmoe_r01/retention_lifecycle_performance/analysis.json)、[命令与执行状态](../20260912_wisp_olmoe_r01/retention_lifecycle_performance/results/execution.json)和全部raw已回传；结果归档SHA256 `f5dc1a7534360e51a383ee3de636e8cd6f2c012a39c5a120ba8b1d3d53aa530b`，代码归档SHA256 `67c8670a006cd1d1867ddb2b0f9900992c0b74df9c8a6093d1f5d5d2648565a1`。回传后GPU查询为空，本轮无后台任务。

本轮分类是当前完整frequency保护formulation的净收益未成立，而非paging或资源调度问题被否定。单模型/同三篇文章的相关重复不支持总体显著性、生产SLO或方法GO。接下来的动作保持频率排序，只在当前层保护计划不增加分组数时执行；它约束已经测到的执行代价，不新增预测器或保护身份。

随后完成的[CPU结构探针](../20260912_wisp_olmoe_r01/retention_lifecycle_performance/group_budget_opportunity.json)标为`POST_HOC_STRUCTURAL_OPPORTUNITY_ONLY`。对当前层active集合A、入口cache C、原frequency选择P，令a=|P∩C|；K=24时，非空调用的保护最少组数gP=max(1,ceil((|A|−a)/(K−a)))，普通执行g0=ceil(|A|/K)。只考虑gP=g0且保护计划不同于普通执行的动作。全部6144个实际测量层调用通过下界、覆盖、容量检查，并在两种合法淘汰顺序下核对入口缺失仅加载一次；频率臂实际P和plan与重建相同。

| 每次384调用的实际入口状态 | eligible | 候选计划不同于普通执行 | 不增组且改变plan | 因多组排除 |
|---|---:|---:|---:|---:|
| none/early自己的轨迹 | 192 | 190 | 126 | 64 |
| frequency/late自己的轨迹 | 192 | 190 | 112 | 78 |

每模式8次结构计数一致。两行来自不同策略的实际状态，不能串起来模拟guard的未来cache，也未计算guard的反事实bytes或时间。它只证明这个受限动作不是空集；guard尚未接入runtime或GPU执行。新增86行[分析器](../../../experiments/admission_capacity/analyze_group_budget_opportunity.py)已和本轮冻结planner身份一同留存。唯一下一实验是在同一资源/记录路径下比较none、完整frequency/late和该“不增组才保护”动作，独立推进后续状态，检验其完整请求收益和旧请求代价；即使组数不增加，选择、ensure触碰、记录和后续cache代价仍须计费。

本轮限定独立复核完成：reused reviewer / same-family / provisional WARN，P0=0、P1=0。全部请求、6144层row与frequency选择、每层组数下界、8配对的16项指标、flush实物、计时交集与资源检查通过；CPU结构数也独立复得。审查时指出尾部合同尚未同步，执行者已将本轮48请求和唯一guard下一动作写入；不扩展新的审计轮次。

### 不增当前分组的保护约束：动作成立，完整净收益仍未确认（2026-09-13）

**同一频率候选集合增加当前组数约束后，实际少搬0.840%，但完整请求收益仍不足以确认。** [冻结协议](../20260912_wisp_olmoe_r01/group_guard_frozen/protocol.json)先规定两个新engine、每格6个episode，none/full/guard按ABC CBA及CAB BAC执行。none是原有普通expert分组和release8，full是完整frequency/late，guard保留同一frequency候选P，仅在当前gP=g0时执行保护；拒绝时实际P清空。cap24、KV1GiB、budget160、BF16后端、三篇文章、五段预热、默认GC及episode写出相同。各策略真实推进其后续状态，不复用另一策略的未来路由。

独立资格先完成3请求/40输出及5份共同预热。首候选call1664会从2组增为3组，确实回退；首个实际applied调用1665（step4/layer1、10个mixed rows）保持2组，finite/allclose(.01,.01)通过，maxabs0.03125、relative L2为0.00540626、非位级相等。它仅是同层当前输入/top-k下的数值诊断，不是全模型质量或正式性能。资格轨迹reject69/applied121；不得混用为性能计数。

性能12/12 episode完成并退出0：36次请求执行/480输出，60份预热/156请求/1944输出全部保留。两格各11744条连续call ID，其中2304条测量；4608个测量层调用的身份、当前P、容量、入口miss-once和guard约束均通过。30个GPU边界PASS，7份运行来源与5份封存分析源码一致，allocated峰值均6,889,629,696bytes。每种模式4次完整trace与输出相同，12次输出相同；跨策略row top-k仍有差异。未比较物理KV内容，边界快照不证明连续独占。

| 每个测量episode | none | full | guard |
|---|---:|---:|---:|
| 实际分组 | 636 | 714 | 637 |
| tensor payload bytes | 70,388,809,728 | 68,702,699,520 | 69,797,412,864 |
| 实际active专家出现次数（按层/调用累计） | 10,697 | 10,689 | 10,695 |
| 入口resident命中次数 | 5,103 | 5,229 | 5,148 |
| 真实加载次数 | 5,594 | 5,460 | 5,547 |

guard每次192个候选mixed调用，拒绝71、实际applied119，另2个候选不改变执行。其每个调用均等于自身当前g0；全程比none多1组，不与局部约束矛盾。四个配对均从step4/layer2开始出现row top-k差异；唯一组数不同的调用在step16/layer5，此时old已完成，guard路径active49而none48，因此3组对2组，该层未执行保护。详见[实际轨迹补充](../20260912_wisp_olmoe_r01/group_guard_performance/trajectory_addendum.json)。首个已观测top-k差异不是数值根因的完整定位。

守恒账为 loads=active occurrences−entry resident hits：guard相对none少47次加载，来自实际并集累计少2、入口命中多45，每份12MiB，合计591,396,864bytes（−0.840%），保留full节省的35.075%。这不是固定未来轨迹下纯缓存因果收益。旧请求仍活跃的后续mixed阶段，两者均351组，guard少搬352,321,536bytes；old完成后的prefill为46对45组、少搬75,497,472bytes，随后decode少搬163,577,856bytes。

所有预定配对如下，负数表示耗时较短；同块基线和方向在表中明确，未用跨域负控最大差标注可分辨性。

| engine/block | 比较 | capture wall | repeat cycle | old max-ITL | old完成 | new TTFT | new完成 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0_forward/0 | guard vs none | +0.226% | -1.649% | +2.833% | +0.347% | +0.605% | +0.480% |
| 0_forward/0 | guard vs full | +0.679% | +0.553% | -0.099% | +0.897% | +0.459% | +0.243% |
| 0_forward/0 | full vs none | -0.449% | -2.190% | +2.935% | -0.545% | +0.145% | +0.237% |
| 0_forward/1 | guard vs none | -0.306% | -0.918% | +1.311% | -0.143% | -0.369% | -0.475% |
| 0_forward/1 | guard vs full | -2.271% | -1.276% | -8.381% | -2.638% | -2.982% | -2.764% |
| 0_forward/1 | full vs none | +2.011% | +0.363% | +10.578% | +2.563% | +2.693% | +2.353% |
| 1_rotated/0 | guard vs none | -0.774% | +1.820% | +0.127% | -0.499% | -1.463% | -1.570% |
| 1_rotated/0 | guard vs full | -2.165% | +0.710% | -3.919% | -1.938% | -3.025% | -3.207% |
| 1_rotated/0 | full vs none | +1.422% | +1.102% | +4.211% | +1.468% | +1.611% | +1.691% |
| 1_rotated/1 | guard vs none | -1.195% | +1.872% | -5.140% | -1.120% | -1.333% | -1.404% |
| 1_rotated/1 | guard vs full | +1.571% | +1.715% | +2.247% | +1.745% | +1.643% | +1.487% |
| 1_rotated/1 | full vs none | -2.723% | +0.154% | -7.224% | -2.816% | -2.928% | -2.849% |

两个engine内各模式两次均值比较，guard对none的capture为−0.039%/−0.988%，old max-ITL为+2.072%/−2.617%，new TTFT为+0.119%/−1.397%；repeat cycle为−1.287%/+1.846%。对full的四个capture配对两升两降。没有稳定超过普通基线或完整保护的净收益证据，也不把小差值直接宣布为零。

完整周期保留reset、五段预热、外围观测、raw写出、trace同步/释放；repeat marker及共享尾部由engine/process总账覆盖，不能按arm分摊。两格进程为90.726460/92.007029s，初始化后cycle64.191886/65.293696s；各7次flush精确写出全部11744条层记录。guard预热wall均值在两engine为8.126/8.372s，none为8.215/8.217s，因此不能把repeat cycle的符号变化全归因于保护。所有原值与GC交集保留在[analysis.json](../20260912_wisp_olmoe_r01/group_guard_performance/analysis.json)，没有删去慢样本、扣除GC或相加重叠CUDA/CPU包络。GC只覆盖observer关闭前的区间。

本轮的实质进展是：验证了可执行的当前组数约束，并量出它同时放弃了约65%的搬运节省；成本模型还必须表达保护后的实际复用及后续路由变化。当前小工作负载仍为MEASUREMENT_ONLY，不否定paging/请求资源问题，不增加第三个selector，不提高任意百分比寻求GO。唯一下一步：从已保留三臂轨迹，按layer统计被保护及被驱逐专家的下一次实际使用间隔与下一次调用入口miss，判断保护寿命是否与复用匹配；仅作事后机制诊断，不将未来使用当在线输入。

本轮限定复核完成：未参与实现的既有只读审查者独立重建全部12配对、逐row/guard/cache、flush实物字节及GC交集，核对新报告与轨迹补充。reused reviewer / same-family / provisional WARN，P0=0、P1=0，无必要修正；停止扩展审计。

性能归档SHA256 `a19d35ce5871e6ee9e64ed40553be4da88de4551903785dc920e48e4a017f0fd`（12,313,920bytes）；资格归档`039b2588ed45aa3ce75d4fa857bfc3f4e03b5af28ab3334d113d022147111dc8`；代码归档`be9492eae4824d6e8e93786e34f18cf1a9058fa8c79f33f258101f10379010a2`。原始包与冻结协议未改。新增的轨迹补充分开标为post-hoc，保留其独立源码。两个engine及driver均已结束，回传后GPU查询为空、远端磁盘余4.1GiB。

### 实际复用与层间静态预算：保护有复用，交换收益相抵（2026-09-13）

对已保留三臂12个性能episode完成[实际复用分析](../20260912_wisp_olmoe_r01/group_guard_performance/retention_reuse.json)。每个episode取192个mixed层调用结束点，只把实际applied的P算保护；ensure-only触碰不算计算使用。沿各策略自己的后续同层group，读取第一次required_experts使用及此前真实驱逐/重载。终点之前未观察到使用记右删失；同一专家在多个起点可重复出现，各模式4次重复不构成独立寿命样本。

| 每个episode的专家出现次数 | full | guard |
|---|---:|---:|
| 实际保护 | 2859 | 1792 |
| 下一次同层调用使用且原驻留仍存活 | 2424（84.785%） | 1514（84.487%） |
| 未观察到后续使用，右删失 | 9 | 7 |
| ensure-only触碰 | 2493 | 1023 |
| 共同实际需求少加载 / 多加载 | 1278 / 1153 | 973 / 931 |
| 加入两侧独有需求后的净加载变化 | −134 | −47 |

这关闭了“主要因为保护对象迟迟不用”这一解释：被保护项约85%下一次调用就使用，但full/guard的入口被驱逐项也分别约73.6%/74.9%在下一次调用被需要。有限槽位交换中的收益与代价大量抵消。共同需求账为Δmiss=added−saved+treatment_only−baseline_only；它使用两策略的真实A/C，包含路由不同，不能解释为纯缓存因果收益。entry被驱逐和计算后未保留两类集合有重叠，不能相加。保护只到当前调用结束，未新增跨调用pin或第三个P选择器。

随后以第一engine的第一none episode作[层预算校准](../20260912_wisp_olmoe_r01/group_guard_performance/layer_budget_calibration.json)。从reset空cache开始，包括短预热和四段完整预热；cap24精确复现1952次调用、3575组、32个slot/map/LRU锚点及逐组load/evict。之后仅在这些固定校准row/top-k上分别演进16层、cap16…32的LRU，动态规划分配总384槽。各层专家均12MiB，因此总scratch保持4,831,838,208bytes。

| 固定校准轨迹上的静态分配 | 测量miss | 测量groups | 预热miss | 预热groups |
|---|---:|---:|---:|---:|
| uniform24 | 5594 | 636 | 22872 | 2939 |
| 优先少miss，测量组数不超uniform | 5513 | 636 | 22751 | 3017 |
| 优先少groups | 5554 | 624 | 22804 | 2982 |

按miss选择少搬1,019,215,872bytes（1.448%），但预热多78组。该结果严格为STRUCTURAL_FIXED_TRACE_CALIBRATION_ONLY：其它容量没有重新生成未来路由，不是GPU加速或serving Oracle。两份CPU分析经未参与实现的既有审查者有界独立复算，reused reviewer / same-family / provisional，P0=0/P1=0；原raw和冻结源未改，新增分析源码单独留存。

这一选择首先补强简单基线。当前[WiSP v2 §3.1](https://arxiv.org/html/2606.21868v2)采用每层统一C表达专家预算，并已提出专家/KV边际价值分配；[公开实现](https://github.com/nokia-applied-research/WiSP)提供pager与静态预算复现。因此这里不把“按剖面分配缓存”本身称为新颖性，也不把离线miss最优当完整请求最优。

唯一下一实验已[冻结](../20260912_wisp_olmoe_r01/layer_budget_frozen/protocol.json)：总384槽和1GiB KV不变，在初始化时选择uniform、少miss、少groups三份静态映射；先做独立逐层数值资格，再以U/M/G/G/M/U六个fresh engine、每格8次none/early/release8执行。评价文档按存量源序取与校准文档不重叠的前三篇（316/480/507），保持P32/P32/P128、输出16/16/8及事件到达。先比较完整capture、旧max-ITL/完成、新TTFT/完成，再同时保留预热/写出/cycle/process成本；不根据评价结果重选容量或SLO。运行状态与真实结果由后续记录给出，冻结配置本身不代表已测。

### 层间静态预算实跑：减量可迁移，完整收益仍受运行变异限制（2026-09-13）

三份分配先在独立资格中完成12请求/96输出，16层×三分配共48个同输入/top-k的全权重对照finite/allclose(.01,.01)通过，非位级相等；这不是质量或性能结论。随后六个预定fresh engine U/M/G/G/M/U全部完成：M为校准少miss，G为校准少groups，每格8次独立推进状态的none/early/release8 episode。48测量包含144次请求执行/1920输出，240预热包含624次请求执行/7776输出；三篇评价文档316/480/507与容量校准分开，仍只是固定自然文本前缀。

[实际分析](../20260912_wisp_olmoe_r01/layer_budget_performance/analysis.json)的93888个连续层call中18432个属于测量；114个GPU边界、7份执行源、实际16层映射/384槽、4,831,838,208B scratch和1GiB KV均匹配。测量allocated峰值均6,889,629,696B。全部48次输出序列相同，跨分配row/top-k不同，各分配内部row/top-k一致。M的完整trace hash不全同，仅因为首repeat的step14/layer4/group1尚未把expert48计为reloaded，而后续repeat的ever_loaded历史已包含它；required/ensure/loaded/evicted、entry/final、实际bytes均相同，不能将该记账差异称为路由漂移。

| 每个实际测量episode | U uniform24 | M 少miss | G 少groups |
|---|---:|---:|---:|
| 实际加载次数 | 5224 | 5195 | 5209 |
| 实际分组 | 631 | 627 | 627 |
| tensor payload bytes | 65,733,132,288 | 65,368,227,840 | 65,544,388,608 |
| 相对U搬运变化 | — | −0.555% | −0.287% |

M在固定校准轨迹上的1.448%少搬没有按原幅度迁移到评价输入，但较小的减量实际成立；G的校准624组也不是评价时的预测标签。M/G在评价中均627组，不能仅用固定trace的校准目标给实际请求时间排序。

下表每行比较各engine内8次均值，负数表示耗时较短。完整cycle含reset、五段预热、观测、raw写出及flush；cycle自身marker和共享尾部保留在engine/process总账。

| block / 比较 | capture | repeat cycle | old max-ITL | old完成 | new TTFT | new完成 |
|---|---:|---:|---:|---:|---:|---:|
| 0 / M vs U | -1.477% | -1.047% | -2.603% | -0.999% | -0.933% | -1.610% |
| 0 / G vs U | -3.636% | -3.208% | -1.276% | -2.858% | -2.814% | -3.662% |
| 0 / G vs M | -2.191% | -2.184% | +1.362% | -1.878% | -1.899% | -2.086% |
| 1 / M vs U | -0.811% | -0.017% | -4.044% | -0.990% | -0.888% | -0.824% |
| 1 / G vs U | -0.396% | -0.355% | -1.641% | -0.527% | -0.445% | -0.334% |
| 1 / G vs M | +0.418% | -0.338% | +2.505% | +0.468% | +0.448% | +0.494% |

M对U的capture、旧max-ITL及新旧完成在两个block均较短，但repeat cycle仅−1.047%/−0.017%，尚不足以确立稳定完整收益。G对M的capture从−2.191%变为+0.418%，不能确认哪种校准目标更适合完整请求。两块共享三篇文本，八次engine内重复和ordinal配对都不是独立样本，未设置噪声倍数门槛或SLO来制造GO。

运行变异必须和均值同时保留：0_uniform repeat3、1_selected repeat3、2_min_groups repeat2的capture为2.870/2.819/2.638s，其余多数约1.84–1.99s；异常episode的第三请求注入前阶段（此时静态层预算已生效）已为0.482/0.480/0.447s，其各自常见值约0.32s。对应GC capture交集仅55/53/71ms，不能由该交集单独解释全程变慢，更不能直接扣除。CPU端点采样约3.9GHz且cgroup未记录throttle，也不足以定位原因。全部慢样本已计入均值；同策略第二engine的capture均值较第一engine下降4.40%–7.51%，这只是已观察漂移，非总体噪声界。没有将这些变化追认为某个外部GPU进程的干扰。

六个成功process为129.380/126.243/112.872/110.727/107.551/107.855s，合计694.627993s。此前两次初始化OOM及一次初始化前ABORT_BUSY零请求退出，仍保留36.185328s失败启动成本；全部尝试process合计730.813321s，资格、归档和回传另计。首两成功格来自attempt02，后四格来自attempt04；[组合索引](../20260912_wisp_olmoe_r01/layer_budget_performance/results/execution.json)保留四份原execution和全部失败日志，不把中断描述为连续独占。首次完整选择规则在读取前两格性能之前写入EXECUTION_ADDENDUM.md。attempt04完整归档31,619,639B，SHA256为8c4036bee48cc5fdcbf338d152496cea9eb02fb8cf363281001aa1a5132fa4c4；原包和493个展开文件逐一匹配。

本轮回答了静态层预算的迁移问题：减量能转移到新文档，但幅度较小，且miss/groups的校准排序不等同完整成本排序。保留M作为已测强简单对照，不改容量去拟合评价文档，不将小幅变化宣称为零或方法GO。主问题仍OPEN。

唯一下一实验是共享专家池kernel接口资格，尚未实跑：在总384权重槽的设计中，每层保留21槽，共336槽，另48槽供当前层临时使用；21+48≥64，因此当前层全专家集合在容量上可容纳。先仅验证现有0.26 Triton对384物理槽和跨段expert_map的读取是否正确，从相同真实pre-call输入/top-k分别执行完整64专家参考和映射池，并加固定错误映射负控；不改变模型实际输出。资格会额外分配reference/shadow，明确不是同预算性能或共享池生命周期实现。完整pager仍需实现所有权、清空与同stream先读后写约束，并实付减少长期缓存后的额外搬运。

这一基线的动机来自当前组数成本，不能包装为全新暂存设计。[WiSP v2 §3.2](https://arxiv.org/html/2606.21868v2)的每层scratch兼作驻留和执行空间，超过容量时分token组；[FluxMoE v3 §3–4.1](https://arxiv.org/html/2604.02715v3)已提供跨层复用的完整层物理块和RAW/WAR约束。这里先关闭现有kernel能否表达更强执行基线的实现不确定性，尚不主张新颖性或请求加速。

静态预算限定复核完成：既有未参与实现的只读审查者重算全部请求/48配对/6均值比较、逐层实际加载和预算、114 GPU边界、flush实物字节、GC交集及4attempt索引；reused reviewer / same-family / provisional，P0=0/P1=0。按意见把“动作前”明确为第三请求注入前（静态预算已生效），并更新基线合同；不扩展审计。

### 共享池索引接口资格：首轮负控失效，新文本按位合同通过（2026-09-13）

[首轮原始资格](../20260912_wisp_olmoe_r01/shared_pool_qualification/attempt01/results/interface/shared_pool_qualification.json)保留FAILED：16层×M{2,10,64}共48个影子输出虽然全部按位等于全权重参考，但固定layer0/M10错误映射仍通过allclose(.01,.01)。该负控maxabs为0.003173828125、relative L2为9.332%，说明其小量级输出下绝对容差未能区分错误；4请求/32输出完成不等于整体资格通过。进程36.706551s和全部原始指标保留，不追溯改判。

随后单独冻结[revision02协议](../20260912_wisp_olmoe_r01/shared_pool_qualification/revision02/protocol.json)，改为纯索引接口合同BITWISE_INDEXING_IDENTITY：正向finite/allclose且按位相同，固定错误映射finite且按位不同；数值误差仍原样报告。采用原来源序号4…7的文档480/507/628/799，与首轮资格文档不交叠；这是既有语料的新资格前缀，不是新语料集。

[第二轮原始资格](../20260912_wisp_olmoe_r01/shared_pool_qualification/attempt02/results/interface/shared_pool_qualification.json)状态PASSED_BITWISE_INDEXING_IDENTITY_ONLY：4请求/32输出完成，16层×M{2,10,64}共48个影子对照均finite/allclose/bit_equal，maxabs和relative L2均0；每层映射的64份权重在调用前后均逐字节匹配参考。全部影子输入来自首请求480的prefill step0，不能称四请求或decode数值路径均已覆盖。layer0/M10按最小两个active ID规则选择[0,3]交换映射，输出finite、bit_equal=false、allclose=false，maxabs0.0121612548828125、relative L2为18.077%。原native部分和验证另有16次finite/allclose通过、均非位级相等，不能混为同一个数值合同。

接口使用384作为物理索引域，逻辑模型仍为64专家；每层21个私有槽加43个临时槽填入真实CPU权重，其余槽填NaN，映射其余320项为−1。验证期间额外分配shadow pool 4,831,838,208B、reference805,306,368B及map/selection；allocated峰值13,597,369,856B。它既不证明同预算性能，也未执行跨层复用生命周期、给出质量保证或推广为所有输入按位相等。三次GPU边界检查通过，属于边界观测。

第二轮进程36.575670s；归档242,123B、SHA256 bfc4d9bbae76010ee82ad72484bfabc6f89ede6f04c9b4c619963ced681d0c1c，23成员/20文件逐一核验，首轮archive b2ef462a82043748989c640874c0b049f85a4b9317e6cbc0c171f7bf8da5a7a2保持不变。

接口资格有界A–F复核完成：既有asset_audit上下文、same-family/provisional，P0=0/P1=0，范围WARN；已明确首请求step0及原部分和非按位边界，不扩展本轮审计。

唯一下一步为同384槽的共享池生命周期资格：普通cap21分组LRU给出本次调用的最终私有缓存状态，先D2D暂存将被覆盖但当前仍需计算的入口驻留专家，再把入口miss各H2D一次到最终私有/临时位置，在同一stream一次执行全并集。保留相同私有LRU状态的split21对照，明确付出长期缓存24→21及D2D成本；不先宣称完整收益或暂存机制新颖性。最小实现已完成，性能UNRUN。CPU符号检查8080次连续调用、30特殊初态及11非法输入通过，错误H2D→D2D顺序造成5114次损坏；命令与结果保留在shared_pool_lifecycle_qualification/cpu_check{_source.py,.json}。

生命周期attempt01在split21初始化执行16层后，因vLLM后续dummy warmup改用其它CUDA stream而触发固定stream-ID拒绝，26.052556s退出；没有测量请求或数值资格，oneshot未运行。完整失败归档SHA256 4ba80dce913583baa1d03c485962d21e04d8833e9439e9b9ecd1196f47920bc3（51,108B）保留。该失败只否定固定stream-ID假设。revision02改为调用后记录event、切流先wait，另用非阻塞host锁拒绝并发访问；CPU顺序回归覆盖0→7→0及异常释放，通过有界复核后启动。原数值/权重判据、四请求输入976/1107/1157/1244均未改，资格已完成，结果如下。实际总池以去重expert_scratch_bytes记384槽，旧scratch_bytes仅336私有槽；H2D和D2D独立记账。

### 真实共享池生命周期通过，四臂完整成本对照已启动（2026-09-13）

[revision02实际资格](../20260912_wisp_olmoe_r01/shared_pool_lifecycle_qualification/qualification_check.json)完成两格共8请求/64输出，各10个measurement step；每格224层调用=32初始化+32预热+160测量。全部160测量调用均与真实call ID/层/step/rows对应，宽度覆盖1/3/4/36/64/160行。这里同时验证后续decode与跨层复用，不再仅取首prefill调用。

| 每格测量 | split21普通分组 | oneshot21共享暂存 |
|---|---:|---:|
| finite / allclose(.01,.01) | 160/160 | 160/160 |
| 同pre-call全权重输出bit_equal | 46/160 | 160/160 |
| 实际权重byte检查 | 320/320 | 640/640 |
| maxabs / 最大relative L2 | 0.0625 / 0.004313 | 0 / 0 |
| 实际groups | 326 | 160 |
| H2D payload bytes | 35,798,384,640 | 35,810,967,552 |
| D2D payload bytes | 0 | 13,526,630,400 |

两格实际均为336私有+48共享=384槽、4,831,838,208B总专家池和1GiB KV，各记录一次初始化切流且先等待前序event。逐调用自己的cap21 LRU计划、加载与缓存连续性核对通过，测量入口16层LRU锚点一致于各自计划。实际resident和权重逐字节核对不等于每调用独立读回完整GPU map；资格有额外参考/读回开销，表中不比较耗时、也不作质量保证。

跨臂的状态不全相同：测量入口仅14/16层cache相同，首次route差异在warmup call33/layer1；输出3/4请求相同，另一条从第5输出token分歧。普通BF16部分和与一次完整执行的数值路径不同，因此“同当前输入保持普通LRU终态”不能外推为跨策略未来cache/route相同。性能必须沿各自真实轨迹测量，不复用固定future trace。

成功process32.565799/32.077871s；原初始化失败26.052556s另列保留。成功归档SHA256 8cdc3ee0a385dce0924dfe1820cc7bbab231f87e2ecb4feff0fa30b884d9ce93，602,798B、44成员/39文件逐一匹配；源9/9与每格3个GPU边界通过。既有源码审查者完成有界数据复核，reused/same-family/provisional、无P0/P1，数值/状态/性能边界保持。

唯一下一实验已冻结并启动：[shared_pool_performance/protocol.json](../20260912_wisp_olmoe_r01/shared_pool_performance/protocol.json)。uniform24、原固定M少miss、split21、oneshot21正反八个fresh engine，每格八次共同episode路径；block0取既有来源新文档1274/1401/1487，block1取1710/1856/1988。保持P32/P32/P128、输出16/16/8、第三请求事件到达、release8、总384槽/1GiB KV；四种原共同预热中的保护路径在两共享模式均执行ordinary21，测量只允许none。原采集/observer/flush保留，新增planner、map、event、D2D及trace开销全部计入，H2D与D2D不合并为PCIe字节。八次同engine是相关重复，反序同时换文档，不能单独估计顺序效应；没有SLO或噪声倍数门槛。代码包88aac0029be609fba9585c76fb0940c22b4b90e20f3dd1c7bfab06ba92d5faf8（129,151B），此为启动时记录；实际完成结果见下节。

相邻动作核对把新颖性边界进一步收紧：[FluxMoE v3 §3–4.1](https://arxiv.org/html/2604.02715v3)已有整层常驻与两个完整层复用块，[MoE-Gen §2/4.2–4.4](https://arxiv.org/html/2503.09716v1)已有持久参数区与expert staging分离；[WiSP v2](https://arxiv.org/html/2606.21868v2)及已检查pager仍以每层C限制单次scratch执行。因此仅“缓存与暂存分开”已被覆盖，当前索引/状态合同也尚未构成新颖性证据。若本轮完整收益成立，最小补强对照为16×20私有+64完整共享stage：一次执行union，保留普通cap20 LRU，计命中项与最终留存项的D2D。此处为补强对照提出时记录，不冒称论文原版复现；它用于区分已知stage收益与21+48紧凑映射的增量，不改当前冻结四臂。

### 共享池四臂性能完成：请求改善，完整实验周期仍有代价（2026-09-13）

[实际八格执行](../20260912_wisp_olmoe_r01/shared_pool_performance/results/execution.json)及[逐请求/调用分析](../20260912_wisp_olmoe_r01/shared_pool_performance/analysis.json)已完成，覆盖原冻结U/M/S/X/X/S/M/U全部格与repeat，失败0。U为uniform24，M为原校准固定层预算，S为private21加预留未用shared48的普通分组，X为private21加shared48的一次完整执行。两block各四个fresh engine、每格八次相关episode；下表为每格八次均值的`100×(X/基线−1)`，负数表示该时间指标下降，old项取每episode两旧请求的最大值。

| block / 对照 | capture wall | repeat cycle | old max-ITL | old completion | new TTFT | new completion |
|---|---:|---:|---:|---:|---:|---:|
| 0 / X/U | −5.542% | +2.715% | −4.509% | −4.635% | −4.089% | −5.147% |
| 0 / X/M | −6.729% | +2.246% | −3.193% | −6.070% | −5.648% | −6.461% |
| 0 / X/S | −12.701% | −4.426% | −11.373% | −12.725% | −12.995% | −13.027% |
| 1 / X/U | −5.931% | +2.871% | −3.351% | −5.155% | −4.392% | −5.528% |
| 1 / X/M | −6.893% | +1.396% | −6.350% | −5.932% | −5.532% | −6.719% |
| 1 / X/S | −12.517% | −4.641% | −11.925% | −12.220% | −12.317% | −12.689% |

64次measurement共192请求执行/2560输出；320份warmup含832请求/10368输出；全部125184层调用中24576次为measurement，152个GPU边界均通过。所有repeat、慢行和原资格失败保留。总专家权重唯一storage为384槽/4,831,838,208B，KV为1GiB；共享臂的336私有槽是视图，另48槽已计费。X相对U的实际H2D反而增加11.449%/10.099%，另付每episode均值32.741/33.924GB十进制D2D；实际分组减少39.812%/38.854%。planner、map、event、观察、写盘和flush均保留，未从wall中扣除任何重叠区间。这里groups计MoE分组调用，内部仍包含多个CUDA kernel。

capture对每种基线的ordinal配对均16/16改善；repeat cycle对U/M/S仅0/16、1/16、16/16改善。parent process相对U为+1.848%/+2.260%，相对M为+1.792%/+0.838%。repeat cycle包含reset、五次warmup、一次measurement及写盘/flush；repeat外工作进入whole-cycle，初始化和尾部进入parent process。**这套五次warmup加一次measurement的实验周期较慢，不是持续服务的摊销性能测量；请求正收益也不等于完整净收益GO。**

对同一封存结果做[五种预热成本定位](../20260912_wisp_olmoe_r01/new_host_20260913/warmup_cost_localization.json)：全部8个ordinal保留。X/U的正增量均出现在frequency-early、frequency-late、decode-late三种protected warmup，合计+0.49608/+0.54043s；此时两臂均实际执行普通expert分组，private容量为21/24，H2D增加17.930/19.148GiB、groups增加337/367，D2D均0。X实际走共享oneshot的普通短预热与none-early反而合计减少0.12713/0.14389s，抵消后得到原总预热+0.36895/+0.39653s。这把成本定位到protected普通分组路径，尚未隔离cap、继承cache/route、包装和环境的因果贡献；不把缺失host_plan字段当作零规划成本，也不把host/CUDA跨度加到wall。此为新分析，不是新GPU或持续服务收益，X/F冻结计划不变。

各臂沿自己的实际route/output/cache推进；逐调用普通LRU终态检查仅使用本臂当前rows，不复用未来trace。X/U输出16/16配对相同，X/M和X/S各8/16相同，三类跨臂route均不同；质量未测。八次同engine不是独立样本，反序同时换文档，不能估计纯顺序效应、总体噪声界或显著性。复用上下文的same-family/provisional复核独立重算全部请求、字节和配对，无P0/P1；GPU检查仅覆盖边界，不代表连续独占。

唯一下一步为普通full-stage同预算基线的GPU资格及性能：16×20私有槽+64完整共享stage，保留普通cap20 LRU，计入命中项搬入stage与最终留存项写回的D2D。它是已有stage思路的适配，不冒称论文原版或新机制。[CPU符号检查](../20260912_wisp_olmoe_r01/full_stage_qualification/cpu_check.json)已通过8096连续调用、30稀疏/tie初态及11非法输入；`full_stage_plan.py`已存在，**GPU资格/性能UNRUN**。该对照用于区分普通完整stage收益与21+48紧凑映射增量，不提前启动新admission/selector。

前一主机阻塞记录：普通full-stage源码、CPU状态检查和11份冻结依赖已准备，GPU资格及后续四格性能包均已上传核验。两次跑前记录发现外部PID85558正在执行Qwen模型资格，占用3650MiB，故本轮保持`BLOCKED_BEFORE_GPU_INITIALIZATION`，0资格请求、0性能请求；没有启动后台等待器。具体接续命令与条件见[execution_status.json](../20260912_wisp_olmoe_r01/full_stage_qualification/execution_status.json)。这是单卡资源阻塞，不是科学负结论。

用户随后提供新主机，旧端点已关闭。新主机实测空闲RTX5090 32GB、driver595.71.05、cgroup内存90GiB/CPU配额25核。已复用现有安装与下载任务，实际Python3.12.3、PyTorch2.11/CUDA13、vLLM0.26、Transformers5.15.1、Triton3.6.0，pip check及WiSP/adapter导入通过；旧0.26记录未保存Transformers/Triton版本，不追认为相同。原11份执行源、输入与性能矩阵不变，[换机附录](../20260912_wisp_olmoe_r01/new_host_20260913/HOST_ADDENDUM.json)要求先对F/X各跑一格逐调用资格，再在新主机完成全部X/F/F/X性能。固定revision的三份正式权重已全部下载并逐文件通过LFS SHA-256校验（合计13,838,721,960B），十篇输入重新编码与冻结token IDs相同。旧PID3326退出后，新Qwen任务PID5127又占卡3650MiB；[有界续跑](../20260912_wisp_olmoe_r01/new_host_20260913/continuation_status.json)PID5178当时已启动并处于WAITING、0 GPU请求。它最多等待6小时，空闲后复核冻结输入/环境/权重，再执行两格资格，逐调用检查通过后才运行X/F/F/X与冻结分析；任一失败保留并停止，不终止他人进程。GPU实验仍UNRUN，新F不得与旧主机X的时间直接配对。

用户再次指定weste:23478后，已切换到该端点：实测GPU UUID为0a66cc34-b091-2000-ba7b-e576b6d3d7d6、5090/driver595.71.05、CPU配额16核、内存92GiB。原FQ/FP包及各11源逐字节匹配、尚无旧results；模型13,838,721,960B逐文件SHA及十篇token IDs再次通过，当前软件版本与westc准备记录相同。旧westc端点关闭，取消请求未送达；旧SSH最后观测仍WAITING，不声称已确认终止。采用[本次换机附录](../20260912_wisp_olmoe_r01/weste_resume_20260913/HOST_ADDENDUM.json)及新weste-r02目录，所有本次比较都在此处重新运行。独立续跑PID4534已启动；准备检查期间仓库rotation_first_swap六格实验占卡，按原占用规则等待。原代码、输入与F/X资格→X/F/F/X矩阵不变。

weste本次两个控制尝试已停止并完整取回：[第一次仅资源预检退出](../20260912_wisp_olmoe_r01/weste_resume_20260913/preflight_abort/retrieval_manifest.json)22978B/16文件；[第二次资格启动失败](../20260912_wisp_olmoe_r01/weste_resume_20260913/revision03/qualification_startup_failure/retrieval_manifest.json)78025B/31文件。第二次fullstage20进程运行8.761357917s，09:57:06.779及09:57:11.969 UTC两个GPU检查均空闲/PASS，但vLLM设备初始化时可用显存已降至17.34/31.36GiB，低于0.9对应28.22GiB，随后退出。未加载pager层、calls空、validation空、无raw请求；oneshot与性能没有启动。这是启动并发造成的资源失败，不是fullstage数值或机制负结论。它也实证说明边界查询不是原子的GPU独占保证；当前需协调其他会话暂缓新提交，再执行新的保留尝试，不降低显存要求来容纳并发，也不覆盖本次失败。

最近接续以[当前执行状态](../20260912_wisp_olmoe_r01/weste_resume_20260913/execution_status.json)为准：1789294392.894 UTC epoch的/proc、ps与nvidia-smi共同确认Qwen PID13275仍存活并占3650MiB，已消费1/16分片。本轮归类verified wait；前一轮的换机核验与启动竞态定位属于progress。同类GPU资源阻塞在三轮接续中持续，现有CPU准备已完成，下一步必须依赖协调独占窗口；主研究仍OPEN/MEASUREMENT_ONLY，fullstage数值与性能仍未测。weste我方控制进程均已终止；恢复时创建新的资格目录，不能重跑含失败results的原目录。

新主机准备期间的具体动作核对收紧了强基线范围：[FreeToken固定提交的decode路径](https://github.com/FlashML-org/FreeToken/blob/953565667f3141c90d0f0eb469bb2655d2407140/python/freetoken/layers/moe.py#L255)已把全局cache中的命中与当前新加载项映射后一起执行；[prefill路径](https://github.com/FlashML-org/FreeToken/blob/953565667f3141c90d0f0eb469bb2655d2407140/python/freetoken/moe/offload_cache.py#L606)借两个完整层buffer，可选地从稳定驻留区D2D取命中；buffer区域旧命中直接失效并重新加载，没有当前逐层canonical LRU终态合同。X的部分暂存与覆盖前保存仍有具体差别，但“直接用驻留cache并做物理映射”已有实现。F是普通完整stage适配，额外支付逐层私有写回，不能代表优化后的FreeToken；X/F测的是两套容量/复制/映射方案的整体差异。下一GPU实验仍完成X/F；形成相邻机制优势主张前，还需考虑同384槽、同runtime的全局按需LRU及完整union映射执行，因为逐层终态约束并非主研究问题本身的必要条件。

在上述两条X的repeat0固定轨迹上另做[global LRU384需求回放](../20260912_wisp_olmoe_r01/new_host_20260913/global_lru_diagnostic.json)：从初始化推进、按已记录reset清空、五段warmup连续，在measurement中分别需要9185/9541次H2D加载，对X实际5860/6258，多56.74%/52.46%，16层均增加。该模型按(layer,expert)键、当前active全保护及(last-call tick,slot)替换，不另设stage/D2D；它只能说明这两条已观察序列的cache需求。没有重新生成G的route/KV/output，不能将额外H2D与X的D2D相减得出性能，也不能据此判死G或代表FreeToken实际结果。

当前最小资源模型说明了为什么比较20与21槽：设层数L=16、逻辑专家E=64、总专家槽B=384。普通完整stage要求`L×C+E≤B`，均匀私有容量最多C=20；当前紧凑映射让仍驻留的本层权重直接参与执行，最坏同时满足`L×C+(E−C)≤B`，可容纳C=21（本实现仍付足384槽）。这只是槽数约束，不是性能上界。完整stage每call需搬入所有当前命中项并写回新留存项；紧凑映射只暂存即将覆盖但当前需要的入口驻留项。两者均支付当前miss的H2D，并沿各自路由推进。新增GPU对照将检验这点容量和D2D差别能否抵消映射、规划和kernel路径的代价；未把固定trace的字节差外推为请求加速。

## 7. 可能形成的论文结构

1. **现象与问题**：普通KV/prefill造成的吞吐—首token—长停顿权衡；进一步定位真实paging下新请求对旧请求的资源影响。
2. **可解释模型**：KV增长/释放、恢复等待、共享专家加载和重载；给出普通状态何时充分、何时缺项的可证伪预测。
3. **最小机制**：同pager下只控制admission和prefill；与普通deadline模型的必要差异，以及失败/超载行为。
4. **真实runtime与同预算实验**：主结果表以B0/B1/B2/B3为行，以steady/bursty的request-goodput、TTFT、max-ITL、完成时间为列；另列同资源/算法不变和运行成本。
5. **适用边界**：resident与offload、并集饱和、H2D隐藏、弱预测、尾部/公平性、兼容成本。单卡权重H2D不能换个带宽常数就外推EP；EP需要重新测token激活、rank/collective/拓扑。

若无方法收益，可形成“恢复等待与资源动作生效边界”的measurement论文资产；尚不能据本轮宣布论文主线已成立。

## 8. 本轮裁决合同

| 字段 | 当前判定 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；主问题与专家信息决策增量仍`OPEN` |
| Evidence type | 历史native raw与CPU结构；4个KV cells；完整WiSP/OLMoE接入及event-triggered native请求干预 |
| What was measured | K0的128测量请求；pager接入6次；0.11注入46次；0.26静态prefill16次；expert执行对照16次及独立资格4次；强底座事件注入16次；同实例重复12次；release基线12次；retention性能24次及资格6次；执行顺序性能48次及资格6次；同程序GC观测24次；trace生命周期96次；共同episode路径的两臂复测48次；不增组保护独立资格3次及性能36次、60份预热；静态层预算独立资格12次及性能144次、240份预热；共享池接口两轮共8请求、64输出，首轮FAILED、第二轮48个按位对照通过；真实共享生命周期另8请求64输出/320次测量层资格；共享池性能八格64测量、192请求/2560输出及320预热/10368输出，全部成本与125184层调用保留；当前分组约束与真实轨迹账；实际copy、KV storage、峰值和完整请求时间 |
| What was not measured | 不增组保护及静态层预算的稳定完整收益、共享池持续服务摊销收益、普通full-stage GPU对照、固定pager下联合动作净收益、expert history增量、真实大模型paging、质量/EP/生产SLO |
| Strongest baseline | 同384槽/1GiB KV、expert互斥分组＋release8的uniform24与固定M层预算；split21为容量/执行对照。X对U/M请求指标改善，但五warmup实验cycle更慢；普通20×16+64 full-stage仍GPU UNRUN |
| Oracle/headroom | 两个合法动作存在可复现等待权衡；未测统一目标的优化空间或Oracle |
| Claim ceiling | 单卡单模型、人工专家池的原生请求表征与受控即时干扰；无部署/SLO/质量/EP主张 |
| Failure category | 小chunk转移等待；同层重载和trace生命周期税已有修正；完整保护多组，受限保护放弃约65%搬运节省；保护项复用收益大量抵消。共享池请求改善伴随H2D/D2D及五warmup实验cycle代价，尚非持续服务净收益结论 |
| Resurrection condition | 真实paging暴露或新的可执行动作改变失败的成本环节；不换名/换seed复活 |
| One next smallest experiment | 普通full-stage 20×16私有+64共享的GPU资格及同预算性能；CPU已通过，GPU UNRUN。计入stage命中搬入与LRU写回D2D，区分已知stage与21+48紧凑映射增量 |

对研究问题的直接回答：**真实paging下，prefill工作量与专家执行组织会改变旧请求等待及完整请求代价。共享池X在两组新文档上相对U使capture下降5.542%/5.931%，旧max-ITL和新TTFT也下降；相对M同样改善请求指标，但X对U/M的五warmup加measurement周期均更慢。该周期不是持续服务摊销测量，当前既不能宣布净收益GO，也不能据此判死共享池或admission问题。主问题与专家信息决策增量仍OPEN；唯一下一步是普通20×16+64 full-stage的GPU资格及同预算性能，CPU已通过、GPU UNRUN。**
