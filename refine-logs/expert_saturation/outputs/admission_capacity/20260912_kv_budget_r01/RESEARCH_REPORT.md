# 从请求准入与 KV 实验走向真实专家 paging 成本

2026-09-12。本地分支 `agent/publish-current-moe-code`，HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`；本次 `git ls-remote` 核实远端仍为指定基准 `d65416c5cd09eb30aa6e46875339b16ebf7f5901`，本地领先三个提交。启动时已有 expert-union 文件改动，全部保留。本文不修改旧裁决或 `docs/current/README.md`，不 push。

**本轮已完成四项 KV 对照：增加 KV 消除了本组秒级暂停，整批吞吐提高3.99%/4.70%，但平均请求完成延迟略增1.02%/0.15%。这是目标权衡与配置测量，不是调度方法 GO。** 当前常驻 OLMoE 的长暂停不需要专家机制解释；真实 expert paging 仍未运行，不能据此判死不同的 offload 运行域。

## 1. 已有资产：9 月原始数据在本地，不必退回 8 月

已读权威入口：[current](../../../../../docs/current/README.md)、[ideas](../../../../../docs/ideas/README.md)、[tracker](../../../EXPERIMENT_TRACKER.md)、[旧初始 cap Gate](../../../experiments/DECODE_CAP_BRANCH_GATE.md)、[sketch README](../../../experiments/gpu_pressure_sketch/README.md)，并追踪下表实际调用。8 月 N0d 是局部数值差异测量；9 月 [admission README](../../../experiments/admission_capacity/README.md) 明确该请求研究不要求先完成 N0e。StableBatch 选择器、JoinStream 当前 formulation、fixed RankLane 的负结果保留。

表中 AC 指 `refine-logs/expert_saturation/experiments/admission_capacity`。所有命令从仓库根目录执行；GPU 入口需要其 prepared 数据和指定环境。

| 实际文件 / 入口 | 实现与实测状态 | 本轮可复用 / 尚缺 |
|---|---|---|
| [run_native_capacity.py](../../../experiments/admission_capacity/run_native_capacity.py) → [native_capture.py](../../../experiments/admission_capacity/native_capture.py) → `engine.add_request / scheduler.schedule / engine.step` | 9 月原生 GPU 实测；本轮 capture CPU 5/5 | 请求/内部 ID、step、receipt；`admission_s` 实际是提交时间，真正首次服务要从成功调度区间读 |
| [admission_feedback.py](../../../experiments/admission_capacity/admission_feedback.py) | 非抢占限制已用于 GPU 实验 | `max(target_cap, active)` 保留运行请求；降 cap 不释放 KV，waiting=0 时没有立即入场动作 |
| [run_prefill_share.py](../../../../independent_ideas_20260911/per_request_prefill_share_r01/run_prefill_share.py) | `set_empty_policy()` 排空后改变 scheduler token 预算/长 prefill 阈值；已实测 | 编译容量保持 1024；不是已完成的任意 in-loop prefill 配额控制 |
| [waiting_order.py](../../../../independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/runtime/waiting_order.py) | 已实测 WAITING 排序 | 只处理从未执行、无输出、未抢占的新请求；不能拿来重排待恢复请求 |
| [native preemption bundle](../20260908_native_preemption_r01/REPORT.md) 的 `native_capture.py / memory_telemetry.py / run_probe.py` | 4/4 episode 与原始 `raw.json.gz` 在本地 | 比通用 capture 更完整的 block、allocation、preemption、重算区间、输出前缀账本；本轮优先复用这一版 |
| [旧 KV attempt](../20260910_kv_budget_r01/await-readiness-state-20260910.json) 与 `execution.tar.gz` | 原 attempt 没有测量；状态是 `STOPPED_FOR_DIAGNOSIS`，不是仍在 WAITING | 原下载进程消失且缺分片；本轮在新机器新目录承接原封不动的包，不改旧 attempt |
| [route_pressure_sketch.py](../../../experiments/gpu_pressure_sketch/route_pressure_sketch.py) | CPU reference + 可选 GPU/Triton 实现；本轮 CPU 5/5 | 实际引用搜索只见测试；README 中三处 serving 接线仍是设计，无紧凑路径净收益 |
| [expert_union_tracker.py](../../../experiments/admission_capacity/expert_union_tracker.py) / `run_expert_union.py` | 已有实现与 CPU 检查；启动时 GPU UNRUN | pure-decode、截断 episode；不等于 pager、H2D 测量或请求级结果。并集也不等于 streaming 下必须同时驻留的字节 |
| [自然 prefill 尾块](../../../../independent_ideas_20260911/natural_prefill_tails_r01/STATUS.json) | 16 篇全文、29,103 prompt tokens 已准备；GPU 0 次 | 可复用独立自然输入；本轮不同时启动另一个 campaign |

对用户 9 月摘要的两点更新已由 raw 复算：

- 原 native32 对保守 cap29：两组吞吐约 +16.99%/+17.08%，全部请求完成；最长暂停 4.473448/4.512153 秒，其中首个重算 call 前为 4.357883/4.396435 秒，重算 call 首尾跨度仅 0.115565/0.115718 秒。这个跨度包含别的请求和埋点，不能写成纯 GPU 重算税。[原结果](../20260908_native_preemption_r01/REPORT.md)、[本轮重算](cpu_analysis/repeat0-native32.json)
- 最新 `per_request512` 相对各自同块 native1024：平均完成延迟 +1.5693%/+0.5111%，wall +1.1296%/+0.6420%。因此早先 pooled ITL 改善的摘要不足以支持完整服务收益，当前固定 512 formulation 保留负结果。[完整来源](../../../../independent_ideas_20260911/per_request_prefill_share_r01/)

## 2. 针对性查新：控制动作已高度重叠，剩余差异必须更窄

截至本次在线核查。表中“未定位源码”只描述检索结果；不声称作者从未发布。论文数值是作者报告，没有在本机复现。

| 工作 / 当前核查版本 | 控制对象、资源与动作前信息 | 目标 / 已测运行域 | 公开实现范围与本题差异 |
|---|---|---|---|
| [WiSP v2](https://arxiv.org/html/2606.21868v2)，2026-08-30 | 专家/KV 共同 VRAM 预算；pager 看当前 route/LRU；live rule 看已完成 turn 的 KV 峰值 | 3090、vLLM 0.11.2 eager、低并发多轮；live turn wall | **README 已过时：源码已发布 online dual resize。** [controller](https://github.com/nokia-applied-research/WiSP/blob/main/src/wisp/dynamic/controller.py) 以 `max(floor, ceil(last_peak×1.15))` 配 KV，剩余给专家，在完全 drained barrier 改池；不是正在生成时的新请求 admission/prefill 决策 |
| [FluxMoE v3](https://arxiv.org/html/2604.02715v3)，2026-09-10 | GPU 压缩/host 层次、驻留与 batch pressure、materialization 成本 | vLLM 0.23；大模型多卡；另有 Poisson 在线评测 | 动态 expert/KV 实验分成 41 次 launch，排除重建 prefill 和重启成本；固定 cap 在线实验不等于 admission controller；本次未定位官方源码 |
| [MoE-Gen v1](https://arxiv.org/html/2503.09716v1) | 模块级 batching、CPU 分工、weights/KV offload，硬件成本搜索 | 大批离线吞吐/BCT，接受更高延迟 | 原仓库转到 [BatchGen](https://github.com/batchgen-project/batchgen)，当前代码不自动等于论文版本；batch 摊薄搬运、成本模型已不是空白 |
| [MoE-Lightning](https://arxiv.org/html/2411.11217v1)，ASPLOS 2025 | CPU/GPU/I/O pipeline、权重/KV 放置、microbatch；给定 workload 长度和服务曲线 | T4/L4 等吞吐实验，含多卡扩展 | [官方 artifact](https://github.com/caoshiyi/artifacts/tree/asplos25)；Roofline/MILP 本身不是创新，尚非本题连续到达的非抢占动作域 |
| [Layered Prefill v2](https://arxiv.org/html/2510.08055v2)，MLSys 2026 | incoming prefill 的 layer groups、decode-first、KV 和等待队列，按长度选分组 | 在线 TTFT–TBT、Poisson/bursty、长上下文 | [实际 scheduler](https://github.com/scale-snu/layered-prefill/blob/main/nanovllm/engine/scheduler.py)；已直接控制新 prefill 对旧 decode 的延迟。主要重复流量是 HBM/GDDR→compute，不是 H2D cache miss |
| [DuoServe-MoE v2](https://arxiv.org/html/2509.07379v2) | prefill 双 stream、decode 路由/流行度/亲和历史 predictor，预取与缓存 | 主实验单请求，batch 1–12 扩展吞吐 | 本次未定位官方源码；分阶段与 batch 扩大专家需求已有研究，不能靠“考虑两个阶段”区分 |
| [FreeToken v1](https://arxiv.org/html/2608.16157v1) | H2D/CPU 带宽、当前 miss，CPU execution vs GPU fill，prefill streaming、弹性池 | 消费硬件、agent 工作流、多种精度/模型 | [当前代码](https://github.com/FlashML-org/FreeToken)；本次未完整审计 reconfiguration 路径，未确认本题的 admission/prefill controller；不能把其跨引擎分叉的工作流当同一轨迹 |
| [Gimbal v1](https://arxiv.org/html/2606.15177v1) | remaining/waiting prefill、KV、EP pressure；DP engine 分发、SJF aging、EP 放置 | 分布式在线 TTFT/TPOT/吞吐 | 论文明确称 pressure-aware admission；实际 admission 主要是选 DP engine，本次未定位官方源码；不能统称前人只做静态放置 |
| [QLLM v1](https://arxiv.org/html/2503.09304v1) | LS/BE×prefill/decode 四队列、expert-level preemption、状态保存 | A100/Mixtral 优先请求 TTFT/turnaround | 已改变已有请求执行顺序；本题最小版禁止这种新增抢占动作，本次未定位新官方 release |
| [Diff-MoE](https://doi.org/10.1145/3712285.3759903)，SC 2025 | **全文未取得，不填实现细节** | 题名/会议信息确认 | DOI/ACM 全文访问失败，未定位官方实现；仍是查新缺口，不能说已排除碰撞 |
| [ELDR v2](https://arxiv.org/html/2607.00466v2)，新增近邻 | 已执行 prefill signature→decode worker locality/load；显式考虑 batch 专家并集 | vLLM/ROCm、最多 40 MI300X | 已覆盖请求路由与已有 batch 专家共享；它在 prefill 后拿到的信息，不能提前供新请求入场使用 |
| [Expert Caching Evaluation v4](https://arxiv.org/html/2608.07911v4#S8.SS3)，新增近邻 | 固定 residency，24 admitted/每步8 served，历史 route 合批、4步公平轮转 | trace 模拟 transfer，不是 GPU/SLO 测量 | 已研究 cache-aware composition 和后续偿还；本题保持已有 decode 正常推进，只动新 admission/prefill，仍需真实 H2D 请求结果 |

WiSP main 已核实 commit `86f69720f0de0647c51728ea2e27e4289bf3dea2`（9 月 3 日）。其 [engine_access](https://github.com/nokia-applied-research/WiSP/blob/main/src/wisp/dynamic/engine_access.py) 和 [kv_resize](https://github.com/nokia-applied-research/WiSP/blob/main/src/wisp/dynamic/kv_resize.py) 要求同步 in-process、全部请求完成，再重建 KV pool。不能将该边界说成 arbitrary live resize。FreeToken 核查 HEAD `953565667f3141c90d0f0eb469bb2655d2407140`（9 月 12 日）。

**查新判断：**“专家/KV 共同预算”“成本驱动 scheduling”“新 prefill 干扰 decode”“cache-aware batch”均已有直接近邻。尚可检验的差异是固定专家缓存配置与后端下，**真实 H2D/cache 状态是否改变新准入/prefill 动作的相对优劣，且动作前可见信息能利用这个差异**。这是剩余问题，不是新颖性已证。

## 3. 一个首选问题与一个不同的备选

**首选：固定专家缓存容量/替换策略/精度/后端，在单卡真实 paging 下，新请求入场与 prefill 配额是否存在可复现的跨请求搬运代价；专家历史能否在强长度/队列/KV/延迟模型之外改变正确动作选择？**

资源是 GPU 显存与实测 H2D 服务能力；动作仅为入场时机/数量和 prefill 量；保持模型 top-k、权重、KV 语义与已有 decode 推进规则不变。新请求可能污染 cache，也可能与已有请求共享权重、扩大 batch 摊薄搬运，两种结果均允许。最强简单 baseline 是校准的固定 cap/prefill 加普通 KV/queue 反馈，同一 pager 同一预算；expert-history 只是最后一项消融。最便宜的反证是受控 paging 下动作曲面完全由 token/KV 解释，或更多 batch 的摊薄始终占优。此时停止该专家特征 formulation，不判死所有 offload。

选择它是因为已有请求账本、KV 测量和合法准入资产可复用，但尚无实际 pager，第一笔实现应关闭运行时兼容与真实传输缺口。只看到 union 或逻辑 bytes 不足以继续建 predictor。

**条件备选：普通 KV 的未来增长与恢复等待，何时使“减少准入目标”的反馈来不及生效？** 用块数/剩余最大输出/完成释放与请求 age 的模型，判断是否应更早保留余量；目标同时报告 TTFT、最大 ITL、完成量。此题不用专家信息，适用于通用 serving。已有 native32/cap29、反馈动作时点与本轮 KV 对照可支撑。反证风险是简单增加 KV 或静态配置已覆盖收益；发生这种情况应保留配置/目标错配的 measurement 结论，不把通用控制包装成 MoE 新方法。

本轮只完成 KV 因果链；不并行实现两个 controller，也不启动 ready-expert 等待合批机制 C。

## 4. 最小系统模型：先定义合法动作，再谈搜索

### 4.1 状态与可见信息

时间边界是完成一次 `engine.step` 后、下一次 schedule 前。状态为：

`S_t = (W_t, P_t, D_t, X_t; request ledger; K_pool/K_free; C_l; Q_H2D; completed history)`。

- W：已到达、从未执行的新请求；P：已准入未完成 prefill；D：正常 decode；X：被原生抢占、等待重算/恢复。未来未到达不进入 W。
- 每请求保留 arrival、prompt length、已计算高水位、已生成 token 数、声明的输出上限、块占用、last receipt/age、阶段、内部 ID。`arrival→submission` 与 `submission→first scheduled` 分开。
- 引擎保留实际 token budget、running/waiting、block size、排除 null 的 usable blocks、抢占事件。`gpu_memory_utilization` 是启动预算参数，不是 KV 字节或 resize API。
- 专家按 `(layer, expert)` 记录 resident、eviction/load、传输队列和历史需求。已有请求只用已完成 step 的历史；从未 prefill 的新请求没有专家 signature，先用长度/阶段的均匀或校准先验。首个 prefill 完成后才允许把其历史用于后续 chunk。
- 服务曲线须按实际 batch/prefill/decode/KV/capture shape 校准。H2D 带宽、copy 启动税、重叠和暴露等待分别实测；旧机曲线可定位采样点，不能直接当新机器真值。

### 4.2 两个动作与运行时边界

`a_t=(A_t, {p_i,t})`：从 W 选择少量新请求入场，给 P 与新 A 分配 prefill token 数。第一版仅枚举 FCFS 前缀 `|A|∈{0,1,2}` 和总 prefill `p∈{0,256,512,1024}` 中合法项，保持 engine/capture 上限1024；这是后续 pilot 起点，不是已测最优网格。已有 P 不能因只关注新 A 被无限拖延。

所有正常 D 保留原生下一 token 推进，预算约束 `|D|+Σp_i+原生恢复work≤T_engine`；缺 KV 导致的原生抢占由引擎决定并进入状态转移，不伪称该约束能保证无暂停。X 的优先级保持原生；若以后要改变它，单列新 action 和成本。

降 cap 只收紧未来 admission；不是释放 KV 的动作。本仓库目前只有排空后的预算切换，**逐 step prefill 配额 hook 尚未实现**。后续 hook 必须在原 schedule 的 token allocation 边界限制 prefill，并保留 decode、computed intervals、skip/recovery 语义；不能拿旧 drain API 当现成 in-loop actuator。

### 4.3 守恒、增长与释放

```text
M_nonexpert + M_expert_pool + M_KV_pool + M_workspace_peak <= M_GPU_budget
K_used(t) = Σ distinct allocated blocks(requests) <= K_usable
K_free + K_used = K_usable = K_total - null_block
```

物理 pool 与池内使用量是不同层级，不能重复相加；expert bytes 是 parameters 的子集，不能再加一遍；Torch reserved/allocated 与物理 storage 也不是可加项。host 常驻权重、pinned 权重/暂存和激活按唯一 storage 计数，另满足主机内存限制。

无 prefix sharing 的当前配置中 `k_i=ceil(computed_i/block_size)` 可作为逻辑块需求；forward 的已计算高水位不等于已输出长度。prefill/decode/recompute 按实际调度区间增长；完成释放全部块；preempt 释放旧 KV、保留已生成 token、把 computed 清零并进入 X。恢复需重新执行历史上下文；不能用 output receipts 反推出重算工作量。多步预测对未来 EOS/输出长度只用提交上限或独立校准分布，执行后用真实事件更新。

### 4.4 batch 共享与时间会计

已知某层 route 后，基础 miss 字节为 `D_l(B,C)=Σ[e∈U_l(B)\C_l]w_l,e`，每个专家最多计一次。一个新请求的增量是 `D_l(B∪{i},C)-D_l(B,C)`，通常不等于它单独运行的成本。后续驱逐损害要沿缓存替换状态推进；容量不足重载、padding/传输粒度、重排/额外复制另计。公式本身不保证所有需要的专家能同时放下。

动作前只能用估计 `q_j,e=P(e included by token j | history≤t)`；独立分析 baseline 为 `E[D_l]=Σ[e∉C_l]w_l,e[1-Π_j(1-q_j,e)]`。它允许一个 token 内 top-k 依赖，但跨 token 独立是假设。均匀例子 `E=64,k=8` 的并集期望在 B=8/16/32 为42.009/56.444/63.108个；B32约98.61%。[CPU解析输出](cpu_analysis/uniform_union.json) 不是实际路由测量，也不能据此否定逐层 streaming。

```text
request latency = observed arrival-to-completion host interval
step model = nontransfer service + exposed transfer wait + scheduler cost
load service estimate = physical bytes / measured effective H2D bandwidth + launch costs
```

`load service estimate` 不能原样再加到含它的 step latency。只有未被 compute/其他 transfer 隐藏的部分进入 exposed wait。恢复等待采用 `last receipt→first recompute call→last recompute return→next receipt` 互斥分段，跨请求间隔可能重叠，不得相加成 episode saving。

### 4.5 目标、预测与求解

先枚举上述少量合法动作，用同一普通模型与加 paging 成本模型预测未来几个 step 的 KV、完成释放、旧请求 age、TTFT。输出候选动作曲面/风险区间；服务曲线缺样本时报告超出校准支持，不自由外推。先看普通模型能否解释动作排序，再考虑部署一个短视或有限时域选择器，只执行第一步并更新状态。没有理由先用 RL、深网络或完整 exact future Oracle。

优化问题可写为：在内存、token预算、最大等待和事前选定的生成间隔风险约束内，选择合法短动作序列，使`E[截止共同T_end的SLO完成请求数 | S_t, 动作序列]`最大。短视实现必须用校准的剩余服务估计补足视野末端尚未完成的工作，不能把它丢弃以奖励无限拖延；该估计尚未拟合。当前交付只输出成本原语与动作设计，未谎称已实现MPC。尚无SLO阈值时先展示TTFT/maxITL/completion的非支配动作曲面，不临时给它们挑权重制造赢家。

对到达窗口 `[0,T_arrival]` 的固定请求集合，另设共同 drain 截止 `T_end`，固定请求级定义：`goodput=在T_end前完成且满足TTFT、meanTPOT、maxITL约束的请求数/T_end`。拒绝、超时和未完成均保留并不达标；未到达请求不进分母。单 token 请求 meanTPOT/maxITL 未定义，单列，不能静默变为通过。首次有限 cohort 同时报告服务起点到最后返回的 wall，避免与固定窗口 goodput 混用。

阈值来自事前应用目标或独立校准，不根据主结果挑选。当前继承5s/200ms仅作参考；最大 ITL 尚无业务门槛，先给请求级连续分布。pooled-token p99 仅辅助；32请求的 p99 不作生产尾部推断。失败的预测、无增量专家信息和简单策略获胜都正常保留。

## 5. 第一批实验及实现顺序

| 顺序 / 唯一待证问题 | 配置与对照 | 必采项 / 结果如何改变下一步 | 当前状态 |
|---|---|---|---|
| E0：普通 KV 是否足以解除秒级暂停 | 本轮固定cap32、OLMoE BF16、3072/1024、50ms、预算1024；95→90→90→95；每项新引擎、同3暖机 | 实际池、全部请求、最大ITL、completion、抢占/重算/恢复；若足量KV改善完整结果，当前常驻域先归因普通配置；若只零抢占而不改善，停止把抢占次数作代理 | 本轮实际运行，见第6节 |
| E1：pager 能否在现有硬件正确执行，并产生可计费的 H2D | 独立 WiSP/vLLM0.11.2；OLMoE BF16 eager，全驻留负控 vs cap8专家池；复用官方5个prompt/8输出 identity入口 | 16层注册、实际miss/eviction必须非零；输出/工作量、physical H2D、暴露等待另核；不支持模型/后端则修适配，不能判科学 NO-GO | **UNRUN，未安装/移植** |
| E2：固定 pager 后，入场/prefill 是否有普通模型解释不了的作用 | 先16–32请求的prospective pilot：预定新到达随机取“现在执行固定chunk”或“仅延后一schedule step”；后续才用独立16请求评价普通→transfer→history策略 | 动作前记录普通X、历史H；检验H是否改变动作效果，普通状态分箱内shuffled-H作负控；已有decode继续；所有轨迹真实独立演化；不因单个相关性直接训练复杂策略 | **UNRUN，依赖E1实际成本** |
| E3：自然超显存模型是否仍有该现象 | Qwen3-30B-A3B BF16、单卡同资源；复用已选最小动作，随后加低负载、近容量、bursty与少量长短混合 | 真实模型不适配或host/disk不够先记录工程阻塞；跨模型只确认已观察残差，不扫庞大全矩阵 | **UNRUN，当前磁盘不足** |

E1 的 OLMoE 限制池仅用于验证 pager/成本模型；不能声称 OLMoE 在32GB部署必须 offload。E2 正式比较必须让每个 policy 独立生成后续 batch/KV/route/完成轨迹；离线 replay 只能帮助定位，不是系统收益。paging阶段选择 WiSP/vLLM0.11.2 同步eager作为唯一待验证主runtime，0.26保留为已测常驻基准，不跨栈计算调度gain。若0.11.2在5090不兼容，再定位移植范围，不能同时建设两套策略系统。

WiSP的通用FusedMoE补丁/配置器包含OLMoE，但当前checkpoint+5090尚未实跑。直接移到0.26不是改版本号：route选择位置、`topk_weights/topk_ids`接口、post-load权重布局与modular kernel路径都变了。先使用隔离版本；官方identity脚本可能在安装失败后只打印warning，因此必须额外确认pager注册和缺页，不能用token相同掩盖未启用。[固定WiSP源码](https://github.com/nokia-applied-research/WiSP/blob/86f69720f0de0647c51728ea2e27e4289bf3dea2/src/wisp/integrations/vllm/fused_moe.py)、[0.26实现](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py)

新颖性 fresh-agent 复核（GPT-5.6-Sol，same-family/provisional）结论是 `NARROW_RESIDUAL_PLAUSIBLE / METHOD_NOVELTY_UNESTABLISHED`。接受其最关键修正：纯admission不能看新请求首chunk的实际route；若先运行chunk再控制剩余prefill，就必须改称有成本的prefill continuation。复核建议的“当前普通KV已解释则停止expert admission”仅作用于本常驻运行域，不外推为超显存paging family失败。没有生成额外`.aris`流程。

同 pager 对照顺序：调好的固定配置 → 普通状态反馈 → 加搬运成本 → 再加专家历史。校准/评价按独立请求集和独立arrival episode拆分，重置引擎/缓存历史，不随机打散同一episode的相邻windows；少量随机动作只支持探索性局部效果，最终选定策略仍需独立episode实跑。改 cache policy、prefetch 或 CPU execution 都是额外能力，逐个启用并消融。必要时把 Layered Prefill 风格的 decode 保护作为近邻动作比较；不同 runtime 的速度不能直接归于本机制。

若 H2D 已暴露但跨请求共享基本饱和，仍可研究 token/phase 服务曲线；只是不再期待专家 identity 改善决策。只有实测显示 repeated loading 且每次服务 token 很少，才讨论 ready expert 有限等待合批，它与 MoE-Gen/Layered Prefill/QLLM 另作 action-level 比较，不作为本轮第三套实现。

预算以执行单元而非假定GPU小时管理：E0只有4个已冻结cell；E1一个全驻留/受限池小对照；E2先一个工作点、两种小动作，再做一次反序重复。编译/加载成本先测，再估下一租期；不为前述UNRUN实验购买或启动新实例。未校准前不承诺完整主线需要多少GPU小时。

## 6. 本轮实际执行结果

**`MEASUREMENT_ONLY / NATIVE_SERVING`：4/4新引擎、128/128主测请求、12/12暖机完整；全部退出0并逐项回传校验。** 32篇输入在4个cell复用，不是128篇独立文本。模型加载/编译与暖机在服务计时外，采集/排队/恢复在服务计时内。[执行记录](execution.json)、[原始账本](gpu_results/)、[完整分析](analysis/report.md)

| 预定顺序 | KV GiB / usable blocks | wall s | 吞吐 req/s | 平均请求完成延迟 s | mean-TPOT p50 ms | 最长ITL s | 抢占 / 重算位置 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat0 / 0.95 | 16.45703 / 8425 | 22.55161 | 1.41897 | 21.24891 | 20.46498 | 0.08315 | 0 / 0 |
| repeat0 / 0.90 | 14.98438 / 7671 | 23.45206 | 1.36449 | 21.03519 | 20.23457 | 4.59064 | 2 / 7685 |
| repeat1 / 0.90 | 14.98438 / 7671 | 23.33146 | 1.37154 | 20.93874 | 20.15543 | 4.55972 | 2 / 7685 |
| repeat1 / 0.95 | 16.55078 / 8473 | 22.28443 | 1.43598 | 20.97066 | 20.24325 | 0.09888 | 0 / 0 |

0.95对0.90：吞吐+3.9928%/+4.6985%，wall−3.8395%/−4.4876%，最长暂停从约4.6s降至小于0.1s；但平均完成延迟+1.0160%/+0.1525%，mean-TPOT p50也略增。不能写成所有请求都更快或所有目标同时改善。尾部完成改善与大多数请求的轻微延迟可同时发生。[mean单独重算](cpu_analysis/completion_means.json)

TTFT p99为0.82609/0.82651s（repeat0高/低）及0.71748/0.80885s（repeat1高/低）；请求max-ITL p99从3.79789/3.77146s降到0.08315/0.09888s。pooled ITL p99只从28.13/28.23ms变到27.63/26.42ms，容易掩盖秒级victim。继承5s/200ms参考SLO四项都32/32通过，不能据它识别暂停改善；本次未事后挑新SLO阈值。

两次低预算最大暂停分别为 `4.590643=4.473617+0.117025+0` 和 `4.559720=4.442875+0.116845+0` 秒，仍主要发生在恢复执行前。重算工作守恒为 `138725=98304首次prefill+32736首次decode+7685重算`；高预算为131040、重算0。[本轮暂停账本](cpu_analysis/current-repeat0-budget90.json)

同repeat两个预算的32/32请求输出token序列完全一致，输入身份/到达配置一致；这只是本组输出一致性，不是质量benchmark。五份执行源码相同、软件相同、engine args仅预算不同；采样前后未见其它GPU计算进程，不声称连续进程监视。GPU UUID在各项初始化前/测量前/测量后记录相同，但未锁频，不声称热/功耗状态相同。四项均为单token返回chunk，`token_level_itl_resolved=true`、无插值。[针对性核查](cpu_analysis/integrity_observations.json)

0.95池在两次初始化相差48块/96MiB，必须保留实际池；两个重复各自都超过完整32请求所需8192块。0.90均为7671块。预算增配分别1.47266/1.56641GiB，不能把此结果写成**同显存预算**的策略收益。

四cell launcher总墙钟约392秒，包含首轮编译/加载、暖机和回传，不是纯GPU利用时长或计费时长。后验GPU进程列表为空。第一次本地分析因`metrics`导入路径缺失退出，补上冻结包`PYTHONPATH`后成功；没有改科学代码或重跑GPU，失败命令见COMMANDS。

[独立复核](EXPERIMENT_AUDIT.md)为`PASS_FOR_QUALIFIED_MEASUREMENT / P0=0 / P1=0`（GPT-5.6-Sol，same-family/provisional）；reviewer重新运行分析，结果与已留存JSON/报告逐字节一致。4个归档及其解包raw/config/log/warmup也再次比对一致。现有检查闭合，不继续增加同类审计。

本轮新GPU `GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f`，RTX5090 32607MiB；Python环境实际导入为 vLLM0.26.0、Torch2.11.0+cu130、Transformers5.15.1；三个模型分片SHA256校验通过。[就绪证据](preflight.json)

`free -h`显示的host可用613GiB不是容器额度；已补查cgroup：memory.max=98,784,247,808字节（92GiB），结束时memory.current约21.47GiB；锁页ulimit为unlimited但实际pinned分配峰值未测。数据盘可用30,272,585,728字节（约28.19GiB）。Qwen3-30B BF16 checkpoint数量级约60GB，当前空间不足，且WiSP约80GiB参考RAM需求必须重新对容器峰值验证。[资源记录](resource_after.json)。未下载新模型/安装新环境，没有新增租赁。

## 7. 已交付并实际运行的 CPU 部分

- [analyze_pause_ledger.py](../../../experiments/admission_capacity/analyze_pause_ledger.py)：兼容 `raw.json/.json.gz`，任意新输出，request/step/call关联、first-service、失败/未来请求留存；4个旧raw均已复算。5项针对性测试通过；重复输出被拒绝。[测试](../../../experiments/admission_capacity/test_pause_ledger.py)
- [paging_cost_model.py](../../../experiments/admission_capacity/paging_cost_model.py)：不可变请求状态、preempt/恢复/完成、cap不释放KV、唯一miss与独立期望、互斥时间桶。工作区并行补充生命周期校验后，本次重读并验证6项测试通过；仅CPU模型原语，未拟合、未接在线策略。[测试](../../../experiments/admission_capacity/test_paging_cost_model.py)
- [run_frozen_kv_remote.py](../../../experiments/admission_capacity/run_frozen_kv_remote.py)：使用已有SSH连接、验证固定模型和包、串行执行/回传；没有自动重试、安装或下载。三个工具及测试当前共506行，其中119行是迁移旧端点/逐项回传，118行是旧gzip账本缺失的可重算入口，137行是状态/成本原语，132行是针对性测试；不是新写500行的GPU方法。实际GPU A/B完全复用既有冻结代码，未改科学实现。

可重跑命令：

```bash
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_pause_ledger.py -v
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_paging_cost_model.py -v
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_pause_ledger.py \
  --raw refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/gpu_results/repeat0-native32/raw.json.gz \
  --output /tmp/pause-recompute-new.json
```

本次实际远端执行命令保存于 [COMMANDS.md](COMMANDS.md)。不直接重跑旧目录；保留所有原始尝试。代码的CPU通过不增加GPU证据层级。

## 8. 可能的论文结构与停止边界

1. **现象与模型**：用请求时间线展示KV增长、等待恢复、prefill干扰；证明哪些local代理不能解释完整目标。真实paging若出现残差，再给共享miss/缓存替换的机制解释。
2. **最小调度机制**：固定pager/预算，只动新入场和prefill；先强普通基线，只有专家信息确有动作增量才加入。中心贡献至多两项：可复现跨请求成本规律，以及捕获它的最小合法动作。
3. **实际runtime**：明确主版本、同步/图执行限制、identity、独立policy状态和完整开销；CPU模拟不替代系统结果。
4. **同预算实验与适用域**：主表请求goodput/TTFT/maxITL/completion，消融ordinary→transfer→history；第二模型与持续到达确认；单卡不外推EP。

如果强简单配置覆盖大部分收益，论文可以收敛成容量/暂停目标的测量边界；不能强行保留一个没有增量的MoE controller。EP以后需重新计费token activation、rank热点、collective/topology，不是把H2D带宽替换为NVLink。

| 最终合同 | 结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；普通KV增配足以移除本组秒级恢复暂停，存在吞吐/平均延迟权衡 |
| Evidence type | 原生vLLM同步in-process请求测量，单模型/单5090；CPU原语与文献核查另计 |
| What was measured | 95/90两组反序重复、实际池、全部完成/暂停/重算、host指标、身份/输出一致 |
| What was not measured | 真实expert paging/H2D、在线新策略、独立holdout、自然EOS、质量、第二模型、EP、生产容量 |
| Strongest baseline | 同cap32/token1024的原生0.90；0.95是更大KV配置对照。历史cap29不是新机同轮baseline |
| Oracle/headroom | 实测配置曲面存在吞吐/暂停改善；同预算策略Oracle、专家历史增量均未测 |
| Claim ceiling | 当前cohort/configuration下的重复方向；不作显著性、Pareto全面支配或MoE方法GO |
| Failure category | 原常驻域秒级损害是KV/恢复等待；不能据此证明expert paging action存在。平均延迟未改善 |
| Resurrection condition | 若固定预算、真实paging产生新的暴露传输且普通状态不能解释动作效果，重新评价专家信息；不同资源域明确新建问题 |
| One next smallest experiment | 独立WiSP0.11.2+现有OLMoE，5prompt/8output，全驻留/受限cap8，先验证16层pager生效、真实传输与输出；未通过前不写controller |

**下一笔GPU时间用在上述pager小实例；下一批代码只补pager激活、copy字节/暴露等待与现有请求账本的最小连接。** 当前常驻域先使用已验证的KV配置来消除这类长暂停；对平均延迟继续保留权衡。Qwen自然超显存验证先解决模型存储与容器峰值内存，不先耗GPU下载等待。
