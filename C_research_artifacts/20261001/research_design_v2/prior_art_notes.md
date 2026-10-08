# 机制近邻、等价性挑战与投稿核验（2026-10-01）

本文件为本轮新增研究笔记，不改写原始结果或权威裁决。已读用户附件、R/AGENTS.md 与现有 M0 退休计划；联网核对下列原文及可访问的官方代码。证据分为文献、独立数学推导、纯 CPU 组件三个层次，未运行 GPU 或 native 性能基线。

## 1. 明确裁决

**当前 M0 不能主张全新的时间容量打包算法。** 在相同输入状态、同一 FIFO 前缀选择、相同候选占用轨迹、相同 native legality 与 progress feasibility 下，逐轮 hard-cap 时间容量可行性与 M0 的峰值公式必然动作等价。退休端点求值只改变计算方式，不改变可行动作集合。这是独立数学结论，不能靠性能重复推翻。

**但仅把 CacheOPT 的预测值换为 hard cap，其具体逐对 embedding 并不与 M0 等价。** 两者甚至存在互不支配的接受状态，已实现下面的两个纯 CPU 反例。因此应同时保留“全局 hard-cap 打包”的最强简单挑战与“CacheOPT embedding”的发表近邻组件，不能用后者较受限的形状代替前者。

**重要限定：当前 M0 对新请求一直计完整上界矩形，并未利用新请求自己的未来退休。** 如果 B3 也这样表示候选，它与 M0 等价；如果 B3 对候选使用可兑现的 prefill/decode 时间轨迹及其退休，它可能比 M0 更宽松。“信息相同”本身不表示轨迹假设、服务约束与动作空间相同。

## 2. 原文对应的机制

| 工作 | 在线状态与动作 | 和本候选的关系 |
|---|---|---|
| CacheOPT | 使用预测长度及 padding、已分配/已用 KV；将短请求嵌入另一个请求暂未使用的预留空间，并结合 SLO 排序、分配削减与抢占 | 同步推进和未来释放复用已存在。M0 的 pooled peak 不等于其逐对 embedding；hard cap 本身只是替换长度信息。来源：[§3.3](https://arxiv.org/html/2503.13773v1#S3.SS3)。 |
| WAIT / Nested WAIT | WAIT 按类型入口累计量决定是否运行，阈值同时约束各 stage 批量；Nested WAIT 根据已观察到的 decode 进度，按存活者进入后续 segment 的队列阈值选择可运行前缀，未选中 resident 保留 GPU KV；launch 前检查投影状态，必要时从未选中的 resident boundary 请求中 LIFO 驱逐 | 它有独立的阶段服务动作，不能简化成固定并发上限。M1 的全体 protected decoder 每步推进与其阶段暂停并非同一动作；“考虑 phase”不是新颖性。来源：[§4–5 与附录 H.1](https://arxiv.org/html/2504.11320v4#S5.SS1)。 |
| PagedAttention / vLLM | 分块、按需物理分配与共享映射；论文版本按 FCFS 服务，资源不足时优先抢占较晚请求，可交换或重算 | 它提供执行层与基线背景，不是全程 hard-cap 安全准入证明。不能把 2023 论文行为直接当成本地最新 pinned scheduler 的代码事实。来源：[§4.5](https://arxiv.org/html/2309.06180#S4.SS5)。 |
| EASY / conservative backfill | EASY 保护队头预约；conservative 为所有已排队作业保留预约，回填不能推迟这些预约 | M2 只保队头属于 EASY 类思想，不应叫“保守回填新算法”。KV 随时间增长、decode token 预算与 prefill 服务同时预约可能是系统适配问题；轮次不延迟仍不等于墙钟 TTFT 不延迟。来源：[原文 §2](https://www.mcs.anl.gov/~kettimut/publications/iwpp02.pdf)。 |

CacheOPT 原文 §3.3.1 的判据为 `a_j − (u_j + s_i^o) − (s_i^p + s_i^o) ≥ b`，并选剩余预留最小的可行 host。预测错误时可能抢占被嵌入请求。最新可访问 arXiv v2 仍采用同一式；原文仍写录用后发布代码，本轮未找到可确认的官方 CacheOPT 源码仓库，因此不能声称官方复现。[v2 机制](https://arxiv.org/html/2503.13773v2#S3.SS3.SSS1)

## 3. 同轨迹等价性的具体含义（独立推导）

设 decoder `i` 的保守当前长度为 `x_i=P_i+O_i`、剩余硬上限轮数 `r_i=M_i−O_i`，块大小 `b`。沿兑现的一 token/轮进度，有

`e_i(k)=ceil((x_i+k)/b) · 1[k≤r_i]`。

按现有 M0 的保守计数，最后 cap 端点仍保留占用；新候选 `j` 的 profile 是常量 `U_j=ceil((P_j+M_j)/b)`。对 FIFO 前缀 `W`：

`∀k: Σ_i e_i(k)+Σ_(j∈W) U_j ≤ C` 当且仅当 `max_k Σ_i e_i(k)+Σ_(j∈W) U_j ≤ C`。

没有退休的区间内各 ceil 项不下降，因此只需初始点及退休前端点。加入相同的 legality/progress 布尔谓词并选择相同最长 FIFO 前缀，两个动作函数相同。这里推导的是现有保守 certificate；不是声称 `P+O` 等于原生物理 KV，也不是重新证明每个原生实现的分配/释放顺序。

M1 若写成 `max E_D + Σ U_P + Σ U_new`，在同一 D/P 分组与相同推进义务下仍是上述打包条件。可能的贡献必须落在跨 phase 仍能履行既有推进承诺、P→D 转换与预算不足后的安全 drain，而不能只落在换一个求和式。

## 4. 已运行的最小 CacheOPT 组件

新增 [prior_art_components.py](prior_art_components.py)，结果 [prior_art_component_result.json](prior_art_component_result.json)。组件仅实现 hard-cap 的**完整需求逐对 embedding + 未分配预留路径**，不包括预测器、SLO 重排、partial cuts、主动分配或抢占。它是自行实现的发表方法组件，**不是完整 CacheOPT，不是官方复现，也不是 native 基线**。native legality 仅作为两个组件共享的输入谓词，未模拟原生 allocator。

`(P,O,M)` 表示 resident 的 prompt、已输出、hard cap；候选写 `(P,M)`。块大小均为 16：

| CPU 状态 | M0 | 逐对完整需求组件 | 区分原因 |
|---|---|---|---|
| resident `(16,16,144),(16,128,144)`；新 `(96,32)`；容量 21 块 | 老请求峰值 13 块 + 新完整上界 8 块 = 21，通过 | 两个 host 各预留 10 块，只剩 1 块未分配；无单个 host 满足 embedding，不通过 | M0 可汇总早退休请求带来的全局容量 |
| resident `(16,16,144)`；新 `(16,16)`；容量 10 块 | 老请求最终 10 块 + 新完整上界 2 块 = 12，不通过 | 候选能在长 host 增长到相应区域前完成，通过 | embedding 利用候选自己的短生命周期；当前 M0 未利用 |

两个分离结果均在 `buffer=0/8/16 token`、`u=P+O` 与 `u=P+O−1` 下保持。共 2 个手工反例、12 个组件参数组合，断言通过；shared legality=False 时均拒绝。不是代表性状态发生率、不是性能，也不证明完整 CacheOPT 接受集合与 M0 互不支配。

复算命令（仅检查，无新文件）：

```sh
python3 C_research_artifacts/20261001/research_design_v2/prior_art_components.py
```

需要保存新结果时可加 `--out <新JSON路径>`；脚本拒绝覆盖已有结果。本轮实际结果用 `--out C_research_artifacts/20261001/research_design_v2/prior_art_component_result.json` 生成。

## 5. WAIT 官方代码状态与尚未完成的原生基线

[作者项目页](https://or4llm.github.io/) 的 Code 指向 [Luoxiaogan/vidur_or](https://github.com/Luoxiaogan/vidur_or)，属于 Vidur 模拟器。可访问的 main 文件有工程变体：

- [nested scheduler](https://raw.githubusercontent.com/Luoxiaogan/vidur_or/main/vidur/scheduler/replica_scheduler/nested_booking_limit_replica_scheduler.py) 使用固定三段，并用 `time.time()` 触发强制调度。
- [general nested scheduler](https://raw.githubusercontent.com/Luoxiaogan/vidur_or/main/vidur/scheduler/replica_scheduler/general_nested_booking_limit_replica_scheduler.py) 包含分段总量与逐 stage 阈值、`force_clear` 相关代码；清空路径的实际可达性与运行配置尚未核对，不能把“存在清空代码”直接报告成实际丢请求。

读取的是可访问的 main 内容；未获得固定 commit，也未证实这些文件与 v4 Algorithm 3 一致。v4 报告额外 SGLang GPU 验证，不等于本轮已拿到该实现。不能直接执行带墙钟触发或可能清空尾部的变体后称为“v4 忠实复现”。

**独立阶段动作 baseline 仍为 UNRUN / 待实现**。最小 native 移植规格：

1. 以 v4 Algorithm 3 为规范；外部等待队列不占 KV，boundary/interior 队列保留 native request 与物理块；不能通过丢 KV 来实现普通暂停。
2. 开发集冻结 segment 边界、阈值与到达率估计；只用已观察输出进度决定 stage，真实未来输出长度不得参与选请求。已知长度 WAIT 的可比变体用 hard caps，名称明确为 WAIT-hardcap。
3. 在每次 completion 后同步更新 stage；选择入口阈值均满足的最大 segment 前缀；每 stage 至多选其阈值数量，并遵守共同 token/sequence budget。prefill 必须按 pinned native 的 chunk 语义映射，不能冒充一个无成本 stage。
4. launch 前由 native allocator 对选中集合检查投影增长，修复仅考虑未选中的 resident boundary 请求；重算损失完整计入。若无合法 victim 仍不够，应 defer，并标明它相对论文伪码的实现补全。
5. 有限 episode 在外部到达结束后使用预声明 drain 规则继续处理残留请求，记录其成本，绝不清空后称作完成；drain/timeout 变体必须单独披露，不能偷换标准 WAIT。
6. 做 16–32 请求的状态与 complete-drain 资格检查后，才进入相同输入独立演化的原生性能对照。此规格需要新 adapter，当前不存在可诚实提供的运行命令，故不编造脚本路径。

M1 入选时 Nested WAIT 是必须补的独立近邻；只完成 CacheOPT embedding 不表示已覆盖全部发表动作。M0 阶段仍应报告该缺口，不能宣称全面超过 WAIT/CacheOPT。

## 6. 投稿来源核验

CCGrid 2027 官方 Track 3 包括 inference/model serving，要求实质系统贡献。[CFP](https://hpcclab.org/ccgrid27-call-for-papers/) 与 [投稿说明](https://hpcclab.org/ccgrid-2027-instructions-for-submission/) 本轮均成功直读：摘要 **2026-11-24 AoE**、全文 **2026-12-01 AoE**；投稿最多 **10 页，含参考文献、图和表**，IEEE letter 模板，双盲。不能将录用后的额外付费页政策当成初投稿页数。

CCF 官方域定向搜索返回了 [体系结构/并行分布/存储系统分类页](https://www.ccf.org.cn/Academic_Evaluation/ARCH_DCP_SS/) 的正文，会议 **C 类**列有 **CCGRID（序号15）和 ICPADS（序号20）**；直接 open 该页仍返回 405。又取得 [2026-03-31 第七版正式公告](https://www.ccf.org.cn/Academic_Evaluation/By_category/2026-03-31/870181.shtml) 的官方搜索正文，确认仅 **Full/Regular 正式长文**计入，Short/Demo/Technical Brief/Summary/Findings/Workshop 不计入，页面记载 4 月 9 日勘误更新；直接 open 同样 405。

所以准确证据状态是：**官方搜索索引正文已核对分类与长文规则；直接网页抓取、正式 PDF 下载核验及学校认定尚未完成**。本轮没有以第三方目录替代官方认定。ICPADS 当前年份 CFP/日期未检查，仍只是待核查备选。

## 7. 对唯一主线的建议

继续 M0 的受限执行契约与完整请求成本研究最省资源；暂停“新峰值调度算法”的叙事。M1 只有在 mixed-phase 自然阻塞足够多且契约可兑现时才成为条件扩展；M2 当前与 EASY 的动作重叠更直接，暂不实现。

最有杀伤力的下一证据有两类：同轨迹 B3 与 M0 动作一致，否定算法新颖性（已有数学理由）；健康 instruction 自然输出下，额外准入无法形成净请求收益，否定当前适用价值。前者要求收缩贡献，后者才可能要求停止这条运行域主线。新的系统贡献仍须独立证明，不能由“与某一个受限 embedding 组件不同”自动推出。
