# 当前 MoE 材料的 CCF B 投稿潜力判断

2026-09-05；仓库 `agent/publish-current-moe-code@2a37765fe522b1d74609a686f1d327ede7619a50`。

**结论：有可按 CCF B 目标投入的候选，尚无已完成的 B 类完整论文证据链。未来投入首选 SLO 并发控制；已有实测材料中，execution conformance 最接近一个独立测量主题。两者不是两篇已成立论文，Verify Precision 也仍只是条件备选。**

本判断只针对当前 MoE 仓库及已核验的相关兄弟 checkout 证据，不覆盖用户其它工程项目；是投稿成熟度和投资优先级判断，不是录用承诺、正式同行评审或新的 scientific verdict。
本轮只读权威入口、最新报告/addendum 与关键 artifact，不跑 GPU、不修改原裁决。之前新增目录仍未跟踪。

## 评价尺度

CCF 对会议目录采用 full/regular paper 口径；排名不能直接作为单篇论文质量结论。
见 [CCF 第七版发布说明](https://www.ccf.org.cn/Academic_Evaluation/By_category/)。
核对的 [2026 年目录 PDF 副本](https://scdm-shu.github.io/ccf/2026-CCF-Ranking.pdf) 中，IPDPS、ICPP、ICS、SoCC 属 B；HPDC 已属 A，不能沿用旧分类推荐为 B。
官网目录页本次直接访问返回 405，类别依据上述原目录 PDF 的表格，未用第三方打分替代。

[IPDPS 2026 官方 CFP](https://www.ipdps.org/ipdps2026/2026-call-for-papers.html) 明确设有 Measurements, Modeling, and Experiments 及 System Software 主题，评审关注正确性、原创性、技术强度、重要性和相关性；要求清楚说明问题动机、最相关先行工作的局限、关键贡献、方法和自身局限。这是审稿标准参照，2026 届投稿已结束，本判断不涉及下一届截稿日。

据此，本次采用的判断是：一个范围聚焦但重要的问题、一个有区别的技术贡献或系统规律、可信对照和与主张相称的验证，可以构成冲 B 的研究计划。复杂模型、8 GPU、完整集群系统或形式化 Oracle 都不是所有此类论文的统一前提；同样，单次 proxy 正结果也不是降低目标后就可省去的证据缺口。

## 材料逐项分档

| 方向 | 当前最强证据 | CCF B 潜力判断 | 当前阻断论文闭合的环节 |
|---|---|---|---|
| MoE SLO 非抢占并发控制 | [真实执行 runner、冻结输入与 CPU 准备](../expert_saturation/outputs/admission_capacity/20260905_pre_gpu_r01/REPORT.md)；新 GPU 容量实验 UNRUN | **最值得投入的 B 类方法候选**，尚无方法实测结果 | 普通 batch/KV/queue/近期延迟后 U/C 是否仍改变并发动作的收益；完整成本后是否进入 goodput |
| Execution conformance / N0d / Longrun A | [N0d](../expert_saturation/N0D_ROUTER_LOGIT_CONFORMANCE_REPORT.md) 的同前态、三进程复现；[Longrun A](../../idea-stage/longrun_A_execution_conformance/SOURCE_LOCALIZATION_REPORT.md) 的 attention/MoE 输出定位及 KV 传播 | **现有数据中最接近独立测量论文主题**，仍不够投稿 | 新规律相对已知浮点/batch-invariance 工作的区别；native transfer、内部算子定位和可用后果 |
| Verify Precision | 纠正后的独立 KV 旧数值机会；[新等预算方案](verify_precision/README.md)尚未 GPU 实跑 | **条件备选，现阶段弱于主线** | 新预算下是否优于强简单策略；真实 low/high backend 及验证成本是否允许部署 |
| StableBatch / SemanticFence | fresh action-space 和冻结 selector 失败；仍为单模型 route/top-k proxy | **有价值的局部测量素材，不是现成 B 类方法** | 两个 selector 失败不证明所有事前选择不可能；免费 Oracle projection 不证明低成本机制 |
| JoinStream | [四 cell 的有界负结果](../../docs/current/JOINSTREAM_FINAL_FREEZE_2026-08-10.md)：overlap window 未转成安全 completion gain | **保留为负结果资产，当前 formulation 不作独立论文主线** | 单卡 tail microbenchmark 的范围，尚不足确立广泛新规律；不重开已冻结机制 |
| Receiver / QuotaEP / Offload | 结构、质量或容量预算材料；系统存在性/成本 Gate 未闭合 | **储备问题，目前不能认定为成熟 B 类候选** | 自然暴露路径、适配后端和完整请求成本；尚未运行不等于科学 NO-GO |
| Rank-tail / Energy-SLO | [跨模型 rank-tail 结构证据](../../docs/ideas/A_rank_tail_fp8/README.md)、[单卡能耗表征](../../docs/ideas/energy_slo/README.md) | **支撑材料，不足直接承担现有系统主张** | 逻辑 payload / 局部功耗与真实 wire / serving 指标之间未闭合 |
| RCBA / BCRD / DEPA | 协议、局部回放或开发 harness | **不能按代码量判断达到 B 类** | 代表性完整请求数据和可执行干预尚缺；不为审稿准备重建通用仿真框架 |

## 为什么主线具备冲 B 的价值

拟回答的中心问题是：在普通服务状态相近时，专家工作集与集中程度，是否改变提高活跃并发的边际收益？如果这种差异可事前观测、持续到动作生效，并在统计与控制成本后改善请求 SLO，就能形成一个聚焦的系统贡献。

动作仅为非抢占接纳 cap，模型、router、top-k、precision 和 placement 固定，能够用真实独立执行对照，因果链短。研究难点是定义并证明有用的决策边界，而不是把 U/C 塞进通用 controller。
已有 [Gimbal](https://arxiv.org/abs/2606.15177) 使用专家压力参与调度，[Scorpio](https://arxiv.org/abs/2505.23022) 覆盖 SLO 接纳与 batching。因此不主张“首次专家感知调度”，也不能只凭控制位置不同形成创新。

我会要求以下三个证据包后，才给出“适合按 B 类完整论文投稿”的判断。它们是针对本研究的评估建议，不是会议统一硬门槛：

1. **问题与边界。** 自然 steady/bursty 负载下的 cap 容量曲线，确认并发、queue 和 SLO 区间有效；随后用可比历史/工作负载下的真实动作对照判断边际响应，不能把单纯压力分组当因果。U/C 与普通状态必须在动作前可用，保留没有额外价值的区域。
2. **贡献与收益。** 在同一 action space 比较 native/default、最优已测试静态 cap、只使用普通状态和近期延迟的强简单规则、最小 proposed；每策略独立未来状态，完整计入采集、决策、排队与自然降 cap 延迟。以请求级 goodput/TTFT/TPOT 为主，真实 observed cap 最优点不命名为动态 Oracle。
3. **适用性与可靠性。** 在一个代表性 serving runtime 确认核心结果；再选择能实际区分机制的第二模型或 routing regime，做受控重复和开销消融。先用少量代表性 cell 决定是否值得升级，避免正信号前铺大矩阵。单 GPU 主张不默认要求 EP；宣称 EP/NCCL 收益才补真实多卡。

若 ordinary-state 策略已经吸收 U/C 的作用，或信号在 cap 生效前消失，就停止当前专家感知控制 formulation。可复现、解释清楚的负结果有测量价值，但不会自动成为可投稿论文，也不能承诺“正负都能发 B”。

## 测量论文的真实机会与边界

N0d 的优势是同前态、独立 cache 与三进程复现。Longrun A 在不同协议下提供六个选定事件、24 个有限 prevalence cases，以及去掉 companions 后的短期 KV/logit/route 传播；它不能被合并成 N0d 的跨协议复现次数。

[Longrun A addendum](../../idea-stage/longrun_A_execution_conformance/REPORT_ADDENDUM.md) 已说明原 `/tmp` source captures 本地丢失、重复只在同一模型进程中、内部 expert grouping 尚未被观测。保存的新输出仍可核对，但原事件选择 provenance 不可完整重放。观察到的后续 predicted-token 差异为零；没有完整请求延迟、任务质量或部署后果证据。

所以合理的潜在论文中心是“dynamic MoE batching 何时破坏可复现执行假设，以及如何定位/约束这种破坏”。它需要相对已有 batch-invariance 工作有新的可验证边界，或者揭示 route replay、精度验证、性能评估中的具体系统错误，并给出实用诊断/修复。仅展示 FP32/BF16 数值不一致不够；这条路线可以没有加速 Controller，但必须有独立的技术贡献。

下一最小 transfer 仍是一组 steady 与 bursty 同前态事件，在代表性 native runtime 观察首个 attention/MoE 内部边界；原来源不完整的事件应重新采集，不能补写来源哈希。此项是条件测量路线，不与并发控制主线同时铺开。

## 把关结果与唯一下一步

单轮学术交叉检查沿用 GPT-5.6-Sol ultra reviewer `/root/independent_scholarly_review`，另由 `/root/capacity_metrics` 核对有界负结果的论文可用性。`review_independence=same-family`，结论 provisional；不记录虚构接收概率，不重复创建审计 trace。

| 固定字段 | 本轮判断 |
|---|---|
| Verdict | `CCF_B_POTENTIAL_CONDITIONAL / NOT_SUBMISSION_READY` |
| Evidence type | 最新报告/verdict/addendum 的桌面评估；既有 custom-GPU 测量与 CPU 准备分开 |
| What was measured | 本轮未产生新实验；核对已完成数据的范围、缺口及当前会议目录/CFP |
| What was not measured | 新容量、等预算 Verify、native transfer、系统收益和录用可能性 |
| Strongest baseline | 并发主线尚缺强 ordinary-state 实跑；conformance 已有同前态 serial 控制 |
| Oracle/headroom | 新容量动作空间未测；已有局部 Oracle 不升级为系统上界 |
| Claim ceiling | 可按 B 类目标投资和组织证据，尚未达到可投稿完整结果 |
| Failure category | 系统贡献证据未闭合；部分旧 formulation 成本/可选择性失败；并非整个问题族失败 |
| Resurrection condition | 真实公平动作收益，或具备 native 适用性与实际后果的新测量规律 |
| One next smallest experiment | 先跑已冻结的非抢占 cap 扫描；有效运行域出现后做最小真实动作对照 |

直接回答：**这些材料支持“有冲 CCF B 的研究基础和候选”，不支持“已经有一篇 CCF B 的结果”。下一笔计算投入首选并发控制主线；现有测量资产保留作条件路线。**
