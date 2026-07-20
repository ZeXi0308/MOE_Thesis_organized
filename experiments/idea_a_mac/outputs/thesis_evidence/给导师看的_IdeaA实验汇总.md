# Idea A 实验报告

---

## 2026-07-14 最新严格结论：QuotaEP-H 取代 R-layout 作为系统候选

R-layout formal 已经证明 rank criticality，但也证明固定 rank 不是 matched-wire 质量 winner。本轮进一步把 combine 的实际 backend wire 粒度纳入：对 hierarchical/high-throughput EP，只在 source topology domain 内先形成 `(token, destination-origin)` grouped partial，然后选择固定 quota 的 FP8 vectors，其余 MXFP4。该候选暂称 **QuotaEP-H**。

两模型、两 expert placement、每组 12 篇独立 WikiText articles、10,000 次 paired document bootstrap 的结论如下：

| setting | rank KL | gate KL | output-aware KL | global-budget KL | random KL |
|---|---:|---:|---:|---:|---:|
| OLMoE top-8 / contiguous | 0.006523 | 0.006261 | **0.005430** | 0.005211 | 0.015398 |
| OLMoE top-8 / round-robin | 0.006621 | 0.006395 | **0.005408** | 0.004830 | 0.025768 |
| LLM-jp top-16 / contiguous | 0.010286 | 0.010705 | **0.008309** | 0.006793 | 0.052695 |
| LLM-jp top-16 / round-robin | 0.008159 | 0.008007 | **0.007108** | 0.006697 | 0.036061 |

可以给导师的硬结论：

- **活下来的主线**：expert output 已知后的 output-aware late binding，在四组都显著优于 rank/gate/random；Holm 校正后仍成立；
- **必须砍掉的包装**：`||Σ go||` 相对 `Σg||o||` 没有稳定显著收益，因此不讲 cancellation novelty；
- **必须正视的代价**：peer/tile fixed quota 相对 global budget 有 `4.2%～22.3%` KL tax，top-16 尤其明显，下一步要测试 bounded borrowing；
- **系统价值仍是假设**：mixed logical wire 相对 grouped BF16 saving 约 `61.5%～61.7%`，相对 uniform FP8 transmitted bytes 再少约 `23.6%`，但没有真实 FP4 codec、RDMA、TPOT/P99；
- **论文定位**：当前足以支持继续做毕业论文；CCF C 仍需真实多 GPU kernel + serving，CCF B 必须做成 backend-level co-design。

完整方法、统计、查重边界与下一步见：`experiments/idea_a_mac/outputs/paper_validation/QuotaEP_H_查重与Grouped生死实验结论_2026-07-14.md`。下方 2026-07-13 R-layout 与 2026-07-11 PLTB 内容保留为演进史和 baseline；冲突处以本节为准。

---

## 2026-07-13 严格修正版：article-level R-layout Stage-1 Formal

旧 signal-comparison 使用 WikiText line 作为样本且 patched combine 的 BF16 accumulation order 不完全等价于原模型，相关 CI 不再进入论文主表。修正版已经完成：patched full 与原模型 logits exact equal（最大/平均差均为 0）；pilot 使用 test `[0:16]`，formal calibration 使用 validation `[0:32]`，formal test 使用从未查看的 test `[16:61]` 共 45 篇、11,475 个有效 next-token；pilot/formal test article hash overlap 为 0；paired bootstrap 5,000 次。

| 策略 | metadata-aware logical saving | mean token KL |
|---|---:|---:|
| uniform FP8 | 49.61% | 0.003359 |
| fixed rank tail-4 MXFP4 | 61.52% | 0.006042 |
| gate threshold MXFP4 | 61.63% | 0.005635 |
| tail-mass MXFP4 | 61.55% | 0.005616 |
| contribution oracle | 61.52% | 0.005559 |
| head / interleaved control | 61.52% | 0.035938 / 0.027707 |

严格结论：

- rank criticality 仍成立：head/interleaved controls 在 45/45 formal articles 上都比 tail 差，KL 是 tail 的 `5.95×/4.59×`；
- fixed rank 不是质量 Pareto winner：gate/tail-mass/oracle 相对 R-layout 的 paired KL delta 为 `-0.000406/-0.000426/-0.000482`，CI 全部排除 0，约降低 `6.7%～8.0%` total KL；
- 因此 R-layout 的唯一待验证优势是定长 buffer、静态 offset 和规整 fused pack/unpack。若真实 kernel 不能明显降低动态 gate selector 的系统开销，主方法应切换为 gate；
- 这仍只有一个模型/一个语料域，没有 native FP4 kernel、all-to-all、TTFT/TBT 或 P99；更多样本、跨域和跨模型证据仍需第二 corpus/model。

完整报告：`experiments/idea_a_mac/outputs/paper_validation/olmoe_r_layout_article_stage1_formal_2026-07-13/Stage1_RLayout_WikiTextFormal结论.md`。系统执行主线已收敛为 `R-layout quality → R-layout kernel → R-layout serving → receiver-aware 条件扩展`；Graceful 与 QTree 不再并行实现。

---

## 2026-07-11 第三轮更新：硬件格式审计改变了主线

旧报告中“FP8-first PLTB rank-segmented combine 是当前核心”的结论已被新的格式对齐实验收缩。旧 PLTB 结果在 per-row symmetric INT4 proxy 下成立，但 MXFP4/NVFP4 block-scaled E2M1 复测显示：fixed tail 的 KL 已从 0.03032 降到约 0.00684/0.00571，PLTB 的额外收益缩小或反转；MXFP4 专用 layer profile 也未稳定优于 fixed tail。因此不再把 PLTB 当成已经成立的核心创新。

当前更可信的主线是：**FP8-first rank-segmented FP4 combine**。在相同 tail-half 预算下，MXFP4 的 head-control KL 是 tail 的 8.75×，NVFP4 是 2.56×；gate threshold/oracle 相对 fixed rank 的优势很小，说明固定 rank 可能以少量质量差距换取规整 two-lane kernel。含 scale metadata 后，uniform FP8、tail-half MXFP4、tail-half NVFP4 的理论 wire saving 约为 49.61%、61.52%、60.69%，仍不等于真实 TPOT/P99 收益。

给导师汇报时请优先使用：

`experiments/idea_a_mac/outputs/paper_validation/论文Idea最新文献与硬件格式审计_2026-07-11.md`

下文 PLTB、INT4 62.5% 与 receiver proxy 内容作为研究演进和历史证据保留；与本节冲突时，以本节为准。

## 2026-07-11 更新说明（含 receiver-aware 混淆剥离结果）

本报告后半部分保留了早期实验的历史数字和图，用于展示 idea 如何收缩；其中部分 KL/PPL 使用旧聚合口径，`receiver_group` 也只是按 expert id 构造的 endpoint proxy。**最终论文主表不应直接引用这些旧数字。** 修正后的 held-out 结果、正确 EP 语义拥塞回放和完整边界见：

`experiments/idea_a_mac/outputs/paper_validation/论文后续验证_2026-07-11.md`

第二轮 layer-budget 创新实验见：

`experiments/idea_a_mac/outputs/paper_validation/论文创新实验_第二轮_2026-07-11.md`

**receiver-aware 混淆剥离与信号分解实验（新增）见：**

`experiments/idea_a_mac/outputs/paper_validation/receiver_isolation/`（`receiver_isolation_report.md`、`receiver_sender_decomposition_report.md`、`expert_popularity_stability_report.md`、`stale_profile_vs_oracle_report.md`）

这批新实验发现，第七节 7.3 中报告的 `recv_aware_hot1/hot2` 以及 `PDF论文的选择.md` 里 `tail_budget_profile_ports ≈ tail_budget_scheduler_receiver ≈ tail_budget_greedy_ports` 的结论，**混入了一个未被识别的 confound**：`run_ep_congestion_sim.py` 的打分函数对所有非 random 策略都硬编码给 remote（跨节点）pair 加最高优先级 bonus，导致三种"不同"selector 在数值上几乎完全相同（如 num_jobs=1 时三者 bottleneck_saving_vs_fp8 均为 `0.22550`），这掩盖了"识别具体哪个端口更热"这个信号本身的真实贡献，也让"离线 profile 有效"这一结论站不住脚。剥离该 confound 后的核心结论：

- 在固定候选池（tail-rank ∩ inter-node，对 hot/cold/random 完全相同）内，`hot` 选择仍稳定大幅超出 `random` 的 95% CI（例如 hotspot/16 jobs/frac=0.5：hot=0.242 vs random=0.121±0.004），`cold` 几乎归零——**热点识别本身是真实信号，receiver-aware 的直觉基本方向没错**；
- 但拆解 `max(sender_load, receiver_load)` 的两个分量后发现：**hotspot 型负载下收益几乎全部来自 receiver 侧**（token-origin GPU 热度，调度器可提前知道），**balanced 型负载下收益主要来自 sender 侧**（expert-owner GPU 热度，占比常年 75%~99%）；
- 用 disjoint calibration/test 检验发现，expert 热度（sender 侧信号所依赖的量）跨样本 Spearman 相关仅 `0.39`~`0.50`，最热 sender-rank 跨集合一致率仅 `25%`——**sender 信号内容依赖、非平稳，不能像 PLTB 的 layer sensitivity 那样离线 profile 一次复用**；
- 直接用 calibration 算出的静态 profile（即论文里 `tail_budget_profile_ports` 实际依赖的信息）去打分 test 场景，效果只比 random 高约 `0.03`，远低于 oracle 上界的 `0.056` 差距，在个别配置下甚至**低于** random（hotspot/4 jobs/frac=0.25：stale=0.031 < random=0.062）——**当前论文里"离线 profile 已经足够"这一表述需要撤回**。

因此，receiver-aware 章节需要重新表述为：**只保留、强调 receiver-side scheduling signal（调度器已知、不需要等本层路由结果的 token-origin 热度）这一支，它在负载不均衡（hotspot）场景下几乎免费拿到全部收益；sender-side 信号虽然在均衡负载下贡献更大，但因跨样本不稳定、依赖本层实时路由同步，只能作为 oracle upper bound 或 future work，不能默认为可离线部署的信号。** 详见第七节 7.3 更新版和新增的"十一、Receiver-Aware 混淆剥离与信号分解"。

最新判断如下：

- WikiText-2 calibration/test 已分离；test 为 32 条文本、1,925 个 next-token 位置；
- 指标已改为 per-token KL、正确累计的 corpus PPL 和 1,000 次 bootstrap；
- fixed rank-tail4 在 62.5% 理论 payload saving 下 KL 为 `0.03032`；
- gate threshold 在 63.13% saving 下 KL 为 `0.01836`，比 fixed rank-tail 低约 `39.4%`；
- cumulative tail-mass 在 62.74% saving 下 KL 为 `0.02130`，低约 `29.7%`；
- contribution oracle KL 为 `0.01813`，说明 gate threshold 已接近质量上界；
- head4 control KL 为 `0.28016`，约为 rank-tail 的 `9.24×`，所以 tail/head sensitivity 仍然成立；
- 多 MoE/多请求回放已改用 `expert_owner_rank -> token_origin_rank`，但仍只是 bandwidth-only trace proxy，不是 TPOT/P99 实测。
- 新增 **Profiled Layerwise Tail Budget（PLTB）**：固定全局 INT4 slots，只在层间按 sensitivity 重分配 `m_l`；
- OLMoE 相同 62.5% saving 下，PLTB 将 fixed-tail KL 从 `0.03032` 降到 `0.01957`，降低约 `35.5%`，paired 差 95% CI `[0.00734, 0.01588]`；
- PLTB 追回 fixed-rank 到 gate-threshold 质量差距的约 `89.9%`，KL 只比 gate threshold 高约 `6.6%`；
- LLM-jp top-16 上，PLTB 将 KL 从 `0.01739` 降到 `0.01397`，降低约 `19.7%`；
- OLMoE mean-KL 最优布局在 calibration 从 8 条扩大到 16 条后完全不变，top-16 在 4→8 条时也保持不变。

这曾是第二轮基于旧 INT4 proxy 的主线判断，现已由本文最前面的第三轮更新取代。当前 R-layout 使用固定 rank split 的 FP8/block-FP4 two-lane；PLTB 不进入默认配置。

---

## 实验来龙去脉

最开始的想法是：MoE combine 阶段要把多个 expert output 传回并加权求和，能不能不要所有 output 都用 BF16，而是按 gate/rank 给不同 output 用不同精度。

**如果 uniform FP8 已经很好用，那从 BF16 出发做复杂 rank-aware mixed precision 就不够有说服力**。所以这次先补了 FP8 fake-quant baseline。结果发现，uniform FP8 确实很强：直接拿到 50% byte saving，而且 KL/PPL 基本不变。

因此当前主线改成更窄、更容易防守的问题：**既然所有 combine output 先用 FP8 已经是合理默认，那么 FP8 之后还能不能安全再压一部分到 INT4？**

这里的两个信号分工需要更新为：

- `rank/gate/tail-mass` 共同回答"哪些 output 更安全"；新 held-out 结果中 gate 信号优于固定 rank，rank 的优势候选是规整 two-lane layout。
- `sender/receiver critical-flow` 回答"有限预算先压哪里"；真实 combine sender 是 expert owner，receiver 是 token origin。旧 `receiver_group` 只能作为 expert-owner/endpoint proxy。

---

## 摘要

本报告围绕毕设方向 A（MoE Combine 通信压缩）完成前置验证。核心发现链：

1. **top-k 内部不是平均分工**：同一个 token 选中的多个 experts 里，后几个 rank 的 gate/contribution 明显更小（3 模型 tail-rank share <10%）。
2. **rank 主要用来判断 INT4 能不能用**：在 INT4 下，压 tail 和压 head 的 KL 差 58×；在 FP8 下只差 2×。说明 FP8 已经稳，rank 的价值是判断谁还能继续降到 INT4。
3. **uniform FP8 是强 baseline**：它直接带来 50% byte saving，且 KL=0.293、PPL 基本无损。论文不能讲"rank-aware 替代 FP8"，而要讲"FP8 之后还能安全多压多少"。
4. **INT4/drop 的风险来自误差逐层传递**：FP8 的误差小，多层叠加仍温和；INT4/drop 误差大，会影响后续层输入，出现明显放大。
5. **FP8 + tail-INT4 可以把 saving 推到 62.5%**：只把低 gate 的 tail ranks 从 FP8 降到 INT4，256 样本 PPL +0.30（约 1.4% relative）；同样 saving 下压 head/random 会崩（+6~7 PPL）。
6. **rank 是便宜的 coarse signal，但尚不能称为够用**：新 held-out 指标下，gate threshold/tail-mass 明显优于固定 rank，gate threshold 已接近 `gate×||output||` oracle；rank 是否值得保留取决于定长 two-lane kernel 的系统优势。
7. **layer sensitivity 让静态 rank 重新变得有竞争力**：PLTB 在固定 payload 下保护敏感层，OLMoE/LLM-jp 分别比 global fixed-tail 降低 35.5%/19.7% KL，并保留每层定长布局。

**本文不再挑战 FP8，而是在 FP8 之后找额外可压缩空间。当前核心候选使用固定 rank split 构造 FP8/block-FP4 two-lane；PLTB 仅作格式依赖的二级增强。多请求拥塞扩展只考虑 receiver-side 可部署信号；当前仍需真实 collective、TBT/P99 验证。**

### 导览

原方案的核心是 `Profile-Guided Receiver-Aware Rank-LUT Partial Combine`：用离线 profile 得到一张 `LUT[layer, receiver_group, rank] -> precision`，运行时只查表，不做在线优化。本次实验先补了 FP8 fake-quant baseline。结果发现，uniform FP8 在 combine 阶段已经能直接带来 50% byte saving，并且 KL/PPL 基本无损。

因此后续实验改成验证一个更具体的问题：

> **在 uniform FP8 已经很好用之后，MoE combine 里有没有一部分 output 可以继续降到 INT4 或直接 drop，而且不会明显伤质量？**

围绕这个问题，后续实验追加了三组更具体的验证：

- **FP8 之后还能不能继续省**：从 uniform FP8 出发，把 tail rank 进一步降到 INT4，观察是否能从 50% saving 推到 62.5%。
- **为什么必须按 rank 选**：在同样 62.5% saving 下加入 tail/head/random 反例对照，验证 INT4 放在 tail 上温和，放在 head/random 上会崩。
- **安全 INT4 预算应该压在哪里**：加入 receiver-aware hot/roundrobin/random/cold 反例对照，验证同样总 saving 下，把 tail-INT4 优先放到热点 receiver/endpoint group 更能降低 max-receiver bytes。这里仍是通信字节近似指标，不是实测 TBT。

---

## 一、研究问题与方法

### 1.1 问题定位

MoE 通信优化集中在 dispatch 和 expert placement；combine 阶段被忽视的独有自由度是——**gate 权重作为重要性信号，免费可得，且在 dispatch 阶段结构上做不到**。

$$y_t = \sum_{e \in S(t)} g_{t,e} \cdot o_{t,e}$$

每个 (token, expert) 对应独立 output，gate 权重已知，按 gate 区分传输精度天然规整。

| 阶段 | 传输对象 | gate/rank 信号是否自然可用 | 本文处理 |
|---|---|---|---|
| dispatch | 同一个 token hidden state 被复制给 top-k experts | rank 已知，但所有副本输入相同；按 rank 做差分量化会破坏规整性，且误差会经过 expert MLP 非线性放大 | 不作为主线 |
| combine | 每个 `(token, expert)` 返回独立 expert output | gate/rank 已知，且最终线性加权聚合；tail output 的误差会被小 gate 缩放 | 主优化对象 |

**文献背景：** 真实 MoE serving 里，expert parallelism 通常会引入两次 all-to-all：先 dispatch token hidden states 到 expert 所在设备，再 combine expert outputs 回原 token 位置。[Lina（USENIX ATC'23）](https://www.usenix.org/conference/atc23/presentation/li-jiamin)指出 distributed MoE 训练和推理低效的主要原因是模型计算中穿插的 all-to-all，并在 A100 testbed 上通过优化 all-to-all 将 P95 inference time 平均降低 1.63×。[NCCL EP / DeepEP](https://arxiv.org/abs/2603.13606) 一类系统工作也直接把 `dispatch` 和 `combine` 做成一等通信 primitive，面向 decoding 的 low-latency 模式和 prefill 的 high-throughput 模式分别优化。[MixServe](https://arxiv.org/abs/2601.08800) 进一步把 MoE EP 的 A2A 通信拆成 Dispatch 和 Combine，并给出同阶通信量 $O(\frac{bs}{d}\cdot hk)$；其 DeepSeek-R1 / Qwen3 serving 实验通过优化 AR/A2A 通信获得 1.03–1.66× ITL 加速和 5.2%–50.3% throughput 提升。

因此本文的保守表述是：**MoE all-to-all 已经被系统论文证明会显著影响 serving latency / tail latency；combine 是其中一条返回路径，而且它比 dispatch 更适合利用 gate/rank 做差分精度。**

### 1.2 方法定位

原方案（rank-aware mixed precision from BF16）在 ≤50% saving 被 uniform FP8 主导。reframe 后：

> **不与 uniform FP8 竞争，在其之上叠加**：所有 output 先用 FP8 拿到 50% saving；然后只把低 gate 的 tail output 继续降到 INT4，把 saving 推到 62.5%。rank 的作用不是微调 FP8，而是防止把 head output 错压成 INT4。

drop 可以作为 aggressive upper bound 或 future work，但主线先采用更规整、更可部署的 FP8/INT4 差分量化。

### 1.3 运行时策略候选

当前不再预设一张大而全的 `LUT[layer, receiver_group, rank]` 已经成立，而是保留两个需要真实 kernel 对决的候选：

- R-layout / PLTB：`LUT[layer] -> num_int4_tail_ranks`，通过离线 sensitivity probe 得到，定长 FP8/INT4 two-lane；
- G-layout：gate threshold 或 cumulative tail-mass 动态 mask，显式计入 metadata、scan、variable-size buffer 与 divergence。

critical-flow budgeting 只有在 selector kernel 跑通后才作为扩展，避免同时堆叠多个未验证机制。

---

## 二、实验环境

| 项 | 设置 |
|---|---|
| 机器 | Mac M5 Pro / 48GB unified memory / CPU-only |
| dtype | bfloat16 |
| 数据集 | WikiText-2 validation（PPL/KL）+ MMLU（下游任务） |
| 主模型 | OLMoE-1B-7B-0924（16 MoE layers, 64 experts, top-8） |
| 跨模型 | LLM-jp E32-k16（top-16）、Mixtral-TinyMistral（top-2） |
| FP8 实现 | E4M3 fake quant，per-token scaling（`torch.float8_e4m3fn`），对齐 MegaScale/DeepSeek 的 per-tensor scaled FP8 |

### 2.1 术语

| 术语 | 本报告中的含义 |
|---|---|
| rank | 某个 token 的 top-k experts 按 gate 从大到小排序后的位次。rank 1 是最高 gate，tail rank 是较低 gate 的 expert。 |
| gate | router 给 selected expert 的权重，combine 时用于 `Σ g·o` 的线性加权。 |
| FP8 默认方案 | 第一步不做复杂策略，所有 combine outputs 都用 FP8 传输，先稳定拿到 50% byte saving。 |
| FP8→INT4 边界 | 第二步才问：哪些 output 还能从 FP8 再降到 INT4。这个边界很危险，压错 head 会崩。 |
| receiver_group | 当前实验里按通信目标/专家归属做的流量分组，用来近似"哪个接收端更热"。它是 endpoint/receiver 近似分组，不是完整 serving 系统里的真实 request receiver。 |
| 误差逐层传递 | 前面 MoE 层的量化误差会改变后面层的输入；如果误差太大，后面层会继续放大这个偏差。 |
| KL / PPL | KL 衡量输出分布形状变化，PPL 衡量语言模型困惑度；二者是诊断指标，最终仍需下游任务和真实 latency。 |

### 2.2 实验地图

| 实验 | 回答的问题 | 结论 |
|---|---|---|
| 实验一 | top-k 内部是否真的有 rank 长尾？ | 有，tail rank contribution share 在 3 个模型上都很小。 |
| 实验二 | rank 在不同精度档位是否都重要？ | INT4 下 rank 决定生死，FP8 下 rank 差异很小。 |
| 实验三 | uniform FP8 是否会打掉原方案？ | 会，所以必须把 FP8 当默认第一步，而不是对手。 |
| 实验四 | FP8 之后还能不能继续压？ | tail rank 从 FP8 降 INT4 可把 saving 推到 62.5%。 |
| 实验五 | 这个结论能否跨模型、能否接热点通信端？ | top-16 仍成立；receiver-aware 能降低最热端字节数。 |
| 实验六 | PPL/KL 退化是否转化成任务退化？ | MMLU 小样本未观察到下降，但仍需生成式任务补充。 |
| 实验七 | held-out 上 rank 是否胜过 gate/tail-mass？ | 否；rank 质量更差，必须用系统规整性证明其价值。 |
| 实验八 | 多 MoE/多请求下预算应该压在哪里？ | trace proxy 支持 critical inter-node flow，但真实 collective/P99 未证明。 |

---

## 三、实验一：Rank 长尾验证

对每个 token、每个 MoE layer、每个 selected expert rank，统计 `contribution = gate_weight × ‖expert_output‖`，`share = contribution / sum(top-k)`。

| 模型 | top-k | tail rank | tail-rank median share | rank1/tail ratio | 判定 |
|---|---:|---:|---:|---:|---|
| OLMoE | 8 | rank-8 | **4.91%** | **5.43×** | 强成立 |
| Mixtral-TinyMistral | 2 | rank-2 | **0.014%** | **14656×** | 极端成立 |
| LLM-jp E32-k16 | 16 | rank-16 | **2.05%** | **9.39×** | 强成立 |

![C1 rank 长尾验证](figures/fig1_rank_contribution_longtail.png)

**图 1**：(a) OLMoE top-8 各 rank 的 median contribution share——rank-8 仅占约 4.9%，rank-1 占约 28%，长尾明显。(b) 跨模型对比：三个模型的 tail-rank share 均远低于 10% 阈值。

第一，top-k experts 不是平均贡献，rank 越靠后，gate-weighted contribution 越小；第二，这个现象不只出现在 OLMoE，一个 top-2、一个 top-8、一个 top-16 模型都能看到 tail 很小。它支撑后续的基本假设：**tail output 不是完全不能动，而是更适合先尝试低精度。**

Mixtral 的 `14656×` 看起来非常大，主要是因为它是 top-2 模型，而且 router/gate 很接近 one-hot：很多 token 上 rank1 几乎拿走全部权重，rank2 的 median contribution 接近 0。这个结果不应作为主论据夸大，只说明"在另一个 MoE 架构里也能看到 tail 很小"。主证据仍然是 OLMoE top-8 和 LLM-jp top-16。

> **C1 成立**：top-k 内部不是均匀的，rank 可以作为低成本的重要性近似信号。

---

## 四、实验二：压 INT4 时，压 tail 和压 head 差多少

做法很简单：每次只改一个 rank 的精度，其他设置不变，然后看模型质量掉多少。

这里还顺带做一个 FP8 对照：

- 如果只是降到 **FP8**，rank1 和 rank8 的差别很小，说明 FP8 本身已经比较稳。
- 如果降到 **INT4**，压 rank1 会明显伤质量，压 rank8 影响很小，说明 INT4 不能随便用。

这组实验的结论是：**FP8 可以作为默认方案；INT4 只能优先给低 gate 的 tail output。**

### 4.1 INT4 档内：rank 差异巨大（58×）

这一步故意只压一个 rank，目的是把"省了多少通信"固定住，只比较"压哪里"会不会影响质量。

OLMoE 是 top-8，每个 token 会选 8 个 expert output。这里有两种对照：

- 压 **rank8**：只把最低 gate 的那个 expert output 降到 INT4。
- 压 **rank1**：只把最高 gate 的那个 expert output 降到 INT4。

因为只压 8 个 rank 里的 1 个，而 BF16 降到 INT4 单个值省 75% 字节，所以总 byte saving 是 `1/8 × 75% = 9.375%`。这样设置后，rank8 和 rank1 的通信收益完全一样，质量差异就主要来自"压的是不是重要 output"。

这张表里的 strategy 只改变"哪个 rank 用 INT4"，其他设置保持一致：

- `rank8_int4`：只把最低 gate 的 rank8 output 降到 INT4，其他 rank 不动。
- `rank1_int4`：只把最高 gate 的 rank1 output 降到 INT4，其他 rank 不动。
- `uniform_int4`：所有 rank 都用 INT4，是一个极端低精度参考。

| strategy | byte saving | KL | PPL Δ |
|---|---:|---:|---:|
| `rank8_int4`（tail） | 9.375% | **0.361** | -0.029 |
| `rank1_int4`（head） | 9.375% | 20.989 | +4.488 |
| `uniform_int4` | 75% | 27.152 | +6.678 |

跨模型一致：OLMoE 58×、Mixtral 41×、LLM-jp 108×。

![Rank sweep](figures/fig2_rank_sweep.png)

**图 2**：将 INT4 逐一施加到 rank 1–8 上（相同 byte saving 9.375%）。压 rank-8 的 KL 仅 0.36，压 rank-1 的 KL 高达 21.0——低 58 倍。

这张图的关键不是"INT4 总体好不好"，而是控制变量：每次只压一个 rank，byte saving 完全相同。结果说明，同样省 9.375% 字节，压 tail 和压 head 的质量代价完全不是一个量级。因此 rank 在 INT4 场景下不是锦上添花，而是决定压缩是否安全的必要条件。

![Cross-model advantage](figures/fig4_cross_model.png)

**图 3**：跨模型验证——三个模型上 tail-rank INT4 的 KL 均比 rank1 INT4 低一到两个数量级（41×–108×）。

图 3 是对图 2 的防偶然验证。它说明"head 不能乱压、tail 相对安全"不是 OLMoE 一个模型的巧合，而是在不同 top-k 结构里都成立。这里的跨模型结果也防住了一个质疑：rank 长尾可能只是某个 checkpoint 的 router 分布特例。

### 4.2 FP8 档内：rank 差异消失（2×）

相同 byte saving（6.25%）下，FP8 rank sweep：

| rank | KL（rankN_fp8） |
|---|---:|
| rank1 | 0.257 |
| rank8 | 0.130 |
| **ratio** | **1.97×** |

这说明在 FP8 这个精度下，压最高 gate 的 rank1 和压最低 gate 的 rank8 都没有造成明显质量问题，二者 KL 只差约 2 倍。换句话说，FP8 已经比较稳，没必要为了"谁用 FP8"做复杂 rank 策略。

### 4.3 这说明什么

FP8 已经足够稳，所以在 FP8 里纠结"rank1 用 FP8 还是 rank8 用 FP8"意义不大。真正危险的是 INT4：如果把高 gate 的 head output 降到 INT4，误差会明显伤模型；如果只把低 gate 的 tail output 降到 INT4，模型效果影响极小。

所以这里的 rank 不是一个复杂调度器，而是一个简单规则：**高 gate 的 output 不动，低 gate 的 output 才考虑继续压到 INT4。**

---

## 五、实验三：FP8 默认方案与误差放大

### 5.1 uniform FP8 主导 ≤50% saving

| strategy | byte saving | KL | PPL Δ |
|---|---:|---:|---:|
| `uniform_fp8` | 50% | **0.293** | **-0.054** |
| `uniform_int8` | 50% | 5.859 | +1.281 |
| `rank8_int4` | 9.375% | 0.361 | -0.029 |
| MILP（旧, ε=0.1） | 55.8% | 9.414 | +1.442 |

这个表的作用是判断"原来从 BF16 出发做复杂混合精度"还有没有必要。这里要特别注意：**`rank8_int4` 是旧策略，不是 `uniform_fp8 + rank8_int4`。**

具体来说：

- `rank8_int4` 表示：原本所有 rank 都按 BF16/full precision 传，只把最低的 rank8 单独降到 INT4。因此它只省 9.375% 字节。
- 后文的 `fp8_r8int4` 才表示：先所有 rank 都用 FP8，再把 rank8 从 FP8 继续降到 INT4。因此它是在 50% saving 的基础上继续省，能到 53.1%。

- `uniform_fp8` 和 `uniform_int8` 都是 50% byte saving，但 FP8 的 KL 只有 0.293，INT8 的 KL 是 5.859。也就是说，同样 1 byte 传输，FP8 比 INT8 更适合 expert output，因为浮点格式有动态范围。
- `rank8_int4` 这个旧策略说明"压最低 rank 是安全的"，但它太保守，saving 只有 9.375%。所以如果不先采用 uniform FP8，单独压一个 tail rank 的通信收益太弱。
- 旧 MILP 虽然做到 55.8% saving，但 KL 到 9.414，PPL 也涨了 +1.442。说明原来的 BF16/FP8/INT4/drop 混合优化，即使多省 5.8pp 字节，也付出了明显质量代价。

因此这个表真正说明的是：**旧的"从 BF16 出发，只挑少数 rank 降 INT4"被 uniform FP8 打败了；但它不否定"uniform FP8 之后继续把 tail rank 降到 INT4"。** 后面的实验四才是在验证新策略：`uniform_fp8 -> fp8_r8int4 -> fp8_r78int4 -> fp8_r5678int4`。

![FP8 vs INT4 Pareto](figures/fig11_fp8_vs_int4_pareto.png)

**图 4**：FP8 frontier（蓝）vs INT4 reference（红 X）vs MILP（紫菱形）。Y 轴 log scale。FP8 frontier 完全主导 INT4/MILP 策略空间。

这张图是本报告转向 FP8-first 叙事的关键。读图时看 Pareto 位置：越往右表示省字节越多，越往下表示 KL 越小；蓝色 FP8 曲线在相同 saving 下 KL 更低，在相同 KL 下 saving 更高。也就是说，原来从 BF16 出发设计复杂 mixed precision 的路线，在 50% saving 附近会被 uniform FP8 直接压住，所以后续主线必须改成"先 FP8，再研究哪些 tail 可以继续 INT4"。

### 5.2 为什么 INT4/drop 比 FP8 危险

- **FP8 良性叠加**：16 层全 FP8，total KL=0.293（~1× 线性）。
- **INT4 灾难放大**：MILP 多层 INT4/drop，actual/predicted KL = **94×**。
- **drift 拆解**：FP8 的 drift fraction（59-73%）反高于 INT4（48%），但绝对 drift 相近（0.176 vs 0.182）——FP8 把数值误差压到极小，drift 才在残差中占比上升。因绝对值小，多层叠加仍可控。

直观理解：MoE 不是只量化一层。前面层的 output 一旦被 INT4/drop 扰动，后面层看到的 hidden state 就变了，后面的 router/expert 也可能跟着变。FP8 的误差太小，传到后面也不明显；INT4/drop 的误差大，就可能一层层放大。

当前方案不试图在线修复这种误差，而是避免在危险位置制造大误差：**head rank 保 FP8，只让 tail rank 进 INT4。**

---

## 六、实验四：FP8+tail-INT4

### 6.1 方法

从 uniform FP8 出发，把 tail rank 从 FP8 升级到 INT4。OLMoE top-8 扫描：

下面表格里，`fp8_r8int4` 表示"全 FP8 后只把 rank8 改成 INT4"，`fp8_r78int4` 表示"全 FP8 后把 rank7、8 改成 INT4"，以此类推。`INT4 ranks` 这一列表示有多少个 rank 从 FP8 继续降到了 INT4。

| strategy | saving | KL | PPL Δ | INT4 ranks |
|---|---:|---:|---:|---:|
| `uniform_fp8`（默认第一步） | 50.0% | 0.293 | -0.054 | 0 |
| `fp8_r8int4` | 53.1% | 0.495 | +0.032 | 1 |
| `fp8_r78int4` | 56.25% | 0.831 | -0.022 | 2 |
| **`fp8_r5678int4`** | **62.5%** | 2.005 | **+0.010** | 4 |
| `fp8_r345678int4` | 68.75% | 4.495 | +0.415 | 6 |
| `uniform_int4` | 75.0% | 27.152 | +6.678 | 8 |

KL 平滑爬升（无突变），PPL 在 62.5% 前几乎不动。

![FP8+tail-INT4 Pareto 与 rank 控制](figures/fig12_fp8_tail_int4_pareto.png)

**图 5**：(a) FP8+tail-INT4 升级扫描——PPL 在 62.5% saving 前几乎不崩；(b) 同 62.5% saving 下 rank 选择造成 tail 无损 vs head/random 崩 +5~6 的巨大差距。

图 5 是当前方法的主证据。左图回答"FP8 之后还能不能继续省"：随着更多 tail rank 从 FP8 降到 INT4，saving 从 50% 往 62.5% 推进，PPL 没有立即崩。右图回答"为什么必须按 rank 选"：同样 62.5% saving，如果 INT4 放在 rank5-8，退化很小；如果放在 rank1-4 或随机 rank，PPL 明显崩。换句话说，收益不是来自"多用了 INT4"本身，而是来自"只在低 gate 的 tail 上用 INT4"。

### 6.2 Rank 选择控制（同 62.5% saving，tail vs head vs random）

这张表固定总 saving 都是 62.5%，只改变"哪些 rank 被放到 INT4"。所以它不是在比较省多少，而是在比较同样省这么多字节时，INT4 放错位置会不会伤质量。

| strategy | KL | PPL Δ | INT4 放哪 |
|---|---:|---:|---|
| **`fp8_r5678int4`（tail）** | **2.005** | **+0.010** | rank5-8（rank-aware） |
| `fp8_r1357int4`（odd） | 23.510 | +4.984 | 随机（无 rank 信号） |
| `fp8_r1234int4`（head） | 24.591 | +6.058 | rank1-4（反 rank-aware） |

**同 saving 下，INT4 放哪里决定结果**：tail PPL +0.01，head/random +5~6。也就是说，额外 12.5pp saving 不是"随便挑一半 output 降到 INT4"就能拿到，必须避开高 gate 的 head output。

### 6.3 扩样本收敛确认（32→128→256）

| 样本数 | tail PPL Δ | head PPL Δ | tail 增长率 |
|---:|---:|---:|---:|
| 32 | +0.010 | +6.058 | — |
| 128 | +0.198 | +5.355 | +0.188 |
| **256** | **+0.296** | **+7.353** | **+0.098（减半）** |

PPL 收敛在 ~+0.30（1.4% relative），增长率减半。head 持续崩 +6~7。**结论在 256 样本下稳定。**

---

## 七、实验五：跨模型与选择信号验证

### 7.1 LLM-jp top-16 跨模型确认

| 62.5% saving | KL | PPL Δ |
|---|---:|---:|
| tail（rank-aware） | 1.296 | +0.350 |
| head（反 rank-aware） | 20.515 | +6.269 |
| odd（无 rank 信号） | 21.457 | +6.326 |

top-16 上 tail 温和（+0.35），head/random 崩（+6.3），KL 差 15.8×（比 OLMoE 的 8.7× 更强）。**模式在 top-8 / top-16 上一致，非偶然。**

### 7.2 为什么不用更精确的 contribution

| 选择信号 | KL | 运行时代价 |
|---|---:|---|
| rank-tail（按位置） | 2.005 | **免费**（gate 排序，dispatch 前确定） |
| contrib-tail（按 g×‖o‖） | 1.401 | 需 ‖o‖（expert 算完才知道） |

`contribution = gate × ||expert_output||` 更精确，但它有一个部署问题：必须等 expert output 算完之后才能知道 `||output||`。这时再决定压缩策略，会打断通信路径，也不适合静态查表。

rank 虽然粗糙，但路由结束后立刻可得，而且效果和 contribution oracle 同量级。因此这里说"rank 够用"的意思是：**它不是最优 oracle，但它足够接近，而且几乎没有运行时代价。**

### 7.3 Receiver-aware：把安全压缩预算放到热点通信端

> **rank 决定能不能压，receiver/endpoint group 决定优先压哪里。**
>
> **⚠️ 2026-07-11 更新：本节以下内容使用的是旧的 expert-id endpoint proxy，且其在多请求拥塞回放中的"离线 profile 有效"结论已被证明主要来自一个未识别的 confound（详见第十一节）。本节数字保留作为历史记录，论文不应直接引用；请以第十一节的剥离后结果为准。**

前面的实验只回答质量问题：哪些 output 降到 INT4 不容易伤模型。系统上还要回答另一个问题：如果只能压一部分 tail output，应该优先压哪里？

这里采用一个保守的通信近似指标：每层看哪个 receiver/endpoint group 收到的 bytes 最多，认为它更可能是该层 combine 通信的瓶颈。然后比较两类策略：

- hot：优先压热点 group 上的 tail output。
- cold/random/roundrobin：压冷点、随机压、均匀轮转压。

下表里各 strategy 的含义：

- `uniform_fp8`：所有 combine output 都用 FP8，是 50% saving 的默认第一步。
- `fp8top4_rest_int4`：所有 group 都采用同一规则，top4 ranks 保 FP8，其余 tail ranks 用 INT4。
- `recv_aware_hot1`：只在每层最热的 1 个 receiver/endpoint group 上，把安全 tail output 从 FP8 降到 INT4；其他 group 保 FP8。
- `recv_aware_hot2`：类似 hot1，但每层压最热的 2 个 group。

| strategy | total save | max-recv save | max-byte save | imbalance | PPL Δ | KL |
|---|---:|---:|---:|---:|---:|---:|
| `uniform_fp8` | 50.0% | 50.0% | 50.0% | 1.183 | -0.054 | 0.293 |
| `fp8top4_rest_int4` | 62.5% | 62.8% | 62.7% | 1.175 | +0.010 | 2.005 |
| **`recv_aware_hot1`** | 53.8% | **56.3%** | **56.3%** | **1.119** | **-0.017** | **0.780** |
| `recv_aware_hot2` | 56.9% | 59.1% | 59.1% | 1.124 | -0.071 | 1.087 |

**关键结果**：

- **recv_aware_hot1 比 uniform_fp8 多降低 6.3pp 的 max-receiver bytes**，且没有观察到 PPL 退化（-0.017 在 32 样本噪声内）。它只压热点 group 的 tail rank，冷 group 仍保 FP8。
- **imbalance 从 1.183 降到 1.119**：最热 group 的流量被压下来，max/mean 差距变小。
- **它仍不是实测 TBT**：这里能证明的是"最热端字节数下降"，不是"真实 P99 一定下降"。真实收益要看 pack/unpack、all-to-all kernel、batch size、拓扑和 overlap。

#### 同 total saving 下的 receiver 选择对照

上表各策略 total saving 不同。为证明 receiver 选择（而非少压）驱动 max-receiver 降低，固定每层压 1 个 group 的 tail-INT4（~53% total saving），比较选 hot vs uniform vs random vs cold：

| strategy | total save | max-recv bytes | vs hot1 | imbalance |
|---|---:|---:|---:|---:|
| **hot1（receiver-aware）** | 53.8% | **1.64B** | **1.000×** | **1.119** |
| roundrobin（均匀） | 53.3% | 1.82B | 1.110× | 1.228 |
| random | 53.4% | 1.81B | 1.100× | 1.218 |
| cold1（反 receiver-aware） | 52.7% | 1.88B | 1.145× | 1.251 |

**同 total saving 下，hot1 的 max-receiver bytes 比其余策略低 10-14.5%**。这说明热点选择确实能降低"最堵通信端"的字节数，而不是因为 hot1 总体压得更多或更少。

> 这部分的防守边界（旧版）：receiver-aware 可以作为系统目标扩展，但目前还不是完整 latency 结论。更稳的说法是：`rank` 负责质量安全，`receiver_group` 负责把安全压缩预算放到更像瓶颈的位置。**该边界描述本身不算错，但第十一节的解析回放（多层、多并发请求）用的是不同脚本 `run_ep_congestion_sim.py`，其中"离线 profile 有效"这一延伸结论存在 confound，已被撤回，具体见下。**

---

## 八、实验六：下游任务（MMLU）

PPL/KL 退化是否转化为任务质量损失？MMLU 200 题（10 科目 × 20 题，zero-shot log-likelihood）：

| strategy | saving | MMLU accuracy |
|---|---:|---:|
| full | 0% | **42.5%** |
| uniform_fp8 | 50% | 40.0% |
| **fp8_r5678int4（tail）** | **62.5%** | **42.5%** |

tail-INT4 在 62.5% saving 下 MMLU 准确率 = full（42.5%），不低于 uniform_fp8（40.0%）。**+0.30 PPL 退化没有转化为任务准确率损失**——PPL/KL 是诊断指标，不等于生成质量。

---

## 九、结论

> **⚠️ 2026-07-11 第三轮更新：以下结论基于旧的 per-row symmetric INT4 fake-quant proxy，已被硬件格式审计（见文档最前面的第三轮更新说明）部分推翻。fixed tail 在 MXFP4/NVFP4 对齐模拟下本身已经很稳（KL 从 0.03032 降到 0.00684/0.00571），第 5、6 条里"rank + receiver_group 两级 LUT"的具体表述、尤其是把 PLTB/INT4 当默认方案的部分需要撤回。tail/head sensitivity（第 2、3 条）在新格式下仍然成立，可保留；receiver-aware 的边界结论进一步被第十一节的混淆剥离结果收紧。本节整体保留作为历史记录，当前可信叙事以最前面的第三轮更新说明和 `论文Idea最新文献与硬件格式审计_2026-07-11.md` 为准。**

**FP8 已经是很强的默认方案。本文现在更适合讲一个更具体的问题：FP8 之后，哪些 combine output 还能继续降到 INT4，以及这些安全的 INT4 预算应该优先用在哪些热点通信端。**

1. **FP8 应该作为默认第一步，而不是被挑战的 baseline。**  
   实验发现 uniform FP8 已经能直接拿到 50% byte saving，且 KL/PPL 基本无损。因此论文不应该讲"rank-aware 替代 FP8"，而应该讲"FP8 之后还能不能继续安全省一点"。

2. **rank 的作用很具体：判断哪些 output 可以从 FP8 继续降到 INT4。**  
   在 FP8 档内，压 rank1 和压 rank8 的差距只有约 2×，说明 FP8 本身已经足够稳；但一旦进入 INT4，压 tail rank 和压 head rank 的 KL 差距达到 58×。所以 rank 不是一个泛泛的"重要性分数"，而是一条简单的安全规则：高 gate output 不压，低 gate output 才考虑 INT4。**（格式审计后仍成立，但倍数在 MXFP4/NVFP4 下变为 8.75×/2.56×，不再是 58×。）**

3. **tail-INT4 有用，head/random-INT4 不安全。**  
   在同样 62.5% saving 下，tail 策略 PPL 基本不动，而 head/random 会带来 +5 到 +7 的 PPL 崩溃。这说明额外的 12.5pp saving 不是随便把一部分数据降到 INT4 就能拿到，必须用 rank 约束在 tail 上。**（tail/head 分离方向在 MXFP4/NVFP4 下仍成立，量级已缩小，见第三轮更新说明。）**

4. **多层误差放大不是靠复杂补救解决，而是靠少碰危险位置。**  
   FP8 多层叠加仍然温和，但 INT4/drop 会明显改变后续层输入。当前方案的处理方式很朴素：不在 head rank 上制造大误差，只在贡献小的 tail rank 上用 INT4。

5. **receiver-aware 目前证明的是热点字节数下降，不是完整延迟收益。**  
   质量安全先由 rank 决定；在已经确定只能压 tail 的前提下，receiver/endpoint group 决定这些安全压缩预算应该优先给谁。同 total saving 下，压 hot group 的 max-receiver bytes 比 random/cold 低 10-14.5%。这支持"热点优先压缩"的系统直觉，但真实 TBT/P99 还要在 A100 serving 或 trace replay 里测。**（这批旧数字用的是 expert-id endpoint proxy；第十一节的混淆剥离结果进一步表明，这里的"receiver_group"实际驱动力经常是 sender 侧信号，且 sender 侧不可离线部署，只有 receiver-side scheduling signal 可保留，详见第十一节 11.5。）**

6. ~~**最终方法是一个保守但可部署的两级策略。**  
   默认所有 combine output 用 FP8；然后用 `rank` 选出可以继续 INT4 的 tail；再用 `receiver_group` 把有限 INT4 预算放到热点通信端。运行时只是查 `LUT[layer, receiver_group, rank] -> precision`，不做在线优化。~~
   **（撤回。当前主线已改为 FP8-first rank-segmented FP4 combine：默认所有 output 用 FP8，tail rank 静态升级为格式对齐的 block-FP4（MXFP4/NVFP4），运行时查 `LUT[layer_id] -> h`（固定 split，不含 receiver_group 维度）；PLTB 与 receiver_group 均降级为可选二级增强或 future work，见文档最前面的第三轮更新说明。）**

---

## 十、局限性与下一步

| 局限 | 状态 | 计划 |
|---|---|---|
| 无真实 serving latency | 目前只有最热端字节数近似指标；它能说明热点字节下降，但不能等价于真实 TBT/P99 | 补 8×A100 trace-replay combine benchmark，测 pack / all-to-all / unpack / P50-P99 |
| 模型偏小 | OLMoE 1B / LLM-jp 920M | 补 Qwen top-4 / DeepSeek-V2-Lite top-6 |

---

## 十一、Receiver-Aware 混淆剥离与信号分解（2026-07-11 新增）

### 11.0 为什么要做这组实验

`run_ep_congestion_sim.py` 和 `run_quality_safe_congestion_frontier.py` 里，`tail_budget_profile_ports` / `tail_budget_scheduler_receiver` / `tail_budget_greedy_ports` 三种"不同"selector 在打分后都会被下面这行代码强制加上 remote bonus：

```python
if policy != "tail_budget_random":
    remote = (rows["sender_rank"] // gpus_per_node) != (rows["receiver_rank"] // gpus_per_node)
    scores = scores + remote.astype(float) * (float(scores.max()) + 1.0)
```

即：只要不是 `tail_budget_random`，所有 remote（跨节点）候选一律排在所有 local 候选之前，不管三种 selector 各自算出的分数是什么。`congestion_simulation.csv` 里 num_jobs=1 时三者 `bottleneck_saving_vs_fp8` 完全相同（均为 `0.22550`）就是这个 confound 的直接证据。这意味着此前"profile/scheduler/greedy 三种复杂度不同的 selector 效果相近"的结论，很可能只是三者共享的"优先压 remote"规则在起作用，而不是"识别热点端口"这个信号本身有效。

为搞清楚真实机制，补充了 4 个新脚本，均基于同一份 OLMoE held-out trace（`olmoe_signal_comparison_n32`），保存在 `experiments/idea_a_mac/outputs/paper_validation/receiver_isolation/`：

| 脚本 | 检验问题 |
|---|---|
| `run_receiver_isolation_experiment.py` | 剥离 remote-bonus 后，"识别热点端口"本身还有没有独立价值？ |
| `run_receiver_sender_decomposition.py` | 热度信号里 receiver 侧和 sender 侧各自贡献多少？ |
| `run_expert_popularity_stability.py` | sender 侧信号（专家热度）跨样本是否稳定，能否离线 profile？ |
| `run_stale_profile_vs_oracle.py` | 论文里已用的离线 profile 策略，剥离混淆后是否仍有效？ |

### 11.1 实验一：剥离 remote-bonus 后，热点识别是否仍有独立价值

固定候选池为 **tail-rank ∩ inter-node**（hot/cold/random 完全相同的候选集合），只改变在这个池子内选哪些 pair 获得 INT4 预算：

- `hot`：按该层当前真实 remote sender/receiver 负载（`max(sender_load, receiver_load)`，逐层计算，不跨层泄露信息）降序选择；
- `cold`：同一指标升序选择（反向对照）；
- `random`：30 个随机种子重复抽样，报告均值和 95% 分位区间。

| origin_mode | num_jobs | budget_fraction | hot | random 均值 | random 95% CI | cold |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 8 | 0.50 | 0.2125 | 0.1084 | [0.1023, 0.1152] | 0.0000 |
| balanced | 16 | 0.50 | 0.2142 | 0.1155 | [0.1127, 0.1178] | 0.0000 |
| hotspot | 8 | 0.50 | 0.2056 | 0.1231 | [0.1182, 0.1276] | 0.0408 |
| hotspot | 16 | 0.50 | 0.2416 | 0.1215 | [0.1176, 0.1257] | 0.0000 |

（完整 24 组配置见 `receiver_isolation_summary.csv`。）

**结论**：`hot` 在几乎所有配置下都稳定、大幅超出 `random` 的 95% CI（fraction<1 时的所有格子均判定"端口感知有独立价值"），`cold` 接近 0。**这推翻了"可能完全是混淆"的担忧——热点识别本身是真实信号，receiver-aware 的基本方向是对的。**

### 11.2 实验二：热度信号拆解为 receiver 侧 vs sender 侧

把 `max(sender_load, receiver_load)` 拆成两个分量单独测，看 `receiver_only` / `sender_only` 各自能拿到 `combined`（相对 `random`）收益的多少比例：

| origin_mode | num_jobs | frac | random | receiver_only | sender_only | combined | receiver 占比 | sender 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 8 | 0.50 | 0.1084 | 0.1584 | 0.1966 | 0.2125 | 0.480 | 0.847 |
| balanced | 16 | 0.50 | 0.1155 | 0.1401 | 0.2121 | 0.2142 | 0.249 | 0.979 |
| hotspot | 4 | 0.50 | 0.1239 | 0.2359 | 0.1124 | 0.2359 | 0.999 | -0.103 |
| hotspot | 8 | 0.50 | 0.1231 | 0.2056 | 0.1293 | 0.2056 | 1.000 | 0.075 |
| hotspot | 16 | 0.50 | 0.1215 | 0.2117 | 0.1986 | 0.2416 | 0.751 | 0.643 |

（完整 18 组见 `receiver_sender_decomposition_summary.csv`；均值：receiver 占比 `0.614`，sender 占比 `0.573`，两者有重叠因为分母不同、且个别格子超出 [0,1]。）

**结论**：

- **hotspot 场景**（少数请求占用大部分流量）：`receiver_only` 几乎独占全部收益（占比 69%~100%），`sender_only` 贡献很小甚至为负；
- **balanced 场景**（更常见的均衡负载）：反过来 `sender_only` 占主导（贡献占比常年 75%~99%），`receiver_only` 只能拿到 10%~50%。

**这说明"receiver-aware"这个命名本身不准确**——收益经常来自 sender 侧（专家所在 GPU 的负载），不是 receiver 侧（token 来源 GPU）。

### 11.3 实验三：sender 信号（专家热度）能否离线 profile

用 disjoint 的 calibration（offset 0，16 条文本）和 held-out test（offset 128，32 条文本）WikiText-2 路由 trace，按层比较 expert 命中分布和聚合到 owner-GPU（sender_rank）负载分布的 Spearman 相关：

| 指标 | 跨层均值 |
|---|---:|
| expert-id 命中分布 Spearman（cal vs test） | **0.503** |
| sender_rank 负载分布 Spearman（cal vs test） | **0.388** |
| 最热 expert 跨集合一致率 | 6.25% |
| 最热 sender_rank 跨集合一致率 | 25.00% |

完整逐层数据见 `expert_popularity_stability.csv`。

**结论**：专家热度跨样本相关性弱（Spearman ~0.4-0.5），最热端口跨集合一致率只有 25%。**sender 侧信号内容依赖、非平稳，不能像 PLTB 的 layer sensitivity 那样做一次离线 profile 就复用**——PLTB 的 layer sensitivity 排序在 calibration 从 8 条扩到 16 条后完全不变（见"2026-07-11 更新说明"一节和第六节 6.1 方法），而这里的专家热度排序在 disjoint 样本间几乎不稳定，两者性质不同。

### 11.4 实验四：论文已用的离线 profile 策略，剥离混淆后是否仍有效

直接对照 4 种策略（同一固定候选池）：`random`（基线）、`stale_profile`（用 calibration 算出的静态负载给 test 场景候选打分，即 `tail_budget_profile_ports` 实际依赖的信息）、`oracle_receiver`（test 场景当前真实 receiver 负载，调度器可提前知道）、`oracle_combined`（test 场景当前真实综合负载，不可离线获得的上界）：

| origin_mode | num_jobs | frac | random | stale_profile | oracle_receiver | oracle_combined |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 8 | 0.50 | 0.1084 | 0.1450 | 0.1584 | 0.2125 |
| balanced | 16 | 0.50 | 0.1155 | 0.1345 | 0.1401 | 0.2142 |
| hotspot | 4 | 0.25 | 0.0619 | 0.0314 | 0.1185 | 0.1185 |
| hotspot | 8 | 0.50 | 0.1231 | 0.1414 | 0.2056 | 0.2056 |
| hotspot | 16 | 0.50 | 0.1215 | 0.1976 | 0.2117 | 0.2416 |

（完整 18 组见 `stale_profile_vs_oracle_summary.csv`。）

汇总：`stale_profile` 与 `random` 平均绝对差仅 `0.0297`，而 `oracle_combined` 与 `stale_profile` 的平均差距达 `0.0555`；在 `hotspot/4 jobs/frac=0.25` 这一格，`stale_profile`（0.031）甚至**低于** `random`（0.062）。

**结论**：与实验三互相印证——**当前论文里 `tail_budget_profile_ports` 之所以看起来"和 scheduler/greedy 差不多好"，主要是三者共享的 remote-bonus 撑住的，离线 profile 本身几乎没有额外贡献**。第 6.1 节 `PDF论文的选择.md` 中"profile_ports 是 deployable only if calibration transfers"这句判断被证实：calibration **不 transfer**，所以 `tail_budget_profile_ports` 目前不可部署。

### 11.5 综合结论与论文修正方向

1. **删除**"离线 profile 有效""profile/scheduler/greedy 效果相近"的表述——这是 confound 造成的假象。
2. **保留、强调**：receiver-side scheduling signal（调度器已知的 token-origin 热度，不需要等本层路由结果）在负载不均衡（hotspot）场景下几乎免费拿到 `oracle_combined` 的全部收益，是低成本、可部署的信号。
3. **降级为 oracle/future work**：sender-side 信号虽然在均衡负载下贡献更大，但因跨样本不稳定、依赖本层实时路由结果，若要使用必须在 dispatch 之后、combine 之前插入一次跨 EP-rank 的专家负载同步——这是额外系统开销，不能默认为免费信号，只能作为 upper bound 分析或明确标注为需要额外同步机制的扩展。
4. **新的候选设计——Two-Tier Congestion-Safe Budgeting**：离线层复用 PLTB 的 layer-sensitivity profile 决定每层安全 INT4 预算总量；在线层只用 receiver-side 热度（调度器已知，不需要等路由）决定这份预算在 remote pair 间怎么分配。这个设计完全绕开了 sender 侧不稳定、不可部署的问题，且与 PLTB"离线定总量、在线定分配"的哲学一致，是比"识别热点端口"更精确、更可防守的系统贡献点。

以上四组实验仍是 **bandwidth-only 解析回放**，`sender_rank`/`receiver_rank` 用的是简化静态 placement 代理，不含 collective、queueing、pack/unpack 或真实调度器行为；receiver-side 信号"可提前调度"的假设仍需在真实 serving 系统里验证。原始数据与完整报告见 `experiments/idea_a_mac/outputs/paper_validation/receiver_isolation/`。

---
