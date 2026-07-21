# Idea A 实验与论文推进 Task List

> 方案源文件：`PDF论文的选择.md`  
> 原始 Mac 设计：`IdeaA_前置实验设计_MacM5Pro.md`  
> 当前实验汇总：`experiments/idea_a_mac/outputs/thesis_evidence/给导师看的_IdeaA实验汇总.md`
> TTFT/TBT 系统验证总清单：`IdeaA_TTFT_TBT_系统验证实验清单.md`

> **2026-07-14 执行重置**：四组 grouped-owner frozen 实验已完成，R-layout 不再是默认 primary。当前顺序改为 `QuotaEP-H backend wire audit → fused grouped kernel → quota/borrowing quality-system Pareto → serving`。output-aware late binding 已通过质量侧生死门；rank/gate final selector、cancellation novelty、“fixed quota 免费”均未通过。详见 `experiments/idea_a_mac/outputs/paper_validation/QuotaEP_H_查重与Grouped生死实验结论_2026-07-14.md`。

> **2026-07-13 执行收敛**：系统验证不再四线并跑。当前固定顺序为 `R-layout article-level 质量 → packed kernel → serving → receiver-aware 条件扩展`；Graceful 与 critical-single/QTree 只根据真实 profiling 二选一。完整资源门、simulator→模型 intervention replay、P99 样本量和校准顺序已补入系统验证总清单。

## 0. 当前结论

### 0.0 2026-07-14 QuotaEP-H 决策门（优先执行）

- [x] 修正 grouped combine 的真实对象：只对 hierarchical/HT 中 source-domain 已归约的 `(token, destination-origin)` partial 做 mixed precision，不声称适配所有 DeepEP/NCCL EP wire layout；
- [x] 双模型 patched-full 与 EP1 grouped exactness：bitwise equal，max logit diff=0；
- [x] OLMoE top-8 formal：contribution 相对 rank/gate KL 降低 `16.76%/13.29%`，document bootstrap CI 排除 0；
- [x] LLM-jp top-16 formal：降低 `19.22%/22.38%`，CI 排除 0；
- [x] round-robin placement stress：两模型方向继续成立，排除 contiguous placement 偶然性；
- [x] 碰撞计数：EP8 下 OLMoE routed-pair/grouped-vector=`1.455～1.475`，LLM-jp=`2.106～2.118`；pair collision fraction 分别约 `31%～32% / 52%～53%`；
- [x] negative result：grouped contribution 相对 pair contribution 的差异不稳定，**不把 cancellation 当创新点**；
- [x] negative result：peer quota 相对 global budget 在四组有 `4.2%～22.3%` KL tax，**不写“regularity nearly free”**；
- [ ] P0：在目标 backend 逐 kernel 确认 HT combine 是 per-expert expanded 还是 source-domain partial；若没有 grouped wire unit，停止该 backend 路线；
- [ ] P0：实现 `grouped BF16 / uniform FP8 / uniform MXFP4 / QuotaEP-H` 四个同 backend kernel；
- [ ] P0：测 fused reduce+score+top-q+pack 的 selector latency、SM occupancy、临时 buffer、actual bytes；
- [ ] P1：sweep global/domain/peer/tile quota 与 tile 16/32/64/128，加入 `peer quota + bounded spill`；
- [ ] P1：真实 serving 至少 10k 请求/点、5 runs，报告 TTFT、TPOT、P50/P95/P99 与 quality；
- [ ] Stage-gated：只有真实 trace 证明 deadline/receiver-incast/topology partial 是主要瓶颈后，才分别打开 Graceful/receiver-aware/QTree，不并行堆叠。

本轮 matched-wire mixed lane 相对 grouped BF16 logical saving 约 `61.5%～61.7%`；相对 uniform FP8 **transmitted bytes** 进一步减少约 `23.6%～23.7%`。该数字包含实验 scale metadata，但不含真实 header/alignment/padding/overlap，不能写成 latency saving。

Mac 前置实验已经完成“是否值得继续”的判断，答案是 **Go，但必须收缩主线**。

当前论文主线不再是从 BF16 出发做 BF16/FP8/INT4/drop 的大一统优化，也不再把 receiver-aware、MILP、oracle、routing drift 都当成并列贡献。经过硬件 FP4 格式审计，主线进一步收敛为：

> **FP8-First Rank-Segmented FP4 Combine**：uniform FP8 作为强系统基线；routing 后按固定 rank split 构造 FP8 head lane 与 block-FP4 tail lane，并与动态 gate layout 正面对比真实 accuracy-wire-latency Pareto。

2026-07-11 第三轮实验发现：旧 PLTB 的强收益具有明显格式依赖。MXFP4/NVFP4 对齐模拟下，fixed tail 已接近 uniform FP8；旧 PLTB allocation 只改善约 `7.6%` 或反转约 `1.7%`，MXFP4-specific profile 也未稳定优于 fixed-tail。因此 **PLTB 降级为 optional enhancement**。与此同时，MXFP4/NVFP4 的 head-control KL 分别是 tail 的 `8.75×/2.56×`，动态 gate/oracle 只比 fixed rank 小幅更好，支持把 fixed rank segmentation 作为当前最值得实现的系统候选。

当前最重要的边界：

- Mac 实验已经证明质量侧现象，尚未证明真实通信收益；
- bit-only `50%/62.5%` 不能当 wire saving；含 scale 后 uniform FP8/MXFP4-tail/NVFP4-tail 约为 `49.61%/61.52%/60.69%`；
- receiver/endpoint group 目前只是分组流量代理，不是真实 combine receiver-port；
- 旧 additive-KL MILP 已被端到端实验否定，不再作为主方法；
- drop、oracle、routing drift 都是 ablation / upper bound / mechanism analysis，不与核心方法并列。
- 2026-07-13 严格复核发现，旧 WikiText line-level offset 会把同一 article 的连续段落当成独立 bootstrap samples；因此 1.8/1.10/1.11 的旧 CI 和 held-out 数字仍是 pilot，必须按 article-level split 重跑后才能进入论文主表。Graceful/QTree 修正版已率先完成该修复，详见 1.12。
- 端到端系统主张的最小执行主线是 R-layout；receiver-aware 只有真实 receiver queue/incast 归因成立后才进入，Graceful/QTree 不再默认实现。

### 0.1 当前执行中的 Stage-1 修复版质量实验

- [x] 修正通用 OLMoE hook 的 BF16 expert-order accumulation，patched `full` 必须与原模型 logits exact equal；
- [x] `run_signal_comparison.py` 默认改为 validation/test article-level split；
- [x] 增加 model/runtime 版本、article hash manifest、paired KL/PPL CI 和 interleaved anti-control；
- [x] 完成 16 calibration + 16 test article、seq=256 的 MXFP4 pilot；patched-full exact，test 4,080 tokens；
- [x] formal 使用 32 个 validation calibration articles + 尚未查看的 45 个 WikiText test articles（offset 16），11,475 test tokens；pilot/formal test hash overlap=0；
- [ ] 再增加第二 corpus/domain，补足跨域和总 article/token 证据，禁止将同一 article 的多窗口当 iid；
- [ ] native FP4/kernel 结果出来前，不把 metadata-aware logical bytes 写成 wire latency 收益。

Pilot 核心结果：uniform FP8 / R-layout / gate 的 KL 分别为 `0.003206 / 0.005849 / 0.005194`。gate 相对 R-layout paired KL delta 为 `-0.000655`，95% CI `[-0.001132,-0.000304]`，在 16 篇中的 14 篇更好；head/interleaved controls KL 为 `0.031741/0.029148`，16/16 篇都比 tail 更差。因此 **rank criticality 成立，但 fixed rank 不是质量 Pareto winner**；后续必须用真实 kernel 证明定长布局的系统开销优势足以换回约 11.2% total-KL 差距。Pilot 使用的 test 前 16 篇从此只作 dev/pilot，不能复用为 formal test。完整报告：`experiments/idea_a_mac/outputs/paper_validation/olmoe_r_layout_article_stage1_pilot_2026-07-13/Stage1_RLayout_质量Pilot结论.md`。

WikiText formal 结果：R-layout / gate / tail-mass / oracle KL 为 `0.006042 / 0.005635 / 0.005616 / 0.005559`；动态 selector 相对 R-layout 稳定降低约 `6.7%～8.0%` total KL，paired CI 全部排除 0。head/interleaved controls 为 `0.035938/0.027707`，45/45 articles 均更差。**严格决策：rank criticality 通过；fixed-rank 的质量 Pareto 不通过；只在真实 packed kernel 能显著低于动态 selector 开销时保留 R-layout 主线。** 完整报告：`experiments/idea_a_mac/outputs/paper_validation/olmoe_r_layout_article_stage1_formal_2026-07-13/Stage1_RLayout_WikiTextFormal结论.md`。

Stage-0 本机资源门已完成：M5 Pro/48GB，Torch CUDA device count=0、MPS unavailable，无 NVIDIA/NVCC/NCCL、无 IB device、无 vLLM/SGLang/DeepEP/Triton；FP4 dtype 的 CPU copy 也未实现。因此本机不能继续 D0/D/E 或声称 TTFT/TBT，下一本地任务只能是第二 corpus/model 质量与 packed-layout correctness。报告：`experiments/idea_a_system/reports/00_resource_gate.md`。

---

## 1. 已完成的前置实验

### 1.1 环境、模型与实验框架

- [x] 建立 `experiments/idea_a_mac/` 实验目录、依赖和输出结构。
- [x] 跑通 `jamesdborin/tiny-mixtral` smoke test。
- [x] 跑通 Mixtral-TinyMistral top-2 支撑模型。
- [x] 跑通 `allenai/OLMoE-1B-7B-0924` top-8 主模型。
- [x] 跑通 LLM-jp E32-k16 top-16 stress test。
- [x] 实现 MoE hook，采集 selected experts、gate、expert output 和 contribution。
- [x] 实现 BF16 / FP8 / INT8 / INT4 / drop fake-quant 策略。
- [x] 实现 rank sweep、layer sensitivity、routing drift、receiver proxy、MMLU 等实验脚本。

### 1.2 Rank contribution 与 INT4 安全边界

- [x] OLMoE top-8：tail rank contribution 明显低于 head rank。
- [x] LLM-jp top-16：tail 安全趋势与 OLMoE 一致。
- [x] Mixtral top-2：观察到极端 gate 偏斜，但只作为辅助证据，不把超大 head/tail ratio 当作主论据。
- [x] 相同 saving 下完成 rank1 vs rank-k 控制实验。
- [x] OLMoE 单 rank INT4：rank8 KL `0.3614`，rank1 KL `20.9892`，tail 约低 `58.1×`。
- [x] 完成 head / tail / random rank control，排除“随便挑一半 output 进 INT4 都可以”的解释。

结论：

> rank 的主要价值不在 FP8 内部微调，而在 FP8→INT4 的危险边界上避免压缩 head outputs。

### 1.3 Uniform FP8 与 FP8 + Tail-INT4

- [x] 补齐 uniform FP8 baseline。
- [x] OLMoE uniform FP8：理论 payload saving `50%`，KL/PPL 基本稳定。
- [x] 完成 `uniform_fp8 -> fp8_r8int4 -> fp8_r78int4 -> fp8_r5678int4` 扫描。
- [x] OLMoE 256 样本：tail-rank 策略理论 payload saving `62.5%`，PPL 约 `+0.30`。
- [x] 相同 `62.5%` saving：head/random PPL 约 `+6～7`。
- [x] LLM-jp top-16：tail PPL `+0.350`，head/random 约 `+6.3`，趋势一致。

结论：

> 旧的“从 BF16 出发只压少数 rank”被 uniform FP8 主导；有效的新问题是“FP8 之后哪些 tail outputs 还能继续进入 INT4”。

### 1.4 Layer sensitivity 与误差传播

- [x] 完成 OLMoE 单层 `rank8_int4` sensitivity sweep。
- [x] 16 层 KL 最大/最小约 `5.74×`，说明 layer 维度值得保留。
- [x] 完成 locked-routing vs free-routing drift attribution。
- [x] 确认低比特误差会通过后续层和 routing 继续传播。
- [x] 确认 drift attribution 只能作为近似机制分析，不能把 KL 差严格解释为可加的因果占比。

结论：

> layer sensitivity 用来决定每层 tail budget；routing drift 用来解释为什么多层误差不能简单线性相加。

### 1.5 旧 MILP 的失败结论

- [x] 完成 per-layer delta profile。
- [x] 完成旧 MILP / rank-only / greedy 离线优化。
- [x] 完成端到端 LUT evaluation。
- [x] 记录关键 negative result：MILP predicted KL `0.10`，actual KL `9.41`，低估约 `94×`。
- [x] 确认旧 MILP 与 rank-only 的实际 saving/质量接近，没有展示出值得承担复杂度的优势。
- [x] 决定将 additive-KL MILP 移出主线。

后续处理：

- 不再假设单层边际 KL 可跨层线性相加；
- 改用每层固定 tail count `m_l`；
- 在 held-out calibration 上构建 cumulative Pareto curve；
- 每个候选策略必须端到端复测。

### 1.6 Receiver/Endpoint Proxy

- [x] 完成按 expert id contiguous/mod 分组的 receiver/endpoint proxy。
- [x] 在同 total saving 下完成 hot / cold / random / round-robin 控制。
- [x] proxy 结果显示 hot group 的 max-group bytes 比对照低约 `10%～14.5%`。
- [x] 明确该结果只支持“热点分组字节数下降”，不支持真实 receiver-port、queueing、TBT 或 P99 已改善。

语义修正：

```text
当前 group        ≈ expert-owner / endpoint proxy
真实 combine sender   = expert_owner_rank
真实 combine receiver = token_origin_rank
```

在补齐 `token_origin_rank -> expert_owner_rank -> token_origin_rank` 的真实通信映射前，receiver-aware 不进入标题和核心 claim。

### 1.7 下游任务 Pilot

- [x] 完成 MMLU 200 题 pilot。
- [x] full：`42.5%`；uniform FP8：`40.0%`；FP8+tail-INT4：`42.5%`。
- [x] 将结论限定为“200 题 pilot 中未观察到下降”。
- [x] 不再写成“已经统计证明下游质量无损”。

### 1.8 Held-Out 信号强对照（2026-07-11）

- [x] 增加可复现的 calibration/test offset、split 和 seed 接口。
- [x] calibration 使用 WikiText-2 validation offset `0` 的 16 条文本；test 使用 offset `128` 的 32 条文本，文本无重叠。
- [x] 将质量指标修正为有效 token 上的 per-token KL 和累计 token NLL 后的 corpus PPL。
- [x] 对 PPL/KL 增加 1,000 次 sample-level bootstrap 置信区间。
- [x] 在 OLMoE top-8 上完成 full、uniform FP8、fixed rank-tail、gate threshold、cumulative tail-mass、contribution oracle 和 head control。
- [x] test 共包含 `1,925` 个 next-token 位置。

关键结果：

| 策略 | 理论 payload saving vs BF16 | corpus PPL | mean per-token KL | KL 95% CI |
|---|---:|---:|---:|---:|
| uniform FP8 | 50.00% | 18.783 | 0.00472 | [0.00377, 0.00589] |
| fixed rank-tail4 INT4 | 62.50% | 19.238 | 0.03032 | [0.02583, 0.03616] |
| gate threshold INT4 | 63.13% | 18.974 | 0.01836 | [0.01566, 0.02211] |
| cumulative tail-mass INT4 | 62.74% | 18.914 | 0.02130 | [0.01903, 0.02421] |
| contribution-tail oracle | 62.50% | 18.947 | 0.01813 | [0.01649, 0.02007] |
| head4 INT4 control | 62.50% | 24.617 | 0.28016 | [0.24957, 0.32610] |

结论：

- head control 的 KL 约为 fixed rank-tail 的 `9.24×`，再次确认 tail/head sensitivity 差异不是随机压缩效应；
- gate threshold 相对 fixed rank-tail 将 KL 降低约 `39.4%`，且理论 payload saving 略高；
- cumulative tail-mass 将 KL 降低约 `29.7%`；
- gate threshold 已接近 contribution oracle，说明固定 rank 仍丢失了有价值的 token-wise gate 信息；
- PPL bootstrap 区间仍重叠，当前主要可信差异来自 paired 输入上的 per-token KL，不能把小幅 PPL 正负变化写成精度提升。

### 1.9 正确 EP 语义与多 MoE/多请求拥塞回放（2026-07-11）

- [x] 从真实 OLMoE selected-expert trace 构造 `expert_owner_rank -> token_origin_rank` combine 流。
- [x] 模拟 EP=8、每节点 4 GPU、16 个重复 MoE layers。
- [x] 覆盖 1/2/4/8/16 个并发请求/replica，以及 balanced/hotspot token-origin 分布。
- [x] 比较 uniform FP8、全部 safe tail INT4、随机有限预算、profile/scheduler/greedy critical-port 有限预算。

在 16 并发下，相对 uniform FP8 的解析型 inter-node bottleneck-byte/time proxy：

| origin | 全部 tail INT4（总 saving 62.5%） | 随机半数 tail（总 saving 56.25%） | critical-port 半数 tail（总 saving 56.25%） |
|---|---:|---:|---:|
| balanced | 23.02% | 11.66% | 约 22.90%～23.02% |
| hotspot | 24.20% | 11.99% | 24.20% |

新候选思路：

> 先用 rank/gate/tail-mass 定义质量安全集合，再把有限 INT4 预算优先用于跨节点关键 sender/receiver flows。该两阶段思路可能用更少的全局低比特 payload 达到相同的关键端口减载。

严格边界：上述结果是 bandwidth-only trace replay，不含 collective 算法、queueing、contention、pack/unpack、kernel launch、compute/communication overlap 或真实 arrival process；绝不能写成 TPOT/P99 已提升。profile/scheduler/greedy 中只有离线 profile 有初步可部署可能，greedy 当前只是上界。

### 1.10 Profiled Layerwise Tail Budget 创新实验（2026-07-11）

- [x] 以 uniform FP8 为 calibration reference，对每层单独施加 fixed-tail INT4 probe。
- [x] profile 只用于层敏感性排序，不把单层 KL 相加成端到端质量预测。
- [x] 保持全模型 INT4 layer-rank slots 总数不变，在敏感层与低敏感层之间重分配 `m_l`。
- [x] 在独立 32 条 held-out test 上端到端复测冻结 LUT。
- [x] 加入 P95-risk、gate-mass 和 anti-sensitivity 等同 payload 对照。
- [x] calibration 从 8 条扩大到 16 条，OLMoE mean-KL 最优布局完全不变。
- [x] 在 LLM-jp top-16 上完成 8 calibration / 32 test 跨 top-k 验证。

方法暂命名为 **Profiled Layerwise Tail Budget（PLTB）**：

```text
uniform FP8
  -> 单层 tail perturbation profile
  -> 按 sensitivity 重分配固定总量的 INT4 rank slots
  -> 冻结 LUT[layer] = m_l
  -> held-out end-to-end test
```

OLMoE top-8、相同 62.5% theoretical payload saving：

| 策略 | `m_l` 结构 | corpus PPL Δ | mean per-token KL |
|---|---|---:|---:|
| fixed-tail4 | 全层 4 | +0.4466 | 0.03032 |
| KL-profile | 3/5 | +0.1948 | 0.02368 |
| P95-profile | 3/5 | +0.3144 | 0.02411 |
| gate-mass profile | 3/5 | +0.3675 | 0.02553 |
| anti-KL profile | 3/5，反向 | +0.6510 | 0.04091 |
| **PLTB** | **2/4/6** | **+0.2067** | **0.01957** |

OLMoE 核心结果：

- PLTB 相对 fixed-tail4 降低约 `35.5%` KL；
- paired `fixed − PLTB` 差为 `0.01075`，95% CI `[0.00734, 0.01588]`；
- 相对 fixed-rank 到 gate-threshold 的质量差距，PLTB 追回约 `89.9%`；
- PLTB KL 只比 gate threshold 的 `0.01836` 高约 `6.6%`，但仍保持 per-layer 定长 rank segmentation；
- anti-profile 稳定恶化，支持“保护敏感层”的机制解释；
- tail gate mass 与真实 layer KL sensitivity 的 Spearman 约 `-0.56`，说明 router statistics 不能直接替代 end-to-end sensitivity probe。

LLM-jp top-16、相同 62.5% saving：fixed-tail8 KL `0.01739`，PLTB 6/8/10 KL `0.01397`，降低约 `19.7%`；paired 差 95% CI `[0.00255, 0.00452]`。跨模型 anti-control 较弱，因此只支持“PLTB 收益跨 top-k 初步复现”，不声称统一机制已完全证明。

第二轮当时判断：旧 INT4 proxy 下 PLTB 比 global fixed-rank 更好；第三轮格式审计已推翻其作为默认核心的外推，当前只保留为历史结果与 optional enhancement。

> 第三轮修正：上述判断只适用于旧 symmetric INT4 proxy。PLTB 当前不再作为核心；完整证伪与格式对齐结果见 1.11。

### 1.11 硬件 FP4 格式与 selector 审计（2026-07-11）

- [x] 实现 MXFP4-style E2M1 + 32-element power-of-two block scale。
- [x] 实现 NVFP4-style E2M1 + 16-element E4M3 block scale + per-vector global scale。
- [x] 计入 FP8/FP4 scale metadata，修正 bit-only saving 口径。
- [x] 完成 OLMoE n=32 INT4 proxy/MXFP4/NVFP4 fixed-vs-PLTB 格式对照。
- [x] 完成 MXFP4-specific layer profile 与 held-out budget 复测。
- [x] 完成 MXFP4 n=32、NVFP4 n=16 fixed-rank/gate/tail-mass/oracle/head 强对照。

关键结果：

| 格式/策略 | scale-aware wire saving | mean token KL |
|---|---:|---:|
| uniform FP8 | 49.61% | 0.00472 |
| fixed tail4 MXFP4 | 61.52% | 0.00684 |
| old-PLTB MXFP4 | 61.52% | 0.00632 |
| fixed tail4 NVFP4 | 60.69% | 0.00571 |
| old-PLTB NVFP4 | 60.69% | 0.00581 |

selector 结论：MXFP4 head/tail KL 比 `8.75×`；NVFP4 为 `2.56×`。MXFP4 gate threshold 比 fixed rank 低约 `10%`，oracle 低约 `19%`；NVFP4 下三者几乎持平。

当前决策：

- fixed rank-segmented FP8/FP4 保留为核心候选；
- PLTB 降级为 format-dependent optional enhancement；
- 下一步不再堆 proxy allocator，优先实现真实 two-lane combine kernel 与 rank-split sweep；
- 详细报告：`experiments/idea_a_mac/outputs/paper_validation/论文Idea最新文献与硬件格式审计_2026-07-11.md`。

### 1.12 Graceful/QTree 严格复核与修正版生死实验（2026-07-13）

- [x] 修复 patched full 的 BF16 accumulation order；与原始 OLMoE logits 逐元素完全一致，max/mean diff 均为 0。
- [x] 改用 article-level sampling：validation 8 篇 calibration，未使用过的 test split 32 篇文章、8,160 个有效 token。
- [x] 使用 article-level paired bootstrap 5,000 次，保存 test article hash manifest。
- [x] 将 Graceful 拆成 sender-side cancellation 与 receiver-ignore-late；后者不再错误计为 wire saving。
- [x] 加入跨策略固定 synthetic token origin，只统计 inter-node combine bytes。
- [x] frozen baseline routes 用于 wire accounting，dynamic routing drift 只进入端到端质量。
- [x] EP8/EP16 使用 matching per-expert FP8 与 hierarchical node-FP8 强基线。

修正版结果：

- 温和 sender-cancel 相对 R-layout 只多省 `2.19%` combine bytes（相对 per-expert FP8 多 `1.68` 个百分点），额外 KL `0.009799`，95% CI `[0.008507, 0.011229]`；32/32 篇文章变差；
- receiver-ignore-late 具有相同质量损失但不减少 wire bytes；
- mass renorm 相对 raw 额外增加 `0.010103` KL；
- two-lane QTree 相对 node-FP8 在 EP8/16 多 `45.47%/15.85%` 字节且 KL 更高，停止；
- critical-single 在 EP8/16 相对 node-FP8 再省 `2.27%/11.41%`，额外 KL `0.000213/0.000964`，仅保留为跨模型/placement 待验证 hypothesis，不进入论文贡献。

完整报告：`experiments/idea_a_mac/outputs/paper_validation/graceful_qtree_corrected_v3_2026-07-13/严格复核后的生死实验结论.md`。

---

## 2. 已经回答的 Go / No-Go 问题

| 问题 | 当前答案 | 论文处理 |
|---|---|---|
| top-k 内部是否存在可利用的贡献差异？ | 是，但跨 top-k 不能只用固定 `<10%` 阈值 | 使用相对 $1/k$ 的 normalized tail share、head/tail ratio 和 cumulative tail mass |
| tail output 是否更适合低精度？ | 是，INT4 下证据强 | 核心 characterization |
| uniform FP8 是否会打掉原方案？ | 是，打掉 BF16 起步旧路线 | FP8 改为默认 baseline |
| FP8 后是否还有额外压缩空间？ | 有，tail-INT4 在小模型上达到较好质量—理论 payload Pareto | 核心方法 |
| drop 是否进入默认策略？ | 否 | aggressive ablation |
| layer sensitivity 是否值得保留？ | 是 | 离线决定每层 tail budget |
| layer-wise 固定预算能否改善 global fixed-rank？ | 旧 INT4 proxy 是；MXFP4/NVFP4 尚否 | PLTB 降级为 optional enhancement |
| additive-KL MILP 是否成立？ | 否 | negative result，不再作为优化器 |
| receiver-aware 是否已经成立？ | 仅 proxy 成立 | 修正通信语义后再决定 |
| Mac 实验是否证明真实加速？ | 否 | 必须补真实 kernel 和多 GPU serving |

---

## 3. 当前论文方法定义

### 3.1 默认策略

默认策略使用全模型固定 rank split $h$：

- ranks `1...h`：FP8；
- ranks `(h+1)...k`：MXFP4/NVFP4 对齐的 block-FP4；
- runtime 不做 per-token precision optimization；
- 用 held-out rank-split sweep 选择 $h$，不使用旧 INT4 PLTB 表作为默认配置。

PLTB 仅为 optional enhancement：若后续格式特定、跨域/跨模型复测稳定成立，再扩展为 `LUT[layer_id] -> num_fp4_tail_ranks`。

### 3.2 目标系统布局

```text
top-k expert outputs
  -> head FP8 lane
  -> tail packed block-FP4 lane + scale metadata
  -> all-to-all combine
  -> fused unpack/dequant + gate-weighted reduction
```

核心系统假设：固定 rank 分段能让每个 token 的 FP8/FP4 数量、buffer size 和 offset 规则固定，从而比逐 token gate threshold 更适合规整 collective kernel。这个假设必须由真实 kernel 对照验证，不能只靠文字成立。

---

## 4. 下一阶段 P0：先修实验可信度

### 4.1 数据划分

- [x] Graceful/QTree 修正版已改为 article-level calibration/test split，并保存 article hash manifest。
- [ ] 将 rank/PLTB/selector 主实验从旧 line-level offset 全部迁移到 article-level split；旧 line-level CI 不进入论文主表。
- [x] 新信号对照禁止 profile 与 evaluation 共同使用同一组文章。
- [ ] 保存所有主实验的 model revision、dataset revision、article id/hash 和完整命令；Graceful/QTree 已完成，其他实验待补。

验收标准：

- calibration/dev 只用于选择 rank split 或 optional `m_l`；
- dev 用于调超参；
- test 只在策略冻结后运行；
- 报告能列出三个集合的样本范围/hash。

### 4.2 指标修正

- [x] 将新主实验 KL 改成有效 token 上的 per-token KL，不再直接使用 sequence-total `batchmean`。
- [x] 将新主实验 PPL 改成累计 token NLL 后统一取指数，不再平均每条样本的 PPL。
- [x] Graceful/QTree 使用 article-level bootstrap confidence interval。
- [ ] rank/PLTB/selector 主实验将旧 sample/line-level bootstrap 迁移为 article-level bootstrap；任务准确率仍待补。
- [ ] 对 MMLU 等 paired prediction 保存逐题结果，支持 McNemar/paired bootstrap。
- [ ] 将跨 top-k 的 C1 指标改成 normalized tail share、head/tail ratio、cumulative tail mass。

验收标准：

- 旧指标与新指标并排核对一次；
- 所有论文主表使用修正后的 held-out 指标；
- 主结论在修正后仍成立。

### 4.3 证据版本统一

- [ ] 重新生成与当前代码一致的 TBT/traffic proxy，删除或归档旧版本冲突数字。
- [ ] 所有 `saving` 字段区分 `theoretical_payload_saving` 与 `actual_wire_saving`。
- [ ] 更新 `给导师看的_IdeaA实验汇总.md` 中过强的 MMLU、receiver 和 latency 表述。
- [ ] 把 `experiments/idea_a_mac/outputs/thesis_evidence/00_proposal/` 明确标记为历史快照，避免与根目录最新版混淆。

---

## 5. 下一阶段 P1：补齐信号与质量对照

### 5.1 必须增加的 Baselines

- [x] full/BF16-reference fake-quant quality baseline。
- [x] uniform FP8 fake-quant quality baseline。
- [ ] uniform INT4 combine。
- [x] FP8 + fixed tail-rank INT4。
- [x] FP8 + gate-value threshold INT4。
- [x] FP8 + cumulative gate-mass INT4。
- [x] head control；odd/random 仍需用同一新指标重跑。
- [x] contribution-tail oracle。

比较原则：

- 同 theoretical/actual payload 比质量；
- 同质量约束比实际延迟；
- 同时报告信号计算和数据布局开销；
- 不预设 rank 一定胜出。

### 5.2 Layer-Wise Tail Budget

- [x] 对每层完成 base-tail 单点 sensitivity probe；完整 `m_l = 0...k` 曲线仍待真实必要性评估。
- [x] 记录 mean per-token KL 与 P95 sample risk；P50/P99 weighted error 仍待补。
- [x] 构建少量固定总 budget 的 3/5、2/4/6 与反向候选。
- [x] 使用 disjoint calibration 选择 `m_l`。
- [x] 在 test 上一次性评估冻结 LUT，并增加 paired bootstrap。
- [x] 与 global fixed tail count 和 gate threshold 对比。

### 5.3 跨模型与跨域

- [x] 保留 OLMoE top-8 主证据。
- [x] 保留 LLM-jp top-16 作为不同 top-k 支撑。
- [x] Mixtral top-2 只作辅助，不用于 PLTB 主结论。
- [ ] 增加至少一个更大、真实 serving 目标 MoE。
- [ ] 增加 general / code / math 至少三个 workload 域。
- [ ] 测 calibration domain -> unseen domain 的 LUT transfer。
- [ ] 测 prefill profile -> decode traffic/quality 的 transfer。

止损标准：

- rank 若在同 payload 下比 gate threshold/oracle 差距过大，主方法改为 gate-aware；
- 静态 LUT 若跨域明显失效，增加保守回退或 drift detector，不声称直接跨部署复用。

---

## 6. 下一阶段 P2：修正真实通信语义

### 6.1 构造 Expert-Parallel Traffic Trace

- [x] 在解析型回放中为每个请求/token 集合分配 `token_origin_rank`。
- [x] 在解析型回放中明确 expert placement 与 `expert_owner_rank`。
- [ ] 记录 dispatch：`token_origin_rank -> expert_owner_rank`。
- [x] 从 selected-expert trace 重建 combine：`expert_owner_rank -> token_origin_rank`。
- [x] 内部构建 `Traffic[layer, sender, receiver, rank]` 并输出聚合 CSV；逐 flow trace 文件仍待落盘。
- [x] 在带宽回放中区分 intra-node 与 inter-node；尚未模拟 NVLink/RDMA collective 细节。

验收产物：

- [ ] `ep_traffic_trace.parquet/csv`。
- [ ] sender-receiver traffic matrix。
- [ ] 每层 max receiver、P95/P99 receiver、imbalance。
- [ ] placement/workload 变化下的热点稳定性报告。

### 6.2 Receiver-Aware Go / No-Go

- [ ] 在相同总 payload 下比较 hot / cold / random / round-robin。
- [ ] 只使用真实 token receiver，而不是 expert-id 分组。
- [ ] 在 trace replay 或真实 collective 中测 receiver completion time。
- [ ] 测 receiver-aware 相对 rank-only 的额外 P99 收益。

Go / No-Go：

- 额外 P99 收益稳定达到约 `3%～5%`：保留为扩展贡献；
- 收益较弱或随 workload 反转：移出主线，保留 future work。

### 6.3 多 MoE/多请求拥塞扩展的 Go / No-Go

- [x] 完成 repeated-layer + concurrent-job 的 bandwidth-only trace replay。
- [x] 初步验证“quality-safe set ∩ critical inter-node flow”优于随机分配有限 INT4 预算。
- [ ] 引入真实 arrival trace、token-level origin、collective schedule、queueing 与 overlap。
- [ ] 评估离线 profile 在不同 workload/placement 上能否迁移。
- [ ] 比较逐 flow 动态选择的布局开销与固定 rank two-lane 的规整性收益。
- [ ] 在真实 EP collective 上测 completion time 和 P99。

保留条件：相对固定 rank-only，在同质量约束下带来稳定的额外 operator/P99 收益，且 metadata/不规则 buffer 开销没有吞掉收益；否则只作为 trace-level motivation 或 future work。

---

## 7. 下一阶段 P3：Two-Lane Combine Kernel

### 7.1 Uniform FP8 Combine Baseline

- [ ] 选定 DeepEP 或 NCCL EP 作为目标通信 primitive。
- [ ] 跑通 BF16 combine microbenchmark。
- [ ] 跑通 uniform FP8 combine。
- [ ] 测 scale、quant/dequant、collective 和 reduction 的真实开销。
- [ ] 确认 workload 是否 bandwidth-bound，以及 uniform FP8 是否真的优于 BF16。

### 7.2 FP8/INT4 Two-Lane Layout

- [ ] 设计 head FP8 / tail INT4 的定长 buffer layout。
- [ ] 设计 INT4 packing 和 scale metadata。
- [ ] 保证每层固定 `m_l` 时 offset 可静态计算。
- [ ] 实现 quant + pack 融合。
- [ ] 实现 unpack/dequant + gate-weighted reduction 融合。
- [ ] 处理 padding、alignment、empty rank 和不同 hidden size。
- [ ] 做数值一致性测试。

### 7.3 Microbenchmark Matrix

- [ ] EP size：8 / 16，有条件再扩展。
- [ ] batch/decode tokens：1 / 8 / 32 / 128。
- [ ] top-k：2 / 4 / 8 / 16。
- [ ] hidden size：覆盖目标模型。
- [ ] intra-node 与 inter-node。
- [ ] BF16、uniform FP8、uniform INT4、rank two-lane、gate threshold。

指标：

- operator P50/P95/P99；
- quant/pack/communication/unpack/reduce breakdown；
- actual wire bytes；
- effective bandwidth；
- SM 占用和 overlap；
- 相对 uniform FP8 的净收益。

Kernel Go / No-Go：

- 相对 uniform FP8，mixed combine operator 净改善约 `10%～15%` 以上：继续端到端集成；
- 收益被 pack/dequant 吞掉：停止堆叠 LUT/receiver，论文收缩为 characterization。

---

## 8. 下一阶段 P4：真实 Serving 闭环

- [ ] 接入 vLLM/SGLang 或目标 serving runtime。
- [ ] 对照 BF16 combine、uniform FP8、rank two-lane、gate threshold。
- [ ] 测 prefill 与 decode，主结论聚焦 decode TPOT/P99。
- [ ] 覆盖不同 request rate、batch、sequence length 和并发。
- [ ] 报告 TPOT/TBT mean、P50/P95/P99、throughput。
- [ ] 报告端到端质量与系统指标的联合 Pareto。
- [ ] 检查收益是否来自真实 combine critical path，而非测量噪声或配置差异。

端到端 Go / No-Go：

- TPOT 稳定改善约 `3%` 以上，或 P99 改善约 `5%` 以上：达到完整系统论文的基本闭环；
- 只有理论字节下降、端到端无稳定收益：不能声称 serving acceleration。

---

## 9. 论文贡献层级

最终论文最多保留三条主贡献：

1. **Characterization**：在 MXFP4/NVFP4 对齐条件下揭示 combine output 的 rank-dependent sensitivity，tail 安全、head 危险，并量化 fixed rank 与动态 selector 的差距；
2. **Method/Layout**：利用 routing 已排序 rank 构造静态 FP8-head / block-FP4-tail lanes；PLTB 仅作为待重新证明的 optional enhancement；
3. **System**：rank-segmented FP8/FP4 two-lane combine kernel，并证明相对 uniform FP8 combine 的真实多 GPU TPOT/P99 收益。

支撑模块：

- layer sensitivity：离线校准；
- routing drift：误差机制解释；
- gate threshold：强 baseline；
- contribution oracle：上界；
- receiver-aware：有真实 P99 增益后才升级；
- drop：aggressive ablation；
- 旧 MILP：negative result。

---

## 10. 当前最小推进路径

如果时间或算力有限，按下面顺序推进，不再增加新的旁支：

```text
P0 修指标与数据划分（OLMoE 初步完成）
  -> P1 rank vs gate/cumulative-mass 强对照（OLMoE 初步完成）
  -> P1.5 硬件格式审计（已完成；PLTB 降级）
  -> P2 正确 EP sender-receiver trace（解析回放完成，真实 collective 待做）
  -> P3 uniform FP8 + FP8/block-FP4 two-lane kernel
  -> P4 8-GPU TPOT/P99
  -> 再决定 fixed-rank R-layout 与 gate G-layout 的系统胜负，并决定 PLTB/critical-port 是否值得恢复
```

短期必须完成的五件事：

- [x] 修正 per-token KL、corpus PPL 和数据划分（OLMoE 初步验证）；
- [x] 补 gate threshold / cumulative mass baseline（OLMoE 初步验证）；
- [x] 完成 PLTB top-8/top-16 held-out 创新实验与 paired bootstrap；
- [x] 用正确 sender/receiver 语义完成解析型 traffic replay；
- [ ] 确定 DeepEP/NCCL EP kernel 接入方案；
- [ ] 获得可运行的多 GPU 环境和明确的实验时间窗口。

在真实 kernel 开始前，不再把新增小模型、更多 proxy 图或更复杂优化器视为优先创新工作。质量侧下一项只保留跨域 transfer 与 rank-split sweep；系统侧优先实现 fixed-rank FP8/FP4 R-layout 与 gate G-layout。
