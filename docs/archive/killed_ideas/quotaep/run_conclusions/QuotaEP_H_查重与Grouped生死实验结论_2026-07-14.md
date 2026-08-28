# QuotaEP-H：查重、Grouped-Owner 生死实验与严格结论

> 日期：2026-07-14  
> 定位：这是在 Mac 上完成的 article-level、端到端 fake-quant 质量实验和 frozen-route wire accounting；不是 native FP4 kernel、多 GPU EP、TTFT/TPOT/TBT 或 P99 证据。

## 1. 审稿式结论

**有一条比旧 R-layout 更值得继续、但尚未达到系统论文闭环的主线：**

> **QuotaEP-H（Placement-Conditioned Fixed-Quota Mixed-Precision Hierarchical Combine）**：面向 hierarchical/high-throughput EP combine，在 source topology domain 内先把同一 `(token, destination-origin)` 的 expert outputs 做 BF16 partial reduction，再在每个固定 peer/tile 内按 output-aware criticality 选择固定数量的 FP8 vectors，其余使用 MXFP4；最后以两条 homogeneous lane 发送并在 destination 端 BF16 accumulation。

它不是“rank-tail 量化”的换名版本。当前真正被实验支持的机制是：

1. hierarchical combine 中存在大量可以在跨节点前合并的 expert-pair；
2. expert output 已经产生后，`||Σ_e g_e o_e||` 或 FP4→FP8 局部误差能稳定识别应保留 FP8 的 grouped vectors；
3. 固定 quota 能形成规整 FP8/MXFP4 lane，但相对全局预算存在可测的 quality tax。

**严格 verdict：Gate B 通过，Gate A 与 Gate C 尚未通过。**

| Gate | 问题 | 结论 |
|---|---|---|
| A：低成本 selector | 只用 rank/gate/input norm，能否接近 output-aware？ | **失败**。两模型、两 placement 下 contribution 都显著优于 rank/gate；不能继续把 rank 当最终 selector。 |
| B：owner-side late binding | 在 expert output 已知后，output-aware selector 是否稳定有效？ | **通过质量侧生死实验**。四个 setting 全部优于 rank/gate/random，文档级 bootstrap + Holm 后成立。 |
| C：系统净收益 | fused reduce+score+select+pack 是否改善真实 TPOT/P99？ | **未验证**。Mac 只能给质量与 logical wire；必须上真实 HT EP backend。 |

因此现在可以写的是“**output-aware fixed-quota hierarchical combine 是值得实现的候选**”，不能写“已经提高 TTFT/TBT”，也不能写“graceful/QTree/receiver-aware 已经有效”。

## 2. 实验设计

### 2.1 模型、数据与隔离

| 模型 | routing | formal test | placement stress |
|---|---:|---|---|
| `allenai/OLMoE-1B-7B-0924` | E64, top-8 | WikiText validation articles `[48:60]` | 相同冻结文档，contiguous / round-robin |
| `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M` | E32, top-16 | WikiText validation articles `[24:36]` | 相同冻结文档，contiguous / round-robin |

- 每个 setting 12 篇独立 article，每篇最多 256 tokens；独立单位是 article，不把 token 当 iid。
- formal offsets 与此前 calibration/confirm 集合分离；每个输出目录保存 `data_manifest.csv`、`config.json` 和 `source_manifest.json`。
- paired bootstrap 10,000 次；当前六个机制比较统一做 Holm correction。
- patched full 与原模型 bitwise equal；EP=1 grouped path 与 patched full bitwise equal，两模型最大 logit diff 都是 0。
- EP=8 grouped BF16 相对原始逐 expert BF16 accumulation 仍有约 `0.001491 / 0.001923` KL；这是浮点加法 association/order 改变，不是量化误差。因此所有 mixed 策略都以 **同 placement 的 grouped BF16** 为主 reference。

### 2.2 匹配预算的策略

- `uniform_fp8`：全部 grouped vectors FP8，系统强基线；
- `mixed_rank`、`mixed_gate_mass`、`mixed_inputnorm_gate`：不依赖完整 output 的廉价 selector；
- `mixed_pair_contribution`：先按 `Σ_e g_e ||o_e||` 评分，再形成 grouped vector；
- `mixed_contribution`：按真正发送的 grouped partial `||Σ_e g_e o_e||` 评分；
- `mixed_qerr`：按该 vector 从 MXFP4 升到 FP8 时减少的局部量化误差评分；
- `global_contribution`：跨所有 peer 的 exact global top-B，仅作为配额自由度 upper bound；
- `mixed_random`：同预算负对照。

主比较的 peer/tile quota 都固定约 50% FP8 vectors，wire 几乎完全匹配。`token_contribution` 因每 token 的 odd group count 使用 floor，实际高精度比例不是 50%，所以只作 exploratory，不能进入 matched-wire 主比较。`mixed_oracle` 是逐 group 单次升级的一阶局部 oracle，并非全局组合最优；它在 top-16 上落后并不说明 oracle 概念失效。

## 3. 结果

### 3.1 碰撞与 logical wire

| 模型 / placement | routed pairs / grouped vectors | pair collision fraction | uniform FP8 saving | mixed saving | mixed 相对 FP8 再减 transmitted bytes |
|---|---:|---:|---:|---:|---:|
| OLMoE / contiguous | 1.455 | 31.26% | 49.90% | 61.70% | 23.56% |
| OLMoE / round-robin | 1.475 | 32.20% | 49.90% | 61.70% | 23.56% |
| LLM-jp / contiguous | 2.106 | 52.52% | 49.61% | 61.55% | 23.69% |
| LLM-jp / round-robin | 2.118 | 52.79% | 49.61% | 61.55% | 23.69% |

这里的 saving 包含实验 codec 的 scale metadata，但不包含真实 backend 的 header、alignment、padding、buffer reservation、pack/unpack 与 overlap。尤其不能把 61.7% 直接翻译为 61.7% latency improvement。

### 3.2 Formal：contiguous placement

| 策略 | OLMoE KL | LLM-jp KL |
|---|---:|---:|
| uniform FP8 | 0.003317 | 0.006454 |
| mixed rank | 0.006523 | 0.010286 |
| mixed gate | 0.006261 | 0.010705 |
| pair contribution | 0.005563 | 0.008619 |
| **grouped contribution** | **0.005430** | **0.008309** |
| global contribution | 0.005211 | 0.006793 |
| qerr | 0.005409 | 0.008235 |
| random | 0.015398 | 0.052695 |

文档级 paired bootstrap：

- OLMoE contribution 相对 rank 降低 `16.76%` KL，delta `-0.001093`，95% CI `[-0.001406, -0.000738]`；相对 gate 降低 `13.29%`，CI `[-0.001179, -0.000446]`。
- LLM-jp contribution 相对 rank 降低 `19.22%`，CI `[-0.002750, -0.001244]`；相对 gate 降低 `22.38%`，CI `[-0.003325, -0.001501]`。
- 上述四项 Holm-adjusted bootstrap sign p 均约 `0.0012`。
- contribution 与 qerr 无显著差异，说明无需包装某一个评分函数为唯一理论真值；更稳妥的 claim 是“**output-aware late binding**”这一信号族成立。

### 3.3 Placement stress：round-robin

| 策略 | OLMoE KL | LLM-jp KL |
|---|---:|---:|
| uniform FP8 | 0.003096 | 0.004914 |
| mixed rank | 0.006621 | 0.008159 |
| mixed gate | 0.006395 | 0.008007 |
| pair contribution | 0.005358 | 0.007315 |
| **grouped contribution** | **0.005408** | **0.007108** |
| global contribution | 0.004830 | 0.006697 |
| qerr | 0.005404 | 0.007001 |
| random | 0.025768 | 0.036061 |

- contribution 相对 rank 降低 `18.32% / 12.88%` KL；相对 gate 降低 `15.43% / 11.22%`，四项 Holm-adjusted p 均 `<0.002`。
- 方向跨 top-8/top-16、contiguous/round-robin 全部保持，排除了“只对某种 expert placement 偶然有效”的解释。

## 4. 三个重要负结果

### 4.1 “group cancellation 是核心新现象”不成立

`||Σ_e g_e o_e||` 相对 `Σ_e g_e||o_e||` 的 KL 差在四组中为约 `-2.4% / -3.6% / +0.9% / -2.8%`，所有 Holm-adjusted CI 均未稳定排除 0。OLMoE round-robin 甚至是 pair score 略好。

所以不能把论文创新包装成“首次利用 expert cancellation”。grouped reduction 是正确 wire 粒度和系统位置；它不是已被证实的独立质量机制。

### 4.2 “固定 peer quota 几乎免费”不成立

相对 `global_contribution`：

- OLMoE contiguous：peer quota KL 高 `4.18%`，多重校正后尚未显著；round-robin 高 `11.98%`，Holm p `0.0186`；
- LLM-jp contiguous：高 `22.32%`，Holm p `0.0012`；round-robin 高 `6.13%`，Holm p `0.0264`。

固定 quota 的规则性确实可能换取 kernel/queue 优势，但它有真实 accuracy tax，尤其 top-16 不可忽略。后续必须画 **quota granularity—quality—kernel regularity** 三维 Pareto，而不是只展示一个 50% 点。

### 4.3 rank/gate 不能承担 final selector

旧 Idea A 把 rank 的静态布局优势放在核心。新结果显示，expert output 可用后，rank/gate 在四组都稳定落后 output-aware selector。除非真实 kernel 证明 late binding 成本过高，论文方法应从“rank segmented”升级为“owner-side output-aware fixed quota”。rank 只保留为零额外 scoring 的强系统 baseline。

## 5. 与公开工作的边界

截至 2026-07-14，不能做以下 broad novelty claim：

| 已被覆盖的方向 | 代表工作 | 对本文的约束 |
|---|---|---|
| 高吞吐/低时延、层次化 EP primitive | [DeepEP](https://github.com/deepseek-ai/DeepEP)；[NCCL EP](https://arxiv.org/abs/2603.13606) | topology-aware / HT combine 本身不是创新；必须落进真实 backend 并与原生模式对照。 |
| EP 中 uniform FP8/FP4 通信与 fused codec | DeepEP、[Practical FP4](https://arxiv.org/abs/2603.02731)、[MoRI](https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html) | “把 combine 变 FP4”不是创新；只能主张同一 backend 内的 mixed-precision placement。 |
| fine-grained mixed precision / activation criticality | [FGMP](https://arxiv.org/abs/2504.14152) | criticality-aware mixed precision 是一般思想，不可声称首创。 |
| gate × activation norm 重要性 | [REAP](https://arxiv.org/abs/2510.13999) | `g||o||` 类信号不是新指标；本文独特性必须来自 EP grouped wire unit、fixed quota contract 与 kernel。 |
| communication-aware placement/pruning | [CAP](https://arxiv.org/abs/2607.05116) | graceful pruning、receiver-aware 或 placement 联合优化不能再作为宽泛首创。 |

当前可防守的窄交集是：

> **在 hierarchical EP combine 的跨域 partial wire unit 上，做 owner-side output-aware late binding，并用 per-peer/tile fixed quota 把动态 criticality 编译成两条规整精度 lane。**

这只是 novelty candidate，不是查重证明。正式投稿前仍要逐段阅读全文、代码和同期工作，不可写 “the first”。

## 6. 下一步最小系统闭环

### P0：必须做，否则停在机制论文

1. 选择一个真实 HT/hierarchical backend（优先 NCCL EP HT；否则在可验证的 DeepEP/Hybrid-EP 路径中找到等价 local-reduce-before-RDMA 点）。
2. 先确认真实 combine wire unit。若 backend 发送 per-expert expanded outputs，则本 grouped 协议不能直接套用，必须改为 backend-specific path 或放弃。
3. 实现四个严格可比 kernel：`grouped BF16`、`uniform FP8`、`uniform MXFP4`、`QuotaEP-H`。
4. 将 BF16 local reduce、norm/qerr score、tile top-q、FP8/MXFP4 pack 融合；把 selector latency、临时 buffer、SM occupancy 单独报告。
5. microbenchmark 覆盖 message size、EP8/16/32、single/multi-node、balanced/skew、prefill/decode。报告 actual NIC bytes、kernel latency、overlap 和 receiver completion time。
6. 端到端 serving 至少 10k 请求/点、5 runs；报告 TTFT、TPOT、P50/P95/P99、throughput 和质量，主比较是同质量/同 arrival trace 的 uniform FP8。

### P1：把固定配额从弱点变成论文问题

- sweep quota granularity：global、node/domain、peer、tile 16/32/64/128；
- 比较 exact sort、histogram/threshold、warp/block top-q；
- 预注册一个允许 borrowing 的 `peer quota + bounded spill`，检验能否追回 global 的 top-16 质量，同时保持有界 buffer；
- decode 若分组碰撞不足或 selector overhead 占主导，明确把方法限定在 prefill/HT，不强行扩张。

### P2：暂不并入主线

- Graceful degradation：CAP 已覆盖 communication-aware pruning，且当前没有真实 deadline/queue 证据；
- receiver-aware：只有旧 bandwidth proxy，需先证明 receiver queue/incast 是 P99 主因；
- QTree：topology-aware accumulation 已是 backend 设计的一部分，除非出现新的 kernel/protocol差异，不单独作为贡献；
- joint LL/HT switching（可暂称 MergeQuant-EP）：只有在真实 profiling 证明 decode 应保留 expanded LL、prefill 应使用 grouped HT 时再立项。

## 7. 当前论文级判断

- **毕业论文：主线已经比旧 R-layout 更扎实，足够继续。**
- **CCF C：若补出真实多 GPU kernel + end-to-end serving，具有现实竞争力；只靠当前 Mac 结果仍没有“保底”。**
- **CCF B：有潜力，但门槛是 backend-level co-design，而不是继续堆离线 selector。需要证明相对 uniform FP8 的增量 wire 能转化为稳定 TPOT/P99 收益，并正面解释 fixed quota 的 quality tax。**

本轮最重要的价值不是得到一个漂亮数字，而是把论文从一个容易被反例击穿的 “rank-tail FP4” 收缩成了一个可证伪、可实现、和真实 EP backend 对齐的系统假设。

## 8. 证据入口

- formal OLMoE：`olmoe_grouped_owner_formal_2026-07-14/`
- formal LLM-jp：`llmjp_grouped_owner_formal_2026-07-14/`
- placement stress：`olmoe_grouped_owner_roundrobin_stress_2026-07-14/`、`llmjp_grouped_owner_roundrobin_stress_2026-07-14/`
- 每个目录的统计报告：`fixed_rate_survival_report.md`
- 实验预注册修订：`GroupedOwnerEP_生死实验预注册修订_2026-07-14.md`
- 实现：`experiments/idea_a_mac/grouped_owner_combine.py`、`run_grouped_owner_combine.py`
