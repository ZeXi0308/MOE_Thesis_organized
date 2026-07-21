# Idea A：TTFT / TBT 系统验证实验清单

> **2026-07-14 主线更新**：质量侧生死实验将 primary 从 fixed-rank R-layout 更新为 **QuotaEP-H**。真实系统实验的先决条件是 backend wire audit：只有 hierarchical/HT combine 在 source domain 形成 `(token, destination-origin)` partial 时，才测试 `grouped BF16 → uniform FP8 → fixed-quota FP8/MXFP4`。output-aware selector 已通过双模型、双 placement 质量门；fixed peer quota 的 `4.2%～22.3%` KL tax 必须通过 granularity sweep / bounded borrowing 明示，而不是隐藏。下文原 H-R 流程保留为强 baseline；与本更新冲突时，以本更新为准。

> 目标：严格验证 FP8/FP4 R-layout、receiver-aware budgeting、Graceful EP 与 topology-aware partial 是否真的改善 TTFT、TPOT/TBT 或 P99。本文不是“多跑一些配置”的列表，而是按 **研究假设 → 隔离实验 → 强基线 → 指标 → 通过/停止门槛** 组织的证伪计划。

> 当前边界：Mac 已验证数值质量与部分 frozen-route inter-node payload proxy；没有真实 all-to-all、queue、pack/unpack、kernel、TPOT/TBT 或 P99 证据。所有 `bytes/BW` 结果只能叫 proxy，不能提前写成 latency improvement。

> **2026-07-13 严格复核修订**：本清单分成“论文级完整协议”和“当前可执行最小闭环”。执行主线固定为 **R-layout 质量 → R-layout kernel → R-layout serving → receiver-aware 条件扩展**。Graceful 与 critical-single/QTree 不再和主线并行推进；只有真实 profiling 指向相应瓶颈时，二者至多选择一个进入实现。

## 0. 最终需要验证的四条假设

### 0.0 新的 primary hypothesis：H-QH

| ID | 研究假设 | 当前证据 | 最终需要的证据 |
|---|---|---|---|
| H-QH1 | hierarchical combine 的 grouped partial 是可利用的真实跨域 wire unit | Mac frozen-route 计数显示 EP8 pair collision 约 31%～53% | backend kernel/code/trace 证明 reduce-before-cross-domain，actual NIC bytes 对齐 |
| H-QH2 | owner-side output-aware late binding 比 rank/gate 更能保护质量 | top-8/top-16、两 placement、document bootstrap + Holm 已通过 | native FP8/MXFP4 codec 下排序不反转 |
| H-QH3 | per-peer/tile quota 的规则性收益能覆盖 quality tax | 只有 logical wire；peer 相对 global KL 高 4.2%～22.3% | quota granularity/borrowing Pareto + selector/pack kernel latency |
| H-QH4 | 相对 uniform FP8 的增量 bytes 能转化为 TPOT/P99 净收益 | logical transmitted bytes 再减约 23.6% | 同 backend、同 trace、同质量下 5 runs/≥10k requests 的 serving 结果 |

执行顺序固定为 `wire audit → codec correctness → fused microbenchmark → serving`。若 H-QH1 不成立，QuotaEP-H 立即停止；若 H-QH3/H-QH4 不成立，则将其作为 negative system result，不回退到用 fake-quant 包装加速。

| ID | 研究假设 | 当前证据 | 最终需要的证据 |
|---|---|---|---|
| H-R | 固定 rank FP8-head/FP4-tail R-layout 相对 uniform FP8 有净系统收益 | 质量—payload proxy 成立 | mixed kernel 和 serving TTFT/TPOT/P99 |
| H-C | receiver-side critical-flow budgeting 在热点下优于相同质量/总字节的 rank-only 或 random | bandwidth-only replay 有正信号 | queue-calibrated replay + 真实 receiver completion/P99 |
| H-G | 稀有拥塞时，head-first + bounded tail degradation 能缩短 barrier tail latency | 只测过固定 cancellation 的质量代价 | 明确 sender/receiver 语义的 queue 与真实 P99/质量联合 Pareto |
| H-Q | 不增加 partial 数的 critical-single topology precision placement 能降低慢链路完成时间 | EP16 payload proxy 约 11.41% | matching hierarchical FP8 kernel、跨 placement/模型、真实 latency |

零假设分别是：收益被 pack/dequant、固定通信开销、节点内 reduction、其他瓶颈或质量约束完全吞掉。实验必须允许零假设成立。

### 0.1 贡献层级与 stage gate

| 层级 | 机制 | 执行规则 |
|---|---|---|
| 核心主线 | H-R：R-layout | 必做；质量、kernel、serving 任一层失败即停止扩大系统故事 |
| 条件扩展 | H-C：receiver-aware | 仅当 R-layout 已通过且真实 trace 显示 receiver queue/incast 是主要 P99 来源时进入 |
| 备选扩展 | H-G 或 H-Q | 根据真实瓶颈二选一：deadline/late-flow 明显才选 H-G；inter-node partial payload 明显才选 H-Q |
| 负结果/未来工作 | 未过 gate 的机制 | 保留证伪结果，不继续堆实现或与核心贡献并列 |

禁止“四条路线同时工程化”。任何阶段只能有一个 primary mechanism 和一个预注册的 strongest baseline。

### 0.2 Stage 0：资源与可行性门

开始真实系统实验前，必须冻结并记录：

- [x] 本机资源已审计：M5 Pro/48GB，无 CUDA/NCCL/IB device，不能运行 D/E；详见 `experiments/idea_a_system/reports/00_resource_gate.md`；
- [ ] 目标 CUDA 集群的 GPU 型号、数量、单/多节点拓扑、NVLink/PCIe 与 RDMA/IB/RoCE；
- [ ] GPU 是否支持目标 FP8/FP4 路径；若不支持，明确 software pack/unpack fallback 及其 claim 边界；
- [ ] 可修改的 runtime（vLLM/SGLang/DeepEP/NCCL）与代码权限；
- [ ] 至少 `2 节点 × 4 GPU` 的可用时间窗口、预计 GPU-hours 和失败 fallback；
- [x] 当前只有本机：所有输出继续限定为 quality/proxy，不声称跨节点 TTFT/TBT；
- [ ] 若没有 native FP4 通信 primitive，则先实现 FP4 packed-buffer microbenchmark，不把 fake quant 当 kernel 证据。

## 1. 实验治理：先冻结规则，再看 test

### 1.1 数据与版本

- [ ] 所有新质量实验使用 article/document-level calibration/dev/test，禁止 line-level bootstrap。
- [ ] calibration 只拟合 rank split、质量阈值和 layer budget；dev 只选系统参数；test 只运行一次冻结策略。
- [ ] 保存 model revision、tokenizer revision、dataset fingerprint、article hash、命令、代码快照和环境版本。
- [ ] serving workload 保存原始 request trace、arrival timestamp、prompt/output length 和随机种子。
- [ ] 同一组系统策略使用完全相同的请求内容、arrival schedule、routing seed 和 warm-up。

### 1.2 预注册 primary comparisons

论文主检验只保留以下比较，其他配置均标为 exploratory：

1. `R-layout vs uniform FP8`；
2. `receiver-aware vs rank-only`，同总 payload、同质量预算；
3. `Graceful head-priority/ignore-late vs wait-all`，同正常格式；
4. `critical-single vs matching node-FP8`，同 EP/topology/placement；
5. 每个候选都与最强动态 gate/tail-mass selector 比较，不只与 BF16 比较。

### 1.3 统计规则

- [ ] 质量以 article/request 为独立单位，paired bootstrap 至少 5,000 次。
- [ ] latency 以 request 为单位，报告 P50/P95/P99；tail percentile 使用 request-level bootstrap CI。
- [ ] 核心 serving 点每个配置累计至少 10,000 个稳定区请求；P99 主张优先使用 30,000～100,000 个请求。1,000 请求只能做 smoke，不能做 P99 论文结论。
- [ ] 每个系统配置至少 5 次独立 run，并保留 run id；对 burst/时间相关 trace 使用 run/block bootstrap，不能把连续 request 当完全 iid。
- [ ] 固定 warm-up，排除模型加载、图编译和 cache cold-start；cold-start 另表报告。
- [ ] primary comparisons 使用 paired arrival trace；多重主检验使用 Holm correction 或预先限定数量。
- [ ] 同时报 effect size、CI 和原始分布，不以单个平均值或单次最好结果做结论。
- [ ] Go/No-Go 同时要求 CI 排除零、超过空跑噪声地板，并达到预注册的 practical effect；阈值不作为脱离噪声的机械判定。

### 1.4 实验规模分层，避免组合爆炸

- **Core matrix（论文主表）**：1 个正式 top-8 模型；prompt 512/2K；decode concurrency 8/32；60%/90% load；balanced/natural skew；uniform FP8、R-layout、strongest gate selector。
- **Stress matrix（外推边界）**：top-2 或 top-16 第二模型；8K prefill、64 concurrency、2× burst、hotspot、第二 topology/placement。
- **Exploratory matrix（不进主检验）**：Graceful、critical-single、oracle、完整参数 sweep。

先完成 Core matrix，再按 gate 打开 Stress/Exploratory；不做模型 × 拓扑 × arrival × 所有策略的全笛卡尔积。

---

## 2. Phase A：质量地基与允许的 degradation budget

这一步 Mac 可完成。目的不是测速度，而是确定系统策略最多可以牺牲多少质量。

### A0. Baseline correctness

- [x] patched full 与原始 OLMoE logits 逐元素一致。
- [ ] 为所有目标模型增加同样的 exact-forward assertion。
- [ ] 验证 fake FP8/MXFP4 与目标 GPU native/cast kernel 的误差分布，报告 saturation、scale 和 rounding 差异。
- [ ] 分离 `quantization error`、`accumulation-order error` 与 `routing drift`。

失败条件：full patched path 不等价，或 fake quant 与 native format 排序明显反转，则相关离线结果全部作废并重跑。

### A1. Article-level 主质量实验重跑

模型至少覆盖：

- [ ] top-8：OLMoE；
- [ ] top-16：LLM-jp 或同类模型；
- [ ] top-2/top-4：一个不同稀疏度模型；
- [ ] 正式 serving 模型至少一个，避免结论只存在于小模型。

冻结策略：

- full BF16/reference；
- uniform FP8；
- uniform MXFP4；
- fixed rank R-layout；
- gate threshold；
- cumulative tail mass；
- random/head anti-control；
- contribution oracle upper bound。

数据：

- [ ] calibration 至少 32 篇文章；
- [ ] dev 至少 32 篇文章；
- [ ] untouched test 目标至少 64～128 篇文章或 5 万有效 token；单一 corpus 不足时增加第二 corpus，不得重复切同一 article 冒充独立样本；
- [ ] general/code/math 三个域；至少一个 unseen-domain transfer。

指标：token KL、teacher-forced NLL/PPL、routing drift、逐文章 P95/P99 quality risk、任务准确率和生成质量。

### A2. 将 KL 映射到真实质量预算

- [ ] 在 MMLU/CMMLU、GSM8K/数学、HumanEval/代码或目标任务上保存逐题 paired prediction。
- [ ] 对多个 KL 档位测任务准确率/生成一致性，建立 `KL budget → downstream risk` 曲线。
- [ ] 报告平均质量与 worst-request/P95 quality loss，避免平均 KL 掩盖少数严重退化。
- [ ] 将后续 Graceful/critical-single 的 `0.001` 等门槛替换为数据校准门槛，而不是人为常数。

Phase A 通过门槛：R-layout 在至少两个模型、两个域上形成稳定 quality–payload Pareto；若 fixed rank 被 gate/tail-mass 在相同质量下大幅支配，主方法改为动态 selector，不继续为固定 rank 包装系统故事。

---

## 3. Phase B：可复现的 EP flow trace

### B1. Trace schema

每条 dispatch/combine flow 至少记录：

```text
request_id, phase(prefill/decode), decode_step, layer_id,
token_id, token_origin_rank, expert_id, expert_owner_rank,
topk_rank, gate_weight, contribution_score,
payload_format, payload_bytes,
ready_time, enqueue_time, start_time, finish_time
```

- [ ] 同时记录 dispatch `origin -> expert` 与 combine `expert -> origin`。
- [ ] 区分 intra-GPU、NVLink domain、intra-node 和 inter-node RDMA edge。
- [ ] 保存 expert placement、replica、EP degree、TP degree 和每节点 GPU 数。
- [ ] 记录 batch formation、token permutation、capacity/drop 和 collective round。
- [ ] 输出 Parquet 原始 flow、sender-receiver matrix、每层 receiver imbalance 和热点持续时间。

### B2. Workload trace matrix

Prefill/TTFT：

- prompt length：128 / 512 / 2K / 8K；
- batch tokens：128 / 512 / 2K / 8K；
- request concurrency：1 / 4 / 16 / 32。

Decode/TBT：

- input length：512 / 2K；output length：128 / 512；
- active sequences：1 / 8 / 32 / 64；
- arrival：closed-loop、Poisson 30%/60%/90% load、2× burst；
- balanced、自然 expert skew、人工 receiver hotspot 三类场景。

模型/拓扑：

- top-k：2 / 8 / 16；
- EP：4 / 8 / 16；
- GPU/node：4 / 8；
- placement：modular、balanced random、frequency-aware、至少 3 个 placement seeds。

### B3. Trace sanity checks

- [ ] flow bytes 与实际 send/recv counter 对齐，误差低于 1%。
- [ ] 每个 token 的 dispatch/combine 数与 top-k、drop/capacity 语义一致。
- [ ] frozen-route 分析与 dynamic-route end-to-end 分开报告。
- [ ] synthetic origin 只用于早期敏感性扫描，正式结果必须使用 runtime 真实 token origin。

---

## 4. Phase C：event-driven queue / collective simulator

这一步 Mac 可完成，但必须在 Phase D 用真实 microbenchmark 校准，不能只使用 nominal `bytes/BW`。

### C0. 系统决策必须回放到模型质量

simulator 与质量实验不能各自使用不同策略后再拼成 Pareto。每次 policy decision 必须输出可重放 intervention trace：

```text
request_id, layer_id, phase, decode_step, token_id,
topk_rank, expert_id, precision, drop_or_ignore,
decision_time, deadline, delivery_semantics
```

闭环执行为：

```text
baseline dynamic run 生成 flow/route
  -> queue replay 产生逐 token/layer/rank intervention mask
  -> 在模型中精确 replay mask，重新计算 logits、quality 与后续 dynamic routes
  -> 用新 routes 重新生成 flow trace
  -> 至少复核一轮 latency/quality 是否稳定
```

frozen-route 只用于隔离直接数值误差和字节变化；最终系统点必须使用 dynamic-route replay。若无法把 simulator 决策映射回精确 token/layer/rank，该策略不能进入联合 quality–latency Pareto。

### C1. 模拟资源

- 每 GPU/NIC sender queue、receiver queue；
- NVLink 与 RDMA 的独立带宽、固定启动延迟和并发 channel；
- pack/quant、unpack/dequant、node reduction service time；
- collective round/barrier、head-ready、full-combine-ready；
- compute/communication overlap；
- late flow 是否继续占用网络、能否取消、取消生效时点。

服务时间使用：

$$T = T_{launch}+T_{pack}+T_{queue}+T_{link}+T_{unpack}+T_{reduce},$$

禁止只用 $bytes / \mathrm{nominal\ bandwidth}$。

### C2. Policy matrix

#### R-layout 基线组

- `U0`：uniform FP8 FIFO；
- `R1`：fixed-rank FP8/FP4，调度不变；
- `G1`：gate/tail-mass 动态布局；
- `R-random`：同 FP4 数量随机 rank anti-control。

#### Receiver-aware 组

- `C0`：rank-only，不看端口；
- `C-random`：同预算随机 remote flow；
- `C-cold`：故意投向冷 receiver；
- `C-profile`：离线 receiver profile；
- `C-scheduler`：只用调度器提前已知的 token-origin/receiver hotness；
- `C-online`：使用 coarse queue bucket，不做逐 flow 连续优化；
- `C-oracle`：完整 sender+receiver 当前层信息，只作上界。

关键隔离：candidate pool、总 FP4 数、总 payload、质量预算必须相同，只改变预算落在哪个 receiver flow。

可部署信号边界：decision 必须发生在 pack/admission 之前；primary online policy 只能使用当时已经可知的 `token_origin_rank`、`expert_owner_rank`、历史/coarse queue bucket 和本层已形成的 batch。使用未来 arrival、真实 finish time 或全局未来队列的版本一律标为 oracle。

#### Graceful 组

- `G0 wait-all`：head/tail 都完成才进入下一层；
- `G1 head-priority wait-all`：只改变优先级，不丢 tail，用来隔离 QoS 本身；
- `G2 receiver-ignore-late`：late bytes 已经发送，不能记为 saving；
- `G3 sender-cancel`：发送前 admission cancellation，才可记为 saving；
- `G4 quality-aware cancel`：在 gate/tail-mass 安全集内取消；
- `G-random`：相同触发率随机 contribution；
- `G-oracle`：按真实 contribution 和未来 arrival 选择，只作上界。

触发率扫描：0.1% / 0.5% / 1% / 5% / 10% token；连续受影响层：1 / 2 / 4 / 8；tail ranks：1 / 2 / 4。重点是 rare P99 event，不再只测 25% 高频 cancellation。

#### QTree 组

- `Q0`：per-expert FP8；
- `Q1`：matching hierarchical node-FP8；
- `Q2`：node uniform MXFP4；
- `Q3`：critical-single；
- `Q4`：two-lane full-dimensional partial，保留为负对照；
- `Q5`：quantize-each-then-sum；
- `Q6`：sum-then-quantize；
- `Q-oracle`：按 partial sensitivity 选格式。

所有 QTree 比较必须使用相同 EP、placement、token origin 和 node aggregation；不得再拿 per-expert FP8 代替 hierarchical FP8 主基线。

QTree protocol 必须固定：谁执行 node reduction、partial buffer 的 owner、何时量化、scale 粒度、跨节点消息数、目的 receiver、最终 accumulation 顺序与同步点。没有这一协议，`partial 数相同` 不等于 transport 可比。

### C3. Simulator 指标

- per-layer MoE completion P50/P95/P99；
- head-ready time 与 full-combine-ready time；
- MoE-attributable TTFT/TPOT proxy；
- receiver queue length、queueing delay、HOL blocking；
- transmitted bytes、useful bytes、late bytes、cancelled-before-send bytes；
- NVLink/RDMA utilization、effective bandwidth；
- 每请求 degradation 次数与 quality budget；
- goodput：满足 latency SLO 且质量预算未超限的 requests/s。

### C4. Simulator 校准与验证顺序

- [ ] `C0`：先用未校准 simulator 做定性区间和实验设计，不报告绝对收益；
- [ ] `D0`：采集 uniform FP8/node-FP8 baseline microbenchmark；
- [ ] `C1`：只用 D0 的 train shapes 拟合固定延迟、有效带宽、并发 channel、pack/unpack 和 reduction cost；
- [ ] `D1`：预留未参与拟合的消息大小、并发和 hotspot 作为 held-out validation；
- [ ] `C2`：只有 held-out operator median 误差低于 10%、P95 低于 15% 后才扫描 policy；
- [ ] 未通过校准时，simulator 只能用于定性敏感性，不得输出“预测 TBT 提升”。

---

## 5. Phase D：真实 kernel / collective microbenchmark

### D0. 最小硬件

- 最低：单节点 4～8 GPU，用于 pack/kernel/NVLink；
- 关键证据：至少 2 节点 × 4 GPU，具有真实 RDMA/IB/RoCE；
- 理想：EP8 与 EP16 两种形态，覆盖节点内与跨节点。

记录 GPU/NIC 型号、互联、驱动、CUDA、NCCL/DeepEP 版本、频率、功耗模式和拓扑。

### D1. Uniform baseline

- [ ] BF16 dispatch/combine；
- [ ] uniform FP8 dispatch/combine；
- [ ] matching hierarchical FP8；
- [ ] 分解 quant、pack、send、recv、unpack、dequant、reduce；
- [ ] 判断每个 shape 是 bandwidth-bound、latency-bound 还是 kernel-bound。

如果 uniform FP8 本身不优于 BF16，必须先解释 primitive/shape 问题，不能继续把 mixed precision 的收益归因于方法。

### D2. R-layout kernel

- [ ] fixed head/tail buffer，offset 静态可计算；
- [ ] FP4 packing + block scale metadata；
- [ ] fused quant-pack；
- [ ] fused unpack-dequant-gate-reduction；
- [ ] empty rank、padding、alignment、不同 hidden size；
- [ ] 与 gate/tail-mass variable buffer 对比 mask、prefix-sum、metadata 和 divergence 开销。

### D3. Receiver-aware scheduler

- [ ] 相同 payload 下构造可控 incast：1/2/4/8 senders → 1 hot receiver；
- [ ] 比较 FIFO、random、cold、receiver-aware、oracle；
- [ ] 测 hot receiver completion、其他 receiver regression、公平性和全局 collective completion；
- [ ] 分离“少发字节”与“改变优先级/发送顺序”的收益。

### D4. Graceful transport

- [ ] head/tail 独立 stream/QP/buffer，但记录是否在底层仍共享同一 NIC queue；
- [ ] 测 head 是否真的绕过 tail HOL blocking；
- [ ] receiver-ignore-late 必须记录 late bytes 继续占用网络的影响；
- [ ] sender-cancel 记录 cancellation decision 到实际停止 DMA 的延迟；
- [ ] 测进入下一层是否真的无需等待 tail，以及 buffer 生命周期/正确性。

### D5. QTree partial

- [ ] node-FP8 是主基线；
- [ ] 测节点内 reduction、partial classification、pack 和跨节点发送；
- [ ] critical-single 与 node-FP8 的 partial 数必须相同；
- [ ] two-lane 同时报告 head-ready 与 full-ready，防止只看总字节遗漏潜在 early-head 收益；
- [ ] 扫 placement seeds，验证 tail-only remote partial 比例是否稳定。

### D6. Microbenchmark matrix

- hidden size：1K / 2K / 4K / 8K；
- top-k：2 / 4 / 8 / 16；
- active tokens：1 / 8 / 32 / 128 / 512 / 2K；
- senders per receiver：1 / 2 / 4 / 8 / 16；
- message size：从 decode 小消息到 prefill 大消息；
- EP：4 / 8 / 16；intra-node / inter-node；
- balanced / hotspot；无 overlap / realistic overlap。

指标：operator P50/P95/P99、breakdown、实际 NIC bytes、消息数、effective bandwidth、SM/NIC utilization、head/full completion、额外 HBM 和 CPU scheduling cost。

---

## 6. Phase E：端到端 serving

### E0. 指标定义与 instrumentation

- `TTFT = first_token_emitted_timestamp - request_arrival_timestamp`；同时拆出 queue、prefill compute、MoE dispatch/combine 与 sampling。
- `TBT_i = token_i_emitted_timestamp - token_{i-1}_emitted_timestamp`；`TPOT` 另按每请求输出阶段总时长除有效 token 数报告，二者不能混称。
- steady-state P99 排除 warm-up 但不排除正常 queueing/batching；cold-start 单列。
- GPU 事件、runtime scheduler 事件与 NIC counter 使用统一 clock/关联 id；若跨机时钟不可直接比较，使用 host-side causal marker 或校准后的 clock offset。
- 同时报告 request-level tail 与 token-level tail，明确统计单位；禁止用大量相关 token 冒充独立 request 扩大样本量。

### E1. 集成与正确性

- [ ] 接入 vLLM/SGLang/目标 runtime，固定相同 model weights、parallel plan 和 scheduler。
- [ ] 每个策略使用同一 prompt、sampling 参数、arrival trace 和随机种子。
- [ ] 校验输出 token、logits/任务质量与离线预测方向一致。
- [ ] 记录每层 MoE 时间线，确认收益确实来自 combine/queue，而非 batch 偶然变化。

### E2. TTFT workload

- prompt：128 / 512 / 2K / 8K；output=1 或固定短输出；
- arrival load：30% / 60% / 90% max throughput；
- batch policy 固定与动态 batching 分开；
- 报告 TTFT P50/P95/P99、prefill throughput、MoE 时间占比和每层 critical receiver。

### E3. TBT/TPOT workload

- prompt：512 / 2K；output：128 / 512；
- 并发 active sequences：1 / 8 / 32 / 64；
- closed-loop、Poisson、bursty 三种 arrival；
- 报告 TPOT、逐 token TBT P50/P95/P99、inter-token jitter、throughput、goodput 和 SLO violation rate。

### E4. 策略组合实验

按照增量方式组合，避免无法归因：

1. uniform FP8；
2. + R-layout；
3. + receiver-aware；
4. + head-priority（不降级）；
5. + rare Graceful；
6. node-FP8 → critical-single；
7. 最终组合。

每一步报告独立增量和 interaction。若组合收益小于单模块之和，需要解释共同瓶颈或开销重叠。

### E5. 质量—系统联合 Pareto

每个系统点同时报告：

```text
(TTFT P99, TBT P99, throughput/goodput,
 transmitted bytes, operator latency,
 token KL, task accuracy, P95 request quality loss)
```

不允许把一个策略的 latency 和另一个策略的质量结果拼成 Pareto。

---

## 7. 必须做的反事实与替代解释实验

- [ ] **同字节不同位置**：receiver-aware vs random/cold，排除“只是多压了一些”。
- [ ] **同位置不同精度**：验证收益来自格式而非 route selection。
- [ ] **同质量不同布局**：R-layout vs gate/tail-mass，验证规整 kernel 的真实价值。
- [ ] **同 flow 不同优先级**：head-priority wait-all vs FIFO，隔离 QoS。
- [ ] **同质量 miss 不同传输语义**：sender-cancel vs receiver-ignore-late，区分省字节和省等待。
- [ ] **冻结 route vs dynamic route**：区分 operator 数值误差与下游 routing drift。
- [ ] **no-pack upper bound**：预量化/预打包，测通信压缩的理论 latency 上界。
- [ ] **no-network control**：本地 copy 替代 network，量出 pack/dequant 固定成本。
- [ ] **no-compute 与 full-overlap**：确定收益是否会被 expert GEMM/attention 隐藏。
- [ ] **placement seeds**：排除 modular placement 偶然性。
- [ ] **balanced vs hotspot**：明确 receiver-aware 只在哪些负载有效。
- [ ] **rare vs sustained congestion**：避免用平均吞吐结论外推 P99 burst。

---

## 8. 各机制 Go / No-Go 门槛

### 8.1 R-layout 主线

Go：

- 至少两个模型上，相对 uniform FP8 mixed-combine operator 改善 `>=10%`；
- 或端到端 TTFT/TPOT 改善 `>=3%`、P99 改善 `>=5%`；
- quality 落在校准预算内；throughput regression 不超过 3%；
- 固定 rank 相对动态 selector 的质量劣势能被更低系统开销补偿。

No-Go：pack/dequant 吞掉 payload 收益，或动态 selector 在质量和系统两侧均支配 fixed rank。此时论文收缩为 characterization，或主方法切换为 gate/tail-mass layout。

### 8.2 Receiver-aware

Go：

- 相对 rank-only，在至少两个 hotspot/高负载场景额外降低 P99 `>=5%`；
- balanced/低负载 regression `<=2%`；
- 同总 payload、同质量预算；
- scheduler 只使用部署时可获得的 receiver-side/coarse queue signal。

No-Go：收益只存在于 sender+receiver oracle、只在单个 placement seed 出现，或 metadata/不规则 buffer 吞掉收益。此时保留为 motivation/future work。

### 8.3 Graceful

Go：

- rare trigger（优先考察 `<=1%` tokens）下，相对 head-priority wait-all 降低 TBT/TTFT P99 `>=10%～15%`；
- 平均与 P95 request quality 都在校准预算内；
- 不发生同一请求连续多层无界 degradation；
- receiver-ignore-late 的 late traffic 不导致后续 token P99 恶化；
- sender-cancel 的 saving 只统计真正未发送 bytes。

No-Go：只有高 cancellation rate 才有收益、质量损失集中在少数请求、必须等待 deadline 导致收益消失，或 head/tail 在 NIC 层无法隔离。

### 8.4 Critical-single QTree

Go：

- 相对 matching node-FP8，至少两个 placement/topology 中真实 inter-node bytes 减少 `>=8%`；
- operator latency改善 `>=5%`；
- partial vector 数不增加；
- 跨模型质量预算通过。

No-Go：只在 modular EP16 有效、节点内 reduction/classification 开销吞掉收益、或 CI 上界超过质量预算。two-lane full-dimensional partial 已是 negative control，不再作为正常执行主方法。

### 8.5 论文级总体门槛

系统主张至少需要：

- 两个 MoE 模型；
- 两种 topology/EP 配置；
- prefill 与 decode；
- balanced、natural skew、hotspot；
- 真实多 GPU operator + 端到端 serving；
- quality、bytes、latency、throughput/goodput 同时闭环。

只通过 simulator 或只降低理论 payload，不得声称 TTFT/TBT acceleration。

---

## 9. 推荐执行顺序与止损点

```text
S0 冻结硬件/runtime/FP4 能力与 GPU-hours
  -> S1 R-layout article-level 质量强对照
  -> S2 uniform FP8 与 R-layout packed-buffer/kernel microbenchmark
  -> S3 R-layout 端到端 serving Core matrix（>=10k requests/config）
  -> S4 真实 trace 归因：receiver queue 是否是主要 P99 来源？
       -> 是：receiver-aware controlled incast + serving
       -> 否：不实现 receiver-aware
  -> S5 只按真实瓶颈二选一：rare Graceful 或 critical-single
  -> S6 第二模型/第二 topology stress validation
```

止损规则：

1. R-layout article-level 质量不形成 Pareto：先停止 kernel 实现或切换 strongest gate selector。
2. R-layout kernel 未打赢 uniform FP8：先停止 receiver/Graceful/QTree 集成。
3. Queue simulator 校准失败：不使用其绝对 TBT/P99 数字，只保留敏感性排序。
4. Receiver-aware 只打赢 random、不打赢 rank-only：不升级贡献。
5. Graceful 只有 oracle 有效：不实现复杂 transport。
6. Critical-single 跨 placement 不稳定：停止 QTree extension。

### 9.1 当前立即执行批次

- [x] 修复 OLMoE patched-full 的 expert-order BF16 accumulation，并加入 exact-logit assertion；
- [x] 将 signal comparison 切换为 validation article calibration / test article evaluation；
- [x] 增加 data manifest、版本记录、metadata-aware bytes 和 article-paired bootstrap；
- [x] OLMoE Stage-1 pilot：16 calibration articles + 16 test articles，seq=256，MXFP4；patched-full exact，test 4,080 tokens；
- [x] OLMoE Stage-1 WikiText formal：32 validation calibration articles + WikiText test 尚未查看的 45 articles（offset 16），11,475 test tokens；pilot/formal test article overlap=0；
- [ ] 第二 corpus/domain formal：补足跨域与总 article/token 证据，article 仍作为 cluster/bootstrap unit；
- [ ] 资源清单确认后再启动 D0；在此之前所有 Mac 输出继续标为 quality/proxy。

本机 Stage-0 verdict：PyTorch CUDA device count=0、MPS unavailable、无 NVIDIA toolchain/NCCL/IB device/serving runtime；FP4 dtype 也没有 CPU copy kernel。D0/D/E 必须等待外部 CUDA 多 GPU/RDMA 资源，Mac 只继续第二 corpus/model 质量与 layout correctness。

Pilot verdict：R-layout KL `0.005849`；gate `0.005194`，相对 R-layout paired delta `-0.000655`，95% CI `[-0.001132,-0.000304]`；head/interleaved controls 为 `0.031741/0.029148`，16/16 articles 均劣于 tail。结论是 **rank criticality 通过，但 fixed rank 在质量侧被 gate 小幅支配；只有真实规整 kernel 能补偿该差距时主线才成立**。Pilot 使用过的 test 前 16 篇从此视为 dev/pilot，不进入 formal CI。详见 `experiments/idea_a_mac/outputs/paper_validation/olmoe_r_layout_article_stage1_pilot_2026-07-13/Stage1_RLayout_质量Pilot结论.md`。

Formal verdict：R-layout / gate / tail-mass / oracle KL 为 `0.006042 / 0.005635 / 0.005616 / 0.005559`。三种 selector 相对 R-layout paired KL delta 为 `-0.000406 / -0.000426 / -0.000482`，CI 全部排除 0；head/interleaved 是 R-layout 的 `5.95×/4.59×` KL，45/45 articles 更差。结论收敛为 **rank criticality 通过、fixed-rank 质量 Pareto 不通过、R-layout 仅作为可能用 kernel 规整性换小幅质量差的系统候选保留**。详见 `experiments/idea_a_mac/outputs/paper_validation/olmoe_r_layout_article_stage1_formal_2026-07-13/Stage1_RLayout_WikiTextFormal结论.md`。

## 10. 最小资源版本与完整论文版本

### 最小资源闭环

- 1 个 top-8 模型；
- 2 节点 × 4 GPU；
- EP8；
- uniform FP8 / R-layout / receiver-aware / head-priority wait-all；
- prefill 512/2K，decode concurrency 8/32；
- balanced + hotspot；
- operator、TTFT/TPOT/P99、quality 联合表。

它足以判断 Idea A 是否有真实系统价值，但不足以支持广泛泛化。

### 完整论文版本

- top-2、top-8、top-16 至少两个正式模型；
- EP8/EP16、两种 GPU/node；
- 三个 workload 域与三类 arrival；
- R-layout 主方法；receiver-aware 作为通过门槛后的第二机制；
- Graceful/critical-single 只有通过各自门槛才进入；
- 完整 quality–TTFT/TBT–goodput Pareto 与机制消融。

## 11. 预期产物

```text
experiments/idea_a_system/
  traces/
    ep_flow_trace.parquet
    request_manifest.json
  simulator/
    run_queue_replay.py
    simulator_calibration.csv
  kernels/
    uniform_fp8_results.csv
    r_layout_results.csv
    receiver_incast_results.csv
    graceful_results.csv
    qtree_results.csv
  serving/
    ttft_results.csv
    tbt_results.csv
    goodput_results.csv
  reports/
    01_quality_budget.md
    02_queue_go_no_go.md
    03_kernel_go_no_go.md
    04_serving_final_verdict.md
```

每个报告必须包含：命令、版本、数据 manifest、强基线、失败配置、原始 CSV、CI、证据边界和最终 Go/No-Go。
