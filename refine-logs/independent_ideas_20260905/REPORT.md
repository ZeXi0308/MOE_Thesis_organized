# 独立方向：实验与实现准备，2026-09-05

**Verdict：优先完成 Verify Precision 的等预算数值资格化；Receiver、Offload 保留有条件的测量准备；不实现同步 full-reference SemanticFence；QuotaEP 等真实 fused backend 后清算系统 Gate。全部新 GPU 实验 UNRUN，没有新增 method GO。**

用户本轮明确授权独立方向的设计、代码准备和把关。既定的 MoE SLO 并发控制仍是主线；这里没有另立多个 Controller，也不更改已封存科学结论。

## 开始时的执行合同

| 项目 | 本轮冻结内容 |
|---|---|
| Repository | `agent/publish-current-moe-code`，HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`，ahead 1 |
| Dirty state | 开始时只有既存 admission_capacity 实验和输出两个未跟踪目录；本轮只在本目录新增文件 |
| Authority files read | `docs/current/README.md`、`docs/ideas/README.md`；SemanticFence、Verify Precision legacy、Receiver README/verdict；QuotaEP/Prefetch 判死文及 `Idea系统纠错审计_2026-07-22.md`；各一份相关 runner/关键 artifact |
| Frozen facts | SemanticFence 预执行 witness 失败但 proxy action space 尚在；旧 Verify H2 masking 无效，兄弟 checkout 的 true-in-loop 修正仍是 `INCONCLUSIVE_RUNTIME`；固定 RankLane NO-GO 不等于 return family NO-GO；QuotaEP 系统 Gate C 未跑；旧 Prefetch formulation 不复活 |
| One research question | 独立方向是否还有一个能被低成本实验区分、且强简单基线未解决的缺口？本轮优先关闭 Verify 的预算混杂 |
| One weakest link | Reactive 的数值改善是否只是比 fixed 多用了高精度，而非选择时机更好？ |
| One next GPU experiment | fresh cohort、32 teacher-forced steps、每策略恰好 4 次 served-high，fixed / reactive / random 三臂各自推进 KV |
| Allowed claim ceiling | 本轮最多代码准备、CPU 正确性和结构容量计算；后续 Verify 最多 double-shadow/QDQ 数值选择信号 |
| Stop / continue / reopen | 不用新测试当研究结果；缺数据保持 UNRUN；预算公平后有稳定信号才升级基线和 native backend；负结果只覆盖所执行 formulation |

使用 experiment-plan 与 research-review 做了一轮针对性检查。fresh reviewer 为 GPT-5.6-Sol，`review_independence=same-family`，意见 provisional；未创建 `.aris` 或多轮 jury。采纳其预算、公平基线和 full-reference 成本意见；未把建议的 3%/5% 自动改成判死阈值，也没有因硬件/采集接口缺失而假装实现了 native runner。

已完成 **11/11 定向 CPU 测试**（Verify 4、Receiver 4、Offload 3），以及 tiny OLMoE 的真实 DynamicCache/QDQ API 检查。Verify 的 [8 篇冻结输入](verify_precision/prepared_r01/COMPLETE.json)已保存到仓库；无 CUDA 时入口返回 UNRUN、零个已完成 policy，并验证输入逐字节复用。完整 [verification.log](verification.log) 留存检查边界。当前全部代码仍是未跟踪新增文件，未提交或 push。

## 哪些优化真正值得做

| 方向 | 明确可改进的环节 | 最强简单基线/当前缺口 | 本轮决定 |
|---|---|---|---|
| Verify Precision | 把 served-high 预算和实际 H/L 调用成本分开；修正策略未来状态和对照预算 | 预声明 fixed-period + budget-matched random；最佳周期相位尚未 calibration 资格化 | 唯一优先进入 GPU 的独立 probe；实现等预算 runner、冻结输入与汇总 |
| SemanticFence | 先限制 verifier 的成本；检验是否可复用已经必算的证据，或有真实可用的独立资源 | serial M1；batch-invariant 执行应作为后续 by-construction 竞争方案 | 同步全参考版本代数上无净收益，不为它另写 GPU runtime |
| Receiver | 直接问 return service 能否推动请求完成；先核实硬件与优化后端兼容 | 硬件支持的 optimized overlap/fused backend；真实跨 rank 因果 trace 缺失 | 实现显式 DAG 校验与结构条件界限；采集器仍未接通 |
| Offload/Prefetch | 先问 KV 与权重是否真的竞争容量、自然 miss 是否暴露 | demand-LRU、独立 calibration 的 static pinning；必要时比较 CPU 执行和简单 cap | 实现 tensor-header + KV 容量检查；暂不写 predictor/prefetcher |
| QuotaEP | 先清算表示、配额和布局在实际 packet/tile 粒度上的固定税 | 同 backend 优化的 uniform FP8，另保留 BF16 correctness；可用硬件须实证支持 | 补清晰的 Gate C 三臂设计；现有代码只能复用质量侧，系统 runner 未实现 |

### Verify Precision：公平比较之前，不解释 selector 收益

兄弟 checkout `毕设论文资料-longrun-B-resurrection-b141` 的纠正记录已现场核对；来源 HEAD、文件 SHA、旧 calibration 阈值和旧样本排除哈希只提取到 [source_lock.json](verify_precision/source_lock.json)，运行不依赖该 checkout。旧 reactive 平均 served-high 4.5、fixed 4.0 的对照不能直接回答等预算选择价值。

新实验从同一 prefill KV 分叉，每个 policy 的每一步真实计算 H/L，选择并提交自己的 cache/logits；下步继续自己的状态。三臂均 32H+32L physical calls、4 次 served-high。Random 的 4 个时刻在运行前抽好；reactive 的阈值冻结，预算耗尽不能超支，剩余步数等于剩余预算时补足，并独立报告 forced count、threshold 请求/执行和预算阻止次数。

**这是新 formulation：32 步、fixed 每 8 步、单步选择；不是旧 16 步/H4 escalation 的复现。** 补足后的 realized KV history 也不是无补足策略的反事实。旧 16-step Oracle 不证明新预算下的 Oracle 空间，新预算 Oracle 保持 UNRUN。当前 fixed phase 0 是预声明简单基线，还不能称超越经过 calibration 选择的最强 fixed phase。

核心指标是每文档、每 repeat 的累计参考 KL 原始差，pooled ratio of sums，以及先跨 repeat 合并后按文档统计的 median/正负数量。请求/document 为独立单位，不能把 32 个相邻 steps 当 32 个独立样本；缺任何预定 run、预算不等或中途失败不输出完整比较。所有 repeat 保留，不选胜出的顺序或文档。

物理双轨、KV clone、KL、decision、commit 都计入成本。该 runner 的低精度是 persistent INT4 QDQ 权重通过 BF16 `F.linear` 执行，并且输入 teacher-forced，故**不能支持 native INT4 速度、自由生成质量、KV 精度压缩或 serving Pareto**。正结果只授权更强简单基线与原生低/高后端资格化；负结果先看该预算下 Oracle 是否还有空间，不能判死整个 Verify family，也不立即增添第三个 predictor。

实现与完整命令见 [verify_precision/README.md](verify_precision/README.md)。首个 GPU pilot 只用一组可用真实文本、两个顺序受控 repeat；样本 freshness 仅针对已提取的旧 calibration/pilot/repeat，并非全仓新样本证明。

### SemanticFence：GPU resident 不会自动消除参考计算的成本

现有 [COST_PROJECTION.json](../../docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01/COST_PROJECTION.json) 给出单 M1 约 41.143 μs、一个 M2 pair 约 42.403 μs。这是旧的局部测量，本轮只重算成本关系：

```text
baseline pair = 2 × M1 ≈ 82.287 μs
synchronous full-reference pair >= M2 + 2 × M1 + verify/commit
                               >= 124.690 μs = 1.515 × baseline
```

上述结论仅适用于同资源、串行付费且没有可复用参考的 formulation；还没加 verify/commit，就已多付 M2。把 CPU/D2H 验证换成 GPU 只能降低那一项开销，不能把 reference 变免费。旧 29.27% 是假定免费知道 semantic label 的 additive expert-stage projection，不是 post-action method 的净收益。

值得保留的优化条件是：证据能来自本来就必须计算的中间量；或者确有不拖慢前景请求的独立资源/自然 slack。前者须给出成立的 fail-closed contract，后者须实测 contention 与 commit 等待。不能用便宜的 exact 输出比较冒充语义证书，exact 不等也不等于语义错误。

若提出实际可计算证书，最小三臂为 A serial M1，B 同输入 canonical state 的 M2+证书+选择性修复，C 保留 B 的证书/调度工作但强制返回 M1 的成本负控。必须在首次 downstream 消费之前提交或修复，禁止事后把错误结果掩掉；对比 whole block completion，再看完整请求。证书、packing、fallback 和 repair 互斥收费。理想零证书成本且按 row 修复时，旧均值下允许修复比例小于约 48.47% 只是必要条件，绝非已测可部署门槛。

另应与 [batch-invariant execution 的原始实现研究](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 区分：已有工作直接处理 batch shape 数值不变性。“验证+回退”组合本身不足以主张新颖性。本轮无新低成本证书，故停止同步 full-reference 实现，只保留原 proxy action-space 结论。

### Receiver：把回传时间变成完整请求的可检验依赖

新 [DAG 分析器](receiver/analyze_exposed_return.py) 接收显式因果边、资源顺序、外生 release、message/request/token 身份和统一时间线，先重建 baseline，再将目标 return service 置零，并保留所有其它约束；非关键路径 service 置零是负控。缺输入或 baseline 无法重建时返回 `NOT_IDENTIFIED`、界限为 null。禁止相加 overlap spans，也禁止用 observed start 伪造外生 release。

输出严格是 **fixed-order structural conditional bound**：即使重建吻合，也不证明因果完备；固定 service duration/order 未模拟 contention、batch、route、调度变化。因此既不是 action-conditioned 系统 Oracle，也不是已实现 Receiver 收益。真实 Nsight/CUPTI+serving identity exporter 尚未接通；这不是“只差 GPU”的端到端 ready 状态。

本次在线核实 [DeepEP main/V2 官方要求](https://github.com/deepseek-ai/DeepEP#requirements)：SM90 或支持其 PTX 的架构、Torch ≥2.10、NCCL ≥2.30.4 等。**不能默认 8×A100 使用当前 V2**；必须冻结 A100 实际兼容的优化 backend，或使用受支持 GPU。单节点 NVLink 与跨节点 RDMA 也分别报告。第一轮一模型一自然 cell、16–32 requests 即可，不先铺开旧 40-run 规格；有已识别 full-request residual 后再补第二模型/运行域和代表性动作实跑。详见 [输入与实验合同](receiver/README.md)。

### Offload：先排除人为制造的 action space

新 [容量检查](offload/inspect_residency.py) 已读取真实 OLMoE tensor headers：权重 12.88794 GiB，专家 12 GiB，1024 个 `(layer, expert)` 对象；checkpoint 的原生上下文上限为 4096。以 32 GiB 为**假设容量**，8×144 tokens 的 weights+KV 下界为 13.02857 GiB，32×4096 为 28.88794 GiB；全部保留 `FIT_NOT_PROVEN`，没有实测 activation/workspace，也没有测到自然缺页。

最小下一实验是自然压力 cell 内实际执行 demand-LRU 与 static pinning，并记录 fetch 的其它依赖 ready 与 consumer start。先证明 miss→暴露等待，再决定预取是否有价值；expert weights 全驻留的 cell 作为负控。压力下不能默认“搬到 GPU”优于“CPU 直接算”或“降低接纳 cap”。[MoE-Infinity](https://arxiv.org/abs/2401.14361) 和 [Fiddler](https://arxiv.org/abs/2402.07033) 已分别覆盖缓存/预取与 CPU/GPU 编排，潜在贡献只能落在新容量边界或基线后 residual，不能只多一个 route predictor。完整预算和采集合同见 [offload/README.md](offload/README.md)；native offload producer 尚未实现。

### QuotaEP：系统未跑，优先清算固定税

[历史纠错审计](../../docs/archive/research_summaries/Idea系统纠错审计_2026-07-22.md) 明确 Gate B 质量有效、fused-kernel TPOT/P99 Gate C 未执行。现有 [grouped owner runner](../../docs/archive/killed_ideas/quotaep/scripts/run_grouped_owner_combine.py) 在 owner-local BF16 reduce 后做 fake FP8/MXFP4，只能复用质量和 collision 统计，不能冒充通信后端。

优化点是让 quota/layout 与真实 backend 的 wire unit、tile/packet 对齐，并将 scale、header、padding、alignment、pack/unpack、launch 和 quota reservation 都计费。若自然流量已被 per-peer 固定 quota 占满，payload 的逻辑减少不会缩短 service；这时应停止该固定 quota actuator，不能靠更聪明的质量 selector 救带宽。

最小实验固定一个经支持的 EP backend 和一个自然工作负载：A optimized uniform FP8，B frozen mixed policy 的真实 fused 编解码，C 同 B 的 metadata/layout/control path 但 payload 统一 FP8 的成本负控；BF16 另作 correctness anchor。保持实际分组和可比完整 wire 预算；记录 transport bytes、pack/collective/combine service 与每 request TTFT/TPOT/completion，分开核验质量和净系统收益。若 backend 不支持相应物理表示，先标 `UNRUN_BACKEND_MISSING`，不要用 QDQ 计时替代。只有 B 比 A 有稳定完整请求净收益且优于最近邻 backend 策略，才讨论方法贡献。本轮没有该 fused adapter，不另实现 selector。

## 结束时的证据合同

| 字段 | 本轮结论 |
|---|---|
| Verdict | `PRE_GPU_PREPARATION`；独立方向下一优先项为 Verify 的等预算 probe |
| Evidence type | 当前代码/CPU 校验 + `STRUCTURAL_CAPACITY_CALCULATION`；旧局部成本仅用于代数复核 |
| What was measured | 本地 tensor-header 真实字节库存；准备输入与 CPU KV/预算、DAG 和公式正确性。最终检查记录见 `verification.log` |
| What was not measured | 所有新 pretrained GPU 结果；native low/high backend；offload fetch stalls；真实多卡 EP；QuotaEP Gate C；任何新 TTFT/TPOT/P99 收益 |
| Strongest baseline | 各方向已明确竞争基线；Verify 当前实现仅 fixed phase 0 + random，尚未证明超过最佳简单周期策略 |
| Oracle/headroom status | SemanticFence 旧 proxy 有空间，同步 full-reference 付费版本无净空间；Verify 新预算 Oracle、Receiver 真系统 Oracle、Offload/QuotaEP 系统 headroom 均未测 |
| Claim ceiling | 实现准备和有条件实验设计，不是科学 GO、论文主线成立或系统 NO-GO |
| Failure category | 预算混杂；同步参考计算成本；原生采集/后端缺口；运行域尚未资格化，分别处理 |
| Resurrection condition | 新预算下公平增益；实测低成本证书；兼容优化 EP 的完整自然 trace；自然 HBM 压力和暴露 fetch；真实 fused 表示。换名字/阈值不算 |
| One next smallest experiment | 主线 GPU scan 之后，先跑 Verify 的冻结等预算三臂，回答“相同高精度预算下，选择时机是否仍有价值”；其它方向先满足各自缺失前提 |

研究问题的回答是：**有明确优化空间，但最值得立即检查的是成本/预算与运行域的错误假设；目前没有证据支持再铺多个复杂机制。**
