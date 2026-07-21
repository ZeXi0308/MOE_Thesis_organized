# 目前最值得推进的 MoE 研究方向：质量约束关键路径通信，先验证必要条件

> **文档整理 2026-07-21**：纠错后仍判死的候选（TokenRace、Prefetch、CreditReduce、PLTB…）正文已迁至 [`../99_archive/killed_ideas/`](../99_archive/killed_ideas/)。活 idea 索引：[`../ideas/README.md`](../ideas/README.md)。

## 执行摘要

目前最应该推进的不是继续设计 receiver classifier、direct-benefit controller 或 bandit，而是完成一个更基础且更决定性的研究问题：真实任务质量约束下，是否存在一个非空区域，使细粒度 receiver-aware 在线策略严格优于 `uniform_full`、`uniform_low` 和 `calib_static`。这应被定义为 `Quality-Constrained Adaptivity Existence Test`，即“在线自适应必要性检验”。它既是当前唯一值得立即在 GPU 上执行的 receiver 实验，也能决定后续是否值得投入真实多 GPU/RDMA 系统。

如果这个必要条件成立，最有潜力的系统主线是 `Critical-Path-Aware, Quality-Budgeted MoE Communication`：用真实 receiver/combine/NIC telemetry 识别关键路径，仅对确实能缩短 P99 TPOT 的 homogeneous low-bit EP lane 动作，并用 action-conditioned task harm、置信上界和 streaming tenant debt 约束质量。如果必要条件不成立，就应停止精细在线控制：低质量价格区间采用 `uniform_low`，中间区间采用校准静态策略，高质量价格区间采用 `uniform_full`。

当前证据最稳的备线是 Energy-SLO Precision EP；单卡可形成新机制的高潜力备线是 `Verify, Don’t Predict`，但后者必须先证明真实 shadow 双算、切换和 wall-clock 成本不会吞掉收益。已有方向中，Expert Prefetch、progressive residual、旧 HHI v3、mean-balanced placement 和双轴合成 POC 不应继续投入。

## 为什么不是马上做 bandit 或多 GPU receiver controller

修正后的质量约束 reward 扫描已经显示三段结构。低 `λ` 时，质量代价便宜，`uniform_low` 往往最优；中等 `λ` 时，价值多数来自 `uniform_low` 与 `calib_static` 的粗粒度切换；高 `λ` 时，加入 reward 恒为零的 `uniform_full` 后，不降级成为正确选择。精细 controller 只在少数代理网格点获胜。

但当前中间策略的质量成本仍使用 `low_frac × KL_uniform_low` 的线性代理。它既没有证明 low fraction 与 harm 线性，也没有证明 KL 与 task accuracy、worst-request harm 或 CVaR 线性。因此目前不能从代理 λ 网格推出“需要在线机制”，更不能据此投入 bandit。

真正需要检验的是：

\[
R_p(\lambda)=S_p-\lambda H_p
\]

其中 `S_p` 必须来自与 policy 实际动作一致的通信收益，`H_p` 必须来自真实 task accuracy/CVaR。细粒度在线机制只有在某个可信质量约束或 λ 区间内，严格优于以下三类粗粒度策略，才有继续研究的价值：不降级 `uniform_full`、全部降级 `uniform_low`、校准后固定策略 `calib_static`。

这也是为什么当前最优先工作已经不是“改善 controller”，而是“证明 adaptivity 确实必要”。仓库中新完成的 GPU harness 正是实验基础设施，还不是新的正结果。

## 第一优先方向：Quality-Constrained Adaptivity Existence Test

### 科学问题

在真实任务质量约束下，MoE receiver-aware 通信是否存在细粒度在线自适应的必要性？或者，所有可实现的 Pareto 点其实都能被三种粗粒度策略覆盖？

这个问题本身具有研究价值，因为它把“在线机制是否有效”从算法竞赛改成了策略复杂度的必要条件检验。若结果为负，也能形成严格的系统经验结论：在质量价格的低、中、高区间，分别使用 uniform-low、calibration-static 和 no-action 即可，复杂 controller/bandit 不提供稳定增量。

### 当前已实现的实验基础

`docs/ideas/receiver_aware/experiments/run_receiver_aware_task_quality_gpu.py` 和 `receiver_lane_policy.py` 已支持五个策略：`uniform_full`、`uniform_low`、`calib_static`、`causal_no_hysteresis` 和 direct-benefit controller。实验在真实 MoE expert combine output 上施加 policy 动作，BF16 reference 测绝对质量，`uniform_full`/remote-FP8 作为 λ 的增量质量基线。

当前实现还包括 MMLU accuracy、逐题 correctness harm、answer-NLL、CVaR、分层 bootstrap CI、动作一致的通信收益、codec 乐观/serialized-tile 敏感性以及 reward 直线的精确 λ 上包络。它解决了先前跨 workload 拼接 saving、截断 MMLU 选项、resume 混跑、BF16 与 FP8 基线混淆等问题。

### 最小决定性实验

先在 OLMoE 和 LLM-jp 上运行冻结配置，不再新增 controller。每个模型至少覆盖 balanced 与 hotspot 两种 workload，五个 policy 使用相同题目、相同 microbatch 顺序和冻结的 validation calibration。

必须报告：accuracy delta、lost-correct count、correctness-harm CVaR、answer-NLL harm CVaR、per-subject 结果、low-pair exposure、selected/optimistic/serialized-codec 三种 saving，以及精确 λ winner 区间。

Go 条件必须同时满足：至少一种真实任务 harm 口径下存在非空 λ 区间；该区间内 `controller` 或 `causal_no_hysteresis` 严格优于三个粗粒度策略；优势在两个模型或两个 workload cell 中复现；相对最佳粗粒度策略的预期 P99/通信收益至少达到 3%，且在 serialized-tile codec 上界下不消失；bootstrap 后 winner 不是由一两道题翻转造成。

No-Go 条件是：所有可信 λ 区间都由三类粗粒度策略覆盖；精细在线 arm 只在 NLL 代理上获胜、accuracy/CVaR 不支持；优势只存在于乐观 codec 假设；或相对 static 增量小于3%。一旦 No-Go，应停止 bandit 和精细 receiver controller，不再通过扩大 λ 网格或换 reward 公式抢救。

### 这里可以提出的创新点

第一，提出 `adaptivity existence region`：不是给定 λ 比 controller，而是精确求解所有策略 reward 上包络，识别在哪些质量价格和 codec 成本假设下在线自适应确有必要。

第二，提出 `action-consistent quality pricing`：质量 harm 和通信 saving 必须来自同一 policy action trace；BF16 用于绝对质量审计，remote-FP8 用于与通信收益同基线的增量 λ 定价。

第三，提出 `complexity ladder`：策略复杂度按 `uniform → calibration-static → causal threshold → hysteresis controller → bandit` 逐级增加，只有上一层的 oracle/真实增量足够大，才允许进入下一层。这比直接展示复杂 controller 胜出更符合系统可证伪性。

上述三点可以形成严谨的方法论或经验分析，但只有在真实任务和真实通信中出现稳定非空区域时，才能升级为系统主贡献。

## 条件成立后的主线：Critical-Path-Aware, Quality-Budgeted Communication

### 最小新颖主张

现有 DeepEP、Comet、UCCL-EP、PROBE、ReaLB 等工作已经覆盖高性能 EP、通信计算重叠、实时负载调整和部分精度动作。单独声称“实时 telemetry”“动态低比特”“在线控制”都不够新。

可防守的最小主张应是：在生产级 EP 通信库之后，使用 receiver/combine critical-path telemetry，而不是 token imbalance 或 regime label，判断 homogeneous low-bit lane 是否能缩短下一 microbatch 的端到端关键路径；动作同时受真实 task-quality frontier 和 tenant debt 约束。

其中 homogeneous lane 不是为了人为强调“同质”，而是为了让 descriptor、packed layout 和通信 collective 保持可执行，避免 per-vector mixed codec 和 progressive residual 已经暴露的 kernel 固定开销。系统创新必须落在“关键路径动作 + 质量预算闭环”，而不是又一个压缩格式。

### 控制架构

快环应是确定性的 safety filter：读取上一 microbatch 的 receiver bytes/backlog、dispatch/combine CUDA event、stream wait、deadline slack 和可获得的 NIC counter 增量，估计 `critical-path time saved - codec cost - λ × task harm`。只有净收益下置信界大于零、quality debt 未越界时才动作，并使用滞回、最小驻留和 FP8 fallback。

慢环才允许使用 conservative contextual bandit，且只调整 threshold、dwell time 或 lane eligibility。Bandit 不应在微秒级数据面探索，也不应成为论文的第一创新点；如果简单 queue threshold 已达到 oracle 大部分收益，复杂学习器应被删掉。

质量控制采用两层结构：软层为 action-conditioned harm 和 uncertainty gate；硬层为 predictor-free streaming debt，按低比特 token/bytes、动作强度和连续降级累计 tenant/request exposure。在线真实 harm 不可得时，debt 记录 exposure，少量 shadow reference 只用于校准动作价格，不能假装每步免费获得 counterfactual KL。

### 多 GPU 决定性门槛

第一个门槛是残余瓶颈存在性：在 DeepEP/NCCL EP 等强基线之后，receiver/shared-cut/incast 是否仍造成至少10%的 P99 excess latency，且上一轮 telemetry 对下一轮 critical-path excess latency 的 held-out 相关性达到约0.5。若不存在残余空间，方向停止。

第二个门槛是动作本身：真实 pack→GPUDirect RDMA→unpack 的 homogeneous low-bit lane，在 matched quality 下是否使 P99 TPOT 改善至少5%，吞吐下降不超过2%。若 codec 后净收益小于3%，控制器没有研究价值。

第三个门槛才是控制器增量：相对最强 queue-threshold+滞回基线，复杂控制器 P99 再改善至少3%，且 SLO violation、task accuracy 和 tenant CVaR 不恶化。只有三个门槛全部通过，它才适合作为独立系统主贡献。

## 第一备线：Energy-SLO Precision EP

从当前证据完整度看，Energy-SLO 是最稳方向。已有真实单 GPU 结果包括 batch 1→64 能耗/token 约17.4×改善、FP8 GEMM 吞吐约2.03×、单 matmul 能耗下降约34.3%，以及两个模型实际 expert FFN FP8 路径 KL 约0.00675和0.00905。

最值得探索的创新不是再证明 FP8 或 batching 节能，而是 `arrival- and communication-aware precision-batching co-control`：在真实 Poisson/bursty arrival、P99 SLO 和 EP communication energy 下，联合选择 batch accumulation 与 compute precision。关键问题是两个单独杠杆联合后是否仍产生相对最佳单轴 controller 的增量，而不是把17.4×和2.03×简单相乘。

单卡先做真实 arrival 的 batch×precision 联合 Pareto。Go 门槛是相同 P99 SLO violation 下，能耗/token 相对最佳单轴 controller 下降至少10%，TPOT/P99 不恶化超过3%。多 GPU 再计入 GPU、NIC、CPU 和通信等待能耗，要求节能仍至少8%。如果只能复现独立 microbenchmark，它只能作为第二贡献。

## 第二备线：Verify, Don’t Predict

这条方向的创新表达比 Energy-SLO 更鲜明：验证 precision sufficiency，而不是预测未来脆弱性，也不是 speculative decoding 中验证 token correctness。仓库中固定周期离线控制已有正结果，而且更晚的细网格可能存在两个模型共同过50% harm-reduction 的配置；但这必须以原始细网格输出重新冻结，不能继续引用过时的 period=4 单点结论。

它当前最大的未知不是算法，而是系统成本。真实 shadow verification 可能要求低精度主路径和 BF16 reference 双算，还包含权重驻留、同步和精度切换。如果这些成本没有计入，离线高精度占比不能转换为 wall-clock 收益。

最值得的创新点是 `budgeted precision-sufficiency verification`：根据已累计的 uncertainty/debt 自适应安排验证，但必须先以固定 period=3/4 做真实 in-loop 基线。Go 门槛应是两模型真实 decode 下 harm reduction 至少40%–50%，端到端 TPOT或能耗改善至少8%，验证总开销低于5%。若净收益低于5%，保留为负结果或质量控制经验分析。

## 当前不应投入的方向

Expert Prefetch 已被正确 cache key、full-top-k 和 oracle 天花板否定；progressive residual transport 在等 wire budget 下质量更差且 codec break-even 远低于现代网络；mean-balanced placement 优化平均量而非瞬时 max/P99；旧 HHI v3 存在因果错误且修复后相对最强固定基线不显著；双轴 POC 使用不可通约的合成代价。它们不应因获得多 GPU 而重启。

Tail-aware placement 和 overlap 只能作为 oracle-gated 探索项。先测事后最优 placement 或强通信库后的 residual bubble；若 oracle P99 空间不足10%，不开发算法。

## 推荐执行顺序

第一阶段只做当前已实现的 receiver task-quality GPU gate。先 smoke test，再在 OLMoE 和 LLM-jp 上冻结样本运行，检查 `task_harm_lambda_exact_intervals.csv`。不要在看到结果前修改 controller。

第二阶段依据结果分叉。如果精细 online arm 没有稳健非空区间，停止 receiver 在线控制，转向 Energy-SLO；如果存在非空区间但只要求 static-vs-uniform 切换，将论文重构为低频 workload-level policy selection，而不是 microbatch controller；只有细粒度 online arm 跨模型/场景获胜，才申请真实多节点 RDMA 资源。

第三阶段在多 GPU 上依次验证 residual bottleneck、homogeneous lane action 和简单 queue threshold。只有简单机制已经 Go，才加入 quality debt、uncertainty gate 或慢环 bandit。

并行保留一条单卡备线：先核对 shadow verification 细网格的最新原始结果，然后实现最简单固定周期 in-loop 版本。Energy-SLO 则作为最稳论文出口，优先补真实 arrival 和联合 Pareto。

## 最终推荐

如果只能选一个当前立刻推进的工作，选择 `Quality-Constrained Adaptivity Existence Test`。它成本最低、决定性最高，并直接回答 receiver-aware 是否值得继续。

如果该 gate 为 Go，主线选择 `Critical-Path-Aware, Quality-Budgeted MoE Communication`，但把 bandit 降为可选慢环，不把它当预设创新点。真正的创新是：由真实关键路径而非 regime 触发动作，由真实 task frontier 而非 KL 代理定价质量，并由 streaming debt 提供跨请求安全约束。

如果 gate 为 No-Go，立即停止 receiver 精细在线控制，优先转向 Energy-SLO Precision EP；若希望追求更鲜明的机制创新且只能使用单卡，则并行验证 `Verify, Don’t Predict` 的真实 in-loop 净收益。

## 局限

当前没有真实多节点 RDMA/IB/RoCE 数据，因此 receiver 系统主线仍是条件性判断。外部 2026 年预印本和系统实现需要投稿前再次核验；微信公众号搜索结果稀疏且“质量公平”查询严重混入教育领域内容，未用于支撑关键结论。仓库新 GPU harness 已通过静态检查，但尚未在 RTX GPU 上执行，不能被当成实验正结果。

## References

1. [DeepEP](https://github.com/deepseek-ai/DeepEP)
2. [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)
3. [UCCL-EP: Portable Expert-Parallel Communication](https://arxiv.org/abs/2512.19849)
4. [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503)
5. [PROBE: Co-Balancing Computation and Communication in MoE Inference](https://arxiv.org/abs/2602.00509)
6. [Fairness in Serving Large Language Models](https://www.usenix.org/conference/osdi24/presentation/sheng)
7. [VoltanaLLM: Energy-Efficient and SLO-Aware Disaggregated LLM Serving](https://arxiv.org/abs/2509.04827)
8. [DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](https://arxiv.org/abs/2508.06041)
9. [本项目 MoE 存活方向重新审计](file:///Users/leandrozhao/Desktop/毕设论文资料/docs/04_research_plans/research_report_moe_surviving_directions_reaudit_2026-07-20.md)
10. [研究现状与 Idea 演进](../01_current_status/研究现状与Idea演进_2026-07-21.md)
