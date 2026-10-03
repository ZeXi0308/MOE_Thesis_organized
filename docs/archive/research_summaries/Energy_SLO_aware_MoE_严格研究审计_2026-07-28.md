# Energy–SLO-aware MoE Inference：严格研究审计与条件化蓝图

> 日期：2026-07-28  
> 文档定位：根据用户指定问题生成的研究审计快照；**不改变** `docs/current/README.md` 的当前权威裁决和唯一执行线。  
> 当前状态：`NO_FORMALLY_VALIDATED_ENERGY_SLO_MECHANISM`。Top 1 只是最值得先证伪的 formulation，不是已成立主线。  
> 证据标签：`[Observed]` 仓库内有边界明确的证据；`[Literature]` 仅为论文作者报告；`[Inferred]` 有限推断；`[Hypothesis]` 尚未验证；`[Planned]` 未执行。  
> 禁止外推：单张 RTX 5090 的 kernel、NVML、virtual-rank 或 trace replay 不能证明 8×A100 EP、NCCL/RDMA、TPOT/P99 或生产收益。

---

## 0. 直接结论

**Verdict：Energy–SLO-aware MoE 值得作为“受硬门约束的候选研究问题”，但截至目前不值得直接宣布为硕士论文主方向。**

原因不是问题不重要，而是 2026 年的相邻空间已经非常拥挤：

- [PALS](https://arxiv.org/abs/2605.21427) 已联合控制 power cap、batch 和 parallelism；其运行时以 500 ms 周期做模型预测与反馈控制，并在 Dense/MoE 上报告能效收益。
- [Festina](https://arxiv.org/abs/2606.30391) 已覆盖 SLO slack、频率、SM 分区、请求 placement、迁移和 scale-in。
- [JITServe](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei) 已处理请求信息不确定下的 SLO-goodput 调度。
- [Director](https://arxiv.org/abs/2607.08782)、[Expert-as-a-Service](https://arxiv.org/abs/2509.17863) 和 [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880) 已分别覆盖预测式在线 placement、细粒度 expert 伸缩，以及带重配置成本的 expert GPU 分配理论；最后一项还给出 NP-hard 结果。
- [AMoE](https://arxiv.org/abs/2505.08944)、[ExpertPlex](https://arxiv.org/abs/2607.18002)、[METRO](https://arxiv.org/abs/2512.09277) 和 [LPLB](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) 已覆盖异步 expert queue/rebatch、persistent expert kernel、activated-expert minimization 和副本间 token 分派。
- [Gimbal](https://arxiv.org/abs/2606.15177) 已联合 frontend request/KV/prefill pressure、expert pressure、source-aware routing 和 placement。
- [FAST](https://www.usenix.org/conference/nsdi26/presentation/lei-yiran) 与 [SwiftEP](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi) 已显著抬高 skew/incast 与 EP collective 的通信 baseline。
- [PagedWeight](https://arxiv.org/abs/2607.16184)、[FluxMoE](https://arxiv.org/abs/2604.02715)、[Alloc-MoE](https://arxiv.org/abs/2604.08133) 和 [Mixture of Precisions](https://arxiv.org/abs/2407.14417) 已占据动态 expert quantization、expert paging、activation budget 和部分精度空间。

因此，以下宽泛命题都已不够新：

1. “联合调 batch、power cap、parallelism 以省电”；
2. “预测热门 expert 后复制/迁移”；
3. “利用 SLO slack 做 DVFS”；
4. “按 expert 热度分 hot/warm/cold”；
5. “动态精度/减少 top-k 以降低能耗”；
6. “路由感知的 admission/placement/controller”；
7. “把这些 actuator 交给 MPC/RL/贝叶斯优化”。

本轮唯一达到“值得先做存在性 Gate”的方向是：

> **E-Wave：Risk-Bounded Expert-Wave Energy Shaping。** 在 logical top-k、模型语义和 exact replica set 固定后，联合决定 routed contribution 到等价副本的 assignment 与极短封批时间，利用实测的 expert `rows → latency, board-energy` 非线性曲线，以最小化 `J/completed-token`，同时对请求 deadline 使用保守风险约束。

它的研究价值来自一个窄而清楚的 MoE 矛盾：**分散 routed rows 可降低单队列负载，却会产生更多小 GEMM、weight/HBM 读取、launch、A2A message 与 fork-join 尾部；合并 rows 可降低单位能耗，却消耗请求 slack 并可能制造热点。** 这不是单独调 batch，也不是离线 placement。

但 E-Wave 仍须先通过三道门：

1. 自然 continuous-decode trace 中存在跨模型、完整分母下的 expert-wave 能量差异；
2. future-known exact/bounded Oracle 相对全部 SLO 合格的强简单 baseline 仍有至少 10% 净空间；
3. 因果策略不能被 fixed timeout、min-finish、LPLB-like、METRO-like 或 AMoE-like 策略捕获 ≥90% Oracle。

任一门失败，停止该 formulation；不得通过加入 FP8、DVFS、offload、RL 或放宽阈值救活。

---

## 1. 统一问题模型

### 1.1 请求、route 与完成 DAG

请求 (r) 在时刻 (t) 的状态写为：

\[
x_t = \{S_r(t), z_{r,l,t}, q_{e,g}(t), c_{g,h}(t), u_g(t), T_g(t), m_g(t)\}.
\]

- (S_r(t)=D_r-t-\widehat R_r(t))：相对 deadline (D_r) 的保守剩余 slack；
- (z_{r,l,t})：已观察到的 layer/token top-k expert、gate、source rank 与 token identity；
- (q_{e,g}(t))：expert (e) 在 GPU (g) 的 queued rows、ready time 与候选 seal time；
- (c_{g,h}(t))：链路队列、可用带宽、A2A message/bytes、overlap 状态；
- (u_g(t))：功率、SM/memory clock、利用率和 throttling reason；
- (T_g(t))：温度与热状态；
- (m_g(t))：KV、expert weights/replicas、EP workspace 和 activation 的显存状态。

请求完成时间不是平均负载，而是 precedence DAG 的 longest path：

\[
C_r = \max_{(l,k)\in\mathcal B_r} C_{r,l,k} + A_r + O_r,
\]

其中 (mathcal B_r) 是该请求各层 top-k expert contribution 的 fork-join 分支，(A_r) 是 attention/非 MoE 路径，(O_r) 是未被 overlap 隐藏的 runtime 开销。任何优化只有改变这条 longest path 或在保持完成集合时减少板卡能耗，才有系统价值。

### 1.2 动作空间

统一动作写为：

\[
a_t=(b_t, x_{i,g}, h_{e,g}, y_{e,g}, f_g, p_g, \pi_{e}, k_i, \rho_i, \alpha_g).
\]

- (b_t)：request/batch composition；
- (x_{i,g})：routed contribution (i) 到合法 exact replica (g) 的 assignment；
- (h_{e,g})：expert microqueue 的 bounded seal time；
- (y_{e,g})：expert replica/residency/activation state；
- (f_g,p_g)：GPU clock/power cap；
- (pi_e)：expert precision；
- (k_i)：token top-k；
- (ho_i)：local/remote/deferred/approximate execution path；
- (alpha_g)：GPU/SM/resource activation。

Top 1 的最小合法动作只有 (x_{i,g}) 与 (h_{e,g})；固定 top-k、expert identity、weights、precision、replica set 和外围 batch budget。这样可把质量风险与 placement 贡献先排除。

### 1.3 目标、SLO 与质量约束

主目标不用任意加权的“多目标分数”，而采用字典序约束：

\[
\min_{\pi} \quad
\frac{\mathbb E[ E_{board}+E_{host}+E_{net} ]}
     {\mathbb E[N_{completed\ tokens}]}
\]

subject to

\[
\Pr(C_r\le D_r\mid\mathcal I_t)\ge 1-\epsilon_r,
\]

\[
\Pr(\Delta Q_r\le Q_r\mid\mathcal I_t)\ge 1-\eta_r,
\]

\[
\mathrm{Goodput}(\pi)\ge (1-\delta)\mathrm{Goodput}(\pi_{perf}),
\]

以及 identity conservation、exact replica legality、显存、通信容量、queue stability 和完整 drain 约束。Top 1 固定 exact semantics，因此 (Delta Q_r=0)；quality 约束只保留为系统统一接口。

板卡能耗必须来自统一窗口：

\[
E_{board}=\sum_g\int_{t_0}^{t_1}P_g(t)dt,
\]

同时单独报告 idle/static、HBM、SM、communication-wait 与 controller/switching 分量；不能把 kernel 局部能耗和整段 idle 能耗重复相加。

### 1.4 可观测、预测与不确定性

| 类别 | 变量 | 处理方式 |
|---|---|---|
| 当下可观测 | 已产生 route/top-k、gate、request deadline、queue、rows、replica map、GPU power/clock/temp、当前链路 counters | 直接 telemetry；必须带 monotonic timestamp 与 identity |
| 可离线测量 | `(expert, rows, phase, dtype, clock) → latency/energy`、A2A size/skew/topology surface、状态切换成本 | 冻结 LUT；只在实测区间保守插值 |
| 需要预测 | 后续 output length、未来 route histogram、arrival、queue service、是否成为 critical rank | 输出分位数/置信集，不使用无校准点估计 |
| 不可消除不确定性 | router drift、突发到达、collective jitter、thermal drift、OS/runtime noise | chance/robust constraint；超出 envelope 立即 fallback |

### 1.5 时间尺度

| 时间尺度 | 允许动作 | 不宜混在同一个在线求解器中的动作 |
|---|---|---|
| 10–100 µs / kernel-wave | assignment、queue admission、是否立即 seal | placement、power cap、模型迁移 |
| 0.1–10 ms / layer-token | bounded coalescing、dispatch ordering、local/remote exact replica | GPU sleep、跨节点模型搬迁 |
| 10–500 ms / decode window | batch composition、rank criticality、clock/power policy（若硬件切换成本允许） | expert weight大迁移 |
| 0.5–10 s | replica activation、placement、GPU consolidation | per-token动作 |
| 分钟级 | capacity planning、电价/碳强度、静态 profile 更新 | request-level deadline控制 |

### 1.6 耦合、复杂度与分解

关键耦合包括：

- assignment 改变每副本 rows，从而同时改变 GEMM efficiency、weight reads、queue、remote bytes 和 join time；
- consolidation 降低 dynamic energy，却可能延长等待并把另一 GPU 变成 critical rank；
- replication 降低 queue risk，却增加 HBM 占用、activated-expert weight traffic 和静态功率；
- power cap/clock 对 compute-bound expert 有效，对 communication/memory-bound阶段可能几乎无效；
- approximation 同时改变 compute、communication、质量与未来 route/KV state，不能用局部 KL 直接闭合。

含 setup cost、可选 batching、异构 machine、deadline 和 fork-join 的离线问题至少包含 generalized assignment / batch scheduling 子问题；最新 [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880) 也证明了其资源分配模型的离线 NP-hard 性。完整在线问题还是部分可观测、非平稳、混合整数且非凸。

可运行的分解是：

1. **慢时标 planner** 冻结 replica/placement/power envelope；
2. **中时标 risk allocator** 给 request/layer 下推保守 slack budget；
3. **快时标 scheduler** 只在合法 exact replicas 内做 assignment+seal；
4. **安全过滤器** 拒绝超出 profile/置信集的动作；
5. **反馈器** 只更新 service residual 与 fallback threshold，不在线改科学门。

---

## 2. 十三个候选 idea 的逐项审计

以下“潜在上限”都是待测来源分析，不是跑数。

### C1. E-Wave：Risk-Bounded Expert-Wave Energy Shaping

1. **核心问题：** 如何在 exact routing 和 strict tail SLO 下，减少 routed rows 被拆成小 expert waves 造成的 energy/launch/HBM/communication 税？
2. **关键假设：** expert `rows→J, latency` 曲线存在可重复的 batching economy，且自然 trace 中有足够 fragmentation。
3. **MoE 特有性：** logical top-k 后同一 expert 的 rows 跨请求、跨 source rank 和等价 replicas 动态分裂；Dense FFN 没有这个 expert-keyed fork-join assignment。
4. **新机制：** identity-complete expert microqueues、energy-service surface、request-DAG slack ledger、risk filter 和 exact fallback。
5. **可控变量：** contribution→replica assignment、每 queue 的 seal time；首版不动 top-k、precision、placement、DVFS。
6. **目标/约束：** 最小化 `board J/completed token`，约束 request deadline chance、exact identity、完整 drain、相同完成集合和 queue stability。
7. **算法：** 10–100 µs 用 marginal batching credit 减去 remote/queue/risk price；1–10 ms seal；离线小窗口 MILP/DP 给 Oracle。
8. **难点：** power window 计费、cross-layer release、top-k closure、surface 插值、action overhead、简单策略可能已足够。
9. **本质差异：** 相对 PALS/Festina 不靠 power/batch 主旋钮；相对 LPLB/METRO 显式优化非线性能量-service 与 request deadline；相对 AMoE 只保留更窄的 exact replica wave action。
10. **优化来源/上限：** 少 launch、少重复 weight/HBM 读取、更高 GEMM arithmetic intensity、更少 activated replicas/messages；上限由完整路径中 exposed fragmentation energy 决定。
11. **5090 最小实验：** 两模型 natural continuous-decode route；真实 expert input；独立 event 的 row-grid energy/latency surface；virtual replica replay；exact Oracle 对 fixed-timeout/min-finish。
12. **8×A100 正式实验：** optimized EP backend、真实 replicas/source-destination、A2A timeline、request TPOT/P99 和 board energy 同窗。
13. **失败条件：** 两模型共同自然 cell 的 fragmentation energy share `<5%`；Oracle `<10%`；或简单策略捕获 `≥90%` Oracle。
14. **贡献潜力：** 建模型强、机制创新强、系统论文潜力高；但只在三门通过后成立。
15. **当前裁决：** **Top 1 / 只授权 Gate，不授权完整 controller。**

### C2. CriticalRank：Routed Fork-Join Critical-Rank Power Shaping

1. **核心问题：** EP ranks 在每个 decode window 对 request completion 的关键性不同，能否降低非关键 ranks 的功率而不把它们推成新 straggler？
2. **关键假设：** critical-rank 身份在至少一个硬件 actuation window 内有持久性，且非关键 ranks 有可兑现 slack。
3. **MoE 特有性：** route skew 和 top-k fork-join 产生 rank-specific criticality；Dense TP 的 rank work 更规则。
4. **新机制：** per-rank criticality posterior、slack-to-power safe set、criticality hysteresis 和 rank-wise fallback。
5. **变量：** per-GPU clock/power cap、可选 SM budget；不改 routing/quality。
6. **目标/约束：** 最小化 cluster board energy，约束每请求 TPOT/TTFT/P99 chance SLO 与 power-switch cost。
7. **算法：** 100–500 ms robust MPC；只在 criticality persistence LCB 大于 actuation delay 时降档。
8. **难点：** A100 power/clock切换粒度、rank criticality快速翻转、NVLink等待的能耗归因、PALS直接碰撞。
9. **本质差异：** PALS按 node/model配置与吞吐控制；C2必须证明 route-derived per-rank fork-join slack 对 tail-SLO 有独立 action value。
10. **上限：** 非关键 rank 的动态功率与等待能耗；若 ranks 长期同步或 power cap 不触发，上限接近零。
11. **5090：** 测 clock/power surface、切换延迟和 route-derived virtual-rank criticality persistence；不能验证真实 per-rank fork-join。
12. **8×A100：** per-rank clock/power、same-clock DAG profiling、route drift 和 P99。
13. **失败条件：** persistence `< actuation latency`；per-rank slack LCB `<5%`；相对 PALS 无 `≥5%` 净增益。
14. **潜力：** 建模型强、机制中等、系统潜力中高。
15. **裁决：** **Top 3；高硬件风险，不能作为单卡主线。**

### C3. ElasticExpert：Deadline-Priced Logical Expert Energy States

1. **核心问题：** 如何在 popularity drift 下决定 expert replicas/ranks 的 hot/warm/cold 状态，使静态/HBM/迁移能耗与 queue tail 共同最小？
2. **假设：** 存在秒级 popularity dwell time，且关闭 replica/GPU 的静态节省可覆盖迁移与再预热。
3. **MoE 特有性：** 稀疏 expert popularity 与 replica map 动态变化，Dense 模型没有独立 expert working set。
4. **机制：** 状态机、break-even timer、deadline-priced replica budget、shadow-copy warmup、emergency activation。
5. **变量：** replica count、placement、residency、rank active/sleep；不改模型质量。
6. **目标/约束：** 总能耗+switching cost 最小，约束 HBM、迁移带宽、P99、每 expert 至少一个合法副本。
7. **算法：** 0.5–10 s online primal-dual/competitive allocation；快层用固定安全 placement。
8. **难点：** GPU 不能按 expert 物理断电、共享 rank 静态功率不可分、迁移慢、预测漂移。
9. **差异：** 必须超越 Director/EaaS/MoE-Serving 的 placement/伸缩，证明“可测物理能源状态+deadline break-even”是新增机制，而非换名。
10. **上限：** 所有可关闭 GPU/HBM/static power，候选池中 gross 上限最高。
11. **5090：** 只能测 weight movement、warmup、residency与 idle-energy break-even；核心多 rank 结论不可测。
12. **8×A100：** drift trace、replica migration、rank shutdown/consolidation、tail recovery。
13. **失败条件：** dwell time短于 break-even；shared rank无法关闭；相对 Director/EaaS 无独立收益。
14. **潜力：** 系统型强但新颖性和复杂度风险极高。
15. **裁决：** **未进 Top 3；最高上限，不是最佳论文选择。**

### C4. PortSwitch：Critical-Port Exact Execution Switching

1. **核心问题：** route 已知后，何时应本地执行、远端 replica 执行、短暂聚合或启用更多 ranks，才能最小化 compute+network energy 且不伤 P99？
2. **假设：** 不同 route matrix/rows 下，各 exact plan 的 energy-delay 排名会反转。
3. **MoE 特有性：** token-to-expert fan-out 形成动态 A2A traffic matrix、incast 和 expert fork-join。
4. **机制：** critical-port DAG、exact plan catalog、port/rank shadow price、plan safety filter。
5. **变量：** exact replica destination、dispatch ordering、aggregation、active replica/rank 数。
6. **目标/约束：** `J/token` 最小；deadline、identity、link capacity、HBM 和 transition cost 约束。
7. **算法：** 每层先用 min-cost-flow/robust assignment，秒级更新 shadow price；小窗口 Oracle 用 MILP。
8. **难点：** FAST/SwiftEP 已强化 network baseline；network energy难测；动作容易与 placement/LPLB 重叠。
9. **差异：** 贡献必须是“critical-port energy switching”，而非一般 A2A scheduler 或 replica placement。
10. **上限：** 可同时减少 exposed A2A、incast wait、extra replicas 与 small waves；但依赖真实 EP denominator。
11. **5090：** 只能做 route-matrix census、plan enumeration 和通信模型敏感性；核心不可证。
12. **8×A100：** FAST/DeepEP/SwiftEP 强 backend、真实 network counters、same-clock deletion Oracle。
13. **失败条件：** optimized backend 下 exposed network energy/time `<10%`；简单 min-finish覆盖 Oracle；plan切换成本吃掉收益。
14. **潜力：** 建模型、机制和系统潜力高；资源依赖也最高。
15. **裁决：** **Top 2 / 只有拿到 8×A100 后才值得开 Gate。**

### C5. RouteRisk-MPC：Uncertainty-Calibrated Route-to-SLO Control

1. **问题：** 如何用前若干 token 的 router statistics 估计未来能耗/尾延迟风险，并在误差下仍保证 SLO？
2. **假设：** route entropy/popularity/drift 对 future expert pressure 有跨请求可校准的增量预测力。
3. **MoE 特有性：** future expert identities 和 traffic matrix 是动态隐变量。
4. **机制：** conformal/quantile route predictor、uncertainty set、risk budget与安全动作过滤器。
5. **变量：** 只允许选择 C1/C4 已被证明有效的动作；predictor 自身不是贡献。
6. **目标：** chance-constrained energy minimization，尾部 coverage 必须按 model/workload slice 校准。
7. **算法：** 10–100 ms robust MPC，预测失配时退回 worst-case plan。
8. **难点：** prediction 可能准确但 actionability 为零；drift coverage失效。
9. **差异：** 相对 Director 的 route prediction 和 JITServe 的不确定请求信息，必须证明 MoE route uncertainty 对 energy action 有独立闭环价值。
10. **上限：** 仅为已有 actuator 的可捕获 Oracle 空间，不能超过底层动作上限。
11. **5090：** causal prefix→future route/energy/latency预测；与 no-route、last-window、oracle比较。
12. **8×A100：** burst/drift 下 coverage、SLO、fallback频率与 overhead。
13. **失败：** route features相对 queue/load不增益；coverage under drift失败；fallback过多。
14. **潜力：** 算法/建模型强，独立系统机制弱。
15. **裁决：** 条件组件，不单独成论文。

### C6. RouteCohort：SLO-Safe Expert-Overlap Cohorting

1. **问题：** 能否按 route-set overlap 形成 request cohorts，提高 expert rows复用/合批效率而不增加尾延迟？
2. **假设：** prompt或早期 decode route-set 对短窗口 future overlap 有持续性。
3. **MoE 特有性：** batch 的 expert union 与每 expert rows取决于动态 routing；Dense batch无 expert-set几何。
4. **机制：** causal route signature、deadline bucket、overlap-aware batch builder。
5. **变量：** batch membership、hold time、dispatch order。
6. **目标：** J/token最小，约束每请求 slack、fairness和最大等待。
7. **算法：** locality-sensitive candidate generation + least-laxity matching，微秒至毫秒级。
8. **难点：** AMoE/ExpertPlex/普通 batch shaping碰撞；等待税可能主导。
9. **差异：** 只有“route overlap→measured energy service curve→request SLO”的闭环才算差异。
10. **上限：** small GEMM/activated-expert/launch energy；高负载时可能被自然 batching吃掉。
11. **5090：** 最容易用真实 route+service curve+causal replay证伪。
12. **8×A100：** 验证 A2A union、rank imbalance、P99。
13. **失败：** overlap预测不稳定；EDF/fixed timeout捕获 ≥90% Oracle；净收益 `<5%`。
14. **潜力：** 工程/建模型中等，独立系统论文潜力有限。
15. **裁决：** 不入 Top 3；**全池最易单卡证伪**。

### C7. QualityLedger：Token–Expert Quality-Risk Energy Ledger

1. **问题：** 哪些 token-expert contribution 的精确执行能耗高但任务级质量边际贡献低？
2. **假设：** 可在线构造对 request-level failure 有校准覆盖的贡献风险上界。
3. **MoE 特有性：** top-k expert contributions、gate weights和expert异质性提供细粒度近似单元。
4. **机制：** per-request quality debt ledger、verification probes、precision/skip safety filter。
5. **变量：** expert precision、top-k、skip/low-rank/fallback。
6. **目标：** 最小化能耗，约束 request/task failure probability而非平均 PPL。
7. **算法：** token级保守 knapsack/primal-dual；高风险token exact fallback。
8. **难点：** 局部误差不可加、autoregressive放大、评估昂贵、A100无原生FP8优势边界与5090不同。
9. **差异：** 必须超越 PagedWeight、Alloc-MoE、Mixture-of-Precisions；仅换约束名称不够。
10. **上限：** expert compute+communication中可近似部分，gross较高。
11. **5090：** held-out token/task质量、真实 kernel energy、policy完整轨迹。
12. **8×A100：** EP byte/compute节省、请求级质量尾部、fallback成本。
13. **失败：** safe accept rate `<5%`；uniform低精度支配；任务级质量尾部失守。
14. **潜力：** 机制强但科学风险极高。
15. **裁决：** 当前不做；仓库已有多条近似路线负/阻塞证据。

### C8. PagedState-MoE：Quality-Aware Expert Residency Portfolio

1. **问题：** 如何在 KV增长时选择 expert驻留、paging和precision，使能源/延迟/质量共同可控？
2. **假设：** expert popularity和KV pressure存在足够可预测性，paging节省可覆盖PCIe能耗。
3. **MoE 特有性：** 总 expert weights大而每 token稀疏访问。
4. **机制：** page-level expert state、KV-expert联合预算、prefetch/fallback。
5. **变量：** residency、precision、page quota、prefetch。
6. **目标：** J/request最小；HBM、SLO、quality约束。
7. **算法：** 秒级 receding-horizon cache/knapsack。
8. **难点：** weight movement大、cold miss P99、profile维度爆炸。
9. **差异：** 与 PagedWeight/FluxMoE 正面重叠，必须找到未覆盖动作才可继续。
10. **上限：** memory-pressure regime高，weights全驻留时为零。
11. **5090：** paging energy/latency与KV pressure可测。
12. **8×A100：** distributed cache、link争用和tail miss。
13. **失败：** 目标模型全驻留；paging净能耗为负；已有工作动作等价。
14. **潜力：** 工程型为主。
15. **裁决：** 淘汰独立主线。

### C9. SpecExpert：Risk-Limited Expert Prefetch/Speculation

1. **问题：** 能否在上游层执行时预取或预执行高概率 expert，并及时取消错误工作，从而以少量额外能耗换取可降档空间？
2. **假设：** future expert预测准确、搬运可重叠、错误工作可取消。
3. **MoE 特有性：** layer-wise dynamic expert working set。
4. **机制：** uncertainty-gated prefetch、cancel token、waste ledger。
5. **变量：** prefetch expert/bytes、speculation depth、cancel threshold。
6. **目标：** 净 board energy最小且SLO满足；所有错误工作计费。
7. **算法：** value-of-information threshold，逐层更新。
8. **难点：** speculation通常增能；GPU kernel很难中途取消；Director/predictive prefetch碰撞。
9. **差异：** 必须证明“speculation换取低功率运行”的闭环，不是普通 latency prefetch。
10. **上限：** offload/cold expert regime中 weight movement exposed share。
11. **5090：** prediction、prefetch overlap、waste energy。
12. **8×A100：** link contention、cancel、tail。
13. **失败：** net wasted energy≥saved energy；错误率/迁移税高；无可取消性。
14. **潜力：** 高风险机制型。
15. **裁决：** 不入 Top 3。

### C10. MissCut：Energy-Aware Inevitable-Miss Admission

1. **问题：** 过载时是否应停止为已必然 miss deadline 的 routed work消耗能量，把资源让给可完成请求？
2. **假设：** inevitable miss可保守识别且业务允许 reject/abort。
3. **MoE 特有性：** route-conditioned expert pressure提高可识别性，但问题本体也存在于Dense serving。
4. **机制：** lower-bound completion certificate、admit/defer/reject、energy waste ledger。
5. **变量：** admission、defer、abort、batch membership。
6. **目标：** 最大 SLO-goodput/J，约束公平性、drop budget。
7. **算法：** least-laxity + dual admission price，request到达/每token更新。
8. **难点：** 容易把节能伪装成少服务请求；DEPA/JITServe/Gimbal碰撞。
9. **差异：** 必须在相同 admitted/completed obligation 下比较，且 route pressure有独立价值。
10. **上限：** 仅在过载下的 doomed-work energy。
11. **5090：** causal replay可快速证伪。
12. **8×A100：** 真实 expert pressure/P99/fairness。
13. **失败：** no-route lower bound同样好；节能全来自drop；正常负载无空间。
14. **潜力：** 工程/建模型中等。
15. **裁决：** 只可作 overload policy，不作主贡献。

### C11. ElasticEP：Energy-Proportional Parallelism Switching

1. **问题：** 何时使用更多 GPU 反而因缩短请求和减少拥塞而总能耗更低，何时应收缩 EP/TP？
2. **假设：** plan切换成本可摊薄，且并行度的 energy-delay排序随负载/route变化。
3. **MoE 特有性：** EP A2A和expert capacity使扩展可能增通信、减热点。
4. **机制：** plan catalog、state transfer、break-even controller。
5. **变量：** EP/TP/DP degree、active GPU数、placement。
6. **目标：** J/request最小；SLO、memory、switch成本约束。
7. **算法：** 秒/分钟级 robust plan selection。
8. **难点：** reload/communicator重建；PALS与MoE-Serving覆盖；单卡几乎无核心证据。
9. **差异：** 必须有低开销在线 EP reshape，否则只是配置搜索。
10. **上限：** 静态GPU功率和拥塞，较高。
11. **5090：** 只能建surface，不能验证切换。
12. **8×A100：** 完整计划切换和energy break-even。
13. **失败：** switch time长于regime dwell；最佳静态计划已足够。
14. **潜力：** 工程复杂、创新偏弱。
15. **裁决：** 淘汰主线。

### C12. RouteCarbon：Fleet-Level MoE Power-Budget Arbitration

1. **问题：** 动态电力/碳预算下如何在多个MoE实例间分配power与请求？
2. **假设：** 实例的 route/energy elasticity不同且可预测。
3. **MoE 特有性：** 仅由MoE elasticity增强，本体是通用集群能源调度。
4. **机制：** elasticity bidding、cluster dual price、local controller接口。
5. **变量：** request routing、instance count、power budget。
6. **目标：** carbon/energy cost最小；SLO和rack cap约束。
7. **算法：** 分钟级 online primal-dual。
8. **难点：** DynamoLLM/POLCA/PALS/Festina空间拥挤；需大集群。
9. **差异：** 很难证明MoE必要性。
10. **上限：** fleet static power/时移，gross很高。
11. **5090：** 仅模拟。
12. **8×A100：** 仍不足以代表fleet。
13. **失败：** Dense方法直接迁移同样有效；route features无增益。
14. **潜力：** 集群论文而非当前硕士资源匹配。
15. **裁决：** 淘汰。

### C13. ReturnLite：Quality-Budgeted EP Return Compression

1. **问题：** 对低贡献 expert output 使用差分精度/压缩以降低 combine return energy。
2. **假设：** return在optimized EP中暴露且质量尾部可控。
3. **MoE 特有性：** top-k contribution与receiver return。
4. **机制：** contribution-level codec/LUT/fallback。
5. **变量：** BF16/FP8/INT4/drop、receiver lane。
6. **目标：** J/token最小；quality和SLO约束。
7. **算法：** token-level conservative LUT。
8. **难点：** codec/layout/metadata、uniform FP8强基线、质量不可加。
9. **差异：** 与既有混合精度/通信压缩重叠。
10. **上限：** 受 exposed return Amdahl ceiling 限制。
11. **5090：** 只可测codec与质量，不能测EP return。
12. **8×A100：** 先做return existence gate。
13. **失败：** `[Observed]` 仓库 fixed RankLane 在 `p_return≤20%` 冻结域相对uniform FP8最乐观E2E仅4.1667%，该 actuator已 NO-GO；不得换名复活。
14. **潜力：** 当前只剩 characterization。
15. **裁决：** 淘汰；只有真实8×A100证明新的return denominator超过reopen门，才能提出全新formulation。

---

## 3. 加权评分与严格筛选

分数为 1–10；总分按用户给定权重计算。`单卡`只评价“能否快速证伪必要条件”，不代表能单卡证明多卡系统结论。

| 排名 | ID | Idea | 真实性15 | MoE15 | 系统20 | 数学10 | 上限15 | 单卡10 | 8卡5 | 复杂度5 | 顶会5 | 加权总分 | 主要扣分 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | C1 | E-Wave | 9 | 10 | 9 | 8 | 8 | 9 | 9 | 7 | 9 | **8.80** | AMoE/LPLB/METRO/ExpertPlex可能吃掉 residual；能量会计难 |
| 2 | C4 | PortSwitch | 9 | 10 | 8 | 8 | 9 | 3 | 10 | 4 | 8 | **8.00** | 核心不可单卡测；FAST/SwiftEP/placement直接碰撞 |
| 3 | C2 | CriticalRank | 8 | 9 | 8 | 8 | 7 | 6 | 9 | 7 | 8 | **7.80** | PALS碰撞；power actuation可能慢于route criticality |
| 4 | C3 | ElasticExpert | 9 | 9 | 6 | 9 | 10 | 4 | 9 | 3 | 5 | **7.55** | Director/EaaS/MoE-Serving已占主问题；实现失控 |
| 5 | C5 | RouteRisk-MPC | 8 | 8 | 6 | 9 | 7 | 8 | 8 | 6 | 7 | **7.40** | predictor不是机制；JITServe/Director碰撞 |
| 6 | C10 | MissCut | 9 | 7 | 6 | 8 | 7 | 9 | 8 | 7 | 6 | **7.40** | 节能可能全来自少服务请求；MoE特有性不足 |
| 7 | C7 | QualityLedger | 8 | 9 | 5 | 8 | 9 | 7 | 8 | 4 | 6 | **7.30** | PagedWeight/Alloc-MoE等强先验；质量尾证明困难 |
| 8 | C8 | PagedState | 8 | 9 | 5 | 7 | 9 | 8 | 8 | 5 | 5 | **7.30** | FluxMoE/PagedWeight正面覆盖 |
| 9 | C9 | SpecExpert | 7 | 9 | 7 | 7 | 7 | 7 | 8 | 5 | 6 | **7.20** | speculation易增能；取消机制不现实 |
| 10 | C6 | RouteCohort | 8 | 9 | 5 | 7 | 7 | 9 | 8 | 7 | 5 | **7.20** | AMoE/普通batch shaping覆盖，独立贡献偏薄 |
| 11 | C11 | ElasticEP | 9 | 8 | 5 | 7 | 9 | 4 | 9 | 5 | 5 | **6.95** | PALS/理论工作覆盖；切换成本高 |
| 12 | C12 | RouteCarbon | 9 | 4 | 6 | 8 | 9 | 6 | 8 | 5 | 6 | **6.85** | 本质是Dense/fleet能源调度；资源不匹配 |
| 13 | C13 | ReturnLite | 8 | 10 | 4 | 7 | 4 | 8 | 10 | 6 | 4 | **6.60** | 冻结域已有4.1667%上界NO-GO；核心分母需8卡 |

注：C5 与 C10 同分；前者因 predictor 没有独立 actuator，后者因节能可能来自减少服务义务，均未进 Top 3。严格 Top 3 是 C1、C4、C2。

---

## 4. Top 3 的 reviewer-style 攻击与补强

### 4.1 Top 1：E-Wave

| 攻击 | Reviewer 最强表述 | 必须的补强；做不到即降级/停止 |
|---|---|---|
| 只是已有技术组合 | “这是 AMoE 的 microqueue + LPLB 的 replica dispatch + 一个 energy objective。” | 固定 replica set；给出 action-level 对照；证明 nonlinear 'rows→J,T' 与 request fork-join deadline 使 LPLB/METRO/AMoE 做出可重复的不同决策；源码级 collision audit。 |
| 简单 baseline 足够 | “fixed timeout 或 min-finish 已经捕获全部收益。” | 预注册 'oracle capture <90%' 硬门；若任一强简单策略捕获 ≥90%，删除复杂 controller，只保留测量论文或停止。 |
| 收益来自降低性能 | “J/token 降低只是因为等更久、吞吐下降。” | 相同 arrival、相同 admitted/completed identity、完整 drain；SLO-qualified baseline 才可比；同时报告 goodput、TPOT/P99、J/request。 |
| 收益来自质量变化 | “可能因数值/路由漂移减少工作。” | 固定 logical top-k、weights、dtype 与每 contribution exactly-once；输出/数值门；首版 quality budget 为 0。 |
| 需要不现实硬件 | “只有大量 exact replicas 才有 action space。” | 先报告 realistic HBM 下 replica census；无 replicas 时 idea 明确不适用；不得用 virtual replicas 宣称系统成立。 |
| profiling/控制开销抵消收益 | “微秒调度器比小 GEMM 还贵。” | 热路径只做定长候选表查询；GPU/CPU decision P99 单独计费；将 controller overhead 纳入同一 energy/time window。 |
| workload drift | “profile 和 route 一漂移就误合批。” | 上界插值、out-of-envelope fallback、在线 residual 只校正预测不改变门；分布漂移作为正式 cell。 |
| 普适性不足 | “仅在一个 top-k=16 小模型成立。” | 至少 top-k/专家数不同的两模型共同自然 cell 过门；synthetic skew 只作解释，不作主证据。 |
| serving 稳定性 | “queue feedback 会震荡和饿死冷 expert。” | oldest-age hard cap、least-laxity safety lane、hysteresis、queue Lyapunov drift监控和公平性 P99。 |
| 一票否决实验 | Oracle 相对 best SLO-qualified baseline 的 J/completed-token 改善 95% LCB '<10%'，或所有共同自然 cell fragmentation share '<5%'。 | 直接 'NO_GO_EWAVE'，不加 actuator、不改模型池、不放宽门。 |

### 4.2 Top 2：PortSwitch

| 攻击 | Reviewer 最强表述 | 必须补强 |
|---|---|---|
| 组合已有系统 | “这是 FAST/SwiftEP + Director/LPLB 的 energy-aware cost function。” | 用 fixed optimized backend；动作仅在合法 exact plans 间切换；证明 critical-port shadow price 相对 min-finish/bytes/rank-load 有独立 Oracle residual。 |
| 更简单 baseline | “选择最短预计完成时间即可。” | 强制 min-finish、min-bytes、min-active-ranks、LPLB、METRO、FAST scheduling；简单法捕获 ≥90% Oracle 即停止。 |
| 节能来自少用网络而伤性能 | “减少 GPU/远程执行自然更慢。” | 相同 SLO/completion obligation；绘制 energy–goodput Pareto，但主结论只在相同 SLO attainment slice 比较。 |
| 不现实硬件 | “8×A100 单机没有足够 topology 差异；跨节点又不在资源范围。” | 先冻结 8×A100 是 PCIe 还是 SXM/NVSwitch；若只有均匀单节点拓扑且 plan 排名不反转，停止。 |
| 测量不可归因 | “NVML 不能给 NVLink/网络能耗。” | 主口径用整机/板卡总能耗；network counters只作解释；用 paired plan A/B 与 deletion Oracle，不把估算 NIC energy当主结果。 |
| drift/稳定性 | “route matrix 每层都变，shadow price滞后。” | 快动作使用当前已知 route；慢 price 只估资源拥塞；突变时回 min-finish。 |
| 普适性 | “只对某个 topology/backend。” | 明确 scope；至少两个 MoE、两种负载、两个 backend mode；不声称跨互联普适。 |
| 收益来自质量变化 | “可能通过 reroute/drop 改了语义。” | 固定 logical expert identity、top-k、weights、dtype 与 exactly-once；只在 exact replicas/plans 间切换。 |
| profiling/控制开销 | “critical-port 建模和 plan 枚举本身吞掉收益。” | 重 profiler仅离线；在线候选集预裁剪；plan decision P99、telemetry能耗和额外同步纳入净收益。 |
| serving 稳定性 | “计划切换把拥塞从一个端口搬到另一个端口。” | 加入 max-step、hysteresis、new-bottleneck detector 与 min-finish fail-high；报告切换率和新热点率。 |
| 一票否决 | optimized backend 下 A2A/critical-port exposed share 95% LCB '<10%'，或 plan switching Oracle '<10%'。 | 'NO_GO_PORTSWITCH'；不回退到弱 collective 制造收益。 |

### 4.3 Top 3：CriticalRank

| 攻击 | Reviewer 最强表述 | 必须补强 |
|---|---|---|
| PALS 已经做了 | “PALS 已联合 per-GPU power cap 与 batch；你只是多了 route feature。” | 直接复现/实现 PALS-like baseline；只有 route-derived critical-rank posterior 在 request P99 下产生独立动作和 ≥5% 净增益才成立。 |
| 简单均匀降频足够 | “所有 ranks 同频调低即可。” | uniform cap/clock、utilization-only、queue-only、oracle criticality 全覆盖。 |
| 收益伤性能 | “非关键 rank 被降频后只是把等待转移。” | longest-path DAG 证明该 rank 仍非关键；报告 critical-rank migration、new-straggler rate 和 P99。 |
| 收益来自质量变化 | “不同功率让数值或执行路径变化。” | 固定 route、weights、dtype、kernel和完成 identity；通过 numerical equivalence gate，任何 route/drop差异使 trial无效。 |
| actuation 不现实 | “route变化在毫秒级，NVML切换在百毫秒级。” | 先测真实 actuation latency；要求 criticality dwell-time 95% LCB 大于 '2×switch latency'，否则立即停止。 |
| power cap 不生效 | “decode memory-bound，cap 根本不触发。” | 同时测 locked SM clock 与 power cap；如果两者对目标阶段都无有效 energy-delay frontier，停止。不能借用别的 GPU/模型结论。 |
| profiling overhead | “Nsight/CUPTI/route instrumentation太贵。” | 正式运行只用低开销 counters；重 profiler只离线；telemetry overhead纳入 ablation。 |
| workload变化 | “热门 rank变化会导致振荡。” | dwell-time filter、hysteresis、max step和fail-high fallback。 |
| serving 稳定性 | “控制器在真实 continuous batching 中来回切换。” | 报告控制动作频率、settling time、fallback率、new-straggler率；超冻结频率门自动禁用。 |
| 普适性 | “只有高 skew 时有效。” | natural skew主结果；skew sweep只解释适用域；把 scope 写成条件机制。 |
| 一票否决 | criticality persistence 不超过 actuation latency，或相对 PALS/uniform clock净节能 '<5%'。 | 'NO_GO_CRITICALRANK'。 |

---

## 5. 第一名 E-Wave 的完整研究蓝图

### 5.1 论文式问题陈述（约 250 字）

在线 MoE 推理把每个 token 动态路由到少量专家，并在多副本、多 GPU 上形成细碎、随请求变化的 expert work。现有负载均衡通常最小化 token/rank 数或最早完成时间，异步系统则扩大 expert queue 以提高吞吐；它们没有把“小 wave 导致的重复权重读取、kernel 启动、通信 setup 与板卡能耗”同请求级 fork-join deadline 放进一个闭合模型。简单合并 routed rows 虽能提高单位能效，却会消耗 SLO slack、制造热点并放大 P99。E-Wave 固定模型语义和 replica placement，只在 routing 后联合选择 exact replica 与 bounded seal time，使用实测能量–服务曲线和保守的剩余路径风险估计，在不降低质量、不过度牺牲 goodput 的条件下减少每个完成 token 的总能耗。

### 5.2 可写入 Introduction 的三条贡献

1. **问题与测量贡献。** 首次把 MoE routed-work fragmentation 定义为请求 completion DAG 上的能量–尾延迟问题，并提供 identity-complete 的方法，将每个 expert wave 的 rows、replica、通信、board energy 与 request TPOT/P99 闭合，而非用平均 batch 或逻辑 bytes 代替系统收益。
2. **机制与算法贡献。** 提出 E-Wave：在 fixed exact replicas 内联合进行 contribution assignment 与 deadline-bounded sealing；以实测非线性 energy-service curve 计算 marginal batching credit，并用 uncertainty-calibrated safety filter 保证预测错误时回退到 SLO-safe execution。
3. **系统与经验贡献。** 在单卡先验 Gate 和 8×A100 optimized EP serving 上，系统比较 LPLB/METRO/AMoE/PALS-like 等强 baseline，刻画何时 consolidation、spreading 或增加 replicas 更节能，并公开适用域、失败域和全部控制开销。

这三条只有在 formal Gate 通过后才能用肯定语气；当前均为 '[Hypothesis/Planned]'。

### 5.3 系统架构

~~~text
offline profiler ──> energy-service catalog ─────────────┐
                                                        │
request arrival ──> SLO ledger ──> route/DAG tracker ───┼─> safe action filter
                                     │                  │        │
GPU/network telemetry ─> residual estimator ────────────┘        v
                                                        wave scheduler
                                                              │
                         exact-replica dispatcher <── action ─┤
                                                              v
                                            grouped expert runtime/A2A
                                                              │
                              completion + board energy ──────┘
                                           │
                                           v
                               feedback / fail-high fallback
~~~

| 模块 | 输入 | 输出 | 调用频率 |
|---|---|---|---|
| Offline profiler | model/revision、expert inputs、rows、phase、dtype、clock、backend | 保守 'T_{e,g}(n)'、'E_{e,g}(n)'、A2A/switch surface和有效范围 | 每硬件×模型×revision一次；环境变化重建 |
| Route/DAG tracker | request/token/layer/top-k/slot/source identity、runtime completion | identity-complete contribution DAG、ready events、join state | 每 route event / completion |
| SLO ledger | TTFT/TPOT/request deadline、已用预算、remaining-path quantile | local safe deadline、risk budget | 请求到达与每 token/layer |
| Telemetry | queue、rows、GPU power/clock/temp、link counters | timestamped state summary | 50–500 µs快状态；5–20 ms power样本；重 profiler不在热路径 |
| Risk estimator | state、surface residual、route drift | completion quantile、out-of-envelope flag | 每 wave decision |
| Wave scheduler | ready contributions、合法 replicas、marginal J/T、risk price | assignment、seal/hold/fallback | 10–100 µs 目标上限 |
| Runtime actuator | action、replica map、A2A backend | exact dispatch、grouped GEMM、combine | 每 wave |
| Fallback | profile miss、risk excess、telemetry stale、overhead excess | immediate min-finish/原生 runtime path | 事件触发 |
| Feedback | predicted/actual completion、energy、violations | residual校正、报警；不在线改硬门 | 每 wave汇总，每1–10 s更新 |

### 5.4 数学模型

对 routed contribution \(i\)：

- \(e_i\)：logical expert；\(r_i\)：request；\(a_i\)：ready time；
- \(\mathcal R_{e_i}\)：合法 exact replicas；
- \(x_{i,g}\in\{0,1\}\)：是否分给 replica \(g\)；
- \(h_{e,g}\ge0\)：microqueue seal time；
- \(n_{e,g,w}=\sum_{i\in w:e_i=e}x_{i,g}\)：wave rows；
- \(T_{e,g}(n,\xi)\)、\(E_{e,g}(n,\xi)\)：状态 \(\xi\) 下的保守 latency/board-energy surface；
- \(C_{src_i,g}(bytes_i,\xi)\)：dispatch/combine 成本。

合法性：

\[
\sum_{g\in\mathcal R_{e_i}}x_{i,g}=1,\qquad
x_{i,g}=0\ \text{if}\ g\notin\mathcal R_{e_i}.
\]

wave 完成上界：

\[
\widehat C_{e,g,w}^{1-\alpha}
=\max_{i\in w}a_i+h_{e,g}
+Q_{e,g}^{1-\alpha}
+C_{src,g}^{1-\alpha}
+T_{e,g}^{1-\alpha}(n_{e,g,w}).
\]

请求风险：

\[
R_r(a_t)=
\Pr\!\left[
\max_{i\in\mathcal B_r}\widehat C_i
+\widehat R_r^{1-\alpha}>D_r
\mid\mathcal I_t
\right].
\]

能量：

\[
\widehat E(a_t)=
\sum_{e,g,w}E_{e,g}(n_{e,g,w})
+E_{A2A}(a_t)+E_{wait}(a_t)+E_{ctrl}(a_t)+E_{switch}(a_t).
\]

在线动作采用受约束的 marginal cost：

\[
a_t^*=\arg\min_{a\in\mathcal A_{safe}}
\left[
\Delta\widehat E(a)+\lambda_t\Delta R(a)+\mu_t\Delta B_{remote}(a)
\right],
\]

其中

\[
\mathcal A_{safe}=
\{a:R_r(a)\le\epsilon_r,\ h_{e,g}\le S_r^{safe},\
a\ \text{identity-legal}\}.
\]

若 \(\mathcal A_{safe}=\varnothing\)、输入超出 profile envelope、telemetry 过期或预测 residual 超门，立即执行 fail-high baseline。离线小窗口 Oracle 用 MILP/DP；在线不运行通用 MILP，也不使用黑盒 RL。

### 5.5 核心算法伪代码

~~~text
on REQUEST_ARRIVAL(r, deadline, quality_budget):
    ledger.register(r, deadline, quality_budget)
    admit using frozen outer serving policy

on ROUTE_READY(r, token, layer, topk, gates, source_rank):
    jobs = dag_tracker.materialize_exact_contributions(...)
    assert identity_closure(jobs, topk)
    ledger.update_remaining_path_quantile(r)

    for job i in jobs:
        candidates = []
        for exact replica g in replica_map[expert(i)]:
            state = telemetry.snapshot(g)
            if stale(state) or outside_profile(i, g, state):
                continue

            for hold h in legal_hold_grid(i, g, ledger.safe_slack(r)):
                wave = queue[g, expert(i)] + {i}
                delta_energy = marginal_board_energy(wave, h, state)
                completion_q = conservative_completion_quantile(wave, h, state)
                request_risk = dag_tracker.project_join_risk(r, i, completion_q)
                if request_risk <= epsilon[r]:
                    score = delta_energy + lambda[r] * request_risk
                            + mu * marginal_remote_cost(i, g)
                    candidates.append((score, g, h))

        if candidates is empty:
            action = FALLBACK_IMMEDIATE_MIN_FINISH(i)
        else:
            action = argmin(candidates)
        queue.commit(i, action)

on QUEUE_EVENT(expert e, replica g):
    if oldest_deadline_risk_exceeds_limit()
       or seal_timer_expired()
       or no_positive_batching_credit():
        wave = queue.seal(e, g)
        runtime.dispatch_exact(wave)

on WAVE_COMPLETE(wave, timestamps, energy_window, outputs):
    assert exact_identity_and_numerical_gate(wave, outputs)
    dag_tracker.release_dependents(wave)
    residual.update(predicted_vs_actual)
    ledger.charge_observed_energy(wave, energy_window)
    if residual_or_violation_exceeds_frozen_limit():
        enable_fail_high_mode(for=frozen_cooldown_window)
~~~

### 5.6 实验矩阵

| 维度 | 阶段 A：RTX 5090 | 阶段 B：8×A100 |
|---|---|---|
| 模型 | 仓库已接入的 OLMoE-1B-7B 与 LLM-jp MoE；具体 revision/权重 hash冻结 | OLMoE + Mixtral-8×7B 或 Qwen/DeepSeek serving-scale MoE；至少两种 top-k/专家数结构 |
| serving | native KV-cache continuous decode；不得用一次 'model(**inputs)' 冒充 | vLLM/SGLang + optimized EP（DeepEP/可用强 backend），固定 revision |
| workload | natural prompts为主；synthetic route只作机制解释 | Azure/ShareGPT/ServeGen类真实 arrival/length trace + 合成可控 trace |
| arrival | Poisson steady、MMPP burst、replayed diurnal片段 | 同三类；30%/70% load 与 SLO knee |
| prompt/output | prompt 128/512/2048；output 32/128/512，按模型容量裁剪并预注册 | 短/中/长 prompt × decode-heavy/mixed output |
| route skew | natural per-layer/expert histogram；Zipf/synthetic仅次要 | natural drift、domain shift、热点注入 |
| SLO | calibration baseline P99 的冻结倍率，例如 1.1/1.3/1.5；正式前锁定 | TTFT 与 TPOT 双 SLO，另有 request deadline |
| quality | exact semantics，预算 0；输出/numerical equivalence gate | exact semantics，预算 0 |
| power | board-energy counter优先；统一窗口；clock/temp/thermal gate | 每卡与整机能耗；IPMI若可用作交叉校验 |
| 重复/统计 | AB/BA随机次序；document→seed两层 paired bootstrap；95% CI | request/block paired bootstrap；跨日复现 |

模型选择依据：[OLMoE](https://arxiv.org/abs/2409.02060) 提供完全开放的 1B-active/7B-total 模型；[Mixtral](https://arxiv.org/abs/2401.04088) 提供 top-2、8-expert 的不同结构。RTX 5090 与 A100 的 architecture/precision/energy surface 必须分别测量，不能迁移数值。

### 5.7 Baseline

必须同时覆盖：

1. static max-performance：最高安全 clock/power、runtime默认batch/dispatch；
2. static min-energy：calibration中 SLO合格且 J/token最低的静态配置；
3. power-only / clock-only、batch-only、PALS-like 'power+batch'；
4. 原生 hash/even split/least-load；
5. deterministic random legal replica；
6. LPLB-like token balance；
7. METRO-like activated-expert minimization；
8. min-predicted-finish；
9. fixed rows、best fixed timeout、EDF/least-laxity；
10. AMoE-like per-expert re-batching；
11. frozen expert placement 与 placement-only adaptation；
12. future-known exact/bounded Oracle；
13. E-Wave 去掉 route、slack、uncertainty、energy surface、feedback 的消融。

所有 baseline 参数只在 calibration split 选择；formal sealed split 不重新调参。若某 baseline 不适用于当前 backend，必须解释 action mismatch，不能静默删掉。

### 5.8 指标与主判定

主指标：

- 'board Joules / completed token'；
- 'board Joules / completed request'；
- SLO-qualified goodput 与 throughput/W；
- TTFT、TPOT、request latency 的 P50/P95/P99；
- SLO violation rate；
- 完成 request/token identity集合。

解释指标：

- GPU power/clock/temp/utilization、SM active、HBM throughput；
- A2A bytes/message count/link counters、idle/wait energy；
- expert rows/wave、activated replicas、queue age、fork-join critical path；
- controller decision P50/P99、telemetry和switch cost；
- latency/energy prediction误差与分位数 coverage；
- fallback率、new-straggler率、oscillation/fairness。

正式 GO 条件建议冻结为：

1. 两模型共同自然主 cell 中，Oracle 相对**每个** SLO-qualified强 baseline 的 J/completed-token 改善 95% LCB '≥10%'；
2. causal E-Wave 在所有主 cell 为正，至少两个 common cell 95% LCB '≥5%'，目标净改善 '≥10%'；
3. goodput 不低于 performance baseline 的冻结容忍度，SLO violation不恶化；
4. best simple baseline 捕获 '<90%' Oracle；
5. controller+telemetry能耗/时延 '<20%' gross saving；
6. identity、数值、环境、统计任一门不闭合则结果 'INVALID'，不是正/负结论。

### 5.9 消融

| 问题 | 消融 | 必须观察的量 |
|---|---|---|
| route 是否必要 | route features→仅 rows/queue | Oracle capture、J/token、P99 |
| SLO slack 是否必要 | 固定 timeout / 无 request ledger | violation、等待、goodput |
| uncertainty 是否必要 | point estimate vs quantile/robust | drift cell coverage、fallback、P99 |
| expert-level actuator 是否必要 | 只调外围 batch/power | 相对 PALS-like residual |
| energy model 是否准确 | latency-only/min-finish | energy prediction error与动作差异 |
| feedback 是否防震荡 | frozen model/no hysteresis | switch rate、new-straggler、P99 |
| assignment贡献 | 固定 default assignment，只seal | J/token 与 wave rows |
| sealing贡献 | immediate seal，只assignment | J/token 与 queue age |
| remote cost贡献 | 忽略 src/dst | A2A bytes、critical path、P99 |

### 5.10 阶段 A：RTX 5090 快速证伪

#### 代码模块（只实现 Gate，不先写完整在线 controller）

建议新目录：'docs/ideas/energy_slo/ewave/experiments/'。在 Gate 通过前，不加入 current mainline。

| 模块 | 作用 | 可复用/必须重写 |
|---|---|---|
| 'capture_continuous_decode_routes.py' | native continuous decode 的 request/token/layer/top-k/slot/row identity | 参考 BCRD capture schema；现有一次 full-forward producer不能复用为 formal |
| 'capture_expert_events.py' | 保存独立 input event、row→token/slot映射与 tensor hash | 不能只保存 pooled tensor |
| 'profile_energy_service_surface.py' | 同一窗口测 rows→latency/board energy；输出 raw power/counter/temp/clock | JouleQueue v1 meter与surface runner未签字，按 review要求重写 |
| 'build_dependency_jobs.py' | route→job、cross-layer policy-dependent release、top-k closure | 统一 string identity，禁止手写布尔自证 |
| 'solve_ewave_oracle.py' | 小窗口 exact MILP/DP；大窗口给可验证上下界 | 必须支持至少 top-k=16×两层 fixture |
| 'run_common_gate.py' | current/random/fixed/min-finish/LPLB/METRO/AMoE-like 同口径 replay | 参数由 calibration冻结 |
| 'analyze_paired_results.py' | document→seed paired bootstrap和机器可读 decision | completed identity必须相同 |

#### 实验顺序

1. **A0 能力门：** native continuous decode、identity closure、正式 energy meter、环境/thermal门、完整 denominator；缺一停止。
2. **A1 surface：** 两模型、自然 event、rows grid；验证 batching energy与latency是否有稳定非线性。
3. **A2 census：** 不运行 policy，统计自然 trace 的 rows/wave、fragmentation、actionable share和完整路径 exposed fraction。
4. **A3 Oracle：** exact/bounded future-known assignment+seal，相对全部强简单 baseline。
5. **A4 causal prototype：** 只有 A1–A3 通过才实现最小 E-Wave；不开 FP8/DVFS/offload。

#### 预期图表

1. 'rows → latency/row, J/row'，按 model/layer/expert/event 展示 CI；
2. natural 'rows/wave' 与 fragmentation energy-mass CDF；
3. Oracle/baseline 的 'J/token–P99' Pareto；
4. Oracle capture ratio 与 simple-baseline residual；
5. predicted vs observed completion/energy calibration；
6. action breakdown：assignment、hold、fallback、remote proxy；
7. Amdahl waterfall：local wave gross saving → controller/wait/denominator → net。

#### 阶段 A 停止条件

- 只有 synthetic skew或单模型存在 headroom；
- 两模型共同自然 cells 的 exposed fragmentation energy share '<5%'；
- row surface近线性，coalesced energy 95% CI 不优于 separate；
- Oracle相对best SLO-qualified baseline '<10%'；
- fixed timeout/min-finish/LPLB/METRO/AMoE-like捕获 '≥90%' Oracle；
- controller允许开销预算已大于 gross saving；
- identity、energy window、thermal或cross-layer DAG无法闭合。

阶段 A 通过只授予 '8xA100_CANDIDATE'，不能写“已降低 MoE serving 能耗”。

### 5.11 阶段 B：8×A100 正式系统实验

1. 冻结 A100 型号（40/80GB、PCIe/SXM）、NVLink/NVSwitch拓扑、driver/CUDA/NCCL、vLLM/SGLang/DeepEP revision、clock/power policy。
2. 在 fixed placement 与 realistic replica budget 下建立真实 EP contribution identity、dispatch/combine timeline、request completion DAG。
3. 用 same-clock Nsight/CUPTI 低频 profiling 确认 expert wave、A2A、wait 位于 exposed path；正式能耗运行关闭重 profiler。
4. 实现 backend-native exact replica assignment 与 bounded queue seal；不通过额外 host copies、同步或弱 collective获得收益。
5. 运行两模型×load×arrival×SLO×route drift 矩阵；按 request/block paired bootstrap。
6. 分别报告“使用更多 replicas/GPUs更节能”和“consolidation更节能”的区域边界，包含静态/HBM/communication/switching成本。
7. 与 PALS-like、LPLB、METRO、AMoE-like、FAST/DeepEP backend 和 Oracle做同环境比较。
8. 只有跨模型自然 cell、完整能耗分母、TPOT/P99 和简单 baseline residual 同时通过，才允许把 E-Wave 定义为论文主机制。

---

## 6. 接下来一周的唯一合理工作

本周不实现完整 controller，也不并行推进 Top 2/Top 3。

### Day 1：冻结问题与直接 collision audit

- 把 E-Wave 的 action 固定为 'exact contribution→replica assignment + bounded seal'；
- 逐动作阅读 PALS、AMoE、LPLB、METRO、ExpertPlex、Gimbal、Director 的全文/代码；
- 输出 action/state/objective/assumption 对照表；若已有同构工作，立即停止或进一步收缩。

### Day 2：统一 schema 与 accounting boundary

- 对齐 BCRD route identity、JouleQueue energy accounting 与 current common Gate；
- 冻结 request/token/layer/top-k/slot/replica/source identity；
- 冻结 't0/t1'、completed-token denominator、idle/static accounting、完整 drain和thermal gate。

### Day 3：完成 CPU/fake-backend 反例测试

- string identity round-trip、missing/duplicate sibling hard fail；
- top-k=16×两层 policy-dependent release；
- 固定测量 envelope在不同 repeats 下不得制造 energy saving；
- 相同 count 但不同 completed identity 必须 fail；
- telemetry stale/out-of-profile 必须 fallback。

### Day 4：native continuous-decode producer dry-run

- 证明不是一次 'model(**inputs)' full forward；
- 生成小型 natural trace、source/data manifest和完整 route closure报告；
- 不打开 sealed formal数据。

### Day 5：energy-service meter calibration

- 在 5090 测 board-energy counter availability、采样间隔、clock/temp/thermal drift、AB/BA顺序效应；
- 只用 fake/smoke inputs验证估计量，不产科学数值。

### Day 6：Oracle 与 baseline design review

- 证明 exact Oracle 对 toy/小窗口的最优性；
- 明确大窗口上下界；
- 锁定 fixed timeout、min-finish、LPLB-like、METRO-like、AMoE-like baseline 与 calibration过程。

### Day 7：独立签字

- 逐项关闭 P0；生成 source/data/protocol hashes；
- 只有 Gate 0 全绿才批准 A1 surface GPU运行；否则输出 'BLOCKED_MISSING_FORMAL_EVIDENCE'。

一周交付物应是：一份冻结协议、一份 code-review attestation、一套 CPU/fake 反例测试、一个 native route smoke 和一个 'GO/NO_GO_TO_SURFACE' 决策。不是新的 controller，也不是论文结果。

---

## 7. 进入 GPU 实验前的审查清单

### 7.1 代码审查硬门

- [ ] route producer 是 native continuous KV decode，不是 one-shot full forward；
- [ ] request/token/layer/top-k/slot/source/replica identity 全程可逆且 closure 完整；
- [ ] 后层 release 随前层 policy completion变化，不用静态 arrival伪造 cross-layer DAG；
- [ ] energy counter、power samples、CUDA events、workload边界和 denominator有统一时钟/窗口；
- [ ] paired arms 使用同一 logical repeat denominator；固定 envelope 不改变差值符号；
- [ ] 原始 counter、power、clock、temp、throttle、UUID、driver、model/revision/hash全部落盘；
- [ ] 独立统计单位是 input event/request/block，不把 inner repeats当独立样本；
- [ ] surface coverage按自然 route energy mass计算，不对 selected experts等权冒充全模型；
- [ ] 未测 rows只用预注册 upper envelope；越界立即 fallback；
- [ ] Oracle支持真实 top-k 和至少两层 closure，或给可验证上下界；
- [ ] baseline与candidate约束、完成集合、max-age语义完全相同；
- [ ] calibration、formal、sealed数据物理/逻辑隔离；
- [ ] capability来自外部 artifact/hash验证，不是 config布尔值自证；
- [ ] telemetry/controller后台异常会向主进程 fail closed；
- [ ] 单卡 virtual replicas 明确标为 replay，不进入 EP/NCCL结论。

### 7.2 实验设计硬门

- [ ] 主假设、主指标、自然 workload cells、模型 revision、seeds、SLO倍率和停止阈值预注册；
- [ ] 两个结构不同的公开 MoE；只在一个模型成立不得写普适机制；
- [ ] 完整路径 denominator独立测量，不从 proposed saving反推；
- [ ] 最强 optimized backend与强简单 baseline齐全；
- [ ] energy比较使用相同 arrival、admission obligation、completed identity和完整 drain；
- [ ] quality保持 exact，或若未来引入近似，需另立 request/task-level质量协议；
- [ ] AB/BA随机顺序、warmup、cooldown、thermal、clock和跨日复现规则已冻结；
- [ ] document→seed/request-block层级 paired bootstrap与95% CI已实现；
- [ ] synthetic skew只用于解释，不替代 natural主结果；
- [ ] 单卡通过只定义为必要条件，不写多卡 serving/TPOT/P99结论；
- [ ] 失败后不换阈值、pool模型、改指标、删baseline或增加新 actuator。

---

## 8. 最终七问

1. **Energy–SLO-aware MoE 是否值得作为硕士论文主方向？**  
   **现在不值得直接定为主方向；值得给 E-Wave 一周 Gate 预算。** 只有共同自然现象、Oracle和简单策略 residual 过门，才升级为主方向。

2. **最大研究风险是什么？**  
   不是 controller做不出来，而是 **MoE-specific 可兑现能量空间在 optimized serving 的完整 denominator 中太小，或已被 batching/LPLB/METRO/AMoE/PALS 捕获**。预测精度和算法复杂度都是次级风险。

3. **哪个 idea 优化上限最高？**  
   **C3 ElasticExpert** 的 gross 上限最高，因为理论上可减少 replica/HBM/static GPU energy；但它与 Director/EaaS/MoE-Serving直接重叠、核心不可单卡验证，因此不是首选。

4. **哪个最容易单卡证伪？**  
   全池是 **C6 RouteCohort**；Top 3 中是 **C1 E-Wave**。单卡只能证伪 batching-energy、route fragmentation和Oracle必要条件，不能证明 EP 系统收益。

5. **哪个最具系统论文潜力？**  
   条件成立时是 **C1 E-Wave**：问题边界最窄、MoE因果链最完整、单卡可先判死、8卡能形成真实系统闭环。PortSwitch的理论潜力也高，但资源依赖和collision更大。

6. **下一周做什么？**  
   只完成 E-Wave 的 prior-art action审计、schema/accounting冻结、P0反例测试、native continuous-decode smoke、energy meter calibration和 Gate 0签字；不写完整 controller，不跑 sealed formal，不做Top 2/3。

7. **GPU前必须审什么？**  
   必须审 route identity/cross-layer DAG、统一能量窗口、独立样本、surface coverage、Oracle可执行性、强baseline、calibration/sealed隔离、环境/thermal provenance和失败后不可救活规则。当前 JouleQueue v1 已被审计为 'BLOCKED'，不能直接拿它跑 formal。

---

## 9. 证据边界与最终裁决

'[Observed]'

- 当前仓库没有正式验证的 Energy–SLO 系统机制；BCRD/DEPA 也尚未完成共同自然现象 Gate。
- fixed RankLane 在冻结的 'p_return≤20%' 域相对 uniform FP8 的最乐观 E2E 上界为 4.1667%，不能作为新 Energy–SLO 主线。
- JouleQueue v1 的正式路径因 energy window、统计独立单位、route→job、cross-layer DAG、Oracle规模、baseline和provenance 等 P0 缺陷处于 'BLOCKED'。
- 5090 的本地 MoE inference-time/codec/route证据不等于8×A100 EP return-path、NCCL/RDMA或request tail证据。

'[Literature]'

- PALS、Festina、Director、Gimbal、AMoE、ExpertPlex、METRO、PagedWeight 等结果均是论文作者报告；未在本仓库复现，不能当作本机事实。

'[Hypothesis]'

- E-Wave 的 natural fragmentation energy headroom、Oracle空间、causal policy收益和跨模型普适性全部未验证。

**最终裁决：'CONDITIONAL_GO_TO_EWAVE_GATE0_ONLY'。**

这不是对 Energy–SLO-aware MoE 的乐观批准，而是一个窄授权：用一周成本判断“固定语义、固定副本集内的 expert-wave energy shaping”是否存在。如果 Gate 0/1/Oracle 任一失败，应转回当前权威的 BCRD/DEPA common phenomenon gate 或换题，不再构造更大的 Energy controller。
