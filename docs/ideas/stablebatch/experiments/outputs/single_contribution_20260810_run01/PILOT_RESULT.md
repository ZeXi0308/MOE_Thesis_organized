# StableBatch-MoE 单贡献传播 Pilot 结果

**实验状态**：`COMPLETE`  
**冻结判定**：`SUPPORT`  
**独立完整性审计**：`PASS`，`provisional_same_family`  
**当前方向判断**：继续沿当前机制推进  
**当前阶段**：阶段 E 的局部收益传播验证；尚未进入在线策略或系统收益验证

## 当前研究状态

- **研究方向**：StableBatch-MoE——对 batch composition / expert execution shape 引起的数值不稳定进行选择性、在线可控的 route stabilization，而不是把整个系统退化为固定 batch 或串行执行。
- **Bottom-line problem**：continuous batching 改变 expert GEMM 的执行形状；即使请求输入和模型权重不变，局部数值差异也可能改变后续 routing，破坏 request-level 可复现性与隔离，并让调度行为反过来影响模型轨迹。
- **Must-solve bottleneck**：必须先证明 execution-shape 引起的单个 expert contribution 差异能够越过 gate-weight/combine，并传播到更后层 route 或 token；否则为它设计稳定化 action 没有系统意义。
- **当前核心机制**：只识别并保护 route-fragile contributions，在有限 action budget 下选择性稳定执行形状；完整 Idea 仍是在线 batching/coalescing 控制，不收缩为纯测量工作。
- **Target outcome**：在保留 batching 效率的同时，降低 batch-context-induced route divergence，后续再验证 request quality、latency 和 serving stability。
- **Non-goals**：本轮不证明特定 CUDA/cuBLASLt algorithm，不评估 latency/throughput/energy，不覆盖 continuous decode、EP/NCCL/RDMA、租户隔离或生产环境。
- **Constraints**：单 RTX 5090；OLMoE 固定 revision；BF16 eager prompt forward；16 个 16-token workload；M=1 对 M=64 repeated-row side-call。
- **已有最强证据**：单个 raw contribution 差异在 32/32 个冻结目标上穿过 combine，在其中 12 个目标、8 个 victim 上三次重复均改变下游 top-k membership；另有 1 个目标稳定出现 greedy token `337 → 608`。
- **已有最重要负结果**：20/32 个富集目标虽有 raw 与 combine 差异，却没有下游 membership-set 改变；因此不能采用“所有 shape delta 都值得保护”的 blanket action。
- **当前最大未知**：不使用 future next-layer margin 或 M1/M64 双算结果时，一个在线可观测 selector 能否以相同 action budget 找到真正会传播的 contribution。

## 本轮新增结果解释

### 实际观察

1. 候选扫描覆盖 `16 victims × 15 layers × top-8 = 1,920` 个唯一 contribution；所有候选的 M1/M64 raw BF16 hash 均不同。
2. 冻结 selection 从四个 layer band 各取 8 个目标，并限制每个 victim 最多 2 个；共 32 个目标。
3. 32/32 个目标的 raw contribution 发生变化，32/32 的 target MoE post-combine 输出发生变化，所有 arm 各 3 次重复 bitwise 稳定。
4. 12/32 个目标在更深层出现可重复 top-k membership-set 变化，分布于 8/16 个 victim；20/32 没有 route membership 传播。
5. 1/32 个目标三次都出现 greedy token `337 → 608`；该目标没有 membership-set flip，说明最终 token 数值变化不必以集合变化为唯一中介。
6. native self-replacement 对 target MoE、全部记录 route 和 final logits 均为 bitwise no-op；输入、attention mask、target router/top-k/weight、upstream routes、native target raw 与所有 non-target contributions 均闭合。

### 支持的判断

本轮支持以下窄因果箭头：

> execution-shape M 改变一个 raw expert contribution  
> → 差异穿过 gate-weight/combine  
> → 在部分预冻结富集目标上改变下游 victim route membership。

这已经回答上一轮最关键问题：局部 shape-sensitive 数值差异不只停留在 isolated expert output，它可以传播到后续模型行为。因此 StableBatch 的“选择性 route stabilization”机制值得继续。

### 被削弱的判断

- 被削弱的是 **blanket stabilization formulation**：32 个有 local/combine delta 的目标中只有 12 个出现 route membership 传播，不能把所有 delta 当成等价风险。
- 没有被削弱的是完整 StableBatch 方向；当前结果反而说明 selector 是核心，而不是可选优化。

### 尚不能得出的结论

- `12/32` 不是自然发生率：目标由 local delta 和 next-layer margin 富集。
- M=64 repeated-row 不是自然 heterogeneous batch，也不锁定某个 kernel algorithm。
- 尚未执行 StableBatch controller、post-route coalescing action 或 continuous-decode serving。
- greedy-token flip 只证明一次冻结条件下的更强传播结果，不等于质量退化、概率或用户可见影响。
- 单卡结果不能证明 EP、多租户、通信路径或生产收益。

## 更新后的因果链

| 因果箭头 | 状态 | 当前证据 |
|---|---|---|
| 动态 batch composition → expert execution shape M | 有较强系统依据，本轮固定操纵 M | 本轮只操纵 side-call M，不运行自然 batcher |
| execution shape M → raw expert numerical delta | 已支持 | candidate 1,920/1,920；正式目标 32/32 |
| raw delta → post-combine target MoE delta | 已支持 | 32/32，且 exactly-once gate weight 与 non-target closure 通过 |
| post-combine delta → downstream route membership | 已支持于富集冻结目标 | 12/32 targets，8 victims，3/3 repeats |
| route/numerical propagation → token outcome | 有初步线索 | 1 个稳定 greedy-token flip；非发生率证据 |
| 在线可观测信号 → 高价值 contribution selector | 待验证 | 当前 next-layer margin 是 offline enrichment，不可直接作为在线 selector |
| selector → 有限预算稳定化 action | 待验证 | 尚未运行 actual policy/action |
| 稳定化 action → request/serving 指标 | 尚未验证 | 无 continuous-decode、latency、quality 或 EP 证据 |

## 假设更新

- **保留的主假设**：batch-aware execution-shape stabilization 可以减少会传播的 MoE route divergence，同时不必全局固定 batch shape。
- **被支持的子假设**：单 contribution 的 execution-shape 数值差异能够穿过 combine 并改变后续 routing。
- **被削弱的子假设**：任意可测 local delta 都值得执行稳定化 action。
- **当前最主要替代解释**：offline next-layer margin enrichment 提高了命中密度；它不推翻 12 条已闭合因果 trace，但可能让当前命中率远高于自然候选池。
- **新出现的控制变量**：route boundary proximity 确实与传播目标选择相关，但下一步必须寻找 intervention 时刻可观测的替代信号。
- **是否修改机制**：保留 action 目标，把机制从 blanket shape stabilization 明确演化为 budgeted、online-observable selective stabilization。

## 当前方向判断

**继续沿当前机制推进。**

原因不是“32 个目标中过了门槛”，而是最薄弱的必要箭头已被因果隔离地观察到；方向现在需要从 existence probe 前进到 selector/action probe，而不是继续扩大当前微实验审计。

## 下一轮唯一核心问题

> 在固定 workload、相同 action budget、相同 M/padding/FLOPs/signature 下，只使用 action 时刻可观测信号的 selector，是否比 action-budget-matched shuffled selector 更有效地阻止下游 route divergence？

## 下一轮最小实验

- **核心假设**：传播风险集中在可由 current-layer router boundary、gate weight 与已知 execution-shape state 识别的少量 contributions；选择性保护优于随机保护。
- **本轮问题**：在线可观测 selector 是否具有 action value，而不仅是离线相关性。
- **自变量**：`observable-selector` 与 `seed-frozen shuffled-selector`；两者每个 victim 保护相同数量的 contributions。
- **因变量**：相对 matched unprotected arm，被避免的 downstream top-k membership divergences。
- **Baseline**：action-budget-matched shuffled selector。
- **必要对照**：selector 与 shuffled 使用完全相同的 M surface、padding、side-call 数、替换数、full-forward 数和执行顺序；只交换被保护 contribution identity。
- **固定变量**：OLMoE revision、RTX 5090、BF16/eager、token windows、attention mask、M1/M64、每 arm 重复数和 action budget。
- **主指标**：`avoided_route_divergences / protected_contribution`；跨 distinct victims 的有效保护数。
- **可复用代码**：当前 candidate capture、single-contribution replacement、hash closure、route membership comparator、manifest/status 协议。
- **最小新增代码**：一个不读取 future next-layer outcome 的 frozen selector；支持同一 victim 的 multi-target protection；paired shuffled assignment ledger。
- **数据**：优先使用 sealed manifest 中未参与本轮 selection 的 held-out windows；不得用本轮 12 个 positives 调阈值。
- **实验步骤**：先离线冻结 observable score、action budget 和 shuffled seeds；执行 no-action、observable、shuffled 三臂；逐 victim 配对复算 route divergence；只报告冻结主指标。
- **资源**：单 RTX 5090。
- **预计成本**：实现约 2–4 小时，GPU Pilot 约 30–90 分钟；不需要搭建完整 serving 系统。

## 预定义结果解释

- **支持当前机制**：在机制实际触发、完整性闭合时，observable selector 的 `avoided_route_divergences / action` 至少为 shuffled 冻结均值的 `1.5×`，且增益覆盖至少 4 个 distinct victims；授权进入 continuous-decode 小型 action Pilot。
- **削弱当前机制**：observable selector 与 shuffled 持平或更差，且其动作确实覆盖有 M-sensitive raw delta 的 contributions；这削弱当前 selector formulation，不否定 execution-shape 传播现象。
- **无法判断**：只有 selector 使用了未来信息、两臂 work/signature 不匹配、action 未真正改变 contribution、same-arm 不稳定或 artifact 损坏时使用。
- **正向后下一步**：在已有 continuous-decode harness 中运行一个固定 action-budget 的小型 StableBatch controller。
- **负向后优先修改**：保留稳定化 action，替换 online signal；优先检查 current-layer margin 是否与真正的 downstream boundary 错位。
- **本轮不能外推**：自然发生率、质量收益、serving latency/throughput、EP/通信、跨模型/跨 GPU 泛化。

## 当前非阻塞问题

- P1：固定 M1→M64 顺序可在后续 reversed/interleaved 复测中增强鲁棒性。
- P1：raw vectors 未持久化；当前 decisive hash/membership 复算已闭合，但后续可保留向量支持更细数值分析。
- P2：需要独立或 cross-family review 才能把 `provisional_same_family` 升级为非暂定接纳。
- P2：测试模块顶层依赖 Torch，不影响当前环境 5/5 测试，但可在工程清理阶段拆分纯聚合测试。

## 审计停止判断

> 当前实验已经足以推进下一轮探索，停止继续扩展审计项。

## 下一步

只做一件事：冻结并实现 `observable-selector vs action-budget-matched shuffled-selector` 的 paired action Pilot；不再扩展本轮 single-contribution 审计。

## 证据入口

- [冻结配置](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/single_contribution_pilot_v1.json)
- [V2 冻结锁](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/FROZEN_PILOT_LOCK_V2.json)
- [正式汇总](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/summary.json)
- [32 个目标原始结果](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/target_results.jsonl)
- [实验审计](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/EXPERIMENT_AUDIT.md)

