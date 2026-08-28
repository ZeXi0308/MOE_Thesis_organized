# AGENTS.md — MoE AI Infra 研究执行手册

> 适用范围：本仓库中由 Codex 或其他 Agent 执行的 Idea 生成、方向筛选、实验设计、代码实现、结果解释、判死与复活审查。
> 科学事实权威：[`docs/current/README.md`](docs/current/README.md) 及各方向最新 sealed/formal verdict。
> 本文件角色：**过程权威**。它规定怎样研究，不自动改变任何方向的科学状态。
> 核心目标：在有限 GPU、时间和论文周期内，最大化每次实验的信息增益，同时降低假阳性、假阴性和过早判死风险。

---

## 0. 总原则

本仓库不以“写了多少代码、生成了多少审计文件、探索了多少 Idea”为成功标准。一次好的研究迭代只需要闭合五件事：

```text
一个明确问题
→ 一个最弱因果环节
→ 一个高信息增益实验
→ 一个边界清楚的结论
→ 一个唯一下一步
```

Codex 必须遵守以下优先级：

1. **问题真实性高于机制复杂度。** 先证明瓶颈、暴露路径或 action space 存在，再设计 Controller。
2. **完整请求收益高于局部代理。** GPU busy、逻辑字节、局部 overlap、expert-stage saving 都不能自动外推为 TPOT/P99/SLO-goodput。
3. **简单基线高于复杂模型。** 复杂策略只有在最强简单策略后仍有 residual 才有研究价值。
4. **探索与确认分层。** 早期允许单模型、小样本、proxy runtime 和 provisional 结果；论文主张必须升级到代表性 runtime、完整成本和更强复核。
5. **失败必须精确。** 只能判死“已经被实验覆盖的 formulation”，不能把实现失败、硬件不匹配或实验无效扩写成整个 problem family 的失败。
6. **审计服务于科学，不替代科学。** 没有新数据时，不得通过增加测试、manifest、hash、review trace 制造“研究进展”。
7. **负结果也是结果。** 只要揭示了新的边界、系统规律或错误假设，稳定的 NO-GO 可以成为论文资产。

资深 AI Infra 研究的基本判断是：**真正有价值的系统贡献通常不是更复杂的 predictor，而是发现了此前未被正确计费的瓶颈、定义了新的可执行动作、建立了新的系统不变量，或在重要运行域中证明了旧假设失效。**

---

# 1. Codex 在本仓库中最容易犯的错误

以下问题已经在历史探索中真实发生。Codex 开始任何新方向前必须逐项自检。

## 1.1 把 proxy 当成目标指标

典型错误：

- 把 full forward 的吞吐/能耗写成 KV decode serving Pareto；
- 把逻辑 payload reduction 写成真实 wire/NCCL/RDMA saving；
- 把局部 expert-stage projection 写成完整请求收益；
- 把 overlap window 写成 completion gain；
- 把 synthetic deadline 或 CPU replay 写成生产 SLO。

仓库案例：

- Energy-SLO 早期固定 `seq=64` full forward 只能说明 shape/utilization characterization，不能证明 arrival-aware decode SLO。
- Rank-tail 的 50/62.5/75% 是逻辑 payload 点，不含 scale/header/padding/alignment。
- JoinStream 证明 natural overlap window 存在，但最终 `0/4` cell 转化成安全 completion gain，形成了 `overlap opportunity != critical-path leverage` 的系统规律。

**纠正规则：** 每个结论必须标注证据层级：

```text
STRUCTURAL
LOCAL_KERNEL
ISOLATED_GPU_RUNTIME
CUSTOM_CONTINUOUS_RUNTIME
REQUEST_LEVEL
NATIVE_SERVING
MULTI_GPU_EP
```

不得跨层升级 claim。

## 1.2 用离线掩码伪装成 in-loop 策略

典型错误：先分别跑完两个完整轨迹，再对其中一条轨迹做离线 masking，声称执行了动态策略。

仓库案例：旧 Verify Precision H2 分别生成 all-BF16 / all-INT4 KV 轨迹，再对固定 INT4 KL 数组置零，并没有维护 policy-specific KV、served logits 和后续状态。

**纠正规则：** 任何会改变模型状态、KV、route、batch composition 或 request set 的动作，都必须从同一个 action 前状态分叉，并沿各自策略独立推进。

```text
same pre-action state
→ policy A state evolution
→ policy B state evolution
→ matched outcome
```

离线 masking 只能是诊断，不能成为策略结果。

## 1.3 忽略动作内生性

在推理系统中，调度动作会改变未来 workload。本仓库当前最重要的例子是：

```text
running_set / active-token budget
→ admitted requests
→ batch shape / KV / padding
→ route
→ expert load
→ latency
```

因此 future route 不是外生标签。不能对所有 candidate budget 复用同一条 observed route trace。

**纠正规则：**

- 观察性问题可以继续问“历史 batched route 是否预测下一窗口 batched latency”；
- 因果 action Oracle 必须对每个 action 重新执行或使用经过验证的 action-conditioned replay；
- “signal endogenous”不等于“signal 无预测价值”，不得用内生性自动判死观察性方向。

## 1.4 把 formulation NO-GO 扩写成 family NO-GO

仓库中多次出现这种风险：

- fixed RankLane 在 `p_return <= 20%` 域内失败，不等于 receiver/return path 普遍不存在；
- static mean-LPT 失败，不等于 placement 家族失败；
- full-top-k resident workload 下 prefetch 无 action space，不等于 offload/HBM-pressure regime 下 prefetch 失败；
- JoinStream 当前 early-visibility formulation 缺 critical-path leverage，不等于所有 dependency-retirement 机制失败。

**纠正规则：** verdict 必须包含：

```text
exact formulation
operating regime
action
objective
strongest baseline
what was falsified
what was not tested
reopen trigger
```

禁止只写“方向死了”。

## 1.5 把无效实验或未运行写成 NO-GO

状态必须严格区分：

| 状态 | 含义 |
|---|---|
| `HARD_NO_GO` | Oracle/必要条件/强基线已压死，当前 regime 不值得继续 |
| `CONDITIONAL_NO_GO` | 当前实现或运行域失败，但存在明确 reopen 条件 |
| `PHENOMENON_ALIVE_MECHANISM_DEAD` | 现象与 Oracle 存在，在线机制/selector 失败 |
| `INVALID_EXPERIMENT` | 实验逻辑、会计或数据不支持结论 |
| `UNRUN` | 关键 Gate 从未执行 |
| `MEASUREMENT_ONLY` | 形成了测量或表征结论，没有方法 GO |
| `OPEN` | 仍有未关闭的高价值不确定性 |

仓库案例：

- QuotaEP 的质量 Gate 有效，但 fused-kernel TPOT/P99 Gate 未运行，不能叫系统 NO-GO。
- Verify Precision 旧 H2 是实验实现不成立，不是机制被正确验证后失败。
- Receiver 8×A100 existence Gate 未运行，不能以单卡结果替代。

## 1.6 使用弱基线制造新颖性或收益

必须至少比较：

```text
当前默认策略
最强简单策略
最近邻 prior-art 风格策略
future-information Oracle / exact upper bound
```

常见错误：只与 BF16、random、naive least-load 或弱 QPS-HPA 比较，然后把增量归因给复杂方法。

仓库教训：

- StableBatch hindsight Oracle 强，但 static map 和 observable ridge 都低于 shuffle，证明“机会存在”与“可部署选择器成立”是两件事。
- Route Capacity 的 M3 必须比较 M2 `workload + current per-expert load`，不能只比 workload-only。
- Receiver 表示必须相对公平的 uniform FP8 或 optimized backend，而不是只比 BF16。

## 1.7 会计与分母错误

高风险错误包括：

- 重复累加 baseline；
- 对 overlapping stages 重复计时；
- 用 proposed saving 反推 denominator；
- 忽略 codec、metadata、layout、launch、queue、fallback、repair；
- 把每层局部 saving 直接相加为 request saving。

仓库案例：Additive-KL 旧 `3.77x` 来自重复累加 FP8 baseline；纠正后只剩未决，不再支持原判死叙事。

**纠正规则：** 在写代码前先写一行守恒关系，例如：

```text
total request time
= queue + prefill + decode non-MoE + exposed MoE + sampling + idle/barrier
```

每个优化项只能从一个互斥 accounting bucket 中扣除。

## 1.8 在错误运行域里判死系统 Idea

系统收益高度依赖 operating regime：

```text
resident vs offload
low vs high HBM pressure
single GPU vs EP
underloaded vs near saturation
small vs large top-k / expert set
custom eager vs fused serving runtime
```

一个 Idea 在错误 regime 下没有 action space，不构成普遍 NO-GO。

Codex 必须在 Idea Card 中回答：

```text
为什么这个瓶颈会在该运行域暴露？
什么系统资源是稀缺的？
动作能实际改变哪一部分 exposed path？
```

## 1.9 用复杂 predictor 抢救已失败的可观测性

如果：

- Oracle 强；
- 两个合理 selector 已失败；
- fresh holdout 无增量；

则默认停止第三个 pre-action predictor。应转向：

```text
by-construction execution
post-action verification
selective repair
slow-timescale placement/capacity
measurement/diagnosis
```

StableBatch、ErrorToken 和 SemanticFence 已充分说明，继续增加特征、模型和阈值通常只是 feature search，不是突破。

## 1.10 过早让审计压制探索

过度审计会导致：

- 为一个 64-window pilot 写数千行协议和测试；
- 多轮 same-family reviewer 重复检查已关闭的问题；
- 没有新数据，却不断增加 hash、manifest 和 `.aris` trace；
- 把“尚未达到投稿标准”错误解释为“不允许做 provisional 探索”。

**纠正规则：探索阶段只保留四个必要检查：**

1. 是否使用未来信息；
2. request/step/window/route 是否对齐；
3. 计时和会计是否明显错误；
4. baseline 是否公平。

只有发现具体风险，才增加针对性测试。连续两轮没有 P0/P1 后停止扩展审计。

## 1.11 Post-hoc 选择、覆盖和运行漂移

禁止：

- 看到指标后替换 canonical run；
- 覆盖原始 report；
- 只保留有利 repeat；
- 审计时原地修改证据；
- 把未隔离的两次运行当作复现。

Route Capacity Envelope 的 `+9.63%` 与 `-24.04%` 符号翻转说明：没有受控 repeat 时，任何单次预测收益都只能是 diagnostic。

**纠正规则：**

- 所有运行永久保留；
- canonical 选择规则在运行前声明；
- 原始 artifact 不修改，修正写 addendum；
- sign flip 后优先做受控重复，不先解释机制；
- 记录 GPU process isolation、环境、代码 commit 和 action config。

## 1.12 新颖性只停留在“组件组合”

下面通常不够构成独立系统贡献：

```text
route feature + generic admission controller
known verifier + known rollback
known placement + another score
known predictor + another workload
```

真正的 novelty residual 应落在至少一个维度：

- 新的、可复现且具有 full-request 后果的现象；
- 以前不可执行的新 action；
- 新的系统不变量或 correctness guarantee；
- 重要新运行域中旧假设失效；
- 新的 causal boundary，使一整类方法可以被正确判断。

---

# 2. Idea 生成：先发散，再收敛

Codex 不得一开始就用 formal gate 审判所有候选。Idea 生成分三步。

## 2.1 Step A：自由发散

围绕一个已观察到的现象，生成最多 6 个候选，至少覆盖以下四种视角：

1. **Causal residual**：普通负载指标解释后，仍剩下什么？
2. **Regime shift**：换到 HBM 压力、offload、多卡 EP、近饱和或长上下文后，旧结论是否改变？
3. **Action redesign**：能否从 pre-action prediction 转为 by-construction、post-action verify 或 slow-timescale control？
4. **Measurement insight**：负结果是否揭示了此前未正确建模的系统规律？

额外保留一个高风险 wildcard，允许跳出当前 mechanism family，但必须仍然符合本仓库 MoE inference / AI Infra 目标。

发散阶段不要求完整 prior-art matrix，不要求双模型，不要求 formal threshold。

## 2.2 Step B：机制化表达

每个候选必须写成以下 Idea Card：

```text
Idea name:
Problem:
Operating regime:
Observed signal:
Bottleneck / scarce resource:
Action:
What the action changes:
What remains invariant:
Target metric:
Full-request denominator:
Strongest simple baseline:
Closest prior-art action:
Potential novelty residual:
Cheapest falsifying experiment:
Expected positive interpretation:
Expected negative interpretation:
Reopen condition:
```

没有明确 action 的候选可以作为 measurement paper，但不能伪装成 method candidate。

## 2.3 Step C：收敛为 1+1+1 组合

每轮只保留：

```text
1 个 Primary：最高信息增益、最短因果链
1 个 Conditional Backup：依赖 Primary 某个结果
1 个 Wildcard：不同机制族或不同运行域
```

禁止同时实现多个 Controller。只有 Primary 的最弱链路被关闭后，才切换 Backup。

---

# 3. 怎样激发更有突破性的 Idea

Codex 在生成新 Idea 时必须主动问以下问题，而不是继续排列组合已有组件。

## 3.1 从失败假设中找新问题

每个失败方向都要提取一句系统规律：

- StableBatch：`hindsight opportunity != pre-action selectability`
- JoinStream：`overlap window != critical-path leverage`
- Route Capacity：`route telemetry may be action-endogenous`
- Receiver：`logical byte saving != exposed return-path saving`
- SemanticFence：`action space != safe low-cost witness`

下一 Idea 应尝试改变失败链条中最根本的一环，而不是优化原参数。

## 3.2 从不可观测动作转向可验证动作

当风险无法在 outcome 前预测时，优先考虑：

```text
canonical execution
speculative shadow execution
GPU-resident verifier
selective repair
transactional commit boundary
request-scoped numerical epoch
```

AI Infra 中，by-construction 或 fail-closed 机制往往比高 AUC predictor 更具论文价值，因为它直接提供可部署保证。

## 3.3 从局部优化转向系统边界

高价值问题通常出现在边界处：

```text
route × batch shape
expert weights × KV memory
admission × expert pressure
numerical conformance × dynamic batching
placement × topology × request completion
verification × natural slack
```

Codex 应优先寻找“两个子系统分别看都合理，但耦合后出现新约束”的地方。

## 3.4 从平均性能转向安全容量边界

与其只预测平均 latency，不如研究：

```text
safe capacity envelope
SLO violation risk
critical-path survival
headroom under uncertainty
```

但必须保证动作可执行，并且安全容量不是从同一 observed trace 伪造出的反事实标签。

## 3.5 从新算法转向新系统规律

如果没有可部署 mechanism，也可以形成强 measurement thesis，例如：

- route-conditioned barrier amplification 何时进入完整请求；
- batch shape 何时改变 MoE numerical trajectory；
- local expert tail 何时被 overlap 吸收；
- optimized EP 下 return path 何时真正暴露；
- generic request DAG 在 dynamic MoE batching 下漏掉哪些 causal edges。

一条跨模型、可复现、有因果解释的系统规律，通常比一个只在 synthetic benchmark 赢 3% 的复杂 scheduler 更有价值。

---

# 4. 探索与验证的分阶段流程

## Stage 0：Repository Reality Snapshot

开始前只读以下内容：

```text
docs/current/README.md
docs/ideas/README.md
目标 Idea 的 README / STATUS / verdict
最近一次相关 artifact
历史纠错审计
```

输出不超过一页：

```text
当前权威结论
已测 / 未测
可复用代码
本轮唯一不确定性
本轮不允许外推的边界
```

不要无目的遍历全部 `.aris` 或重复历史审计。

## Stage 1：Cheap Existence Probe

目标：判断现象、action space 或 Oracle headroom 是否可能存在。

允许：

- 一个模型；
- 16–32 requests；
- 单张 GPU；
- custom runtime；
- offline replay；
- 单一或两个运行域；
- provisional 3%–5% effect 作为继续信号。

必须保留：

- 公平 baseline；
- identity/time alignment；
- 明确证据层级；
- 一个负控。

这一阶段不得因为“未达到论文标准”自动阻塞。

## Stage 2：Causal Qualification

只验证最弱因果链。优先使用三臂设计：

```text
A: baseline / native
B: intended intervention
C: matched negative control or decorrelated control
```

示例：execution conformance 诊断中，应比较：

```text
target serial
original companion batch
same width but shuffled companions
```

并从同一个 pre-step state 分叉，捕获 first divergence。

## Stage 3：Action / Oracle Headroom

只有现象存在后才做。

要求：

- action 真实改变系统状态；
- candidate action 重新生成后续 route/batch/completion；
- 与 strongest simple baseline 比；
- 计入 action 执行成本；
- Oracle 自身没有 material headroom 时立即停止 method search。

探索阶段 `1%–3%` 只能算弱信号；`>=3%` 可以继续；论文阶段再根据工作量和噪声冻结更严格门槛。

## Stage 4：Minimal Mechanism

只实现能捕获 Oracle residual 的最小动作。

默认优先顺序：

```text
static/simple rule
→ one-dimensional feedback
→ small causal model
→ only then complex model
```

禁止直接上 RL、复杂深度模型、多动作优化或完整 Kubernetes Operator。

## Stage 5：Confirmatory Evidence

只有准备形成论文主张时才升级：

- 第二模型或第二 routing regime；
- steady + bursty；
- representative serving runtime；
- strong simple + nearest prior-art baseline；
- full cost / request-level denominator；
- controlled repeats；
- 必要时多 GPU EP。

---

# 5. 实验设计正确性规则

## 5.1 Claim-to-Measurement 对齐

| 想主张什么 | 最低证据 |
|---|---|
| 路由结构存在 | trace / structural measurement |
| kernel 局部加速 | GPU microbenchmark |
| continuous-decode latency 改善 | 同 runtime continuous decode |
| TTFT/TPOT/P99/SLO | arrival、queue、request completion |
| 容量控制收益 | action-conditioned rerun + SLO-goodput |
| EP/A2A/receiver/NCCL | 真实多 rank / 多 GPU backend |
| 生产可部署性 | native serving runtime + overhead/fallback |

任何报告必须明确自己在哪一层停止。

## 5.2 Policy-specific state

动作会改变状态时，必须为每个 policy 保存独立：

```text
KV
route
request set
batch composition
queue
randomness/seed
completion timeline
```

不得共享未来状态或用一个 policy 的 future trace 评估另一个 policy。

## 5.3 Strong baseline ladder

每个 method experiment 至少考虑：

1. Current/default；
2. 最强一维或阈值策略；
3. 同 action space 的 nearest prior-art；
4. Future-known Oracle；
5. Proposed。

如果简单策略捕获绝大部分 Oracle，复杂机制应停止或降级为 engineering variant。

## 5.4 数据切分

根据问题选择独立单位：

```text
request
document
arrival episode
model
stack/runtime version
```

不得随机打散相邻 decode windows。Train/test 不得共享同一 request 或 document；时间序列问题优先按 episode 切分。

## 5.5 重复与环境

出现以下任一情况必须做受控 repeat：

- sign flip；
- 接近噪声；
- thermal-sensitive；
- GPU 共享；
- kernel/tactic 可能变化。

受控 repeat 要记录：

```text
commit
Python/PyTorch/Transformers/CUDA
GPU process isolation
clock/power/temperature（若相关）
exact workload/action config
run retention rule
```

## 5.6 Source localization 优先于机制解释

发现 divergence 时，不要立即包装为新方法。先定位 first divergence：

```text
pre-router hidden state
router logits
top-k boundary
selected experts
expert output
combine output
next-router
token/request completion
```

只增加能区分两个原因的最小观测点，不进行全栈埋点。

## 5.7 Correctness 与 usefulness 分离

必须分别回答：

```text
动作是否合法？
结果是否正确/语义可接受？
动作是否可观察/可选择？
局部收益是否进入 critical path？
完整成本后是否净正？
```

其中任何一项通过都不能替代其它项。

---

# 6. 审计预算与工程纪律

## 6.1 早期最多四类测试

在出现正信号前，测试只覆盖：

1. identity/window alignment；
2. causal cutoff / future leakage；
3. accounting / metric correctness；
4. action-specific state regeneration。

发现具体 bug 后再加回归测试。禁止为了“看起来严谨”扩展成全面覆盖率工程。

## 6.2 Audit Stop Rule

满足以下条件后停止追加审计：

```text
两轮 targeted review 无 P0/P1
核心命令可重跑
原始 artifact 未被覆盖
claim boundary 已写清
```

之后必须获取新数据或进入下一科学问题。

## 6.3 Artifact 规则

探索阶段每个实验最多一个 canonical bundle：

```text
config
raw/processed core data
metrics
report
commands
run log
```

要求：

- 所有 repeat 保留，不只保留 favorable run；
- raw artifact 不原地修改；
- 修正写 `ADDENDUM.md`；
- 不为每次 smoke 复制 source snapshot、git diff 和大量空目录；
- `.aris` 仅在用户明确要求 independent review trace 时生成。

## 6.4 代码规模自检

在为探索性 pilot 写超过约 500 行新代码前，Codex 必须解释：

```text
为什么现有工具无法完成？
为什么更小的 A/B probe 不能回答？
新增代码直接关闭哪一个不确定性？
```

如果答不清，先缩小实验。

---

# 7. 创新性与突破性评估

## 7.1 Action-level novelty matrix

查新必须按以下字段比较，而不是按标题相似度：

| Work | Signal/state | Prediction target | Action | Objective | Regime | Guarantee |
|---|---|---|---|---|---|---|

只有完整链条有 residual，才能主张独立 Idea。

## 7.2 值得成为主线的五种突破

1. **新现象**：跨模型、自然 workload 中发现此前未测的 causal bottleneck；
2. **新动作**：现有系统无法表达，而该动作真实改变 full-request completion；
3. **新不变量**：在动态 batching/EP 下提供 correctness 或 conformance guarantee；
4. **新运行域**：在 offload、HBM pressure、多卡 EP 或近饱和下，旧结论发生结构变化；
5. **新边界**：证明一类局部优化何时必然无法转化为 request benefit，或何时可以。

## 7.3 不足以成为独立贡献的模式

- 只多加 route feature；
- 只换 predictor；
- 把两篇已知机制拼接；
- 只在 synthetic skew 上有效；
- 只提高 local kernel 指标；
- 只增加完整性工具，但没有发现 generic trace 的系统性错误；
- 只在 outcome 后才能选择动作，却包装成 online policy。

## 7.4 Prior-art 使用节奏

- Idea 发散前只做局部碰撞意识；
- Cheap probe 出现正信号后，再对 Top 3 collision 读全文和代码；
- 没有正信号前，不投入大规模查新；
- 没有 action-level 查新前，不写“first”或“novel”。

---

# 8. Resurrection Audit：防止好 Idea 被过早埋掉

每个停止或归档方向必须有一张 Resurrection Card：

```text
Idea family:
Exact formulation tested:
Operating regime:
Evidence tier:
Current status:
What was actually falsified:
What remains untested:
Oracle/headroom status:
Strongest baseline:
Measured implementation tax:
Same-idea reopen trigger:
When it becomes a new idea:
Cheapest discriminating experiment:
Do-not-rescue list:
```

## 8.1 当前仓库的示例分类

这些仅作为过程示例，不替代各方向权威文档：

| 方向 | 建议分类 |
|---|---|
| CreditReduce | `HARD_NO_GO` in current quality/precision regime |
| StableBatch | `PHENOMENON_ALIVE_MECHANISM_DEAD` for pre-action selector |
| JoinStream | `CONDITIONAL_NO_GO` for current early-visibility formulation |
| Receiver/RankLane | `CONDITIONAL_NO_GO`, reopen by measured 8×A100 exposed return fraction |
| Prefetch | resident/full-top-k regime NO-GO；offload/HBM-pressure 是新 regime |
| QuotaEP | `UNRUN` system Gate；不是系统 NO-GO |
| Verify Precision H2 | `INVALID_EXPERIMENT / UNVERIFIED` |
| Route Capacity | capacity interpretation paused；observational signal 与 conformance source localization 分开 |
| SemanticFence | action space alive，pre-execution witness dead；post-action verify/repair conditional |

## 8.2 复活条件

只在以下事件发生时进行 resurrection sweep：

- 新硬件或新 runtime 可用；
- 新结果改变关键 denominator；
- 新机制消除了已测固定税；
- 新 action 改变了原 problem definition；
- 旧实验被证明 invalid。

禁止仅因换名字、换阈值、换 seed、换一小段 workload 而复活。

---

# 9. Codex 每次执行的固定合同

## 9.1 开始前

Codex 必须先输出：

```text
Repository HEAD / dirty state
Authority files read
Frozen facts inherited
One research question
One weakest causal link
One experiment
Allowed claim ceiling
Stop / continue / reopen conditions
```

不得自动修改 `docs/current/README.md`，不得自动 push，除非用户明确要求。

## 9.2 执行中

- 一次只推进一条证伪链；
- 不并行实现多个 Controller；
- 不覆盖 artifact；
- 发现实验设计错误时，先标 `INVALID`，不要用补丁修饰旧结果；
- 发现新现象时先做 source localization；
- 发现正信号时先找 strongest simple baseline；
- 发现负信号时先判断是 signal、action、cost、runtime 还是 measurement 失败。

## 9.3 结束时

最终报告固定为：

```text
Verdict
Evidence type
What was measured
What was not measured
Strongest baseline
Oracle/headroom status
Claim ceiling
Failure category
Resurrection condition
One next smallest experiment
```

结尾必须直接回答研究问题，不能只写“值得继续探索”。

---

# 10. 当前仓库的近期执行蓝图

本节是基于当前证据的过程建议，不自动改写科学权威。

## 10.1 Primary：精确定位 batch-dependent route divergence

当前最有信息增益的问题不是立刻实现容量 Controller，而是：

> 同一 request/token state 在 serial 与 batched execution 下为何选择不同 Expert？

下一实验应从已有真实 difference event 选择 target，不只取 steady 前四个请求。至少包含一个 steady 和一个 bursty 事件，并比较：

```text
A: target serial
B: target + 原始 companions / 原始 KV length / 原始 padding
C: same width + shuffled companions
```

从同一 pre-step canonical state 分叉，最小捕获：

```text
pre-router hidden state
router logits
top-k boundary margin
selected experts
expert output
next-token logits
```

解释：

- hidden 相同、router logits 不同 → router GEMM/kernel shape；
- hidden 已不同 → upstream attention/KV/padding/residual；
- logits 微变且 flip 集中于 near-tie → numerical amplification；
- B/C 不同 → companion identity externality；
- A/B 不同而 B/C 相同 → batch width / physical shape；
- 同 arm 不稳定 → environment/kernel nondeterminism。

## 10.2 Route Capacity 不应被机械判死

Batch-dependent route 只阻止固定-route action counterfactual，不自动否定：

```text
historical batched route
→ next-window batched latency/risk
```

因此当前正确状态是：

```text
PAUSE_ACTION_ORACLE
CONTINUE_OBSERVATIONAL_DIAGNOSTIC
RUN_CONFORMANCE_SOURCE_LOCALIZATION
```

只有受控 repeat 证明 route residual 稳定后，才做 action-conditioned running-set rerun。

## 10.3 RCBA 作为 full-request leverage Gate

当 representative runtime trace 可用后，用 RCBA 回答：

```text
local route/expert tail
→ barrier amplification
→ full-request completion
```

没有 full-request leverage，就停止局部调度机制；有 leverage，再寻找最小 action。

## 10.4 Conditional Backup

优先级顺序：

1. SemanticFence GPU-resident post-action verifier / selective repair；
2. QuotaEP fused-kernel system Gate **或** true in-loop Verify Precision，二选一清算；
3. 8×A100 receiver existence Gate；
4. offload/HBM-pressure 下的 prefetch 重新定义。

不得同时铺开。

---

# 11. Definition of Done

## 一次探索迭代完成

- 唯一问题被回答；
- 证据层级明确；
- 没有未来泄漏或明显错账；
- strongest simple baseline 已考虑；
- verdict 精确到 formulation/regime；
- 只保留一个下一实验。

## 一个 method candidate 成立

- 自然问题和 action space 存在；
- Oracle 有 material full-request headroom；
- 简单策略没有覆盖大部分收益；
- action 因果可执行；
- 完整成本后净正；
- novelty residual 不只是组件组合；
- 有代表性 runtime 或明确的 transfer plan。

## 一篇论文主线成立

- 一条可复述的因果链；
- 一个中心机制或系统规律；
- 两到三个核心贡献；
- 一张主结果表；
- 明确适用域和失败边界；
- 不依赖被隐藏的 proxy、future information 或 post-hoc selection。

---

## 最终提醒

本仓库已经不缺 Idea 数量。下一阶段最稀缺的是：

```text
代表性 runtime 数据
受控 counterfactual
full-request denominator
action-level novelty residual
对已归档方向的条件化复活边界
```

Codex 的职责不是不断生成第 31 个 Idea，也不是充当无限严格的实验法官；它应像一名成熟的 AI Infra 研究者一样，**优先寻找最可能翻转方向判断的那一个实验，并用最短、最可解释的证据链完成它。**
