# Corrected FJRC：B0/Q/J/R 信息格实验设计

状态：`EXPERIMENT_DESIGN ONLY / NOT GPU APPROVED`

## 1. 研究问题

在 route、任务集合、deadline、service、exact receiver queue map 和当前 ready set 相同
时，**per-join 已完成 sibling phase** 是否仍提供 aggregate receiver state、EDF、
least-laxity、request-FCFS 与 join-remaining-work 无法恢复的信息，并改变 bounded
receiver admission 的第一个动作？

这不是“receiver queue 是否有用”，也不验证 controller。主 estimand 是 join phase 的
条件信息价值：`R(Q+J) - Q`。

## 2. 可证伪假设

在两个模型的 request-disjoint holdout matched fixtures 中，相对拥有 exact current
q-map、deadline、service、topology、request identity 与当前 ready candidates 的 `Q`
oracle，额外拥有 causal sibling completion bitmap 的 `R` oracle同时满足：

1. unique optimal first-credit action flip rate `>=25%`；
2. selected-join deadline miss rate relative reduction `>=10%` 且 absolute `>=2pp`；
3. normalized tardiness CVaR90 reduction `>=5%`；
4. 所有简单 join-aware-without-bitmap baseline 捕获的 `Q-to-R` gain `<90%`。

任一模型失败即停止，不实现 credit controller。

## 3. 决策 locus 与动作

- locus：一个 receiver ingress credit arbiter；不是 sender内队列重排。
- 决策时刻 `t0`：receiver 有 `B` 个空 credit slot，主门固定 `B=1`，`B=2/4` 只作
  sensitivity。
- candidate：来自至少两个 sender、至少两个 joins 的 causal-ready contribution。
- 动作：选择至多 `B` 条 candidate 发 credit；正 service 下 idle 被 dominance删除。
- credit完成后进入相同的 `sender pack -> cut -> receiver unpack -> once-per-join combine`。
- 所有 arms full drain；任务、bytes、precision、route和工作量不变。

`B=1` 先隔离首动作信息；若只在 B=1成立而 B=2/4完全消失，最多说明窄队列现象。

## 4. 信息格

| Arm | 可见信息 | 禁止信息 | 用途 |
| --- | --- | --- | --- |
| `B0` | ready candidates、request/join identity、route、deadline、service、topology | exact q-map、sibling bitmap | sender/request基础信息 |
| `Q` | `B0` + exact current receiver qdepth/queued work/availability | sibling完成bitmap/missing count | 最强 queue-aware对照与主 baseline |
| `J` | `B0` + keyed completed/missing sibling bitmap | exact q-map | 分解 join-state单独价值 |
| `R` | `B0 + Q + J` | future release/service/outcome | 完整 causal information oracle |
| `C` | full future | 无 | ceiling，不参与通过判定 |

主差值只有 `R-Q`。`Q-B0`、`J-B0` 用于解释互补/替代关系，不得代替主门。

## 5. Matched-world 构造

### 5.1 真实部分

- OLMoE top-k 8、LLM-jp top-k 16 的 clean-v2 native route identity；
- request、layer、token、slot、expert、sender、receiver、gate weight不重标；
- 每个 join保留完整 siblings；service 后续来自同一 5090 LUT。

### 5.2 受控部分

从两个共享同一 receiver、且至少两个 common senders 的真实 joins构造一对 worlds：

- `world0`：join A 的其他 sender siblings 已推进到缺 1/2 个，join B 缺口较大；
- `world1`：交换 A/B 的**prior sibling completion identity**；
- decision candidates、deadline、service、q-map、queued-work多重集与映射、sender
  可见 history完全相同；完整 task universe相同；
- 被交换的 prior sibling 在另一 world 中成为尚未 release 的 fixed-after task。两 world
  的 future **work/service/resource/release-time 多重集**完全相同，但 task identity按
  prior-completion swap互补；否则会造成同一 task重复完成或永不完成；
- prior completions 必须由时间 `<t0` 的合法 pack/cut/unpack history重放得到；不能直接
  写 bitmap；每条 prior task只能出现一次；
- 两 world 的 receiver busy availability必须逐 receiver完全相同，而非只保持多重集。

若无法构造至少 16 对 request-disjoint、可达且 q-map exact相同的 fixtures，输出
`BLOCKED_INSUFFICIENT_MATCHED_SUPPORT`，不得放宽条件。

### 5.3 防止构造自证

pair选择只读 identity、route与common-sender结构，禁止读取任何 arm outcome。join phase
模板在 selection split 冻结；holdout 不按 action flip或gain筛选。另设：

- equal-phase：两 world bitmap相同，gap/flip必须为0；
- shuffled-key：phase随机映射到无关 join key，aggregate gain/flip必须为0；
- fanout-1：无 join phase自由度，gap/flip必须为0；
- Q-only：只改变 q-map、不改变phase，用于证明信息格方向正确，但不进入主 claim。

## 6. 自变量、因变量与控制变量

### 自变量

- information arm：`B0/Q/J/R/C`；
- credit capacity：主 `B=1`，敏感性 `2/4`；
- phase模板：主 `{1, top_k-1}` missing contrast，敏感性较弱 contrast；
- arrival load只在 selection split选择一个非退化值，holdout冻结。

### 因变量

- unique optimal first-credit action及action-set；
- selected-join deadline miss；
- normalized tardiness CVaR90；
- join closure P50/P95；
- makespan、starvation、full-drain conservation；
- 每个简单 baseline 对 exact `Q-to-R` gain 的 capture ratio。

### 控制变量

模型revision、输入/route manifest、placement、EP8逻辑拓扑、payload、BF16、service LUT、
deadline、arrival、task universe、random seeds、pair set、tie-break、credit capacity与solver
tolerance。两 arms间不得重新采样 service。

## 7. Baseline

Baseline API 可读 request/join identity，这是公平比较所必需；只禁止读取 prior sibling
completion bitmap。

1. FCFS/oldest-ready；
2. RR-by-sender credit；
3. global request EDF；
4. least-laxity；
5. task SRPT；
6. request/join remaining-unadmitted-work SRPT；
7. request-FCFS/group-by-request；
8. EDF then remaining work；
9. exact q-map projected finish；
10. aggregate qdepth/backpressure credit；
11. `Q` exact causal oracle；
12. full-future `C` ceiling。

其中 request/join remaining work只按当前 task universe与top-k计算，不得减去未向该 arm
公开的 prior completed siblings。若 request-FCFS 捕获 `>=90%` exact gain，novelty门失败。

## 8. 数据划分与统计

- calibration/selection 与 holdout 按 request identity完全不相交；每 receiver均分；
- selection只选择一个 arrival/deadline非退化 cell与固定phase模板；
- holdout每模型至少32 requests/16 disjoint pairs，不替换 invalid pair；
- world 是同一 pair 的反事实状态，不作为独立样本扩增；统计单位是 request/pair；
- 报全部 pair结果，不删除零增益或反向样本；
- 对 pair做 paired bootstrap 10,000次，报告 gain 95% CI；
- first-action flip用 exact枚举后的集合判定，只有两 world均 singleton且不同才计 flip；
- 多模型使用 AND gate，不池化；敏感性不用于调主阈值。

## 9. Level 规划

### Level 0：信息隔离与可达性 fixture

- **输入：** toy DAG + native route identity；暂用确定性归一化service。
- **实现：** B0/Q/J/R API、合法history replay、exact action enumerator、负对照。
- **输出：** invariant report、action sets、matched-world fingerprints。
- **停止条件：** q-map不完全相同、bitmap可由Q特征恢复、past/future partition不能对
  完整 task universe恰好一次覆盖、history不可达、负对照非零。
- **进入条件：** 全部不变量通过；这只是代码/问题有效性，不是科学正结果。

### Level 1：5090-calibrated trace replay

- **输入：** native routes + reviewed pack/unpack/combine/expert-ready service LUT。
- **实现：** request-disjoint selection/holdout、完整baseline与exact oracle。
- **输出：** per-pair raw trace、bootstrap、gate decision。
- **停止条件：** 任一主门失败，或 simple baseline capture >=90%。
- **进入条件：** 两模型AND通过；结论仅为L2信息headroom。

### Level 2：单GPU简化原型

- **输入：** Level 1冻结candidate与baseline；多个CUDA streams模拟sender/receiver资源。
- **实现：** bounded credit消息、codec/tax、staleness/fallback；不得声称物理rank。
- **输出：** executable overhead与headroom retention。
- **停止条件：** 加税后保留 <50% zero-tax gain，或时序由同卡串行化主导。
- **进入条件：** 只允许申请多GPU timed trace，不形成serving claim。

### Level 3：真实多rank

独立GPU ranks + Nsight/CUPTI统一时间轴；先跑RR-credit自然incast census，再跑candidate。
单卡结果不得替代本级。

## 10. 成功与失败解释

- `R-Q=0`：join phase不提供增量，FJRC/Join-Deficit Credit终止。
- 有risk gap但无unique first action flip：信息只改变后续排序，不支持early credit机制。
- request-FCFS覆盖：收益来自普通request grouping，不是receiver join feedback。
- 只在强synthetic phase contrast成立：最多是构造性upper bound，不进入系统实现。
- Level 1正、Level 2负：信息存在但不可部署；论文claim停在分析，不包装系统收益。

## 11. 当前是否允许写代码/GPU

允许下一步实现 **Level 0 CPU/reference code**，但须先做代码审查；不允许启动正式GPU。
Level 0 通过后再设计/审查5090 LUT consumer，不能用GPU结果反向证明信息隔离正确。
