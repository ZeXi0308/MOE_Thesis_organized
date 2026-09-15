# RouteShield-MoE Gate-0 冻结协议

> 版本：`routeshield-gate0-v1`
> 冻结日期：2026-07-29
> 当前裁决：`PLAUSIBLE_SECURITY_HYPOTHESIS / PROTOCOL_ONLY / BLOCKED_PROTOCOL_NOT_AUTHORIZED`
> 权威边界：本文是候选资格协议，不改变 [`docs/current/README.md`](../../current/README.md) 的当前主线。

## 0. 直接结论

RouteShield 的问题动机真实，但附件中的 `A+` 和“5090 victim-P99 Gate”定性过早。当前只允许这个结论：

> **尚未找到与“将 tenant 的 expert/physical-rank demand 作为多资源公平对象，并在 exact MoE semantics 下做 victim isolation”完全同构的已知系统；但这是有限检索后的可信缺口，不是完整查重结论。**

这个缺口同时被三类工作夹住：

1. RepetitionCurse 已给出 route-induced DoS、共批 victim 传染、静态 vulnerability-aware placement、PPL 过滤和 dynamic EPLB 讨论；
2. VTC 和 FairServe 已给出 token/service-counter 级多租户公平；
3. Gimbal、METRO、UltraEP 等已覆盖 expert pressure、placement 和 exact-load balancing。

因此 RouteShield 必须证明的不是“route 信息有用”，而是：

> **在一组真正可执行、会删除 victim 对 attacker 的 completion-DAG 依赖边的 action space 中，expert-footprint 相对最强 route-blind 公平/限流/分区方法仍有稳定增量。**

## 1. 引用核验与必须修正

### 1.1 附件的 7 个 arXiv 条目

| arXiv | 核验 | 与本方向的真实关系 |
|---|---|---|
| [METRO 2512.09277](https://arxiv.org/abs/2512.09277) | `verified` | memory-bound decode 中平衡 activated experts 而非 token；压缩了普通 balancing 新颖性。 |
| [RepetitionCurse 2512.23995](https://arxiv.org/abs/2512.23995) | `verified` | 证明黑盒 adversarial text 可制造 route concentration 和 TTFT DoS。 |
| [Gimbal 2606.15177](https://arxiv.org/abs/2606.15177) | `verified` | 联合 DP-engine、KV、queue 和 expert pressure；论文页中未见 tenant/fairness/adversarial 机制，但这不等于完整 collision audit。 |
| [EEP 2605.10670](https://arxiv.org/abs/2605.10670) | `verified` | 明确排除不触发 timeout 的 transient performance degradation 与 silent data corruption。 |
| [UltraEP 2606.04101](https://arxiv.org/abs/2606.04101) | `verified` | post-gating exact-load 的逐层、逐 microbatch 重平衡；是 RouteShield 的强 performance baseline/collision。 |
| [ExpertPlex 2607.18002](https://arxiv.org/abs/2607.18002) | `verified` | tile-level adaptive persistent scheduling 和 isolation；进一步压缩通用 scheduler 新颖性。 |
| [Ekka 2606.04594](https://arxiv.org/abs/2606.04594) | `verified` | 支撑 silent serving error 是真实问题，但主要对应 SparseCommit，不是 RouteShield 的直接证据。 |

### 1.2 对 RepetitionCurse `3.063×` 的修正

`3.063×` 是该文表 3 中 Mixtral-8×7B 商业/API TTFT 放大比的 **95% 置信下界**，不是它本地 8-GPU 实验的 point estimate。论文本地 mixed/attack batch 实验报告的 8-GPU Mixtral TTFT 放大最高为 `2.48×`。两者都是有价值的外部证据，但不得混成本仓库的系统结果。

### 1.3 附加的必要相关工作

- [VTC: Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588)：token-cost 公平和 continuous-batching scheduler。
- [FairServe](https://arxiv.org/abs/2411.15997)：应用特征感知的 throttling 与 weighted service counter。
- [Rethinking Latency DoS](https://arxiv.org/abs/2602.07878)：说明 modern continuous batching 可以对部分算法级 latency attack 产生逻辑隔离；因此 strong backend/chunked-prefill 必须是负控和 baseline，不能只与素朴 FCFS 比。
- [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880)：从在线算法角度研究 expert 的 GPU 配置，是与本题相邻但不同的 placement 路线。

## 2. 唯一允许的主张

本轮只研究 **prefill RouteShield-P**，primary metric 是 victim request-level `TTFT P99`。

- RepetitionCurse 的已有安全证据主要是 prefill/TTFT。
- decode route persistence 可以在 5090 上作探索性表征，但不与 prefill 结果 pooling，不得把它直接写成 TPOT isolation。
- 如果以后 decode 自然现象独立通过，再另立冻结协议；本文不预先授权。

研究问题是：

> 在一个身份可验证、无 Sybil 的多租户 MoE prefill 服务中，当黑盒 adversarial text 使真实 expert/physical-rank demand 集中时，能否在不 drop/饿死 attacker、不改变模型语义的前提下，用可执行的 batch/lane 隔离为 victim 提供显著优于 token/service-counter/DRF 的 TTFT 保护？

不再使用“P99 可证明下界”这个表述。延迟隔离应讨论 victim latency/slowdown 上界或 goodput 下界；实验 P99 本身不是 worst-case 数学保证。

## 3. 冻结 threat model

### 3.1 攻击者

- 单一 authenticated tenant；Sybil/multi-account 明确 out of scope。
- 只有黑盒文本 API；不知道 router 权重、expert-to-rank placement 或后端实时状态。
- 不能直接注入 expert ID/route trace，不能请求特权 GPU 或绕过公开 admission 规则。
- 每个 paired block 固定 request 数、arrival、input-token budget 和 `max_new_tokens=1`；attack 与 matched-benign cotenant 的付费/资源预算相同。
- 攻击可以是人工 adversarial text。“合成”不是安全实验的自动判死条件；真正无效的是 direct route injection、无限请求、无限重复或使用特权 placement 信息。

### 3.2 victim 和系统

- victim 来自 document-disjoint natural sealed holdout，与 calibration、attack discovery 和历史实验零重叠。
- 目标是 optimized EP prefill backend，固定 engine/backend commit、EP size、expert placement、replica set、chunked-prefill 参数和 scheduler。
- 两个负载 cell：预检后实测容量的 `30%` 低负载负控和 `70%` primary load。排队必须稳定；不允许用开环过载人为放大 P99。
- 不允许 drop attacker、无界 starvation 或修改输入/输出 token 来制造 victim 收益。

## 4. 数据、模型和 split

### 4.1 固定模型

| key | model@revision | 角色 |
|---|---|---|
| `olmoe` | `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5` | E64/top-8 |
| `llmjp` | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055` | E32/top-16 stress model |

两模型分开出 verdict，不按 token、layer、request 或 cell pooling 抢救任一模型。

### 4.2 四类输入

| class | 用途 | 能否让 Gate GO |
|---|---|---|
| `NAT_BENIGN` | 普通自然 victim/cotenant | 是，作基线和良性税负 |
| `NAT_PATHOLOGICAL` | 代码、模板、日志、合法重复结构 | 是，作 false-positive 硬门 |
| `ADV_TEXT` | 只通过公开文本 API 产生的攻击 | 是，作 primary security cell |
| `SYN_ROUTE` | 直接手造 route 的压力夹具 | 否，只能做 code/sensitivity smoke |

当前仓库中“WikiText-103 尚未使用”的历史假设已不可用。formal run 前必须建立新的全仓历史 prompt-hash registry，并固定真正未暴露的 natural/structured dataset revision。

### 4.3 split 和最小数量

- calibration：只选 attack trigger、route threshold、quota/DRF 参数和 service fit。
- sealed evaluation：一旦生成，formal verdict 前不得打开或根据其结果修改任何阈值。
- route census 每个 `model × class × prompt-length` 至少 128 个独立 document/prompt cluster；主长度为 512 和 2048 input tokens，长度严格 matched。
- 正式 P99 cell 需要预先功效分析；不论结果如何，每个 cell 不少于 30 个 paired blocks 和 10,000 个 completed victim requests。
- bootstrap 单位是 paired arrival block，同时按 request/document cluster 保持相关性；不得把 token/contribution 当独立样本。

## 5. Gate-0 证据层级

### G0-Q：资格预检

状态优先级是：授权位未打开时为 `BLOCKED_PROTOCOL_NOT_AUTHORIZED`；授权后任一项缺失仍为 `BLOCKED_MISSING_FORMAL_EVIDENCE`：

1. prior-art 矩阵和冻结的 novelty 主张；
2. threat model、attacker budget、victim workload、arrival 和容量 cell；
3. 两模型、tokenizer、dtype、seed、dataset revision 和 prompt ordered hash-of-hashes；
4. target engine/backend commit、EP size、physical expert-to-rank placement 和 replica snapshot；每个 `(expert_id, target_rank)` 必须验证为 snapshot 中的合法 membership，不假设一个 expert 只能位于一个 rank；
5. tenant-qualified native route producer、预期 token/chunk/layer event manifest 和 exactness runner；
6. 与目标 backend 绑定的 service/full-path evidence，不得从 proposed saving 反推 denominator。
7. hash-bound raw paired blocks、completion ledger 与重算器；必须从原始 `B/A/O/S` 样本重算数量、P99、estimator 和 bootstrap CI，手填 aggregate JSON 不得发出科学 GO/NO-GO。

### G0-R：RTX 5090 route-footprint census

5090 只直接裁决：

- expert concentration；
- 按 `route_observed_us` 的因果时间顺序，在完整 chunk action boundary 上达到前 25% prefill route contributions 后，对剩余 expert footprint 的持续性；
- 在至少 50% 剩余 **service work** 完成前，是否已有可执行的因果观测点。count-based contribution fraction 只是 route census，不得代替 service-weighted work。

单卡根据 placement 推导出的 `target_rank` 只能标记为 exploratory placement-derived replay，不是物理 rank 观测。formal rank 指标必须来自 target backend 的 native executed-dispatch ledger，并同时闭合 `(target_rank, replica_instance_id, device_uuid)`、唯一 dispatch event ID、冻结 placement snapshot 和预期 token/chunk/layer manifest。仅证明 expert 在 snapshot 中可以属于某 rank，不等于证明该次 dispatch 真的在那个物理副本执行。

### G0-D：完整 request-DAG 回放

回放必须包含所有 prefill chunks、layers 和真实资源依赖，且显式计入：

- nonlinear batching/grouped-GEMM service；
- launch、router、dispatch、A2A、expert、combine、collective/barrier；
- batch split/re-batch/lane 税负；
- admission、queue 和 downstream request completion。

只算各 rank 的负载和、只回放 single-layer window 或把 5090 expert curve 直接叫“victim P99”均为 `INVALID_REQUEST_DAG`。任何单卡产生的延迟必须命名为 `REPLAYED_TTFT_P99`。

### G0-S：强 simple baseline gap

只在 G0-D 有 exact legal Oracle 后运行。Gate-0 不实现 Dominant-Expert Fair Queueing controller，只判断是否还存在值得实现的空间。

## 6. contribution 和 route ledger

每个 contribution 的不可变身份至少是：

```text
model_revision
tenant_id / request_id / document_id / prompt_hash
phase / chunk_id / decode_step
token_position / token_id
layer_id / expert_id / topk_slot / gate_weight
placement_id / target_rank
rank_binding_stage / replica_instance_id / device_uuid / dispatch_event_id
```

`topk_slot` 是 router 选中槽位，`expert_id` 是逻辑 expert，`target_rank` 是特定 placement 下的 EP rank，三者不得混用“expert-rank”一词。formal 行还必须标明 `EXECUTED_DISPATCH` 阶段和物理副本/设备身份；但这些字段仍需要 hash-bound native producer 证明来源，不得由离线 placement 脚本自行填充。

合法 action 不得改变 contribution multiset、expert/top-k slot、gate weight、activation、combine 顺序和 output token。完整 exactness 需要：

- routed/dispatched/executed/combined 四阶段 multiset exactly-once；
- hidden/logit 容差和 deterministic argmax/token 一致；
- matched completion set 和 output hash 一致。

只跑离线 replay 而没有真实 model/backend 执行时，“输出一致”是空命题，不能关闭 exactness Gate。

## 7. 两个必须分开的 Oracle

### 7.1 `D`: delete-attacker attribution Oracle

- 删除 attacker 节点及其资源边；
- victim route、arrival、service 和 request DAG 不变；
- 只回答“这部分 victim harm 是否由 attacker 造成”。

`D` 不是可部署策略，不得用它计算 policy 的 Oracle capture。

### 7.2 `O`: future-known legal scheduling Oracle

- 保留全部 attacker work，不 drop、不修改语义；
- 可看到小规模冻结 episode 的未来 route/arrival/service realization；
- 只在真实 backend 支持的 action point 重排：例如未启动 prefill chunk 的 batch partition、lane 和 order；
- 已经 launch 的 wave 不得改写；
- 必须付出 batch split、launch、A2A 和丧失 batching efficiency 的真实税负。

Oracle 必须 exact。超出 solver budget 时输出 `UNSOLVED_EXACT_STATE_LIMIT`，不得用 heuristic 冒充上界。

如果 backend 仍在一个全 batch barrier 后才能让 victim 继续下一层，那么 per-rank/per-contribution WFQ 只是换执行顺序，没有删除 completion-DAG 依赖边；该 action 立即判死。

## 8. 冻结 baselines

所有 baseline 使用同一 arrival、request/completion set、backend、placement、capacity 和 output budget：

1. target backend 的 production/default FCFS + chunked-prefill scheduler；
2. per-tenant request/concurrency quota；
3. per-tenant input-token quota/rate limit；
4. exact repetition detector 和论文中的 PPL-filter 思路，同时测 legitimate code/template false positive；
5. VTC 类 token-cost fairness；
6. FairServe 类 weighted service-counter + throttling；
7. canonical DRR/DRF，resource vector 分别使用 expert 和 physical-rank work；
8. fixed per-tenant capacity partition；
9. RepetitionCurse vulnerability-aware placement 与 dynamic EPLB snapshot；
10. 若可审计复现，Gimbal/strong expert-pressure scheduling。

“ordinary WFQ”不是足够精确的 baseline 定义；必须写明服务单位、resource vector、权重、quantum、admission 和 batching 交互。

## 9. 指标与统计门

对每个模型、负载和 `ADV_TEXT` cell 单独计算。记：

- `B`：attacker 替换为 matched-benign cotenant 时的 victim `REPLAYED_TTFT_P99`；
- `A`：存在 attacker 时的 victim `REPLAYED_TTFT_P99`；
- `O`：future-known legal Oracle；
- `S`：最强 simple baseline；
- `P`：未来 proposed policy，Gate-0 不要求实现。

\[
H=\frac{A}{B}-1
\]

\[
G_O=\frac{A-O}{A}
\]

\[
R_O=\frac{A-O}{A-B}
\]

\[
CH_S=\frac{A-S}{A-O}
\]

\[
G_{PS}=\frac{S-P}{S}
\]

`O` 是 legal action space 的 exact optimum，所以任何合法 `P` 都不得比 `O` 更快。所有 CI 使用 10,000 次 paired-block bootstrap，报告 95% LCB/UCB；formal evaluation 不得改变 estimator、block 定义或符号。

计算细节同时冻结为：

1. request-level P99 使用 nearest-rank，位置为 `ceil(0.99N)`；
2. 每个 replicate 用同一 multiplicity vector 联合重采样 B/A/O/S 的 paired arrival blocks，然后在 replicate 内重新 pooling request 并计算 P99，不对 block P99 取平均；
3. base seed 为 `20260729`，cell seed 由 evaluator version/model/cell 确定性派生；interval 为 percentile two-sided Type-7 95% CI；
4. `S` 只能在 calibration 上选举并在 sealed evaluation 前冻结；
5. goodput 使用 paired-block ratio-of-sums，每块 wall clock 由原始 request 的 `last_completion-first_arrival` 重算，不信任生产者自报 duration；
6. queue stable 由块内 service-work 起点、终点和 arrived work 重算，容许增长上限固定为 2%。

raw capsule 还必须绑定 config bytes、evaluator source bytes、每个 artifact 的 hash/size/row-count 和 config 中的 SHA 字段；自报的 request-world/arrival/victim/budget hash 必须从 request 行重算。详细合同见 [`Raw_Capsule_Contract_v1.md`](Raw_Capsule_Contract_v1.md)。

### Gate-0 通过阈值

每个冻结模型在 primary `70% load × ADV_TEXT` 上必须同时满足：

1. `H` point estimate `>=20%`，95% LCB `>10%`；
2. `G_O` point estimate `>=10%`，95% LCB `>5%`；
3. `R_O` 95% LCB `>=50%`；
4. `CH_S` 95% UCB `<90%`；
5. 合法隔离动作在 `NAT_BENIGN` 上的 goodput loss 95% UCB `<5%`；
6. queue stable，无 drop/starvation，exactness 全通过；
7. `30% NAT_BENIGN` 负载负控与 `70% NAT_PATHOLOGICAL` false-positive control 均不得超过同一 `<5%` goodput-loss 上界；两者的 latency 数字不计作正向 Gate 证据。

任一模型失败，最多保留 `MODEL_SPECIFIC_OBSERVATION`，不得 pooling 成跨模型 GO。

Gate-0 不要求 `G_PS`，因为在现象、legal Oracle 和 simple gap 之前不应实现 controller。只有进入后续 policy Gate 后，才要求 `G_PS` 95% LCB `>5%`、且 `P>=O`。

## 10. fail-closed verdict

### `INVALID`

- tenant/request/contribution identity 不闭合；
- top-k sibling、gate weight 或四阶段 multiset 不守恒；
- placement 未固定却报告 rank 结论；
- service surface 越界外推或未绑定 model revision/dtype/hardware/backend；
- 没有 full request-DAG；
- online policy 看到未来 token/layer route、future arrival 或 future service；
- exactness、completion set 或 queue stability 失败。

`INVALID` 只能修测量，不得解释正负效果。

### `NO-GO / KILL`

- matched-work 后两模型无稳定 route-specific victim harm；
- 只在任意选择的 virtual placement、direct route injection、无限攻击预算或开环过载下成立；
- causal observation 出现时剩余可改变 work `<50%`；
- exact legal Oracle 没有足够 headroom；
- quota/VTC/FairServe/DRF/partition/repetition/placement 中的最强简单方法捕获 `>=90%` Oracle；
- backend 没有能删除 victim-attacker completion-DAG 依赖边的可执行调度点；
- batch split、launch、communication 和 isolation 税负吞掉收益；
- 收益依赖 drop 或无界 starvation attacker；
- 只在 `SYN_ROUTE` 中成立，真实 API adversarial text 不成立。

formal evaluation 打开后，不得改 attack intensity、阈值、模型、负载、split 或 pooling 规则救活方向。

### 唯一正向输出

Gate-0 全部闭合后只输出：

```text
QUALIFIED_FOR_8XA100_EXISTENCE_GATE
```

它不是 `SYSTEM_GO`、不是论文主结果，也不授权把 5090 `REPLAYED_TTFT_P99` 写成真实 serving P99。

## 11. 8×A100 后续存在性 Gate

只有 Gate-0 通过后才进入：

- 真实 optimized EP backend 和冻结 physical placement；
- continuous arrivals、真实 chunked prefill 与可执行 lane/batch action；
- 随机交错 paired blocks；
- 真实 request-level TTFT P99，TPOT P99 只作 secondary；
- per-tenant/total goodput、fairness、batching tax、A2A/collective overhead；
- 同一 harm、Oracle、simple-gap、goodput-loss 和 exactness 门重新通过。

5090 的阈值结果不传递为系统证据。8×A100 失败时直接 NO-GO，不回到 5090 改阈值。

## 12. 当前已知阻塞

1. 没有 tenant-qualified native continuous-prefill producer。
2. 现有 cached-decode capture 是 batch size 1、顺序 request、synthetic arrival，且 metadata 正确标记 non-formal。
3. 现有 route-v3 没有 `tenant_id`、traffic class、prompt family 和隔离域。
4. 没有 target backend/placement snapshot；单卡上的 rank 压力无法直接观测。
5. 没有两租户 matched full-path denominator。
6. 现有 service curves 主要是 isolated expert 且不完整，不能表示 mixed-tenant interference 或 batching loss。
7. BCRD 的当前 Oracle 明确只是 single-layer window，formal Gate 2 因 full request-DAG 缺失而硬返回 invalid。
8. 没有 RouteShield 的 quota/VTC/FairServe/DRF/partition 可审计 baseline，也没有 tenant-aware exact legal Oracle。
9. natural/structured dataset revision、历史 hash registry 和 attack-generator implementation hash 尚未冻结。
10. expected token/chunk/layer schema、tokenizer 绑定、截断检查和 placement/replica membership verifier 已实现，但真实 manifest、tokenizer hash、prompt-bytes/token-ID manifest、placement snapshot 和 tenant route ledger 仍全部未解决。
11. hash-bound raw request/block loader、provenance 重算、P99 和 joint paired-block bootstrap 已实现为 development-only harness；formal flag 仍为 false，且 full-DAG、tensor exactness 和 Oracle certificate 验证器未闭合。
12. 当前的 `EXECUTED_DISPATCH` 合同只能检查行身份、唯一 dispatch ID 和 snapshot membership；尚无 target backend native producer artifact 证明这些行来自真实物理 dispatch。

所以现在的正确输出是“协议、合同与开发态 raw 重算夹具已建立，formal Gate-0 未运行”，而不是用旧 RouteSlack/BCRD 产物填充数字。
