# FJRC Phase 2 Frozen Oracle Pilot

状态：**BLOCKED BEFORE FORMAL IMPLEMENTATION / SUPERSEDED SCIENTIFIC OBJECT / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

> 2026-07-23 实现前对抗审查发现：本版 `R0` 只比 `B` 多 keyed
> receiver queue availability，因而实际测量的是普通 receiver-qmap 在 fork-join
> objective 下的价值，不是 Phase 1 所声称的 per-join sibling-deficit/correlation
> 信息增量。此外 arrival/background/pairing/simple-baseline 的 normative generator 仍未
> 唯一冻结。因此不得使用本文档启动正式 GPU 实验；已写 LUT/core 仅作
> 开发材料。后续若继续，必须新建 `B0 / Q(queue) / J(join phase) / R=Q+J`
> 信息格协议，并使 matched worlds 保持 q-map 相同，只交换可达的 keyed join phase。

## 1. Scientific object

本轮只检验 receiver keyed state 对 request-level fork-join deadline risk 的信息价值，不写
credit controller。起点是 expert output ready 后的 combine-return sender path；固定 route，
不允许 drop/reroute/replica/precision adaptation。

每模型使用 clean-v2 calibration native route 的 request-disjoint split：每个 virtual
receiver 的 8 requests 按 `sha256("fjrc-split" || request_id)` 排序，前 4 只用于选择
arrival/deadline 参数，后 4 为 holdout。holdout 共 32 requests，按结构 hash 配成 16 个
互斥 matched pairs；不补样、不替换 invalid pair。

每 request 选一个 assigned replay layer 上的 token join：
`argmin sha256("fjrc-join" || full_join_identity)`。必须保留该 join 的全部 native top-k
siblings。每 pair 只允许两 request 具有至少两个共同 sender；每个共同 sender 在 t0
拥有两个 ready contributions，分别属于两个 request，action 是先 admit 哪一个。其他
siblings 的 completed/ready/future 状态由冻结 arrival generator 产生并完整 replay，禁止
按 miss/outcome 构造 last-missing。

## 2. Required 5090 LUT

在 RTX 5090 上对 OLMoE hidden 2048/top-k 8 与 LLM-JP hidden 512/top-k 16 测量：

- row-1 BF16 sender pack：冻结 `index_select`，20 warmup + 100 measured；
- row-1 BF16 receiver unpack：冻结 `index_copy_`；queue depth 每个整数 `0..16`，每点
  20 warmup + 100 measured，并保存每个 invocation completion timestamp；
- once-per-join canonical BF16 combine：20 warmup + 100 measured；
- host oracle-policy lookup tax 单列，不进入 zero-tax R0。

raw CUDA-event/wall trials、GPU/runtime/protocol/source SHA、tensor descriptor 与 summary
必须原子保存并由 consumer 重算。depth >16 直接 BLOCKED；禁止插值/外推。

shared-cut 固定：`payload=hidden*2`、descriptor=16、align-up 16B、200 Gb/s analytic L2；
source 精确为 `ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA`。pack、cut、unpack、combine 各计一次；
combine 每 join 仅一次。

## 3. Arrival, deadline and reachable histories

arrival 是合成但预注册的 post-expert-ready continuous trace。selection half 比较
deterministic 与 Poisson 两个冻结 family、rho `{0.6,0.8,0.95}`；primary family/rho 由
selection-half strongest-B 的非全零/非全一 miss 可辨识度选择，tie 用 canonical name。
deadline `D_j = arrival_j + kappa * isolated_critical_path_j`，kappa grid
`{1.25,1.5,2.0,3.0}`；selection half 选择使 strongest-B miss 最接近 50% 的 kappa，tie
取较小值。holdout 不得再改。

两个 matched worlds 共享全部 foreground task/join/sender/receiver identity、release、
deadline、service、future arrivals、action domain 与 sender history。背景 receiver unpack
histories从空状态按冻结 generator replay可达；queued-work/depth多重集相同，只交换其到
receiver identity 的映射。保存 raw histories、replayed availability 与恒等式证明。

背景深度主格点固定为 `(2,8)`；`(1,4)` 与 `(4,16)` 只作 sensitivity；equal-map `(0,0)`
是负对照。两个被交换 receiver 由 pair structural fingerprint 排序取前两名，其他
receiver 使用较小 depth。depth 不得按 oracle gain、miss 或模型单独调整。

## 4. Event model and request accounting

每 sender 非抢占串行：`pack -> analytic cut`。receiver 按 arrival timestamp、task-id tie
break 串行 unpack。全部 siblings unpack 后：

`join_ready=max(sibling_unpack_completion)`；

`join_close=max(join_ready, combine_available_receiver)+combine_us`。

request completion 在本 pilot 精确等于所选 native join 的 close；claim 必须写成
`selected-layer selected-token join deadline risk`，不是端到端 TTFT/TPOT。全部 foreground
任务最终完成，同一 contribution 与 risk saving 不得重复计费。

每 pair 的共同 senders 同时作一次 joint first-admission 决策；随后每个 sender 的第二个
竞争 task 被强制发送，其他 causal events按冻结 replay执行。`IDLE` 因无新决策前信息且
positive service 被 dominance 删除并记录证明。

## 5. Information arms

- `B`：完整 sender queues、released route/bytes/deadline、service LUT、topology、正常 causal
  ACK、receiver queued-work/depth多重集与所有简单-baseline state；不知道 keyed
  `receiver_rank -> availability` mapping。一个 joint action vector必须跨 matched worlds
  相同。
- `R0`：仅比 B 多决策前 causal keyed receiver availability/map；不看未来。
- `C`：full future ceiling，单列且不参与 gate。

由于本 pilot 只有一个 joint decision node，joint nonanticipativity 是完整 action vector
跨 world 共享；之后没有第二个可选 action，不存在把 ACK adaptation错误冻结的问题。
枚举所有共同-sender二元 action vector，并用 permutation-selection MILP 独立核对 B。

## 6. Metrics and frozen gates

统计单位是 32 个 holdout request，不把 siblings 或模型池化扩样本。主指标：

`miss_rate = sum(1[join_close_j > D_j]) / 32`。

次指标：32 个 normalized tardiness
`z_j=max(0,join_close_j-D_j)/max(D_j-arrival_j,eps)` 的 empirical CVaR90；tail 至少 4 个
request，不得换回 CVaR99。再以 mean tardiness、makespan lexicographic tie-break。

### 6.1 Matched-world folding and tied optima

对每个 native holdout request `j`，两个 matched worlds 是等概率反事实状态：

`p_j=(1[miss_j,w0]+1[miss_j,w1])/2`，
`z_j=(z_j,w0+z_j,w1)/2`。

聚合时 `miss_rate=sum_j p_j/32`，`mean_tardiness=sum_j z_j/32`，CVaR90 为 32
个 `z_j` 中最大 4 个的算术平均。world 不得展开为 64 个统计样本。

每个 matched pair 按已冻结的 pair-level lexicographic objective 独立优化。若 B 或 R0
有多个精确同目标最优 policy，聚合时取按 `(sender_rank, task_id, world0,
world1)` 序列化的 canonical-minimum policy；该选择只能看 identity，不得看其他
pair 或 aggregate outcome。flip 仍由 canonicalization 前的最优集合定义：仅当 R0
最优 policy 集为 singleton 且 world0/world1 的 joint first-action vectors 不同时，
`strict_unique_flip=true`。canonicalization 不得把 tie 变成 flip。

### 6.2 Aggregate gates

`actionable_rate=actionable_pairs/16`，`flip_rate=strict_unique_flip_pairs/16`。任一
invalid/missing pair 直接 `BLOCKED`，不得删除或替换。每模型 aggregate gate 为：

- `relative_miss_reduction=(miss_B-miss_R0)/miss_B`，`miss_B=0` 时本门失败；
- `absolute_miss_reduction=miss_B-miss_R0`；
- `cvar_reduction=(CVaR_B-CVaR_R0)/CVaR_B`，`CVaR_B=0` 时本门失败；
- `actionable_rate>=50%`，`flip_rate>=25%`。

equal-map、fanout-1 与 shuffled/uninformative-key 对照都使用固定 16 pairs，并同时要求
aggregate miss gap=0、CVaR gap=0、flip rate=0。两模型 AND 与 simple-baseline
capture<90% 由 runner 在同时拥有两模型和 baseline 结果后判定。

两个模型 AND：

1. exact B-to-R0 miss rate relative reduction >=10% 且 absolute reduction >=2pp；
2. matched-pair unique optimal joint first-action vector flip rate >=25%；
3. CVaR90 normalized tardiness reduction >=5%；
4. 至少 50% pairs 为 ACTIONABLE：存在两个合法 action 产生不同 request miss vector；
5. best simple baseline 对 B-to-R0 gain 的 capture <90%；
6. equal-map、fanout-1 与 shuffled credit-key negative controls gap/flip=0；
7. MILP/enumeration/replay一致，solver gap <=1e-7。

任一失败：`NO_GO_FJRC_RECEIVER_RISK_INFORMATION`。全部通过仅为
`PROMISING_FJRC_L2_INFORMATION_HEADROOM`，尚未证明有成本/stale credit。

## 7. Phase 4 P0 checklist

未来 sibling/service/arrival 泄漏；deadline/arrival在holdout重选；B缺少等价 causal
deadline/service/aggregate state；错误 combine-return direction；join key 缺
request/forward/layer/token/slot/expert/sender/receiver/epoch；branch重复消费risk/slack；
不同 arms 改 task universe；invalid pair替换；B action未跨world联合；tie被伪装成flip；
pack/cut/unpack/combine双计或漏计；candidate=last-missing outcome构造；feedback/tax免费进入
executable claim；负对照非零；32-request denominator漂移；sealed读取。

## 8. Evidence boundary

最高 claim：native route identity + RTX 5090 pack/unpack/combine LUT + analytic cut + synthetic
continuous arrival/virtual receiver replay 中，keyed receiver information 对 selected-join
deadline risk 的 conditional headroom。不是实测 receiver queue、RDMA/NCCL、multi-rank
incast、serving、TTFT/TPOT/P99 或 production benefit。
