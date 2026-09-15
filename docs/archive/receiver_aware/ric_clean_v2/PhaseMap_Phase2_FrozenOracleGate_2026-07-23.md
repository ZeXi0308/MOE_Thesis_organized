# PhaseMap-MILP Phase 2 Frozen Oracle Gate

状态：**FROZEN FOR IMPLEMENTATION / NO SCIENTIFIC RESULT**  
版本：`phasemap-v1`  
日期：2026-07-23

## 1. Scientific object

本轮只检验 queue map `Q` 与 join phase `J` 的纯交互信息上限。决策点为 expert
output ready 后的 combine-return first admission；固定 native route、bytes、precision 和完成
集合，不允许 reroute/drop/replica。

最高 claim 为：native route identity + RTX 5090 primitive LUT + analytic cut + causal
synthetic receiver replay 中，`Q×J` 对 selected-token selected-layer join deadline risk 的条件性
headroom。不是真实 RDMA/NCCL/incast/serving/TTFT/TPOT。

## 2. Data, split, selected joins and pairs

每模型仅读 clean-v2 calibration native routes。每 receiver 的 8 requests 按 canonical
JSON UTF-8 编码的 `sha256(["phasemap-v1-split",request_id])` 排序；前 4 为 selection，
后 4 为 holdout，各 32 native requests。不读 sealed namespace。

每 request 在 assigned replay layer 上选：

`argmin sha256(["phasemap-v1-join",full_join_identity])`。

`full_join_identity` 包含 model/revision/request/forward/layer/token_position/epoch 以及全部
top-k sibling `(slot,expert,sender,receiver)`。必须保留全部 native top-k siblings。

pair eligibility 只看 route structure：两 request 的 output receiver 不同；至少两个共同
sender；用结构 hash 固定两个 decision sender 后，每 request 仍至少有 4 个
phase-carrier siblings。eligibility graph 上取 lexicographic-canonical perfect matching：边按
`sha256(["phasemap-v1-edge",min(reqA,reqB),max(reqA,reqB)])` 排序，通过依次尝试包含边并使
剩余图仍存在 perfect matching 得到唯一解。selection/holdout 任一侧不存在 16-pair
perfect matching 即 `BLOCKED_ROUTE_SUPPORT`，不得补样、换 join 或删除 pair。

## 3. Frozen 2x2 worlds and reachability

每 pair 有四个等概率 world：`(q0,j0),(q0,j1),(q1,j0),(q1,j1)`。两个 request
按 request-id 排序命名 A/B，两 output receiver 同理命名 RA/RB。

- q0：RA=low、RB=high；q1：RA=high、RB=low。
- j0：A=near、B=far；j1：A=far、B=near。
- primary unfinished depth `(low,high)=(8,16)`；sensitivity `(8,12)`；equal-Q `(8,8)`。
- near 恰好有 1 个 phase carrier 未 commit；far 恰好有 4 个 phase carriers 未 commit。
- carrier 按 `sha256(["phasemap-v1-carrier",full_sibling_identity])` 排序选定，不得看
  deadline、last-missing、service 或 outcome。

共同 sender 按
`sha256(["phasemap-v1-decision-sender",pair_key,sender_rank])` 排序，取前两个参与
joint first-admission。若某 request 在该 sender 上有多个 sibling，按
`sha256(["phasemap-v1-decision-contribution",pair_key,request_id,full_sibling_key])`
取最小者作 decision contribution。每 sender 在 t0 同时拥有 A/B 各一个
ready contribution，选定 first 后 second 强制紧随执行。其他 siblings 已在 t0 前发出；
sender 只看到 send-complete，不获得 receiver commit ACK。

未 commit carrier 作为 receiver FIFO 中的 foreground unpack job。背景 row-1 unpack jobs 补齐到
q-bit 规定的总 depth；因所有 job 使用同一模型的冻结 row-1 unpack service，固定
q-bit 翻转 j-bit 后，每 receiver 的 depth、unfinished-work 和 availability 必须逐位一致。
固定 j-bit 翻转 q-bit 后，每 request 的 committed/queued sibling bitmap 必须逐位一致。

决策时刻固定 `t0=0`。设该模型重算后的 row-1 unpack median 为 `s>0`，目标
unfinished depth 为 `d`。对该 receiver 的 `d` 个 unfinished jobs（foreground carrier
与 background 合集）按 `sha256(["phasemap-v1-fifo",world-invariant-job-key])` 排序，令
`eps=s/(4*(d+1))`，第 `i=0..d-1` 个 job 的 arrival 为
`-s/2-(d-1-i)*eps`。因最早 job 在 t0 时尚未完成，该集合恰好有 `d` 个
unfinished jobs。background job key 只由 `(model,pair,receiver,q-bit,j-bit,ordinal)`
确定，不得读决策或 outcome。

已 commit carriers 按 carrier structural hash 排序，arrival 为
`-s*(64+carrier_count-ordinal)`，必须在 unfinished set 的最早 arrival 前全部完成。
不得通过调整单个 job service/arrival 使其刚好变成 last-missing。

所有 background/foreground pre-t0 jobs 从空 FIFO 通过负 arrival timestamp 完整 replay，最后一个
job 在 t0 前到达但在 t0 后完成。产物保存 raw arrivals、FIFO ledger、commit/unfinished census、
sender-history hash、Q/J observation hash 和 reachability certificate。任一不闭合即
`BLOCKED_PHASE_NOT_RECEIVER_PRIVATE`，不得直接手写 bitmap。

## 4. Service and stage accounting

RTX 5090 每模型测 row-1 BF16 sender pack、receiver unpack 及 once-per-join canonical
combine，20 warmup + 100 measured，保存 raw CUDA event/wall trials、tensor descriptor、GPU/runtime/
protocol/source SHA。consumer 必须重算 summary 与 self-hash。

cut 仅作 analytic L2 proxy：`payload=hidden*2`、descriptor=16B、16B align、200Gb/s，来源
必须写 `ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA`。每 post-t0 decision contribution 各计一次
pack/cut/unpack，每 join 仅计一次 combine。pre-t0 commit 不得重复计费，unfinished receiver
job 只计 t0 之后的 residual service。

receiver 按 `(arrival_timestamp,full_sibling_identity)` FIFO；join 在全部 native siblings commit 后 ready，
同 receiver 的 combine 按 `(join_ready,full_join_key)` 串行。request completion 精确定义为该
selected join close。

## 5. Deadline

`D_j = arrival_j + kappa * isolated_full_join_critical_path_j`，`kappa` grid 固定为
`{1.25,1.5,2.0,3.0}`。isolated path 是在空 sender/receiver、无 background、同一 stage ledger 下
该 request 的全部 native siblings 在 t0 ready，同 sender 按 full-sibling identity 排序时，full selected
join close 减 t0。记该值为 `L_j`，冻结 request arrival 为 `arrival_j=-L_j`，因而
deadline 为 `(kappa-1)*L_j>0`。四世界不得改变 arrival/deadline。

只在 selection artifact 上，以 pooled 两模型 `B0` miss 最接近 50% 选一个全局共用
kappa；tie 取较小值；选择模块不得读 Q/J/R gain。两模型的 selection B0 miss 都必须在
`[20%,80%]`，否则 `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`。冻结 selection manifest 的 source/data/
route/LUT/config hash 后，holdout runner 以该 manifest 为唯一决策输入；可同时读取原 selection
bundles 做完整性 replay，逐项重算 B0 rows 与已冻结 linear artifact 的网格最优性，但不得调用
fit/tuning API、不得返回新权重或替换 `selected_kappa`。完整性 replay 与 manifest 任一字节不一致即
BLOCKED。相邻 kappa 只作 robustness，要求 miss gain 不为负。

## 6. Information arms and exact optimization

- `B0`：见 sender-local ready queues、native route/bytes/deadline/service/topology 以及 Q/J 多重集；
  不见 keyed mappings，四世界共用一个 action。
- `Q`：额外见 `receiver -> depth/work/availability`；同 q-bit 的两世界共用 action。
- `J`：额外见 `full_join_key -> committed/queued sibling bitmap/deficit`；同 j-bit 的两世界共用
  action。
- `R`：同时见 Q/J，可按四个联合 observation 选 action。
- `C`：见 full future，只作 ceiling。

action space 是两个 decision sender 各从 A/B 中选 first，共 4 个 joint actions。每 pair/
arm 按其 observation partition 枚举全部 deterministic policies，并用 MILP 独立核对。pair-level
lexicographic objective 固定为：

1. 四 world expected miss count；
2. 四 world expected normalized-tardiness sum；
3. 四 world expected join-close sum；
4. canonical serialized policy identity。

不得用 pair-local CVaR 选 policy，不得为全局 CVaR 在 tied pair optima 中事后挑 policy。

## 7. Folding, metrics and gates

对每 native request：`p_j=sum_w miss(j,w)/4`，
`z_j=sum_w normalized_tardiness(j,w)/4`。统计单位始终为 32 native requests：

- `miss_rate=sum_j p_j/32`；
- `CVaR90=32 个 z_j 中最大 4 个的均值`；
- `mean_tardiness=sum_j z_j/32`。

世界、siblings、kappa sensitivities 都不得扩充样本量。`best_single` 为 Q/J 中按冻结的
aggregate lexicographic comparator 较优的 exact arm：`miss_rate -> CVaR90 normalized
tardiness -> mean normalized tardiness -> mean join-close -> arm identity`。因此如 Q/J 一个
miss 更低而另一个 CVaR 更低，仍以 miss 优先；只在 miss 完全相同后比较 CVaR。该 comparator
同时用于 miss 与 CVaR 两项增益的共同参照，禁止为每个指标分别事后挑选 Q/J。

两模型 AND：

1. R 相对 best_single miss relative reduction >=10% 且 absolute >=2pp；
2. CVaR90 normalized tardiness reduction >=5%；
3. actionable pairs >=50%；
4. dual-conditioned strict interaction flip >=25%：每个计入 pair 必须同时存在固定 q 换 j
   与固定 j 换 q 的 singleton-optimal joint first-action flip；tied optimum 不算；
5. strongest simple baseline 对 `best_single -> R` oracle gain 的 capture <90%；
6. 相邻 kappa sensitivity 的 miss gain 不为负；
7. negative controls gap/interaction flip 为 0；
8. enumeration/MILP/replay 一致，solver gap <=1e-7。

失败：`NO_GO_PHASEMAP_QUEUE_JOIN_INTERACTION`。全过：
`PROMISING_L2_QUEUE_JOIN_INTERACTION_HEADROOM`。

## 8. Frozen simple baselines and controls

全部 baseline 只见 causal R information，tie 均按 canonical task identity：

1. request FCFS；2. EDF；3. qwork-first；4. remaining-siblings/last-missing-first；
5. least-laxity；6. lexicographic `(slack,remaining_siblings,receiver_work)`；
7. selection-only 拟合的 separable linear `slack+qwork+deficit`；
8. myopic predicted-join-close greedy（不看 future）。

separable-linear 三个 feature 在每个 decision observation 的 4 个 ready tasks 内分别作
min-max normalization；常数 feature 映射为 0。weight 限定为非负 simplex，网格为
`(w_slack,w_qwork,w_deficit)=(a,b,4-a-b)/4`，其中 `a=0..4`、
`b=0..(4-a)`。只在 selection 上按 miss、tardiness-sum、join-close-sum、weight tuple
的 lexicographic objective 选一组全局 weights；落盘 grid/source/example hash 后 holdout 只读该
冻结选择；允许只作 exact integrity replay，不允许重新拟合或采用 replay 产生的替代选择。

capture 的分母是 `best_single -> R` exact miss gain；分母 <=0 直接失败；baseline 比
best_single 差时 capture 按 0 计但保留 raw negative value。

负对照：

- equal-Q `(8,8)`：R 必须等于 J；
- equal-J：两 request 都 near，R 必须等于 Q；
- fanout-1：J 增量和 interaction 为 0；
- no-conflict：无共同 decision sender/无限 credit 时所有信息 gap 为 0；
- shuffled/uninformative join key：J 与 interaction 为 0。

## 9. Implementation acceptance tests

1. 四 world 的 native request/join/full-sibling/action census 一致。
2. 固定 q 翻 j 后 Q observation byte-identical；固定 j 翻 q 后 J observation byte-identical。
3. B0/Q/J/R observation classes 数分别为 1/2/2/4。
4. 改 post-t0 future 而保持 pre-t0 observation，决策 action 不变。
5. j-world 间 sender send/ACK history byte-identical。
6. 删除/重复 sibling、commit、pack、cut、unpack、combine 任一项即失败。
7. carrier/pairing 不能读 outcome、deadline、service 或 last-missing。
8. 输入行顺序扰动不改变 canonical perfect matching。
9. holdout runner 不得调用 selection fit/tuning API；只读 integrity validator 可重放 selection
   examples 并证明已冻结结果仍是完整网格的唯一 canonical optimum，但不得生成替代选择。
10. 四世界折叠后 request count 严格为 32；128 world rows 作独立样本必须拒绝。
11. tied policy 不得制造 interaction flip；pair policy 选择不得读 aggregate CVaR。
12. equal-Q/equal-J/fanout-1/no-conflict/shuffled-key 精确通过。
13. receiver key 必须是 output receiver rank；换成 expert/sender identity 必须失败。
14. LUT raw-summary/source/protocol/self hashes 、depth census 和 stage accounting 全部重算。

## 10. Required artifacts

`selection_manifest.json`、`holdout_instance_manifest.json`、`lut.json`、`per_pair.jsonl`、
`per_request.jsonl`、`baseline_results.json`、`controls.json`、`milp_crosscheck.json`、
`decision.json`、`environment.json`、`source_manifest.json`、`summary.md`。所有文件不覆盖历史目录。
