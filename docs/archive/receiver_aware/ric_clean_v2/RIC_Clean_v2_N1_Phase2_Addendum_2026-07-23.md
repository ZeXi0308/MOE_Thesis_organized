# RIC-Clean-v2 N1 Phase 2 冻结补充协议

状态：**FROZEN FOR N1 IMPLEMENTATION / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

本文件只细化 `RIC_Clean_v2_Phase2_冻结实验协议_2026-07-23.md` 的 N1。
它不修改已经冻结的 calibration data manifest；N1 producer、实例、解与 lock
必须同时绑定 base protocol、本 addendum 与
`RIC_Clean_v2_ServiceLUT_Phase2_Addendum_2026-07-23.md` Revision 2 的 SHA-256，并且
只允许从已独立验证的双模型 `release_manifest.json` 进入 LUT。sealed、N2 与 RDMA claim
仍未开放。

## 1. 输入与 outcome-blind 选择

- 只允许 clean-v2 calibration manifest、两模型 calibration route、placement 与
  calibration service LUT；禁止读取 sealed 路径。
- 每模型从同一个结构 eligibility pool 选择 16 pairs；排序键为
  `sha256(n1_pair_seed || structural_pair_fingerprint)`。
- eligibility predicate、pool denominator、全部 rank 与未入选原因必须在求解前写入；
  invalid/ambiguous pair 不得跳过补样。
- contribution 身份固定为 `(request_id, layer_id, token_id, token_block_id,
  topk_slot, expert_id, sender_rank, receiver_rank, epoch)`。

## 2. Matched worlds 与可达历史

每个 pair 是两个等概率世界。两世界共享同一实际 route task universe、当前 ready
task IDs、action domain、release、payload、冻结 service tuple、resource demand、placement
与当前 aggregate resource state。禁止交换或重标 join key、request、token、slot、
expert 或 contribution identity。

唯一允许差异是两条不同的 prior receiver history。每条 history 必须从空状态经确定性
replay 可达：release 前不执行；每 contribution 的
`pack -> shared-cut -> unpack` 顺序完整且每 stage 至多一次；missing mask 单调清除；
全部 top-k 到齐后每 join 只 enqueue并执行一次 canonical combine；容量、冻结仲裁与
canonical microstep 顺序成立。expert execution 只形成 ready offset，不作为资源 service
重复计费。该定义由
`RIC_Clean_v2_ServiceLUT_Phase2_Addendum_2026-07-23.md` Revision 2 明确
**SUPERSEDES** 本文件先前的“四段 service / per-contribution receiver-apply”措辞，并与
base protocol 的 `3N+J` full-drain 恒等式一致。replay 终态必须逐字段等于 world
snapshot，并使两个当前 candidate contribution 的 `last_missing_status` 互换。

实例必须保存完整 prior histories、history SHA、replayed keyed join state、current resource
state，以及 task-universe/immutable-fields SHA。下列恒等式任一失败即
`BLOCKED_INVALID_MATCHED_WORLD`：

- current ready task/action domain 相同；
- immutable task fields、route、placement、service tuple 相同；
- sender-local observation history 相同；
- aggregate receiver/qdepth/port/shared-cut observation history相同；
- keyed receiver state 不同且仅 last-missing status 互换。

## 3. 信息投影与联合求解

所有 view fingerprint 必须从 raw history/state 重算，不能相信 producer 的布尔声明。

- `S`：sender-local ready tasks、endpoint、payload/service、age/slack/fairness 与稳定
  tie-break；禁止 keyed join/missing/combine state及其可反查表。
- `B`：在 S 上加入 aggregate receiver qdepth、port/shared-cut state、topology 与
  non-keyed contention；仍禁止 per-join state。
- `R0`：允许当前零延迟 keyed receiver join state；禁止未来 release/service/state。
- `C`：可用未来信息，仅作 ceiling，不参与 N1 gate。

`S/B` 必须在两个 world 的单个联合 MILP 中等权求解。对整个决策树中 observation
fingerprint 与 action domain 相同的节点，所有 action variables 跨 world 相等；不能只
耦合 first action，更不能逐 world 单独求优。`R0/C` 仅能按相应合法 view 拆 observation
node。

## 4. 目标、CVaR 与 flip

严格 lexicographic：先最小 closure-budget violations，再最小 joint empirical
CVaR99，最后最小 mean closure。每一级最优值固定后才能进入下一级，禁止加权和。

CVaR99 对两个 world 完整 drain 后的全部 token-block closure latency 等权合并，使用
Rockafellar--Uryasev 线性化：`u_j >= L_j-z, u_j >= 0`，
`CVaR99 = z + sum(u_j)/(0.01*M)`。B/R0 的 causal samples 与权重必须相同；P99 或
top-k mean 不能替代。`CVaR99_B <= 0` 直接 BLOCKED。

每 pair 的 normalized gap 为 `(CVaR99_B-CVaR99_R0)/CVaR99_B`；16 个值的 median
是排序后第 8、9 个值的算术均值。

R0 first-action flip 必须用 fix-and-resolve 证明：先获得完整 lexicographic optimum，
再逐个固定 candidate first action 重求三阶段目标；只有两 world 的 optimal-action set
均为单例且单例不同才计 flip。任一 world 多解则按 base protocol
`BLOCKED_AMBIGUOUS_FIRST_ACTION_OPTIMUM`，不得换 pair。

## 5. Solver 与独立重放

每个 information arm 和 lexicographic stage 记录 solver/version/parameters/seed/
threads/status/primal/bound/absolute-gap/relative-gap/time/optimum。必须 `OPTIMAL`；每级
relative gap `<=1e-6`，零目标时 absolute gap 也 `<=1e-6`；禁止接受 time-limit feasible。
独立 replay 必须复算 action sequence、latency、violations、CVaR 与 mean，并与 solver
一致。

## 6. 最小产物

- `matched_world_pairs.jsonl`：pair rank/pool、task universe、两条 history、replayed
  states、S/B fingerprints、所有 invariant、base/addendum/config/data/route/LUT hashes；
- `milp_solutions.jsonl`：joint-world 标志、observation partition、nonanticipativity
  matrix hash、三阶段 solver records、CVaR/mean/violations、optimal first-action sets 与
  independent-replay hash；
- `oracle_status.json`：完整 pair denominator、两模型 summary、所有 BLOCKED reason；
- `lock.json`：仅在两模型各 16 pair、median gap `>=0.05`、flip rate `>=0.25` 且所有
  solver/replay gates 通过时写 `N1_GO_LOCKED`。

## 7. 一票否决

以下任一项为 P0：outcome-dependent pair selection；invalid pair 补样；不可达 history；
identity 重标；两 world task/action/immutable state 不同；S/B view 泄漏 keyed state；
R0 使用未来；S/B 分 world 求解；nonanticipativity 未覆盖后续相同 observation nodes；
CVaR sample/权重/full-drain 错；B/R0 world 不同；未 fix-and-resolve 证明唯一 flip；solver
非 OPTIMAL 或 gap 超门；lexicographic 未逐级冻结；无独立 replay；任一模型不足 16
pairs；ambiguous pair 移出分母；median/flip/model-AND 算错；读取 sealed；B/R0 间改变
route/LUT/placement/payload/arrival；用 C 替代 B→R0 gate。
