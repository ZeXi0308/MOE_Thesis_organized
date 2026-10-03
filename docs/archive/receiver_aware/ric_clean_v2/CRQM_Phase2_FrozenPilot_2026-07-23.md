# CRQM Phase 2 Frozen Exploratory Pilot

状态：**FROZEN FOR PILOT IMPLEMENTATION / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

## 命题

`Cross-Receiver Queue-Map (CRQM)` 检验：在同一 expert sender 的共享 combine-return cut
上，若 sender 可见任务、route、service 与 receiver queue-work 多重集相同，仅
`receiver_rank -> current queued work` 的映射不同，零延迟 receiver queue map 是否会
改变唯一最优 first admission，并相对不知道该映射的联合最优基线降低 contribution
receiver-apply completion CVaR99。

本轮是新的 early receiver-congestion existence pilot。它不复用已判死的 keyed
post-arrival join-tail claim，也不验证 controller、RDMA 或 serving。

## 输入与方向

- 仅使用 clean-v2 calibration native route；禁止读取 sealed。
- 通信方向固定为 expert-owner `sender_rank` 到 token-owner `receiver_rank` 的
  combine-return contribution。
- 每个窗口含同一 sender 的 6 个 ready contributions，至少覆盖 3 个不同 receiver；
  task、join、expert、sender、receiver 与 epoch 身份不得重标。
- calibration/holdout 按 request cluster 先切分；窗口按结构 fingerprint 排序选择，
  不读取 outcome。

## 5090 校准

对 OLMoE `(hidden=2048, top-k=8)` 和 LLM-JP `(hidden=512, top-k=16)` 测量冻结 BF16
receiver primitive：row-1 BF16 unpack（`index_copy_`）。queue depth 固定为
`0,1,2,4,8,16`；每点 20 warmup + 100 measured CUDA-event trials。记录 median/p95/max、
wall time、GPU identity 与 producer hash。主 pilot 用 depth 2 与 8 的 measured median
queue-drain work；depth 0 是负对照，1/4/16 仅 sensitivity。

这些是单 GPU serial CUDA-stream virtual receiver 测量，不是 NIC/RDMA 或多 rank queue。

**实现前会计澄清：** `depth=d` 的 CUDA event 精确包围 `d` 个在候选 task
到来前已排队的 receiver primitive，是 backlog-only drain，**不包含候选 task
自身**。raw 保留 depth-0 的真实 CUDA-event/wall harness 读数；summary 的唯一
consumer 字段为 `backlog_only_queue_work_us`，depth 0 强制为 `0.0`，depth > 0
取该 depth measured CUDA-event median。pilot runner 另行且仅一次计入 candidate
的 shared-cut finish，并以 depth-1 primitive median 计一次 candidate receiver-unpack service；
禁止将 candidate 加入 backlog event 或重复计费。

shared-cut 主口径冻结为：`payload_bytes=hidden*2`（BF16 row-1），
`descriptor_bytes=16`，`transport_bytes=align_up(payload_bytes+descriptor_bytes,16)`，
`cut_service_us=transport_bytes*8/200000`，带宽单位为 Gb/s、时间单位为 us，
`source=ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA`。100/400 Gb/s 不进入本 pilot verdict，也不在
结果后补 sensitivity。

## Matched worlds 与信息投影

每窗口构造两个可达 queue histories：从空 receiver stream 开始，按冻结深度 enqueue
相同 primitive。两世界共享全部 current tasks、action domain、service、sender history、
每 receiver aggregate queue depth 多重集与总 queued work；唯一差异是将 depth-2 与
depth-8 histories 在两个 receiver 身份间交换。history 和 replayed queue map 必须落盘。

- action 是从当前 ready set 选择一个 contribution 进入共享 cut，即
  `ADMIT(task_id)`；执行后继续选择，直到 full drain。`DEFER` 在所有 task 已 ready、
  无新信息到达的本 pilot 中被 dominance 删除，并保存该证明。
- `B`：看到完整 sender view、拓扑、queue-depth/work 多重集，但看不到 keyed
  receiver mapping；在两世界使用同一个 admission sequence。
- `R0`：额外看到决策前已经存在的 keyed receiver queue map，可按 world 选择
  permutation；禁止未来 arrival/service/state。

本 pilot 没有新的外生 observation，因此 joint nonanticipativity 等价于 B 的
单个跨世界 admission permutation。每个 permutation 均由完整 `6!` enumeration 计算；另用
permutation-selection MILP 求 B，并与 enumeration 精确核对。

## 会计与指标

receiver backlog 从 `t=0` 开始并行排空。令 candidate 的 shared-cut arrival 为 `a_i`，
该 receiver 当前 availability 为 `A_r`，则 contribution completion proxy 为
`c_i=max(a_i,A_r)+receiver_service_i`，随后更新 `A_r=c_i`。禁止使用
`a_i+initial_queue_work`，也禁止对同一 receiver 的多个 candidate 重复附加初始 backlog。
这里的 `c_i` 只表示该 contribution 的 receiver unpack/apply completion；没有 missing
mask、last-missing 或 combine claim。两个 world 的全部 task 等权合并计算
empirical CVaR99，再以 mean completion 作 lexicographic tie-break。所有 arms 使用同一 task
和 full-drain 分母；receiver work 只计一次。

主门槛按两个模型 AND：

1. holdout 8 windows 的 median `(CVaR99_B-CVaR99_R0)/CVaR99_B >= 5%`；
2. 两 world 唯一最优 first-action singleton 不同的比例 `>= 25%`；
3. MILP status optimal、gap `<=1e-7`，且 replay/enumeration 一致；
4. depth-0 负对照的 gap 与 flip 必须均为 0；
5. depth 1/4/16 sensitivity 不得产生负 median gap。

任一失败：`NO_GO_CRQM_L2_EARLY_RECEIVER_CONFLICT`。全部通过也只记
`PROMISING_CRQM_L2_HEADROOM`，不得直接进入 RDMA/bandit；下一阶段仍需实现并计费
receiver feedback。

## 一票否决与停止规则

- sender/receiver 方向错误、receiver 少于 3、跨 sender 拼窗、identity 重标；
- keyed mapping 泄漏给 B，或 R0 使用未来；
- queue work 不是来自本轮 5090 raw measurement；
- 两世界 task/service/action/multiset 不同；
- invalid window 被替换、删分母或 outcome-dependent 选窗；
- producer summary 无法从逐窗口记录重算；
- 通过扩大 synthetic depth、增加 seed、放宽 flip/gap 或只报告正 oracle gap 抢救。

## Claim boundary

最高证据为：native MoE route identity + RTX 5090 receiver primitive calibration +
virtual receiver queue replay 下的 early-information headroom。禁止称为真实 receiver
queue、RDMA/NCCL、多节点、TPOT/P99 或 production benefit；`scientific_result=false`。
