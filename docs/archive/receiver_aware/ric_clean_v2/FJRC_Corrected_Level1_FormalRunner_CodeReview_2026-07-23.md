# Corrected FJRC Level-1 Formal Runner Code Review（2026-07-23）

## 结论先行

`CPU LOGIC APPROVED / NATIVE CPU DRY RUN BLOCKED / GPU RUN NOT APPROVED`

本轮已修完 formal runner 的已知阻塞性代码问题；mock CPU 测试、回归、编译、产物闭包均通过。
但是本地没有真实 clean-v2 route/LUT artifact，远端 SSH 在认证前后立即关闭，且本机缺少 torch，
因此尚未完成 native artifact CPU dry run 和 GPU-specific smoke tests。不得启动正式 GPU capture，
不得把 mock 结果解释为科学结果。

## 1. Scientific question、mechanism 与 claim 边界

### Scientific question

在 exact receiver queue state `Q` 相同的条件下，keyed fork-join completion state `J` 是否能为
下一个 receiver admission 决策提供可重复的增量价值？

Primary estimand：

`Q-only exact policy risk - (Q+J) exact policy risk`

### System mechanism under test

- 一个 receiver-local、全局 credit `B=1` 的 next-contribution admission node；
- action 是两个不同 sender、不同 join 的 ready contribution 中选择一个先发；
- `Q` policy 在 matched worlds 中必须采取同一 action；
- `Q+J` policy 允许依据 completed-sibling identity 分支；
- pack、cut、unpack、combine 使用显式四阶段事件账本。

### Measurable hypothesis

对每个模型独立要求：

1. holdout request miss-rate 绝对下降至少 5 percentage points；
2. matched-pair bootstrap 95% CI 下界严格大于 0；
3. 16 个 matched pairs 中至少 4 个具有唯一、严格的 world-dependent action flip；
4. OLMoE 与 LLM-jp 必须分别 PASS，不允许 pooled rescue。

### 最大可形成的当前 claim

即使 PASS，也只能声称：

> 在 native MoE route identity 上构造的、selection-calibrated logical timing workload 中，
> join completion state 在固定 receiver queue state 后仍存在条件性调度价值。

不能声称真实网络 headroom、TPOT/P99、NCCL/RDMA 收益、生产调度收益或真实到达分布下的收益。

## 2. 代码执行链路

1. `load_verified_joins()`：验证 data/placement/parity/signoff/route trace 的 hash 与笛卡尔积闭包。
2. `load_service_lut()`：重算 primitive LUT summary，验证 RTX 5090、模型和 analytic-cut 边界。
3. `request_split()`：每 receiver 8 requests，确定性拆成 4 selection + 4 holdout。
4. `select_split_scenarios()`：每个 split 固定生成 16 pairs / 32 request-disjoint requests。
5. `calibrate_deadline_on_selection()`：只读取 selection 的 exact-Q miss rate；禁止读取 R outcome，
   从冻结 grid 中选取最接近 50% 且落在 `[20%, 80%]` 的 deadline factor。
6. `materialize_replay()`：保留 native task/join/sender/receiver identity；生成明确标注为 synthetic 的
   arrival、deadline 和 background workload。
7. `receiver_state()`：从 common prior history、swapped prior identity 和 background jobs 重建 `Q`；
   两个 world 的 `Q` 必须逐字段相同。
8. `simulate()`：执行 `B=1` admission，并生成 pack/cut/unpack/combine ledger 与 request outcome。
9. `optimize_information()`：精确枚举 Q 与 Q+J 的 first action policy。
10. `evaluate_baselines()`：FCFS、EDF、least-laxity、SRPT、projected-finish、join-credit 共用同一引擎。
11. `aggregate_campaign()`：16 pairs 聚合为 32 个 distinct native requests。
12. `paired_bootstrap()`：以 matched request pair 为 cluster 重采样。
13. `decide_campaign()` / `decide_two_model()`：执行冻结 gate 与双模型 AND。
14. `write_artifacts()`：非覆盖式生成 config、metrics、raw results、environment、source manifest、
    stdout 和 summary。

## 3. 已确认正确的关键实现

- selection/holdout 为 32/32 requests、零重叠；
- pair selection 不读取 latency、miss、R-Q 或任何策略 outcome；
- selection deadline calibration 只调用 `optimize_q_only()`；
- calibration 若找不到非退化 Q-risk 区间会 fail closed；
- holdout Q miss 全 0 或全 1 附近会返回 `INVALID_WORKLOAD_IDENTIFIABILITY`，不会当作科学 FAIL；
- common completed siblings 在两个 worlds 中完全一致，只交换一个同 sender/receiver/service sibling；
- future work/resource multiset 在两个 worlds 中完全相同；
- prior ledgers 逐条串行，且全部结束于 background arrival 和 `t0` 之前；
- `Q` map 在两个 worlds 中逐字段相等，否则 fail closed；
- Q-only baseline 不能依据 world/J 分支；
- 四阶段 task census 和 join combine census 精确；
- `projected_finish` 不再重复计入 candidate service；
- aggregate 严格要求 16 unique scenarios、32 unique requests；
- bootstrap unit 是 matched pair，而不是把两个 world 当独立样本；
- 双模型决策严格使用 `OLMoE AND LLM-jp`；
- 输出目录存在时拒绝覆盖；
- source manifest 绑定代码 hash、LUT hash、validated route metadata 和证据边界；
- JSON/YAML 写入禁止 NaN/Inf；
- runner 明确记录 `cuda_execution=false`、`gpu_measurement=false`。

## 4. 本轮发现并修复的问题

### 阻塞性问题（已修复）

1. 初版将 pack+cut+unpack 合并为一个 service；现已拆成四阶段 ledger。
2. 初版直接输入 `q-map=0`；现从 prior/background receiver events 重建。
3. 初版两个请求 deadline 相同；现 request arrival/deadline 异质且确定性生成。
4. 初版多数 siblings 在 `t0` 后到达，结构性抹掉 J 的关键路径价值；现 common siblings 在两个
   worlds 中共同完成，只保留 two candidates + one symmetric future sibling。
5. 初版多个 prior events 可能重叠；现生成严格无重叠的 prior ledger。
6. 初版默认 workload 在 mock campaign 中 32/32 miss；现用 selection Q-only calibration，
   holdout 不用于调参。
7. 初版 `projected_finish` 重复计 candidate service；已修复。
8. 初版没有完整 artifact writer 和 source binding；已补齐。

### 仍存在的高风险边界（不是代码静默错误）

1. clean-v2 route trace 没有真实 wall-clock arrival/deadline；本实验仍是 native identity + synthetic timing。
2. `B=1` global admission credit 会放大排序权，尚未证明 `B>1`、continuous batching 下仍有 headroom。
3. cut 来自 200 Gbps analytic proxy，不是 NCCL/RDMA measurement。
4. stage service 使用 model-level median，未包含 expert、message size、queue depth 和 contention variance。
5. combine 与 unpack 在逻辑上使用独立资源，尚未建模 GPU SM/HBM 竞争。
6. primary gate 使用 exact R oracle；implementable `join_credit` 仅作为 baseline，尚未成为 pass gate。
7. negative controls 是信息分区/事件账本的 structural sanity check，不是独立真实 workload。
8. selection-calibrated deadline 回答“非退化 SLO 下是否存在信息价值”，不能估计自然流量中的收益频率。

## 5. Baseline 公平性

所有 baseline：

- 消费同一 pair、world、stage service、arrival、deadline、background 和 task universe；
- 只改变 first-admission action；
- 使用相同 deterministic tie-break；
- 不允许修改 future arrivals 或 completed-sibling truth；
- Q-only baseline 在两个 matched worlds 中必须选择同一 action；
- join-credit 是唯一可见 J 的可部署启发式；
- exact Q 与 exact Q+J 是 information-value oracle，不冒充在线算法。

## 6. 数据泄漏与统计审查

- request split 在任何 timing outcome 前冻结；
- pair selection 只读取 route structure；
- deadline factor 只用 selection exact-Q risk 校准；
- R outcome 不进入 selection calibration；
- holdout request 完全不参与 parameter selection；
- primary risk 按 request 折叠两个 matched worlds；
- bootstrap 按 matched request pair cluster 重采样；
- CVaR90 在 32 requests 上取 worst `ceil(10%) = 4`；
- 不允许跨模型 pooling；
- mock fixture 只用于逻辑验证，不计入 scientific result。

## 7. CPU / 小样本 dry run

### 自动测试

- 61 tests passed；
- 3 tests skipped：本机无 torch，均为 route GPU tensor/parity 相关测试；
- `py_compile`：PASS；
- `git diff --check`：PASS；
- CLI `--help`：PASS；
- artifact bundle：7/7 files，48 JSONL records，全部可解析；
- overwrite protection：PASS。

### Mock selection-to-holdout dry run

- selection 选择 deadline factor：`0.75`；
- selection Q miss rate：`0.50`；
- holdout Q miss rate：`0.53125`；
- holdout R miss rate：`0.50`；
- Q-R absolute reduction：`0.03125`；
- strict flips：`2/16`；
- bootstrap 95% CI：`[0.0, 0.078125]`；
- frozen decision：`FAIL`。

这是预期且有价值的负 dry run：它证明 gate 不会因“代码能构造 action flip”而自动 PASS。
该结果不是模型实验结果。

## 8. Source SHA-256

- `fjrc_corrected_level0.py`：
  `4f88d97ca6110a8e66e09f9e3cdd265fca0e9570190c987d620e4c97dba79611`
- `fjrc_corrected_level1.py`：
  `0fe27fe06ba7dfa58b2262c08e352246be00339e8923a16c595abcd560831faf`
- `fjrc_corrected_replay.py`：
  `fa2de0a871896c5a168184d9841d9794d15e665b903846703702233e580fdcf9`
- `run_fjrc_corrected_level1.py`：
  `6505c8c4d7bd09ecfd15a42628df8bf24d82afa1336ef403e89b60558a9027a7`
- `test_fjrc_corrected_replay.py`：
  `0c6231444d3febb662a9272e37c80220b157bdd2abd964b726be0d7b75475e65`
- `test_run_fjrc_corrected_level1.py`：
  `ce6d4d93a5a7dadfb3e682e1f977d7cae521a81387446fa12bcc45f63bf45f8a`

## 9. 必须修改项与建议修改项

### 必须修改项

代码逻辑层面当前无已知未修 blocker。

进入 GPU capture 前仍必须完成：

1. 恢复可用 SSH；
2. 在远端 torch/CUDA 环境补跑本机 skipped 的 3 个 tests；
3. 定位或重新生成 OLMoE、LLM-jp 的 validated native route artifacts 和 primitive LUT；
4. 对两个模型分别执行 native artifact CPU dry run；
5. 检查 7 件输出、32 request denominator、Q-map equality、calibration audit 和所有 negative controls；
6. 重新生成 source SHA 并确认远端代码与 review source 完全一致。

### 建议修改项（不阻塞 Level-1 existence gate）

1. 增加固定 background depth `{0, 2, 4}` 和 service factor `{0.5, 1.0, 2.0}` sensitivity sweep；
2. 把 implementable join-credit 相对 strongest Q baseline 的差值加入下一阶段 gate；
3. 扩展 `B={1,2,4,8}`，检查收益是否只存在于人工单 credit；
4. 获取带真实 sender-ready、receiver-arrival、unpack-complete、join-close timestamp 的 multi-rank trace；
5. 在真实 trace 到位前，不推进 topology 或 physical headroom claim。

## 10. 是否允许进入 GPU 实验

**否。GPU Run Approved：NO。**

原因不是 formal runner 的已知逻辑错误，而是 native CPU dry run 尚未完成、3 个 GPU-specific tests 尚未执行，
且远端 SSH 当前仍在 TCP 建连后立即关闭。

下一次允许改变结论的最小证据：远端 3 个 skipped tests 全通过 + 两模型 native CPU dry run 完整产物通过复核。
