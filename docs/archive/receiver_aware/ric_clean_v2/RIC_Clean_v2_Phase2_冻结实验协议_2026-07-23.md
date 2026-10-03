# RIC-Clean-v2 Phase 2 冻结实验协议

状态：**FROZEN FOR IMPLEMENTATION / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

## 1. 命题与边界

在 route、service time 与 payload 固定时，若 receiver-side queue/join
state 能使发送顺序在 matched worlds 中发生可执行翻转，并在完整
signal cost 后保留相对最强 join-blind 基线的 tail-latency 改善，
则 RIC 值得进入在线控制器实现。

最高 claim 仅为 `L2_CALIBRATED_VIRTUAL_EP` 的 necessity/headroom 与
charged replay。禁止声称 RDMA/NCCL 实测、serving TPOT/P99 或生产泛化。

## 2. Clean reset

1. `RIC-v1 + Amendment Q + formal_signoff/v6` 全部 `SUPERSEDED`；
2. 新 authoritative bundle 与旧 bundle 物理分离；
3. 新 bundle 在 manifest 生成前不得存在 route/scenario/oracle/result；
4. venv、model cache 与 dataset cache 位于 bundle 外；
5. 新 sealed 文本 SHA-256 与历史 calibration/sealed 并集零交集；
6. 冲突不得跳过后补样，直接 `BLOCKED_DATA_SPLIT`。

## 3. 数据预注册

- dataset/revision/model revisions 沿用 `configs/ric_v1.json` 的冻结身份；
- selection seed：`2026072301`；
- calibration window：`[100000,104000)`，64 documents；
- sealed window：`[120000,124000)`，128 documents；
- canonical text：仅将 `CRLF` 与孤立 `CR` 统一为 `LF`；不 trim、不做
  Unicode normalization，以 UTF-8 原字节计算 `canonical_text_sha256`；
- 选择：`sha256(str(seed) || canonical_text_sha256)` 升序；
- sequence length 128；两 tokenizer 均需至少 129 tokens；
- calibration/sealed 分别生成 manifest 和 selected-hash list；
- 打开 sealed manifest 前 Phase 4 必须 `SIGNED-OFF` 且无 P0。

## 4. 决策链

### N1 Receiver information necessity

两模型 matched-world MILP 比较 sender-only `S`、最强简单 join-blind
`B`、零延迟 receiver oracle `R0` 与 full-information ceiling `C`。
每对 worlds 必须在**同一个联合 MILP**中求解：`S/B` 在相同
observation-history node 上共享同一 action variable，不得逐 world 单独
求最优；两 world 等权，共用 lexicographic objective。`R0/C` 只在
协议明示允许的 keyed/full state 上拆分 observation node。

N1 GO 需两模型同时满足：

- `median((CVaR99_B-CVaR99_R0)/CVaR99_B) >= 0.05`；
- `R0` 在两 matched worlds 中的唯一最优 first action 不同的
  pair 比例 `>= 0.25`；
- exact MILP solver gap `<= 1e-6`。

任一模型失败立即 NO-GO，不写更复杂 controller。

Matched-world **不得重标 contribution/join key 或 application identity**。唯一允许的
差异是：从两条各自可达、合法且可重放的 prior receiver history 出发，使当前时刻的
`last_missing_status` 互换；当前 task/DAG、route、receiver placement、payload、
service tuple、release time、sender-local observation 与 aggregate port state 必须逐项
相同。实现必须保存两条 prior history、重放后的 keyed join state 与上述恒等式证明；
任一历史不可达或恒等式不成立则 `BLOCKED_INVALID_MATCHED_WORLD`，不计入 headroom。

### N2 Charged receiver signal

仅 N1 GO 后执行。将 receiver feedback 的 codec、字节、传播延迟与
decision tax 全部计入。两模型 × 两主 workload cell 均需：

- family-wise 校正后的 CVaR99 relative-reduction LCB `>= 5%`；
- closure-budget violation absolute-reduction LCB `>= 3` percentage points；
- charged arm 对 zero-tax R0 information headroom 的 retention LCB `>= 0.5`；
- full drain 与 `3N+J` 会计恒等式成立；
- 相对每个冻结简单基线不得反向劣化。

### N3 Sensitivity

仅 N2 GO 后报告 100/400 Gbps 代理链路。不改变主 verdict，
不表述为 RDMA 实测。

## 5. 基线、固定项与停止规则

冻结 join-blind 基线集合与 config 逐字对齐：`sender_fcfs`、
`sender_edf`、`sender_srpt`、`sender_age_service_drr`、
`sender_remaining_work`、`sync_token_order`、`receiver_qdepth`、
`topology_projected_finish`、`receiver_contention_joinblind`。
`calib_best_joinblind` 仅是从上述集合在 calibration 上选出的冻结别名，
不参与选择自身；另保留 sham feedback。route、top-k、precision、payload、
placement、canonical reduction 与 arrival seeds 在 sealed 不得修改。

- N1 失败：不再换 predictor/bandit/RDMA 叙事抢救；
- N1 GO 但 N2 失败：记为“信息有价值，当前 signal contract
  无系统净收益”；
- 设计/代码错误：产物 `SUPERSEDED`，回 Phase 2/3，禁止改门槛
  后直接重跑 sealed。

## 6. Phase 4 验收

1. clean bundle 初始无 route/scenario/result/decision；
2. 新 split 对历史文本 SHA 并集零冲突；
3. formal API 内部执行 one-shot reserve，不依赖 CLI；
4. identity 含 request/layer/token/topk-slot/expert/sender/receiver/epoch；
5. sender-only 视图不得间接读取 receiver queue/join state；
6. primary/sensitivity 复用完全相同 causal world；
7. accounting/full-drain/sham-cost/hard-rollback/one-shot 反例全通过；
8. 任何 P0 未关闭时禁止打开 sealed manifest。

## 7. 预期产物

`clean_v2/data/{calibration,sealed}`、`clean_v2/routes/`、
`clean_v2/scenarios/calibration/`、`clean_v2/oracle/`、
`clean_v2/calibration_lock/`；仅 Phase 4 签字后生成
`clean_v2/scenarios/sealed/` 与 `clean_v2/results/sealed/`。
