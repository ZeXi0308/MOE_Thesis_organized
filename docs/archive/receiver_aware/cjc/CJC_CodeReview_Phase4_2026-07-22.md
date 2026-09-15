# CJC v1 Phase 4 严格代码审查

状态：**BLOCKED / P0 OPEN / 禁止 Phase 5 正式运行**  
审查日期：2026-07-22  
对象：`cjc-v1` 冻结协议、配置、route producer、LUT/merge、calibration、replay core/runner 与测试。  
证据边界：远端 65 个 unit tests 仅证明若干实现分支可执行；**不是 GPU 测量、G0/G1 证据或科学结果**。

## 1. 一票否决结论

当前代码不能安全地产生 `GO_TO_STREAMING_RUNTIME_PROTOTYPE` 或 `NO_GO_CJC_V1`。这不是“结果不好”，而是 formal runner 仍可在数据隔离、统计单位、G1、ACK 因果路径和冻结敏感性缺失时生成科学 verdict。

| Scope | 状态 | 说明 |
|---|---|---|
| ACK 32 B codec、基础 route schema/closure validator、join-blind API 隔离、dev verdict 标为 `NOT_TESTED` | **SIGNED-OFF（仅 unit utility）** | 可作为修补后的底层组件；不授权打开 sealed data |
| G0 identity-complete / native-route provenance | **BLOCKED** | batch 身份碰撞、完整 layer 集未验证、patched/native parity 未证明、formal topology 参数未冻结 |
| G1 ready-block reorderability / early-release effect | **BLOCKED** | 仅信任 environment 中的布尔值，没有可执行 capability evidence |
| G2 zero-tax necessity screen | **BLOCKED** | replay load 与 A1 抽样不自洽，CI 统计单位错误，offline ceiling 与关键 sensitivities 缺失 |
| G3 charged deployability | **BLOCKED** | replay 绕过真实 ACK codec/hash/epoch 路径，coordination 会计不能证明 deployability |
| LUT / calibration formal artifacts | **BLOCKED** | provenance 与 calibration 口径仍有 P0/P1；尚无可审签 formal artifact |
| 任意 Phase 5 formal capture/LUT/calibration/sealed replay | **BLOCKED** | P0 关闭并重新审签前不得启动 |

允许的动作只有：修改 Phase 2/3、跑 unit/dev smoke，或在**不接触 sealed split**的 calibration/toy 输入上做非证据调试。任何这类输出必须保持 `NOT_TESTED`。

## 2. P0 缺陷

### P0-01 — sealed 数据可被 dev 路径提前打开，并可覆盖/反复生成结果

类型：**代码做错 / selection leakage**。

- `run_cjc_oracle.py:447-458` 在 dev 模式不要求 `sealed=True`，但仍强制 route 对上 `sealed_manifest_sha256`；因此 dev 可直接消费 sealed route。
- `run_cjc_oracle.py:565-704` 即使 dev 也计算全部 action、episode metric 和 bootstrap；`NOT_TESTED` 标签不能消除研究者已经看到 sealed outcome 的泄漏。
- `run_cjc_oracle.py:520-533` 使用 `mkdir(..., exist_ok=True)` 并覆盖正式产物，不满足一次打开/不可覆盖。

后果：可以先看 sealed 结果再改 score/代码/叙事，之后重新签字；formal evidence 不再可证伪。

修复门：dev runner 对 sealed registry/hash **hard-fail**；formal 使用一次性 run manifest/nonce，输出目录必须不存在并原子落盘；sealed artifact 首次打开即写不可变 consumption record。需要新增“dev + sealed 必失败”和“已存在 formal output 必失败”测试。

### P0-02 — G0 formal gate 可接受错误 topology/placement 与不完整 route identity

类型：**代码做错 / 错误对象身份 / provenance**。

- `capture_cjc_routes_gpu.py:61-73,175-217` 允许 formal 调用者改变 `--ep-size` 和 `--placement`，而 capture signoff 只绑定协议、source、data hash，不绑定这两个参数。
- `run_cjc_oracle.py:180-190` 忽略 placement 文件中的 `ep_size/gpus_per_node/placement`，也不重算 contiguous expert placement 与 route-blind origin LPT；任意有利 mapping 可通过自哈希。
- `capture_cjc_routes_gpu.py:261-263` 所有独立 forward 都写 `batch_id="batch-0"`；validator 只检查一个 forward 内 batch 一致，未检查 batch 是否跨 forward 复用。
- producer 只检查 observed layer 不重复（`capture_cjc_routes_gpu.py:263-279`），不检查冻结模型应有的完整 MoE layer 集。四个 observed layer 即可进入 A1。
- `route_source="native_model_forward"` 实际来自 `capture_moe.py` monkey-patched forward；现有测试没有与未 patch 的 native router 做逐层 top-k/weight parity。

后果：错误 placement、漏层或 patched-route drift 都能被标成 G0 PASS，直接改变 headroom。

修复门：formal 主配置强制 EP=8/contiguous；sensitivity 使用独立、显式 role；signoff 绑定全部 CLI/config。`batch_id` 全局唯一并做双向 batch↔forward 校验。由 model config/独立 hook 固定 expected MoE layer set。至少在 calibration fixture 上比较 unpatched native hook 与 producer 的逐层 `(token,slot,expert,weight)` hash；不通过不得使用 `native_model_forward` 标签。

### P0-03 — A1 抽样把未选 token 从系统负载中删除，headroom 不再对应真实 128-token forward

类型：**Phase 2–3 共同错误 / 问题定义偏移**。

- runner 先在 `run_cjc_oracle.py:565-570` 把每 request/layer 缩到 16 token，再把这个子图传给 task builder。
- `cjc_policy.py:551-577` 的 expert row count、sender FIFO 与 ready time 只来自抽样后的 group；`cjc_policy.py:581-617` 的 combine-ingress bytes/service 也只包含这些 token。

这不是“在完整负载下只抽样 metric”，而是一个移除了 112/128 token 及其 compute/queue contention 的稀疏系统。A1 虽把结论缩到 sampled closure，但没有授权把 background load 删除；`route-real/LUT-calibrated` 会因此被误解。

修复门：用完整 128-token route 构造 compute/ingress load，只对 outcome-blind 16 token join keys 计主 metric；或构造经过恒等式验证的 exact background-load aggregation。LUT row grid需覆盖真实 group rows。若坚持当前子图，必须回 Phase 2 改成“synthetic sparse-subgraph proxy”，且不能用原 5%/3pp system gate。

### P0-04 — G1 与 canonical reduction 只靠自报布尔值，runner 可在无证据时 GO

类型：**代码做错 / claim–implementation mismatch**。

- `validate_environment` 只读取 `capabilities[name] is True`（`run_cjc_oracle.py:250-276`）；测试 `test_cjc_policy.py:453-473` 也只验证 false boolean 会失败，并未实现冻结验收用例要求的 full-barrier 零价值 fixture。
- `canonical_reduction_signature()` 只是孤立 utility；formal runner 从未调用它，也没有 contribution payload/numerical output。`run_cjc_oracle.py:741-775` 的 GO 条件没有 G1 evidence hash 或 canonical equality。

后果：没有可重排真实路径、早闭包没有 buffer/HoL/downstream 效果、或 reduction 数值不一致时，runner 仍可输出 GO。

修复门：environment capability 必须引用有 schema/self-hash 的 probe artifacts，并由 signoff 绑定。实现 barrier-vs-reorderable capability fixture及早闭包可释放的明确系统状态。canonical equality 要么捕获真实 sampled contribution output 并执行固定 slot/expert reduction 对比，要么回 Phase 2 删除这一不可执行 gate、收窄 claim；不得用布尔声明替代。

### P0-05 — CI 把同一拥塞队列中的 request 当成独立 episode

类型：**实验逻辑错误 / 统计错误**。

- 每个 model/cell/seed 先将全部 128 requests 一起建立 arrival trace 和共享 sender/receiver queues（`cjc_policy.py:502-577`、`run_cjc_oracle.py:565-633`）。一个 request 的 latency 取决于其他 requests。
- 随后 `episode_metrics()` 按 request 拆行（`cjc_policy.py:1059-1086`），bootstrap 又在 `cjc_policy.py:1167-1174` 独立重采 request/seed，却不重跑被重采后的 queue。

后果：128 个相关 request 被伪装成独立样本；目前真正独立的 workload realization 至多是 5 个 seeds。P99/violation LCB 可能严重过窄，5%/3pp 门不可信。

修复门：回 Phase 2 重定义统计单位。推荐以**完整 workload trace/seed**为 cluster，并增加足够独立 trace；若要 document bootstrap，必须在重采 document 后重建 arrivals、queue 和 replay，不能重采 post-hoc latency。冻结前做 coverage/power smoke，禁止把 contribution 或相互依赖 request 当独立样本。

### P0-06 — `bursty_rho80` 不是所声明的 MMPP，且 calibration P99 口径错误

类型：**代码做错 / 指标会计错误**。

- `fit_two_state_mmpp()` 用 inter-arrival median 二分并估计“每次 arrival 的切换概率”（`prepare_cjc_calibration.py:125-150`）。
- `_request_arrivals()` 只在 arrival 边界尝试切换状态（`cjc_policy.py:502-520`），不是状态独立连续时间演化的 CTMC MMPP；以 rate 的算术平均归一也不保证 time-average arrival rate 等于 base rate。因此标签 `rho80` 不成立。
- calibration 声明 `pooled_token_closure_p99`，但 `select_calibration_static_arm()` 先算每 seed P99，再取 P99 的均值（`prepare_cjc_calibration.py:221-254`），不是跨 calibration token/seed 的一个 pooled P99。
- calibration 选择时 `starvation_us=inf`，sealed runner 对所有 arm 用 `starvation_us=slo_us`；SLO 与 baseline 实现口径不一致。

后果：workload cell、SLO threshold 和 violation metric 均可能被系统性改写。

修复门：实现连续时间两状态 MMPP，按 stationary time distribution 精确缩放到 bottleneck `rho=0.80`，并在生成 trace 后校验 realized utilization；或回 Phase 2 改名并冻结 Markov-renewal proxy。SLO 必须从同一 calibration policy semantics 下的 pooled token latency 一次求 P99；starvation bound 在冻结协议中显式给定并前后一致。

### P0-07 — G3 replay 绕过了声称被审计的 ACK wire/identity 路径

类型：**实验逻辑错误 / 因果与会计错误**。

- `simulate()` 在 completion 后直接构造带 Python `join_key` 的 `AckEvent`（`cjc_policy.py:965-987`）；没有调用 `encode_ack_message/decode_ack_message`，没有用 64-bit token hash 做 collision-checked lookup。
- duplicate/stale/out-of-order/malformed wire record 没有进入真实 parser/epoch path；`malformed_task_ids` 只是直接切换 event.valid。
- ACK benchmark 的 `_timed_us()` 把 `hash(operation())` 也计入 build/serialize/parse/lookup（`prepare_cjc_calibration.py:257-286`），组件名与实际被测代码不一致。

后果：G3 得到的是“收费的理想 direct join-key oracle”，不是冻结的 ACK protocol；既可能乐观（无 collision/parser failure），也可能因 harness/hash 与全串行收费而错误判死。

修复门：明确 controller location。若为 remote controller，replay 必须真实 encode→wire event→decode→collision-checked identity table→epoch apply，并注入 duplicate/out-of-order/malformed fixtures，失败后按冻结 fallback 且税保留；若为 receiver-local，则回 Phase 2 删除虚构 wire ACK。microbenchmark 使用独立 sink/harness-baseline，保存 raw repeats。

### P0-08 — formal verdict 缺少冻结 arm、sensitivities 与 hard gates，却仍可 GO

类型：**代码做错 / 协议偏离**。

`run_cjc_oracle.py:565-753` 仅跑主 200 Gbps、block size 1、单一 placement，未实现/判定：

- `offline_clairvoyant` ceiling；
- 100/200/400 Gbps sensitivity；
- block size 8 sensitivity；
- round-robin placement dependence；
- actual canonical reduction equality；
- “不依赖单一 seed”的显式 gate；
- P50/P95/CVaR、request/layer closure、fairness、queue utilization 等冻结报告项。

GO 只检查 G2/G3、20 us positive 和 action collapse（`run_cjc_oracle.py:723-753`）。因此当前 formal verdict 不等价于 Phase 2 的 Go/No-Go。

修复门：把每个冻结 gate 变成 machine-checkable decision field；缺 artifact/arm/cell/sensitivity 时 `formal_run_valid=false`，不是默认 PASS。补 offline ceiling 与全部 sensitivity 产物；主结果和 sensitivity 分栏，不得挑最好值。

### P0-09 — formal provenance 可绑定错误文件或 dev upstream artifact

类型：**代码做错 / provenance**。

- `validate_source_manifest()` 只按 basename 判断 required file，允许用任意路径下同名文件满足集合，而不保证绑定实际 import/executed repo path（`run_cjc_oracle.py:230-245`）。
- formal oracle 只读合并 `lut.csv`，不要求/校验 merged `lut_metadata.json`；也不验证 calibration 的 `status=CALIBRATION_ONLY`、`mode=formal`、model revision 和 provenance hashes。
- `calibration_entry()` 只抽取数值/字符串（`run_cjc_oracle.py:294-351`），所以手写或 dev calibration 可被 artifact signoff 误接入。

后果：hash gate 看似 fail-closed，实际可能不绑定执行代码和正式 upstream producer。

修复门：source manifest 使用固定 repo-relative allowlist并逐项与 `Path(__file__)`/import module resolve 后的路径一致；formal runner 同时接收并验证 merged LUT metadata、calibration schema/status/mode/provenance/self-hash，且将完整 provenance DAG 写入 `protocol.json`。

### P0-10 — data registry 会“声明”冻结字段，却不验证输入 manifest，historical exclusion 还可为空

类型：**代码做错 / 数据隔离错误**。

- `build_cjc_data_registry.py:22-46` 只检查 split/self-hash/model revision 相等。
- `build_cjc_data_registry.py:58-74` 不读取 input manifest 的 seed/window/count/tokenizer threshold，而是无条件写死冻结值；使用 `--seed` 改出的 manifest 也会被 registry 错标为 seed 20260722。
- historical registry 是 optional（`build_cjc_data_registry.py:47-52`），空列表可 vacuous pass，无法证明与仓库历史 evidence 隔离。

修复门：逐项校验两个 input manifest 的 dataset/split/window/seed/selection/count/sequence/tokenizer revisions，请求数量、request/text hash 唯一性和实际 source row 范围；historical registry 必填并绑定仓库全历史 manifest 扫描产物的 hash/count。加入错 seed/错窗口/空 historical hard-fail 测试。

## 3. P1–P3

### P1

1. **LUT activation 身份不匹配。** `run_cjc_lut_gpu.py:286-359` 把一个 layer 中最先执行的任意 expert activation 拼成公共 pool，再拿同一 pool 测 SHA 选出的四个 experts；不是各 `(layer,expert)` 自身 routed activation。应按 route/expert 身份分别采样，保存 raw trial，不只存 median/max。
2. **GPU provenance 太弱。** LUT merge 只比较 `gpu_name`（`merge_cjc_luts.py:107-145`），未绑定 UUID、driver、CUDA/PyTorch/Transformers revision、clock/power state 与 background load。
3. **topology baseline 退化未显式报告。** 每次只在一个 receiver resource 内调度，`resource_backlog` 对 ready tasks 是共同常数；`topology_join_blind` 基本退化为 SRPT+EDF（`cjc_policy.py:712-714`）。若 action trace 等价，应作为 baseline collapse 明报，而不是仍称独立 topology-aware baseline。
4. **reserve seeds 流程未实现。** CI 跨门后的 06–10 预留 seed 没有一次性、条件触发的 runner 路径。

### P2

1. `simulate()` 在多 resource 循环结束后只检查最后一个 resource 的 `actual_missing`（`cjc_policy.py:865-998`）。当前 exactly-once/full-count 通常使其冗余，但应改为逐 resource 或全局 closure map。
2. `report.md` 只写状态和边界（`run_cjc_oracle.py:827-833`），不足以人工复核 formal 决策；即使 CSV/JSON 齐全也应生成 gate matrix 和失败原因。

### P3

无仅样式级问题值得在 P0 修复前优先处理。

## 4. 关闭条件与最小重审顺序

1. **回 Phase 2 amendment**：先修统计单位、A1 background load、controller location、MMPP/arrival model 与 starvation bound；这些不是局部补丁能诚实解决的实现细节。
2. **Phase 3 修补**：关闭 sealed leakage、G0 topology/identity、真实 ACK path、provenance DAG、完整 arms/sensitivities/decision gates。
3. **重新 unit/integration review**：新增上述 hard-fail 与 queue-level CI tests；65 个旧 unit tests 不能替代。
4. **重新 Phase 4**：先签 calibration/toy producer code；只允许 calibration formal artifacts。审过 upstream artifact 后，再签 sealed one-shot capture/replay。
5. **Phase 5 门**：只有新的 review 状态为 `SIGNED-OFF` 且 P0=0，才可打开 sealed route；在此之前不得把任何 GPU/dev 输出解释为 G0–G3 或 No-Go。

本轮没有可信科学结果，因此无需标 `SUPERSEDED` 的数字；需要被阻断的是尚未发生的 formal verdict。
