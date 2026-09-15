# CJC / JouleQueue Phase 4 交叉审计

> **历史交叉审计。** 本文保留 CJC/JouleQueue v1 被阻断的完整原因，不构成当前主线；当前状态见 [`../../current/README.md`](../../current/README.md)。

日期：2026-07-22  
角色：独立指标、会计、因果与 provenance 交叉签字；**只审查，不修改实现**  
审计对象：`cjc-v1`、`joulequeue-v1` 冻结协议及其 Phase 3 代码/测试

## 1. 结论先行

| 子系统 | 当前最高可信产物 | 最高未关闭缺陷 | Cross-sign | Phase 5 |
|---|---|---|---|---|
| CJC route/capture/LUT | calibration-only capture、5090 LUT、CPU dev replay | G1 与 formal capability 可自报；控制 locus/全局 ready set 未收费；placement/data/arrival provenance 未闭合；canonical reduction 未实际验收 | **BLOCKED** | 仅允许 calibration/dev 子运行；禁止 sealed scientific replay |
| JouleQueue activation/surface | calibration activation capture、isolated expert surface diagnostic | 正式 job/arrival/dependency/queue-energy executor不存在；oracle 容量与 LLM-jp top-k 冲突；surface CI 伪重复；主分母与 artifact 绑定未闭合 | **BLOCKED** | 仅允许 `--mode dev` 的 activation/surface diagnostic；禁止 E1/E2/E3 与 GO/NO-GO |

这不是科学 No-Go。当前只能判定：**Phase 3 尚未实现冻结协议所需的正式证据链，因此不能进入会生成科学结论的 Phase 5。** 不得通过把 capability 布尔值改成 `true`、手写 `SIGNED-OFF` JSON 或把 isolated surface 称为 queue execution 来绕过。

### 本地可复现检查

```text
CJC pure policy/calibration tests       32 passed
CJC torch capture test                  NOT RUN: local Python has no torch
JouleQueue pure policy tests             18 passed
JouleQueue torch capture/surface tests   NOT RUN: local Python has no torch
```

这些测试证明若干局部恒等式存在，不证明 formal 路径完整。

## 2. CJC 交叉审计

### C-P0-1：G1、early-release 价值与 canonical reduction 都能由环境 JSON 自报

冻结协议要求先证明 ready-block 可重排，且 early closure 能释放 buffer、解除 HoL 或推进下游；canonical reduction 也必须实际一致（协议 `docs/archive/receiver_aware/cjc/CJC_Phase2_冻结实验协议_2026-07-22.md:42-49,151-152,192-200`）。

实现只读取：

```text
environment.capabilities.ready_block_reorderable = true
environment.capabilities.token_block_early_release_effect = true
environment.capabilities.canonical_reduction_hash_match = true
```

并检查 GPU 名称/H2D 标签字符串，见 `docs/archive/receiver_aware/cjc/experiments/run_cjc_oracle.py:259-276`。环境文件没有硬件探针、自哈希或可复算 evidence reference。`canonical_reduction_signature()` 只存在于 `cjc_policy.py:1193-1206` 和单测；正式 runner 从未对 candidate/baseline payload 执行或比较它。最终 `decision.json` 还直接写入 `full_drain=true/task_set_equivalence=true/data_bytes_equal=true`，见 runner `:755-775`，而 GO 表达式只消费统计门与 action-collapse，见 `:723-753`。

影响：手写一个 capability 全 true 的环境文件即可把未证明的 G1/数值能力变成 formal 前提。closure proxy 变快不等于系统关键路径真的推进。

关闭条件：G1 必须消费一个由受审 runtime fixture 生成的 artifact，至少含 `before/after frontier、buffer release/HoL unblock、action trace、output hash、producer SHA`；formal runner 重算并校验，不接受裸布尔值。

### C-P0-2：`global_causal_join` 的控制 locus 与协调会计不闭合

`simulate()` 先把所有 sender 的任务按 receiver-node resource 汇入一个全局 `ready` 集合，再由单一策略选择下一任务，见 `cjc_policy.py:861-939`。candidate 因而免费获得跨 sender 的 ready-task census、task size、deadline 与 dispatch authority。实现只在 contribution 完成后收一条 ACK 的 build/serialize/wire/parse 与 lookup 税，见 `:943-991`。

这里存在二选一矛盾：

1. 若 scheduler 在 receiver ingress 本地，它只能在 contribution 已到达后看见任务，无法用该状态重排尚未传输的 sender data；此时 completion ACK 对本地排序又没有必要。
2. 若 scheduler 在 sender/集中控制端，跨 sender ready-state 上报、调度命令与 command staleness 都必须收费；当前完全免费。

因此当前 G3 不是“完整协调税后的 deployability”，只是给一个全局集中式 ready-set oracle 加了 completion ACK 税。协议 `:137-150` 所列协调账本没有覆盖 ready-state acquisition 与 action dissemination。

关闭条件：先冻结控制位置与消息方向；明确 `ready announcement -> decision -> grant/dispatch -> receiver completion` 四段消息、bytes、时间戳、重试/epoch，并在 replay 中逐事件收费。若改成 receiver-local，只能在真实 ingress queue 中重排已到达 blocks，并重新定义能改善的关键路径。

### C-P0-3：formal placement 可偏离冻结 contiguous/LPT 主口径

协议冻结 EP=8、两节点、contiguous expert placement 与 route-blind token-count LPT origin（协议 `:101-105`）。producer 却允许 formal CLI 使用 `--placement round_robin` 和任意 `--ep-size`，见 `capture_cjc_routes_gpu.py:61-73`。runner 的 `load_placement()`：

- 不检查 manifest 自报的 `placement` 是否为 `contiguous`；
- 不检查 manifest 的 `ep_size/gpus_per_node` 是否与 config 一致；
- 不重算 expert→sender contiguous 公式；
- 不重算 request→receiver LPT。

它只验证任意映射的 self-hash，随后照单使用，见 `run_cjc_oracle.py:162-191`。因此 round-robin 或人为挑选 origin 也能进入主 formal cell。

关闭条件：formal capture 禁止覆盖 topology/placement 参数；runner 从冻结 config 和 canonical request manifest 重算两张映射，逐项比较；round-robin 只能进入显式 sensitivity artifact。

### C-P0-4：data registry 会“硬写正确字段”，却未验证输入 manifest 真满足这些字段

`build_cjc_data_registry.py:22-36` 只检查 split 标签与 self-hash；随后在输出中固定写入 seed、窗口、数量与 tokenizer 长度规则，见 `:58-75`。它没有验证输入 manifest 的：

- dataset/config/split；
- candidate window；
- selection seed/method；
- 64/128 request 数；
- 每条 source row 是否属于窗口；
- 两个 tokenizer 的真实长度。

`--historical-hashes` 还是可选的，见 `:13-18,47-52`。capture 的 `_load_manifest()` 也只检查 self-hash、128 tokens 与非空 requests，不检查正式 split/window/count，见 `capture_cjc_routes_gpu.py:76-91`。formal runner 最终只验证 registry 中那些被硬写的字段，见 `run_cjc_oracle.py:194-227`。

影响：任意自哈希 prompt 集合可被 registry 包装成“冻结 sealed 数据”；历史实验重用也可通过省略 registry 静默发生。

关闭条件：registry builder 逐字段验证两个原 manifest；historical registry 改为必需且绑定版本；formal runner 同时绑定并读取原 calibration/sealed manifests，而不是只读派生 registry。

### C-P0-5：arrival trace 缺证据，且来源标签可伪装

仓库当前没有 `cjc-arrival-trace-v1` 的真实/public artifact；只有 loader 与测试。`load_arrival_trace()` 接受任意单调 timestamps，只要 JSON 自称 `measured_same_run_host_serving` 或 `public_trace_calibration_slice` 并自哈希，见 `prepare_cjc_calibration.py:98-122`。没有 source URI、license、raw artifact SHA、采集工具 SHA、request 定义或 calibration slice rule。

影响：完全 synthetic timestamps 只需换一个字符串即可成为 formal MMPP calibration；“bursty workload 由公开/实测 trace 固定”不可复核。

关闭条件：补只读 raw source artifact 或明确版本化公开 trace；manifest 绑定 URI/license/raw SHA/slice rule/单位/请求语义；由 producer 生成而不是人工填写 source 枚举。

### C-P0-6：sealed route 完整性与跨层因果没有闭合

正式 runner 只要求每模型有 128 个 request_id，并要求每个**已出现的** `(request,forward,layer)` 有 positions 0..127，见 `run_cjc_oracle.py:490-501`。它没有要求：

- 每 request 恰好一个 forward；
- 所有 request 拥有相同 MoE layer set；
- 每个 request 都包含 A1 选中的四层。

`select_replay_routes()` 只从全局 layer union 选四层，再对存在的 request/layer 选 token，见 `:354-399`。缺层 request 会静默变短。

更重要的是，任务 release 直接定义为：

```text
request_arrival + layer_id * calibration_layer_period
```

见 `cjc_policy.py:563-577,592-612`；later layer 不依赖 prior-layer closure。queueing 使上一层延迟时，下一层仍可不可能地提前进入 compute/ingress。协议虽已把 claim 缩到 sampled token-block closure，但第 6 行仍称“真实 MoE routing DAG”；当前只有 route identity，不是 dependency-correct DAG。

关闭条件：sealed validator 对每 request 强制 one-forward/full-layer census；要么显式把各层作为独立 exogenous episode并删除“routing DAG/跨层”表述，要么把上一层 token/request frontier completion 接到下一层 release。

### C-P0-7：source manifest 只按 basename 点名，没有绑定 canonical 实现路径

`validate_source_manifest()` 接受 manifest 中的任意路径；相对路径拼到 repo root，绝对路径原样使用，最后只把 `path.name` 加入 required-name 集合，见 `run_cjc_oracle.py:230-256`。因此一个 repo 外文件只要也叫 `cjc_policy.py` 或 `run_cjc_oracle.py` 就能替代 canonical scientific source 满足点名；正式执行的实际 repo 文件未必是被签字的那个文件。

`SIGNED-OFF` 本身也只是普通 JSON 的 `status` 与若干 hash，没有 reviewer identity、review report SHA 或不可变生成链，见 runner `:279-291`。人工流程可以补足信任，但代码不能宣称 attestation 已自证。

关闭条件：source manifest 使用冻结 repo-relative canonical path 白名单并拒绝绝对路径/重复 basename；signoff 绑定 review report SHA、commit/worktree diff、全部 transitive producer/tests/configs与 signer identity。

### C-P0-8：A1 不只是抽样 metric，而是删除了 112/128 token 的背景负载

runner 在构造 task 前先调用 `select_replay_routes()`，见 `run_cjc_oracle.py:565-570`；`build_tasks_from_routes()` 的 expert row count、sender FIFO、ready time、ingress bytes与service全都只来自这16个token，见 `cjc_policy.py:551-617`。因此当前系统不是“完整128-token forward中观测16个token”，而是只剩1/8 token load的稀疏子系统。

影响：compute/queue contention与headroom都会改变；A1虽缩小 metric对象，却没有把 background work删除写成冻结 workload。当前不能称 route-real 128-token replay。

关闭条件：完整128 token都进入 compute/ingress queue，只对SHA选中的16个join keys计主metric；若为性能聚合背景任务，必须证明bytes/service/queue恒等式。否则回Phase 2把问题改名为 sparse-subgraph proxy，原5%/3pp门失效。

### C-P0-9：bootstrap把同一共享队列中的相关request当独立样本

每个 model/cell/seed 的128 requests共享同一arrival trace、sender FIFO与receiver ingress queue，见 `cjc_policy.py:502-577`。`episode_metrics()`事后按request拆行，`paired_hierarchical_bootstrap()`再独立重采request与seed，却不按重采后的request集合重建queue，见 `cjc_policy.py:1059-1086,1118-1176`。

影响：一个request的latency受其他request影响，不能被当独立episode IID。当前真正独立的queue realization最多是5个seed；LCB可能严重过窄。

关闭条件：统计cluster改为完整 workload trace/seed并增加独立realizations；或每次document bootstrap后重新生成arrival并重跑queue。禁止对post-hoc共享队列latency做request-level伪重复。

### C-P0-10：`bursty_rho80`、calibration P99与sealed policy语义不一致

`fit_two_state_mmpp()`按inter-arrival median二分并估计“每次arrival切换概率”，`_request_arrivals()`也只在arrival边界切状态，见 `prepare_cjc_calibration.py:125-150` 与 `cjc_policy.py:502-520`。这不是有time-dwell CTMC的MMPP；low/high rate的算术平均也不保证time-average utilization=0.80，生成后没有realized rho校验。

同时 calibration 先算每seed pooled P99再对P99取均值，见 `prepare_cjc_calibration.py:221-254`，与 config 声称的单一 `pooled_token_closure_p99` 不同。calibration baseline还用 `starvation_us=inf`，sealed runner却给所有arm `starvation_us=slo_us`，见 calibration `:249-251` 与 runner `:588-629`。

影响：workload cell、SLO与baseline行为均漂移，violation门无可信共同口径。

关闭条件：回Phase2冻结真实CTMC MMPP或诚实改名为Markov-renewal proxy；按stationary time分布缩放并校验realized bottleneck utilization。SLO用相同policy semantics的一次pooled token latency求P99。

### C-P0-11：G3绕过实际 ACK codec/hash/epoch 应用路径

formal replay在completion后直接构造携带完整Python `join_key` 的 `AckEvent`，见 `cjc_policy.py:965-987`；没有调用 `encode_ack_message()/decode_ack_message()`，也没有64-bit token hash collision table。malformed只由 `malformed_task_ids` 直接把event标false，未经过真实parser、duplicate/out-of-order与epoch apply。

影响：被计费的是理想direct-key oracle，不是协议所写的ACK wire path。codec单测通过不能签G3。

关闭条件：若remote control，replay必须执行 encode→wire→decode→collision-checked lookup→epoch apply；若receiver-local，回Phase2删除虚构wire ACK。两种语义不能混用。

### C-P0-12：formal GO没有消费冻结的完整arms与sensitivities

runner只执行200Gbps、block=1、单一placement；没有 `offline_clairvoyant`、100/400Gbps、block=8、round-robin dependence与canonical numeric artifact，见 `run_cjc_oracle.py:565-753`。最终GO只检查G2/G3、20us positive与action collapse；reserve seeds条件流程也未实现。

影响：即使代码输出 `GO_TO_STREAMING_RUNTIME_PROTOTYPE`，也不等价于协议 `:192-202` 的GO。

关闭条件：每个冻结arm/cell/sensitivity成为machine-checkable required artifact；缺任一项必须 `formal_run_valid=false/BLOCKED`，不得默认通过。

### C-P0-13：dev路径可以提前打开sealed并反复覆盖输出

dev mode虽然最后标 `NOT_TESTED`，仍会加载与 `sealed_manifest_sha256` 匹配的route并计算完整action、metrics和bootstrap，见 `run_cjc_oracle.py:447-458,565-704`。output使用 `mkdir(...,exist_ok=True)` 并覆盖文件，见 `:520-533`。

影响：研究者可先看sealed outcome再改代码/叙事；状态标签不能撤销selection leakage。

关闭条件：dev对sealed registry/hash hard-fail；formal使用不存在的输出目录、一次性nonce/consumption record与原子提交。P0关闭前禁止任何sealed capture/replay。

### C-P1-1：部分“强基线”在当前 workload 下名义强、实际退化

- `largest_flow_first` 读取 `wire_bytes`，但同一模型 main block=1 时所有 contribution bytes 相同，退化为 ready-time tie-break；见 `cjc_policy.py:579-616,708-709`。
- `topology_join_blind` 的 projected backlog 对同一 receiver-node ready set是共同常数，只剩 service/slack；没有 sender/NIC/link queue，见 `:712-714,930-937`。
- `receiver_qdepth` 在单一 node-ingress server 内只按 rank ready-count，不预测 sender contention。

这不必然是代码 bug，但会削弱“相对最强简单基线”的 necessity 解释。若 candidate 只赢这些退化臂，不足以支持 novelty。

关闭条件：加入不读 sibling bitmap、但使用同一控制接口的 sender-ready greedy、age×service、per-link projected-finish/DRR 基线；先在 calibration 固定，无权 sealed 调参。

### C-P1-2：ACK microbenchmark 测的是 toy codec，不是完整 controller path

`measure_ack_components()` 只计时构造固定 dataclass、编码/解码一个固定 record 和固定 dict lookup，见 `prepare_cjc_calibration.py:257-314`。未计：token identity hash、epoch/state update、aggregation、ready census、grant command、真实 queue lookup规模与 Python/transport boundary。协议却要求 build、aggregate、serialize、transfer、parse、policy lookup 全栏（协议 `:141-149`）。

允许把它称为 codec/host microbenchmark；不能把该数值称为完整 coordination tax。

### C-P2-1：5090/H2D/RDMA 边界标签正确，但不能支持 deployability

LUT 对 expert/pack/reduction 使用 CUDA event，对 host staging 使用 pinned H2D，并明确 `NOT_RDMA`；replay另加 analytic 100/200/400 Gbps link，见 `run_cjc_lut_gpu.py:340-411` 与 `cjc_policy.py:579-591`。没有发现 Mac timing 混入主路径。

这是诚实的 L1/L2 proxy 边界；但 `pinned H2D + analytic link` 不是 RDMA data path，二者串加也不是经过实测校准的 transport pipeline。可用于筛选 headroom，不能支持 G3“已部署可行”或任何 NCCL/RDMA/P99 claim。

## 3. JouleQueue 交叉审计

### J-P0-1：正式 Phase 5 runner 尚不存在

配置正确地把四项 formal capability 全部设为 `false`，见 `docs/ideas/energy_slo/joulequeue/experiments/configs/joulequeue_v1.json:69-74`。即便以后全部改为 true，`run_joulequeue_oracle.py` 仍只会：

1. 加载一份 jobs 和一份 surface；
2. 调用 `run_development()`；
3. 固定输出 `PARTIAL_DEVELOPMENT_ONLY`、`scientific_result_eligible=false`。

见 runner `:396-425`。它没有实现冻结协议要求的 calibration operating-point selection、两模型×Poisson/MMPP 四 cells、五 seeds、paired hierarchical bootstrap、E0/E1/E2/E3 decision、reserve-seed 停止规则，也不输出正式 action/power/accounting/per-episode/decision artifact 套件。

当前 fail-closed 是正确的；不得把 capability 改 true 当成实现完成。

### J-P0-2：job schema 与 route producer 不兼容，也不能表达 multi-row token membership

`_load_jobs()` 只信 job artifact 自报四个 metadata 布尔值，随后把 `forward_id/token_id` 强制为整数，见 `run_joulequeue_oracle.py:302-333`。CJC native producer 使用字符串 forward/token identity。已有 `validate_route_closure()` 又要求字段 `placement_sha256`，而 CJC producer 输出 `placement_manifest_sha256`；该 validator 还根本没有被 runner 调用，见 `joulequeue_policy.py:845-880`。

`Job` 只有一个 token identity，却允许 `rows>1`，见 `joulequeue_policy.py:35-95`。如果一个 job 代表真实 expert invocation 的多行，它没有保存这些 rows 对应的 token集合；如果坚持一 contribution 一 job、`rows=1`，LLM-jp 单 token 的 top-k=16 已超过 exact oracle 的 `max_exact_jobs=12`，见 config `:51-56` 与 policy `:499-512`。

影响：当前没有一种 job 映射能同时满足 route identity、completed-token denominator 和 exact oracle规模。

关闭条件：冻结 job granularity。建议一个 job 保存不可变 `row_members=[request,forward,layer,token,slot]`，completion 可逆回写每个 token；oracle按 expert episode分解并提供 exactness gap，而不是全 episode 12-job上限。

### J-P0-3：model/data 身份未进入 JobIdentity，surface 与 jobs 可跨模型错配

`JobIdentity` 不含 `model_revision/data_manifest_sha256`；`SurfaceCatalog` 只按 `(layer_id,expert_id)` 查曲线，见 `joulequeue_policy.py:35-58,187-212`。formal surface metadata 虽有 model revision，但 `_load_jobs()` 不读取对应 model revision，runner也不比较二者。

影响：只要 layer/expert id 重叠，OLMoE surface 可被用于 LLM-jp jobs，甚至不同模型 job 可进入同一 expert queue。此时所有 energy/latency与分母均失去对象身份。

关闭条件：job artifact 顶层和每条 identity均绑定 model/data/route/arrival manifests；runner强制一个 run 只含一个 pinned model，并与 surface revision、hidden size、expert count逐项一致。

### J-P0-4：最强基线、SLO calibration 与“hierarchical bootstrap”未进入执行链

`run_development()` 把 timeout/row threshold固定为 config 中的单点 `20us/32 rows`，却把 arm命名为 `best_fixed_timeout/best_static_rows`，见 `run_joulequeue_oracle.py:336-376`。`select_calibration_only()` 只是未被 runner 调用的工具，见 `joulequeue_policy.py:790-822`。

`paired_hierarchical_bootstrap()` 也没有 document/seed identity，只对一个平面 `ScheduleMetrics` 列表 IID重采，见 `joulequeue_policy.py:698-754`，且 runner完全不调用它。不存在“全部 SLO-qualified baseline 中 J/token 最低者”比较。

影响：即使 dev oracle看起来节能，也不能回答 E2 necessity，更不能与 10%/1.03/1pp 门比较。

### J-P0-5：surface 的统计单位违反冻结协议，E1 route-mass gate也未实现

capture 确实保存真实 BF16 expert activations，且核对 hook row count 与 native routing，见 `capture_joulequeue_expert_inputs_gpu.py:163-249`。但 surface runner 对每个 expert 把所有 capture events拼接成 pool，所有 10 trials都复用同一个 `pool[:rows]`，见 `run_joulequeue_expert_surface.py:637-672,696-768`。随后把这 10 个重复测量窗口当作 per-expert curve 的 independent trials做 bootstrap，见 `:394-471`。

冻结协议规定 surface CI 的独立单元为 `(layer,expert,input event)`，inner repeats不能当独立样本（协议 `docs/ideas/energy_slo/joulequeue/JouleQueue_Phase2_冻结实验协议_2026-07-22.md:189-191`）。当前只能估计同一 activation prefix 的测量噪声，不能估计 input-event异质性。

此外 E1 要求两个 row bins 在两模型均有正 LCB，且各覆盖 calibration expert-energy mass ≥10%（协议 `:118-119`）。surface artifact不消费 route trace，也没有 energy-mass coverage或 E1 decision。

关闭条件：以冻结 hash选不同 capture events/slices作为 trial cluster；bootstrap先按 expert/event，再汇总模型；另由 calibration route census × surface E(m)计算并绑定 energy-mass coverage。

### J-P0-6：E0 环境/thermal gate不完整，metadata含自证明字段

协议要求记录 driver、power limit、clock、温度、模型/权重 hash，并执行 thermal/repeatability gate（协议 `:99-106,120-140`）。surface metadata目前只有 GPU name、UUID、energy source、gap、window与CV；没有 driver/power cap/clock/temp/weight hash，见 `run_joulequeue_expert_surface.py:490-602`。

`background_sampler_exceptions_propagated` 直接从 config contract复制到 metadata，而不是从运行 artifact推导，见 `:590-592`。`native_activations/formal_eligible/phase4_signoff_verified` 也由 artifact自报；formal loader只检查这些字段和 signoff SHA“长得像64位hash”，没有读取 signoff内容，见 `run_joulequeue_oracle.py:160-220`。测试甚至手工构造全 true surface并证明 formal loader接受，见 `test_run_joulequeue_expert_surface.py:80-139,217-225`。

formal runner 的 hash manifest又只列 protocol/policy/runner/policy test/config，没有绑定 capture/surface producer、surface config、job/arrival builder和实际 jobs/surface artifacts，见 `joulequeue_v1.json:75-83`。

影响：当前 attestation链无法证明 surface来自受审 producer，也不能打开 E0。

### J-P0-7：4×4 sampled surface无法驱动完整 route token closure

formal loader只接受4层×每层4个experts的16条曲线，见 `run_joulequeue_oracle.py:269-275`；完整route却可能命中其余layers/experts。formal路径没有default curve，缺profile会hard-fail；若删掉未profile route，则top-k sibling closure与token completion失真。

关闭条件：回Phase2冻结“profile全部replay所需queues”或把E2缩为per-queue episode oracle并删除token-level/cross-layer P99主张；不得用pooled curve或selected-only routes补洞。

### J-P0-8：total-energy counter window 含未量化 envelope，且不同 repeats 可制造假节能

`NVMLWindowMeter.measure()` 的 counter start发生在首个 power sample、thread start和 workload start之前；counter end发生在 workload结束、thread stop/join、末尾 power sample之后，见 `run_joulequeue_expert_surface.py:159-224`。因此 `energy_j/repeats` 包含 sampler/thread envelope。separate/coalesced各自选择不同 repeats以达到约2秒，再分别除以 repeats，见 `:281-297,685-717`。

固定包围能耗为 `E_o` 时，每arm估计含 `E_o/repeats_a`；通常separate更慢、repeats更少，差值会被系统性抬高。长窗口可能让偏差很小，但实现没有 empty-envelope测量、boundary delta或上界，不能假设小于10%主门。metadata诚实标记 `SEQUENTIAL_BRACKETING_NOT_ATOMIC`，这是正确边界，不是 formal同窗证明。

关闭条件：counter在 synchronize后的 workload t0/t1尽可能紧邻读取；采样线程在 t0前稳定启动、t1后停止；输出 counter envelope与logical workload两套时间，做空窗/上界检查并要求偏差相对能耗差小于预注册比例。

### J-P0-9：所谓 conservative interpolation没有 upper-bound 保证

`SurfaceCurve.estimate()`对相邻point的energy/latency UCB做线性插值并标记 conservative，见 `joulequeue_policy.py:143-183`。端点各自为UCB不代表连线是未测row的upper bound，也不保证单调；oracle可能因插值低估改变动作偏序。

关闭条件：formal只允许grid点、越界/非grid回退immediate；或在Phase2预注册并验证真正的单调upper envelope。

### J-P1-1：surface-total + 固定30W idle只是 dev模型，不是 real queue board energy

调度器把 `total_during_launch` surface逐 launch相加，再用 config 的 `idle_power_w_dev_only=30` 给 defer gap收费，见 `joulequeue_policy.py:475-493,602-620` 与 config `:30-37`。单测正确避免了 busy idle double-count，但并未证明：

- isolated back-to-back launch energy可加到动态 queue；
- defer期间真实 board idle功耗为30W；
- clocks/temperature/residency与surface一致；
- attention/router/KV/CPU/NIC未执行时的J/token可称主 serving分母。

配置的 `real_board_energy_queue_executor=false` 正确承认这一点。该路径只能做 oracle modeling smoke，不能作为 E2 主能量会计。

### J-P1-2：`completed output token` 名称超出当前证据

`schedule_metrics()` 按唯一 `(request,forward,token)` 计 denominator，见 `joulequeue_policy.py:643-678`。在一 contribution 一 job且完整 route membership时，这可作为 routed-token denominator；但当前 capture是128-token prefill expert input，job builder不存在，`rows>1`又没有成员表。因此它尚不能证明 denominator是“completed output token”。

正式报告必须先称 `expert-stage J / unique routed-token identity`；只有 integrated independent-KV decode、完整 drain和输出token ledger闭合后，才能升级为 `J/completed output token`。

## 4. 允许与禁止的下一步

| 子运行 | 当前决策 | 允许的表述 | 禁止的表述 |
|---|---|---|---|
| CJC calibration manifest/route capture smoke | **RUN_ALLOWED（dev/calibration only）** | producer capability、identity diagnostic | G0 formal pass、sealed route-real result |
| CJC 5090 expert/pack/H2D/reduction LUT | **RUN_ALLOWED（LUT_ONLY/dev）** | same-GPU component measurement；H2D=`NOT_RDMA` | RDMA/NCCL/service P99 |
| CJC CPU dev replay/ACK codec microbench | **RUN_ALLOWED（NOT_TESTED）** | policy/accounting smoke、upper-bound diagnostic | G2/G3、科学 GO/NO-GO |
| CJC sealed route capture/replay | **BLOCKED** | 无 | 任何 formal verdict |
| JouleQueue calibration activation capture | **RUN_ALLOWED（dev）** | native BF16 activation capability | E0/E1 pass |
| JouleQueue isolated expert surface | **仅最小 `dev` mechanical smoke；完整grid暂不建议** | shape/UUID/counter plumbing diagnostic | E0 formal pass、headroom、queue J/token、Energy-SLO |
| JouleQueue E1 route-mass/nonconvexity gate | **BLOCKED** | 无 | E1 pass/fail |
| JouleQueue E2/E3 oracle/causal runner | **BLOCKED** | 无 | 10%/1.03判定、GO/NO-GO |

若现在使用 GPU，建议只运行表中 `RUN_ALLOWED` 且强制 dev/calibration 输出；这些结果可用于修实现和估计量级，**不得打开 sealed，也不得进入论文结论表。**

## 5. 最小重审顺序

### CJC

1. 先冻结并实现物理控制 locus与完整 message flow；关闭全局 ready-set免费信息。
2. 修 data/placement/arrival/source/environment validators；所有 capability改为可复算 artifact。
3. 闭合 one-forward/full-layer与 dependency语义；执行 canonical output comparison。
4. 加入真正不读 sibling bitmap但使用同一物理接口的强基线。
5. 只重跑 calibration；P0全关后重新 Phase 4，之后才允许 sealed capture/replay。

### JouleQueue

1. 重定义可逆 multi-row job identity，加入 model/data/arrival/dependency manifests。
2. 将 exact oracle按可验证 episode分解，覆盖 LLM-jp top-k=16；实现 full dependency与四 workload cells。
3. 修 surface独立单元、环境/thermal/weight provenance与counter envelope；补 E1 route-energy-mass gate。
4. 实现 calibration-frozen全部强基线、real board-energy queue executor、document→seed bootstrap与正式 decision。
5. 把全部 transitive producer/config/artifact SHA纳入 signoff，独立重审后才可 Phase 5。

本审计未修改冻结阈值，也未把代理实验或建模结果升级为科学结论。
