# JouleQueue v1 Phase 4 严格代码审查

状态：**BLOCKED / NOT SIGNED-OFF / 禁止 Phase 5 正式实验**  
审查日期：2026-07-22  
审查范围：JouleQueue v1 activation capture、expert surface、NVML 会计、route/job replay、强基线、oracle、统计与 formal gate。  
审查者动作：只读审查；**未执行实验，未修改实现**。

## 0. 一票否决结论

| 子范围 | Phase 4 状态 | 允许动作 | 禁止动作 |
|---|---|---|---|
| 真实 activation capture | **DEV-ONLY** | 可做最小机械 smoke，产物必须是 `NOT_TESTED/CAPTURE_ONLY` | 不得称 route-joined 或 formal surface evidence |
| E0 surface measurement | **BLOCKED** | 修复后重审 | 当前脚本不得用于正式能耗曲线或 E0 通过判定 |
| E1 non-convexity / coverage | **BLOCKED** | 修复统计单位和 route coverage 后重审 | 不得从当前 `delta_energy_*` 判 headroom 或 No-Go |
| E2 oracle necessity | **BLOCKED** | 回 Phase 2/3 补全 job/dependency/profile-coverage 设计与实现 | 不得打开 sealed，不得跑 formal oracle |
| E3 causal approximability | **BLOCKED** | E2 通过且 causal policy 参数由 calibration 冻结后再审 | 不得生成 Go/No-Go |

当前报告发现 **9 个未关闭 P0**。因此即使 65 个单元测试全部通过，也只说明一部分纯函数和 fail-closed 分支可执行；它们不构成 E0、E1、E2 或 E3 的科学证据。

没有可被本审查“翻转”的正式科学结论：仓库当前仍是 `NO SCIENTIFIC RESULT`。如果已经由当前 surface/replay 代码生成数字，只能标为 **DEV_ONLY / SUPERSEDED_BY_CODE_REVIEW**，不得引用为 Go、No-Go 或论文数字。

## 1. 审查基线

审查时文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `JouleQueue_Phase2_冻结实验协议_2026-07-22.md` | `ecd512a88849f72f412c146912d3afe530e4f61cd5850dc29803549778f98da6` |
| `experiments/configs/joulequeue_v1.json` | `19c8eeed55ab0fb1f91f204446eaeeb709f30872ea9655f38dd225a23cd06de2` |
| `experiments/configs/joulequeue_surface_v1.json` | `0daab80ba0977b1151b5779b1195a572a3a255ea6538c1363720433feaea355f` |
| `experiments/capture_joulequeue_expert_inputs_gpu.py` | `5bc7ed85cfee362fece49a51aa6c1281c81f17ba98d4e09812577c841dabe092` |
| `experiments/run_joulequeue_expert_surface.py` | `e799ba6e82d24dc9d7a31ee8de1f7e5d2f537bc46ec1528bff66351d2f26d083` |
| `experiments/power_accounting.py` | `bcf8cb2adb9cd02189b5aaa4037da1d2bf466798f6bbb6e87a5244dbcaaad2d3` |
| `experiments/joulequeue_policy.py` | `64ff8d8cedf74dea754c167dd53071e66d7ad54e986ee9a2493ca9e06cd66959` |
| `experiments/run_joulequeue_oracle.py` | `b826ef9f9da1bd71812c3beff678a2a72702c069b1dbf94fa1bd6cc2f2a7bb89` |
| `experiments/test_capture_joulequeue_expert_inputs_gpu.py` | `f25a2fb7eec21a3f14b6a37b798429e0f20a5cf897755075d4826b804b3c244f` |
| `experiments/test_run_joulequeue_expert_surface.py` | `2f9a2a3dfae41805244d7233cc763eeb9fe95bff8765937f3d340a209fc7be03` |
| `experiments/test_joulequeue_policy.py` | `4c647986d5f3f22ac338e5d025bd75281468f4ba73f1ae5a3de0d16689d0b467` |

任一文件变化后，本报告不能作为新版本签字。

## 2. P0 缺陷

### P0-JQ-01：两个 arm 的 total-energy 分母不同，能系统性制造“合批节能”

**类型：代码做错 / 能耗会计错误 / 可翻转 E1 符号。**

冻结协议要求 counter 与 sample 的 workload boundary 统一，并以同一窗口总板卡能耗比较（协议 122--140 行）。实现实际做法是：

1. counter start；
2. 额外读一次 power sample；
3. 记录 `workload_start_ns` 并启动 sampler thread；
4. workload 完成后停止 thread、再读一次 power sample；
5. counter end。

证据：`run_joulequeue_expert_surface.py:159-249` 明确把关系命名为 `SEQUENTIAL_BRACKETING_NOT_ATOMIC`；formal loader 反而把这个字符串列为可接受条件（`run_joulequeue_oracle.py:186-194`）。这不是冻结协议中的“同一 boundary”。原始 counter 值和原始 power samples也没有落盘，无法离线复核窗口。

更严重的是，`separate` 与 `coalesced` 分别选择达到 2 s 的 repeat 数（`run_joulequeue_expert_surface.py:685-695`），随后把各自整个 counter window 能耗除以各自 repeat 数（714--715 行）。令固定的前后包围能耗为 `E_o`，则 arm `a` 的估计中含有 `E_o / repeats_a`。通常 `separate` 单次逻辑 batch 更慢、repeat 更少，于是它承担更大的每 batch 固定开销，`separate - coalesced` 被系统性抬高。

**关闭条件：**

- 明确定义唯一计费窗口，并让 counter/sample/raw trace 都绑定该窗口；若硬件接口只能顺序读取，必须在协议中预注册边界误差处理，不能把 “bracketed” 重命名为 “unified”。
- paired arm 使用同一 logical repeat denominator，或给出经审计、不会随 arm 改变的窗口开销消除方法。
- 落盘 counter start/end 原值、读取时刻、所有 power samples、workload 起止、repeat 数和 CUDA event。
- 加入 fake backend 验收：注入任意固定前后开销后，已知相同能耗的 A/B 两 arm 仍须得到零差；交换 repeats 不得改变差值符号。

### P0-JQ-02：E0 hard gate 未实现，formal eligibility 可在无 thermal/environment 证据时自判通过

**类型：协议未写全 + 代码未实现。需先回 Phase 2 补 amendment。**

协议把 thermal gate 列为 E0 hard gate，并要求记录 driver、power limit、clock、温度、模型/权重 hash（协议 39、99--106 行）。当前 surface metadata 只检查 GPU 名称、UUID、source、CV、窗口、次数和数值门（`run_joulequeue_expert_surface.py:505-601`）；没有读取或 gate：

- driver；
- power limit；
- graphics/SM/memory clocks；
- trial 前后温度及热漂移；
- throttling reason；
- 模型权重 hash；
- idle calibration 窗口和 CI。

此外，`maximum_energy_cv = 0.1` 在 `joulequeue_surface_v1.json:11` 新增，但冻结协议未预注册 10% 阈值；协议也未给 thermal 门的数值定义。Phase 3 无权事后补选 formal gate。

**关闭条件：** Phase 2 amendment 先冻结 repeatability/thermal/clock gate、冷却/随机化规则和环境字段；Phase 3 再实现并加 fail-closed 测试。未完成前，`formal_eligible=true` 无效。

### P0-JQ-03：surface CI 使用错误独立单位，E1 的 route-energy-mass coverage 完全缺失

**类型：代码做错 / 统计错误 / 选择对象错误。**

协议指定 surface CI 的独立单位为 `(layer, expert, input event)`，inner repeats 不能当独立样本（协议 191 行）。实现对每个 `(layer,expert,rows)` 固定取一次 `pool[:rows]`（`run_joulequeue_expert_surface.py:670-672`），十个 trial 反复测同一 activation；随后直接 bootstrap 这十个窗口（394--470 行）。这些窗口可以估测仪器重复性，但不能替代十个独立 input events，也不能估测输入异质性。

E1 还要求两个 row bins 各覆盖 calibration expert-energy mass 至少 10%（协议 118 行）。当前汇总只对 16 个被选 experts 等权平均（`run_joulequeue_expert_surface.py:773-829`），未读取 route trace、未计算 actual invocation row bins、未按 energy mass 加权，也没有 coverage artifact。

**关闭条件：**

- calibration 前按冻结 SHA 规则选择独立 input events；每个 event 的 inner repeats只用于该 event 的测量均值。
- CI 以 event 为单位；若跨 expert/layer 汇总，明确层级 bootstrap。
- 生成可审计的 `route_bin_coverage.csv`，逐模型报告每个 row bin 的 invocation count、routed rows、estimated energy mass及覆盖率；E1 gate 从该文件计算。

### P0-JQ-04：没有 route→job producer，现有 loader 与冻结 CJC producer 的真实 schema 不兼容

**类型：代码缺失 + 身份/因果错误。**

协议明确沿用 CJC identity-complete producer（协议 71--87 行）。CJC producer 实际输出字符串：

- `forward_id = "<request>:prefill:0"`（`receiver_aware/experiments/capture_cjc_routes_gpu.py:261`）；
- `token_id = "<request>:tok:<position>"`（289 行）；
- placement 字段为 `placement_manifest_sha256`（293--321 行）。

JouleQueue loader 却把 `forward_id`、`token_id` 强转 `int`（`run_joulequeue_oracle.py:302-333`）；closure helper要求另一个字段名 `placement_sha256`（`joulequeue_policy.py:845-880`）。runner 没有调用 closure helper，也没有 job/arrival builder，只信任 job JSON 中四个布尔 metadata（`run_joulequeue_oracle.py:309-315`）。任意手写 JSON 都可自称 `native_route/full_dependency_replay=true`。

**关闭条件：** 实现唯一的、hash 绑定的 route→job producer；保留字符串 identity，不允许重编号丢失；真实调用 top-k closure；生成 arrival/deadline/service-sample identity；对 old CSV、字段漂移、重复/缺失 sibling 和伪 metadata 做端到端 hard-fail 测试。

### P0-JQ-05：冻结的 4 layers × 4 experts surface 无法支持声称的完整 token closure replay

**类型：Phase 2 问题定义不闭合 + Phase 3 无法执行。**

formal loader强制恰好 16 条 per-expert curves（`run_joulequeue_oracle.py:269-275`），且不为未 profile 的 `(layer,expert)` 提供 default。route-real trace则包含所有 MoE layers和所有被路由 experts。任何未 profile queue 都会触发 `missing surface`；即便只保留 profile queues，也会丢掉 token 的 top-k siblings，无法计算协议定义的 token completion（协议 173 行）。

这是冻结设计本身未说明的缺口，不能在 Phase 3 临时用 pooled curve、平均 expert 或删 route rows 来补。

**关闭条件：** 回 Phase 2 二选一并冻结：

1. profile replay 所需全部 `(layer,expert,row)`；或
2. 把 claim 明确缩为可审计的 per-queue episode oracle，并放弃 token-level / cross-layer P99 claim，同时给出从抽样队列到全系统 headroom 的统计外推协议。

不得用 selected-only token、缺 sibling completion 或 pooled default curve开启 E2。

### P0-JQ-06：cross-layer dependency 未建模，exact oracle 的规模甚至小于一个 LLM-jp token 的 top-k

**类型：科学机制未实现 / oracle 不可执行。**

`simulate_causal()` 只按 job 中预写的 `arrival_us` 把任务放入 ready queue（`joulequeue_policy.py:407-496`）。后层 job 的 release 不会随同一 token 前层 top-k completion、不同 policy 的 completion 时刻而变化，所以不能实现协议要求的“跨层按冻结 dependency 串联”。一个静态 job JSON 的 `full_dependency_replay=true` 不能证明因果依赖。

`exact_clairvoyant_oracle()` 对全 episode做子集枚举，并硬限 `max_jobs=12`（`joulequeue_policy.py:499-580`；配置 `joulequeue_v1.json:51-56`）。LLM-jp 的一个 token 单层就有 top-k=16 个 contribution jobs；因此按当前 job 粒度，连一个完整 token layer都无法进入 oracle。若把多 contribution 聚成一个 `rows>1` job，当前 identity只有一个 token/slot，不能可逆表示多 token完成集合。

**关闭条件：** Phase 2 先冻结 job 粒度与 dependency DAG；Phase 3实现 policy-dependent release。oracle必须采用可扩展的精确分解、MILP/DP或同时报告可校验 upper/lower bound；至少用 top-k=16、跨两层 fixture证明 exactness/闭合 gap，不能只用两 job toy。

### P0-JQ-07：强基线、calibration、主 cells与统计判定未实现

**类型：代码缺失 / 基线不公平 / 统计实现不符协议。**

runner把：

- `best_fixed_timeout` 固定为 20 us；
- `best_static_rows`、throughput、AMoE proxy、Festina proxy均固定为 32 rows；
- `festina_like_profiled` 只是 `StaticRowsPolicy` 的空子类。

证据：`run_joulequeue_oracle.py:336-362`、`joulequeue_policy.py:296-323`。协议要求 calibration grid 选择并在 sealed冻结（协议 153--169 行）。当前没有：

- calibration immediate P99 生成 SLO/max-age；
- timeout/row grid search；
- Poisson rho50 与 MMPP rho80 arrival manifests；
- 两模型 × 两 workload cells × 主 seeds；
- best SLO-qualified baseline 选择；
- 逐 cell paired Go/No-Go。

名为 `paired_hierarchical_bootstrap()` 的函数只对一个扁平 episode list重采样一次（`joulequeue_policy.py:698-754`），没有“先 document、再 seed”的两层结构；它只校验 completed token数量相同，不校验 token identity集合相同。runner也从未调用它。

**关闭条件：** 生成 calibration artifact并锁定每 cell参数；formal runner逐模型/cell/seed执行所有 baseline；比较完整 identity集合；实现预注册两层 bootstrap和 decision artifact。缺任一强 baseline时必须 `BLOCKED`，不能自动跳过。

### P0-JQ-08：replay 主能耗不是冻结的 board-energy 口径，插值也不保守

**类型：能耗会计错误 / oracle objective 可能偏序错误。**

开发 runner使用拍定的 `idle_power_w_dev_only=30.0`（`joulequeue_v1.json:30-36`），然后把 surface launch能耗与这个常数乘等待时间相加（`joulequeue_policy.py:475-493`）。协议明确要求 idle calibration 的窗口、温度/clock与 CI，不得使用拍定常数（协议 129--140 行）。因此该数只能做纯逻辑 smoke，不能生成主 `expert_stage_board_J/completed_token`。

`SurfaceCurve` 把相邻两个 UCB 做线性插值并称为 conservative（`joulequeue_policy.py:143-183`）。相邻点 UCB 的线性连线不保证是未测 row 的 upper bound，也不保证单调；这违背协议 116 行的“预注册单调/保守 upper interpolation”。oracle可能因低估某个未测 row而选择错误动作。

**关闭条件：** measured idle distribution/CI进入同一环境 manifest；主口径对等待区间使用冻结的保守估计。插值改为经预注册验证的 upper envelope，或正式 run只允许 grid 点、其余回退 immediate。

### P0-JQ-09：formal gate 是可编辑布尔自证，且没有绑定决定结论的完整输入/实现

**类型：过程/溯源错误。**

`validate_formal_capabilities()` 只检查四个 config值是否字面等于 `true`（`run_joulequeue_oracle.py:127-146`）；单元测试甚至明确证明“把四个名字写为 true 即通过”（`test_joulequeue_policy.py:323-336`）。这些布尔值没有与 producer、executor或 evidence artifact建立可验证关系。

核心 `hash_manifest` 只覆盖 protocol、policy、oracle runner、一个测试和 config（`joulequeue_v1.json:75-83`），遗漏：

- activation capture；
- surface runner和surface config；
- power meter；
- route/job/arrival producer；
- data、route、surface、arrival、job artifacts；
- calibration选择；
- 统计/decision输出。

formal surface loader又只检查 JSON metadata中的 self-asserted布尔值和一个“看起来像 SHA-256”的 signoff字符串（`run_joulequeue_oracle.py:160-220`），并不验证产生该 surface 的 attestation内容。配置在实际 Review 前还预填 `review_status: SIGNED-OFF`（`joulequeue_v1.json:75-76`）。

最后，runner无论是否 `--formal` 都把 `scientific_result_eligible` 写为 false并输出 `PARTIAL_DEVELOPMENT_ONLY`（`run_joulequeue_oracle.py:71-90,396-430`）；不存在协议要求的 accounting、per-episode、bootstrap和decision产物链。

**关闭条件：** 去掉自证 capability；由外部 Review attestation绑定完整 source/config/protocol hashes，由 source/data manifests绑定输入 artifacts，并由 runner逐级验证。正式输出必须 fail-closed地产生协议 230 行列出的完整产物；任何缺失都不能写 decision。

## 3. P1--P3 缺陷

| ID | 级别 | 类型 | 证据与影响 |
|---|---|---|---|
| JQ-10 | P1 | activation identity 不足 | capture只保存 request/forward/layer/expert、row_count和整块 tensor（`capture_joulequeue_expert_inputs_gpu.py:170-195`），只按 route bincount核对总行数（224--249 行）；没有 row→token/slot映射、row hash或canonical split manifest，无法做 route coverage join或独立 input-event审计。 |
| JQ-11 | P1 | max-age语义漂移 | StaticRows/AMoE在 launch start 时检查 age（`joulequeue_policy.py:280-319`），oracle却要求 completion-age不超过 max-age（553--557 行）。baseline与oracle约束不等价。 |
| JQ-12 | P1 | TPOT proxy定义不稳 | `schedule_metrics()`按 token id排序后直接做 completion time差（`joulequeue_policy.py:658-677`）；并发下可为负，prefill token顺序也不是 decode TPOT。需要冻结非负且因果可解释的定义，或删除该门。 |
| JQ-13 | P1 | latency/energy非同 trial | CUDA event latency在能耗 workload之前另跑一次（`run_joulequeue_expert_surface.py:703-718`），因此 joint `E_e(m),T_e(m)` 来自不同热状态/调用。若只分别建均值需明确；若做联合风险预算则不成立。 |
| JQ-14 | P1 | 原始证据不可重算 | surface只输出 trial摘要；不输出 raw power samples、counter start/end值、温度/clock或独立 `power_trace.csv`，与协议正式产物契约不符。 |
| JQ-15 | P2 | shared meter仍有旧缺口 | `power_accounting.MonotonicNVMLSampler._loop()`未捕获并在 `stop()`重抛后台异常（`power_accounting.py:240-281`）。当前 surface另写了一套 meter，造成两套口径；该 shared helper不得被 formal路径复用。 |
| JQ-16 | P2 | 测试只验证自报 metadata | surface测试用手写 JSON把 `formal_eligible`、窗口、UUID等直接设真（`test_run_joulequeue_expert_surface.py:80-139`），没有 fake NVML counter/sample、repeat分母、thermal drift或raw trace测试。 |
| JQ-17 | P3 | 过时说明 | `run_joulequeue_oracle.py:5-7` 仍称本 bundle没有 native surface producer；实际已有 producer，但未通过本审查。应改为“producer exists but is not signed off”，避免状态混淆。 |

## 4. 冻结验收用例覆盖判定

| 协议用例 | 判定 | 说明 |
|---|---|---|
| 1 / 3 / 6：identity、同 expert、full drain | **PARTIAL** | toy job纯函数有测试；没有 native route→job端到端。 |
| 2：route closure、旧 CSV拒绝 | **PARTIAL/BLOCKED** | helper存在，但字段与CJC producer不一致，runner不调用。 |
| 4：causal无未来信息 | **PARTIAL** | API没有 future参数；没有 dependency-aware runtime。 |
| 5：calibration/sealed隔离 | **BLOCKED** | 有孤立选择 helper，runner未执行 grid/calibration freeze。 |
| 7：单位恒等式 | **PASS（纯函数）** | 不覆盖真实窗口分母。 |
| 8--11：sampler、同窗、异常、UUID、AB/BA、2s | **BLOCKED** | metadata和部分代码分支存在，但关键同窗/分母、环境、raw trace与meter注入测试缺失。 |
| 12：越界/保守插值 | **PARTIAL** | 越界回退有实现；线性插值不是已证明 upper bound。 |
| 13：等待 idle board energy | **DEV-ONLY** | 逻辑恒等式测试通过；idle功率是拍定30W。 |
| 14：数值门 | **PARTIAL** | 单 activation可测；没有独立 input-event层级及route split identity。 |
| 15：oracle exactness | **BLOCKED** | 两 job toy有效，真实 top-k/cross-layer规模不可执行。 |
| 16：baseline/SLO/完成集合比较 | **BLOCKED** | runner没有 calibration和正式比较器。 |
| 17：Review/hash gate | **BLOCKED** | hash清单和artifact绑定不完整，capability可自证。 |
| 18：dev不得出科学结论 | **PASS** | 当前 runner始终输出非科学状态；必须保留。 |

## 5. 最小修复顺序

### 先回 Phase 2（不是代码补丁能解决）

1. 冻结 E0 的 thermal、clock、repeatability、counter-boundary误差门。
2. 决定 4×4 surface如何与全 route token closure相容；若不能相容，缩小 claim和主指标。
3. 冻结 job粒度、cross-layer DAG与 top-k=16 可执行的 exact/bounded oracle形式。
4. 明确 TPOT proxy、max-age是在 launch start还是completion约束。

上述属于协议问题。若直接在 Phase 3 临时决定，后续结果必须 `SUPERSEDED`。

### 再回 Phase 3

1. 修复 NVML计费窗口和 repeat分母；输出 raw power/counter/environment artifacts。
2. capture并冻结独立 input events，补 row→route identity和E1 coverage join。
3. 实现唯一 route→job/arrival producer和policy-dependent dependency release。
4. 实现逐 cell calibration、全部强基线、best-qualified选择、两层bootstrap与decision。
5. 以完整 source/data/artifact manifests取代 config capability booleans。
6. 新增至少这些反例测试：固定测量开销不制造energy saving；string identity round-trip；top-k=16两层closure；missing surface hard-fail；不同完成identity但相同count hard-fail；thermal/source漂移整对trial失效；flat bootstrap不能冒充hierarchical bootstrap。

### 重新 Phase 4

重新审查时至少需要：

- Phase 2 amendment及其hash；
- 完整 source manifest；
- 所有 P0 的反例测试；
- 一份不打开 sealed 的 calibration/dev artifact schema smoke；
- Reviewer独立生成的 `SIGNED-OFF` attestation。

在此之前，不要通过把 `formal_capabilities` 改成 true、删 baseline、改 pooled default curve、把30W常数称为实测、或把 current unit tests称为 E0/E1 evidence 来“救” JouleQueue。

## 6. 当前可执行边界

- **允许：** 纯单元测试；单请求 hook/shape/UUID 的机械 smoke；输出必须 `NOT_TESTED`，且不得读取 sealed。
- **不建议：** 当前脚本完整跑两模型 × 16 experts × 9 rows × 10 trials。它会消耗GPU时间，但 P0-JQ-01/02/03 使其不能回答 E0/E1。
- **禁止：** `--formal` surface、formal oracle、sealed run、Go/No-Go、论文数字。

最终签字：**BLOCKED**。
