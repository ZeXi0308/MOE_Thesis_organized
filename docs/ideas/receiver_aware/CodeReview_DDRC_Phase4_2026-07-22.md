# DDRC Phase 4 严格代码审查

状态：**BLOCKED**  
日期：2026-07-22  
审查边界：只审查冻结协议及 Phase 3 新增实现；未修改被审代码，未采信任何 dev 输出为科学结果。

## 结论

**不得进入正式 Phase 5。** 当前实现是一个有用的纯策略/账本 reference 和 dev smoke runner，但不是冻结协议要求的端到端 formal 实验实现。核心 sender-local / receiver-credit API 的信息集隔离、credit 编解码、基础字节闭合、late/hard-gate 回滚和 `NOT_RDMA` 边界大体正确；然而真实 route/quality producer 未实现且未纳入 source hash，数据隔离、质量动作绑定、强基线集合、两模型门、分层 bootstrap、G1 先行停止门和完整 G2 会计均未闭合。

配置自己也明确声明四项 formal capability 为 `false`（`configs/ddrc_v1.json:147-153`）。因此本报告不生成 `SIGNED-OFF` attestation；任何 formal 运行只允许得到 `PARTIAL`，不得解释为 GO 或科学 No-Go。

## 审查对象与固定摘要

| 文件 | SHA-256 |
|---|---|
| `DDRC_Phase2_冻结实验协议_2026-07-22.md` | `94a22584d9b67470fe5efa3b2245a8ad57980e6c0d75ac11452342c08d479289` |
| `configs/ddrc_v1.json` | `71870ba0e9efd5c0cc53546985019f06f8ac4c307d0cf08a39f7872ba222d8b1` |
| `experiments/ddrc_policy.py` | `9c14098f50af044102d2ad72478f785cc9394ec3094b51bffcc4524ef9aa6028` |
| `experiments/run_ddrc_existence_gpu.py` | `a4ebe462e00c298f0fc02859f64357fa940fccc1c4421e69247c0594ecf4073d` |
| `experiments/test_ddrc_policy.py` | `40a50140517b9d048288920bd6077c69769b143e5770cca4f1590c937e551ee9` |

这些摘要只标识本次审查快照，**不是签字 attestation**。

## P0：必须返回 Phase 3 修复

### P0-1：没有可审计的真实 route / action-matched quality producer，source hash 也没有覆盖 producer

证据：

- runner 明确只消费外部 route matrix 和 quality CSV（`experiments/run_ddrc_existence_gpu.py:2-24,84-94`），整个文件没有加载冻结模型、捕获 native router output 或执行 FP8/INT4 combine action 的实现。
- `SOURCE_FILES` 只包含 policy、runner 和 test（`experiments/run_ddrc_existence_gpu.py:67-74`）；任何生成 `trace_jsonl` / `quality_csv` 的代码均不在 `source_sha256` 内。
- quality 输入只有聚合 harm、CVaR 和一个调用方提供的字符串签名（`experiments/run_ddrc_existence_gpu.py:504-529`）。该签名只由 lane action 等账本字段构造（`experiments/run_ddrc_existence_gpu.py:488-496`），不绑定 token-row identity、expert output、模型 revision、量化 tensor、logits 或样本身份。
- 配置显式承认 `native_route_capture=false` 和 `native_action_matched_gpu_quality=false`（`configs/ddrc_v1.json:147-151`）。

影响：无法证明系统账本和质量行来自同一真实 route/action；任意外部程序甚至手写 CSV 都可构造匹配签名。因而 G0、质量门和 utility 均不可采信。

关闭条件：实现并审查冻结模型 producer；记录 checkpoint revision、逐 token 的 origin、expert-owner、layer/step、有效/丢弃身份和 action；在同一执行中施加实际 homogeneous FP8/INT4 回程动作并产出 logits/quality。producer 及其配置必须纳入 source manifest；quality 行必须绑定 producer-run UUID、样本 SHA、完整逐层 action-trace SHA 和模型 revision。

### P0-2：formal data manifest 只验两个文件 hash 和一个布尔值，未实现冻结数据隔离

证据：

- `validate_formal_inputs` 只验证 trace/quality 文件 SHA 和 `sealed is True`（`experiments/run_ddrc_existence_gpu.py:817-829`）。
- trace parser 只相信调用方给出的 split 字符串，并验证 lane closure/LPT（`experiments/run_ddrc_existence_gpu.py:217-268`）；它不验证十个 subject、4/12 每 subject、seed、canonical question/choices/answer SHA、历史 receiver registry 不重叠、模型 revision、模型对应 top-k/num-experts。
- `model` 只检查是否为配置中的 key（`experiments/run_ddrc_existence_gpu.py:320-325`），而 `top_k` 没有和该 model 的冻结值比较。
- `LaneMatrix.validate_closure` 允许从 `top_k * valid_origin_tokens` 中减去任意 `dropped_pairs_by_receiver`（`experiments/ddrc_policy.py:318-333`），但主命题预设无 token drop，冻结验收要求精确等式。

影响：错误数据集、重叠样本、错误模型/top-k 或带 drop 的 trace 都能通过 formal gate，直接改变 route 分布、质量和收益。

关闭条件：定义强 schema 并 fail closed；逐项验证冻结 subject/split/count/seed/sample SHA/registry exclusion/model revision/top-k/num-experts，主 balanced cell 强制 `dropped_pairs_by_receiver == 0`。manifest 由受审 producer 生成并包含 selection registry 的内容 hash，而不是仅由人工写入 `sealed:true`。

### P0-3：统计层级不符合预注册；当前 CI 是 trace-id IID bootstrap，P99 没有 block bootstrap

证据：

- quality schema 没有 subject、request、forward 或 burst-block identity（`experiments/run_ddrc_existence_gpu.py:507-528`）。
- bootstrap 以 model 下的 `trace_id` 为总体并 IID 有放回抽样（`experiments/run_ddrc_existence_gpu.py:599-657`），没有 subject/request 分层。
- P50/P95/P99 仅计算点分位数（`experiments/run_ddrc_existence_gpu.py:536-571`），没有冻结协议要求的 forward/burst block bootstrap 或 P99 CI。
- 配置也声明 `burst_block_bootstrap=false`（`configs/ddrc_v1.json:147-151`）。

影响：同一 request、同一 forward 内的 layer/step 相关性会被当作独立样本，CI 可被严重压窄；正式 margin/P99 判定不合法。

关闭条件：producer 落盘 `subject_id/request_id/forward_id/burst_block_id`；质量按 subject→request 的 paired hierarchical/stratified bootstrap，系统尾延迟按 forward/burst block 重采样；为每个模型、lambda 输出有效样本数和 CI，并加“打乱输入行顺序结果不变”的测试。

### P0-4：`causal_prev_step` 被排除在 best strong baseline 之外，且该 arm 自身可按输入顺序读到非因果 trace

证据：

- 冻结协议把 `causal_prev_step` 列为基线（协议 `:123-133`），但 `STRONG_BASELINES` 只含 uniform、static、sender-local（`experiments/run_ddrc_existence_gpu.py:76-81`）；paired utility 和 P99 best-baseline 均使用这个缺项集合（`experiments/run_ddrc_existence_gpu.py:563-569,603-636`）。
- `previous_by_model` 只按 JSONL 遍历顺序更新（`experiments/run_ddrc_existence_gpu.py:409,456-462,498`），没有 stream/request identity，也不验证 step/layer 单调。文件中排在前面的未来 step 或其他 request 可被当成“previous”。

影响：候选可能绕过一个更强的简单基线；同时现有 causal arm 本身可能 look-ahead 或跨请求污染，不能直接加入比较。

关闭条件：把修正后的 `causal_prev_step` 纳入强基线；输入必须有 stream/request identity 和严格事件序，parser 拒绝逆序/重复，previous state 按独立 stream 维护。加反例测试：调换 JSONL 行不能把未来 step 变成过去信息。

### P0-5：decision 没有强制两个冻结模型，单模型输入可以错误地产生 GO

证据：

- decision 从实际 accounting rows 推导 `models`（`experiments/run_ddrc_existence_gpu.py:713-720`），从未读取/比较配置中的 `required_models`（`configs/ddrc_v1.json:128-135`）。
- 随后的“两模型”门只是对当前字典做 `all(...)`（`experiments/run_ddrc_existence_gpu.py:763-770`）。

复现：将四个 capability 和 timing source 临时设为可用，传入仅 `olmoe` 的合成 accounting/bootstrap，`build_decision` 返回 `GO_TO_REAL_TRANSPORT_NEXT_ITERATION`，且错误地把门命名为 `*_both_models=true`。

影响：直接违反“两模型 balanced cell 均复现”的硬门，可翻转科学结论。

关闭条件：要求 observed model set **严格等于** `required_models`，每个模型分别检查 calibration/sealed cells、arm 完整性、样本下限和所有门；缺一模型只能 `PARTIAL`，不得 `NO_GO/GO`。

### P0-6：没有实现冻结的双 codec 口径与同 action-trace 敏感性门

证据：

- main 强制只运行 `serialized_tiles`（`experiments/run_ddrc_existence_gpu.py:845-851`）。
- decision 只检查该字符串是否为 `serialized_tiles`（`experiments/run_ddrc_existence_gpu.py:759-770`），没有 `amortized_once_per_step_proxy` 的成对结果，也没有验证两种口径共享 action trace。
- 冻结 GO 要求 optimistic 口径达门且 serialized 同 trace 优势仍为正（协议 `:224-234`）。

影响：当前代码即使其余缺陷修完，也可能在未满足 codec 敏感性门时错误 GO。

关闭条件：策略只生成一次 immutable action trace；账本对该 trace 同时计算两个明确分栏的 codec 口径，分别 bootstrap，并在 decision 中执行冻结条件。quality 不得重跑或随口径改变 action。

### P0-7：G1 oracle ceiling 没有在 sealed quality 运行前执行停止

证据：

- formal mode 一开始就强制要求 `quality_csv`（`experiments/run_ddrc_existence_gpu.py:855-857`），并在 G1 前加载 quality（`:867-885`）。
- G1 直到 `build_decision` 内才计算（`experiments/run_ddrc_existence_gpu.py:718-733`）。外部 quality CSV 在 runner 启动前已经被生成，说明 sealed quality execution 更早发生。
- 冻结协议要求两个模型 G1 未达 3 pp 时“不启动 sealed quality run”（协议 `:44-48`）。

影响：违反预注册停止规则并无谓打开 sealed quality；也无法证明质量评估未受先看结果影响。

关闭条件：拆成不可跳过的两段 formal state machine：`G1_ROUTE_ONLY` 只消费封存 route trace并落盘通过凭据；只有两个模型均通过才授权 action-matched sealed quality producer。第二段验证第一段 artifact hash。

### P0-8：G2 时间/资源会计缺少冻结字段，当前 net saving 不是完整主口径

证据：

- `StepLedger` 只有 payload/scale/descriptor/padding、一个 aggregate critical wire、pack/H2D/unpack 和 credit（`experiments/ddrc_policy.py:854-883`）。
- `_account_phases` 虽构造 sender/receiver map，最终只保留 aggregate/max（`experiments/ddrc_policy.py:892-935`）；输出也没有 sender egress、receiver ingress、physical shared-cut 的独立账本（`experiments/run_ddrc_existence_gpu.py:933-950`）。
- 冻结要求的 compaction、launch、reduction、stream wait 未实现；每个输出时间字段也没有随行 `measured_same_run | measured_other_gpu | analytic | assumed` 来源（协议 `:163-191`）。
- receiver-resource view 在单进程中免费获得整组 receiver ranks 的 counts（`experiments/ddrc_policy.py:379-397`）；代码只收一个固定 `aggregate_us`，没有对 receiver-rank→coordinator 的计数/同步字节和规模进行闭合。

影响：遗漏或错误聚合的成本可以改变 positive-net hard gate、oracle ceiling 和 3 pp utility margin。当前账本只适合作边界清晰的 proxy smoke，不能打开 G2。

关闭条件：实现逐资源 sender egress/receiver ingress/shared-cut closure；补齐所有冻结时间字段和逐字段来源；receiver-side aggregation 的 bytes/messages/timing 随 active lanes/peers 计费；定义并测试 critical-path composition。若某项仅 analytic，必须在主表显式标注，不能藏在一个 aggregate 数字中。

## P1：P0 关闭后仍必须修复或解释

### P1-1：G0 检查不能证明“增量来自 sender 来源分解”

`g0_by_model` 只检查 DDRC 与 sender-local 的 `requested_lanes` 是否在任意 trace 不同（`experiments/run_ddrc_existence_gpu.py:724-727`）。不同阈值作用域本身就可能产生差异，并不证明固定 receiver 总 rows 时来源分解改变了动作。应增加配对反事实：固定每 receiver 的总 rows、拓扑和 calibration 阈值，仅重排 sender decomposition；验证 DDRC action/utility 随之变化，并对 `K×valid_origin_tokens` 重写基线保持不变。

### P1-2：两个验收测试为空测或同源自证

- `test_receiver_credit_is_local_to_one_receive_resource` 构造的所谓 modified view 与原 view 完全相同（`experiments/test_ddrc_policy.py:183-200`），没有改变另一资源，无法检测泄漏。
- 没有“DDRC 聚合矩阵 vs 独立 CPU reference”的独立实现；现有测试主要调用 production helper 本身。
- future/label 测试只检查函数参数名并用同一输入重放（`experiments/test_ddrc_policy.py:294-313`）；由于真正的 route/quality producer 缺失，它不能证明端到端无 future/label leakage。

### P1-3：origin LPT tie-break 与冻结文本不一致

冻结协议规定 request hash、rank id tie-break（协议 `:135`）；实现按传入 request-id 字符串排序（`experiments/ddrc_policy.py:48-70`）。应显式计算冻结的 canonical request hash，并把 hash 与 assignment 写入 manifest。

### P1-4：冻结质量输出不完整

quality schema 只含 incremental accuracy harm 和 CVaR（`experiments/run_ddrc_existence_gpu.py:507-528`），缺 BF16 absolute accuracy/NLL、lost/gained correct、answer-NLL harm（协议 `:193-220`）。这些字段必须由同一 producer 生成，并在报告与 gate 中使用或明确仅报告不决策。

### P1-5：missing-credit 路径没有可执行的 fault model

`apply_receiver_credit_messages` 能处理 late/duplicate/malformed，但“预期应有而实际丢失”的 message 与“本来无 credit”都表现为空列表（`experiments/ddrc_policy.py:611-699`）。应传入冻结 expected-message/epoch set 或序列号，验证 missing 后 FP8 fallback、状态回滚且已发生成本保留。

## P2 / P3

- **P2**：`status.json.formal_scientific_result` 只在 `decision.go` 为真时为真（`experiments/run_ddrc_existence_gpu.py:965-970`）；合法 formal No-Go 也会被标成 `false`。应区分 `formal_run_valid`、`scientific_verdict` 和 `go`。
- **P2**：测试模块使用工作目录相关的裸 import；从仓库根运行 `python -m unittest docs/.../test_ddrc_policy.py` 会找不到 `ddrc_policy`，从 experiments 目录运行才通过。补包入口或文档化唯一命令。
- **P3**：无阻断性风格问题；长行/输出 schema 可在 P0/P1 修复时统一整理。

## 已通过的局部检查

- 从 `docs/ideas/receiver_aware/experiments/` 运行 `python -m unittest test_ddrc_policy.py`：**19 tests passed**。
- 三个 Python 文件 `py_compile`：通过。
- dev runner 可运行并输出完整文件集合；`decision.json` 正确为 `NOT_TESTED`，未伪造科学 verdict。
- 局部正面证据：sender-local API 不接收 global matrix（`ddrc_policy.py:550-573`）；DDRC 通过序列化 credit 输入 sender（`:576-699`）；credit header/record/alignment、late/duplicate/hardgate rollback、byte closure 和 `NOT_RDMA` 均有单测。

这些局部通过项不能抵消 P0；它们只说明 reference 层可继续复用。

## 重审入口与最小关闭顺序

1. 先补受审的 native route + action-matched quality producer、强 manifest 和 producer/source hash；在此之前不要打开 sealed quality。
2. 修 formal state machine：G1 route-only → 两模型 gate → sealed quality；强制 required models、完整 arms 和 exact split cells。
3. 补 causal baseline 的 stream 时序，纳入 best strong baseline。
4. 实现 subject/request hierarchical bootstrap、forward/burst block bootstrap 和双 codec 同-trace 账本。
5. 补完整 G2 资源/时间会计及 receiver aggregation 成本。
6. 按协议 16 项验收重写非空测，并增加本报告各 P0 的 adversarial negative tests。
7. 全部 P0 关闭后提交新的不可变 source/config/protocol hashes进行独立复审；只有新报告明确写 `SIGNED-OFF` 才可启动正式 Phase 5。

当前允许动作：继续 Phase 3 修补和 dev smoke。  
当前禁止动作：正式 sealed quality、正式 Go/No-Go、RDMA/NCCL/P99 claim。

---

## 返修复审（2026-07-22，23 tests 快照）

复审状态：**BLOCKED（维持）**  
复审原则：本节是对第一次审查缺陷状态的最新判定；上文保留为历史证据。没有生成 `SIGNED-OFF` attestation，也没有授权正式 Phase 5。

### 返修快照

| 文件 | 返修后 SHA-256 |
|---|---|
| `configs/ddrc_v1.json` | `9c77633ca0f2b0c0d31921ed3e1a55720480711b31c18b4b89814fc4caaebeae` |
| `experiments/ddrc_policy.py` | `b42dce241bb9bbc33c9c75e11f2f6e656f4c53c474a711a6fe0235133b8de137` |
| `experiments/run_ddrc_existence_gpu.py` | `c59ebeeea058e65f7c1af5798ecd09e31877c7a7acf9a1a675fd9458c82f4c99` |
| `experiments/test_ddrc_policy.py` | `4e149f2abd8c10594e09627b88859e167ecd40bdbb588ba2cab0a8d096538802` |

冻结协议未改，仍为 `94a22584d9b67470fe5efa3b2245a8ad57980e6c0d75ac11452342c08d479289`。上述摘要仍只标识审查快照，不代表签字。

### P0 关闭矩阵

| 原缺陷 | 返修状态 | 复审证据与剩余阻断 |
|---|---|---|
| P0-1 native route / action-matched quality producer 与 producer hash | **仍开放** | `SOURCE_FILES` 仍只有 policy/runner/test（runner `:71-75`）；没有模型 route/quality producer。`native_route_capture`、`native_action_matched_gpu_quality` 仍为 `false`（config `:148-152`）。外部 quality/action 的端到端因果绑定仍不可签。 |
| P0-2 formal 数据隔离/schema | **部分关闭** | 已在 formal load 强制 model-specific top-k、非空 stream、流内严格顺序，并拒绝 sealed 主 cell 的 drop（runner `:225-274,327-355`）；对应负测见 test `:458-534`。但 manifest 仍只验 trace/quality hash 与 `sealed:true`（runner `:975-987`），未验 subject、4/12 数量、seed、sample SHA、历史 registry、模型 revision/producer；calibration drop 也未拒绝，仍可污染阈值。 |
| P0-3 hierarchical / block bootstrap | **仍开放** | `paired_bootstrap` 仍按 model 内 `trace_id` IID 抽样（runner `:685-786`），schema 仍无 subject/request/forward/burst identity；P99 block bootstrap 未实现。`burst_block_bootstrap=false`（config `:148-152`）。 |
| P0-4 causal strong baseline 与时序 | **代码层已关闭** | `causal_prev_step` 已加入 `STRONG_BASELINES`（runner `:77-83`）；state/previous 按 `(model,stream,layer)` 隔离（`:441-442,472-495,562`）；policy 强制同 stream、同 layer、严格过去 step（policy `:803-843`）；formal parser 拒绝逆序。端到端 stream 身份真实性仍归 P0-1 producer 阻断。 |
| P0-5 两模型硬门 | **已关闭** | decision 强制 observed model set 与 `required_models` 完全相等，缺失或意外模型只返回 `PARTIAL`（runner `:801-819`）；单模型反例负测通过（test `:438-456`）。 |
| P0-6 双 codec 同 action-trace | **部分关闭** | 已添加 sensitivity 配置（config `:22-35`），策略只生成一次，随后用相同 `plan` 对两种 codec 记账（runner `:507-561`），并验证 action signature/cardinality（`:568-612`）。但函数主动标注 `ACCOUNTING_ONLY_NO_SCIENTIFIC_VERDICT`，dual-codec 结果尚未 paired/block bootstrap，也未进入 `build_decision` 的 GO 门（decision `:889-901` 仍只检查 serialized 主口径）。因此不能视为冻结条件 6 已关闭。 |
| P0-7 G1 先于 sealed quality 的状态机 | **仍开放** | formal 仍先要求并加载 `quality_csv`（runner `:1013-1029`），之后才在统一 decision 中计算 G1（`:847-863`）。尚无 `G1_ROUTE_ONLY -> pass artifact -> sealed quality` 的不可跳过两阶段入口。 |
| P0-8 完整 G2 时间/资源会计 | **仍开放** | 返修没有补 sender egress、receiver ingress、physical shared-cut 分栏，也没有 compaction/launch/reduction/stream-wait 与逐字段来源；receiver-side aggregation 仍是未闭合的固定 proxy。`measured_credit_timing=false`（config `:148-152`）。 |

结论上，**2 项已关闭、2 项部分关闭、4 项仍开放**。任何一个开放/部分关闭 P0 都足以阻止签字；配置中的四项 capability 仍全为 `false`，是对当前 `PARTIAL` 边界的正确自描述，而不是可忽略的 TODO。

### 其他缺陷状态

- 原 P2“formal No-Go 被标成非科学结果”已关闭：`build_status` 已把 `formal_run_valid`、`scientific_verdict`、`go` 分开（runner `:923-948`），且有 GO/No-Go 语义负测（test `:536-548`）。
- 原 P1-1 G0 非退化证明、P1-2 空测/同源自证、P1-3 request hash tie-break、P1-4 完整质量字段、P1-5 missing-credit fault model：**仍开放**。
- 原 P2 测试依赖工作目录：**仍开放**；本次仍需从 experiments 目录运行 unittest。

### 复审执行证据

- 从 `docs/ideas/receiver_aware/experiments/` 运行 `python -m unittest test_ddrc_policy.py`：**23 tests passed**。
- 三个 Python 文件 `py_compile`：通过。
- dev runner：`NOT_TESTED`；`formal_run_valid=false`、`scientific_verdict=null`、`go=false`。
- dual-codec dev 账本：28 条 primary action rows、28 条 sensitivity rows，signature closure 通过；状态明确为 `ACCOUNTING_ONLY_NO_SCIENTIFIC_VERDICT`。

### 最终门禁

**最终结论仍为 BLOCKED。** 本轮返修显著收紧了 reference runner，且没有用叙事把局部修补伪装成 formal 能力；但它仍不是冻结协议的正式实验闭环。下一次复审至少必须提交：

1. 纳入 source hash 的 native route + action-matched quality producer及强 manifest；
2. subject/request hierarchical bootstrap 与 forward/burst block bootstrap；
3. 不可跳过的 G1 route-only 前置门；
4. dual-codec 的 paired decision gate；
5. 完整 G2 会计和 measured credit path。

在这些 P0 关闭前，GPU 只能用于 producer/codec/timing 的 Phase 3 开发和 dev smoke，不能执行或采信 formal sealed 结果。
