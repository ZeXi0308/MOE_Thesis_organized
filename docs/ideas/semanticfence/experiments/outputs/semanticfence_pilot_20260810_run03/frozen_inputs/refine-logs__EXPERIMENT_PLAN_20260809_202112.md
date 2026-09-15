# SemanticFence-MoE 第一轮实验计划

> 生成时间：2026-08-09 20:21 +0800  
> 状态：`PLANNED_NOT_RUN`  
> 当前唯一实验：OLMoE / RTX 5090 的 calibration-to-fresh-eval expert rebatching Pilot

## 0. 本轮结论

完整 Idea 保持为 **SemanticFence-MoE: Route-Sealed, Regime-Certified Expert Rebatching**。本轮不做 serving、EP、多卡或完整 runtime，只回答一个必要问题：在固定 route ledger 后，能否仅用 calibration 得到一个非退化的、版本绑定的 executor contract，使 fresh evaluation 中若干真实跨请求 expert rows 在 `M>1` 合批时仍逐 row raw-BF16 等同于隔离 `M=1`，并保留局部延迟收益。

当前没有新 GPU 结果。旧 SpectatorRoute run01 仍是 `INVALID diagnostic-only`，只能提供实验动机，不能生成 allowlist、不能进入本轮结果。

## 1. 稳定研究锚点

- **Bottom-line problem**：MoE serving 为吞吐而跨请求合并 expert rows 时，batch composition / execution shape 可能改变底层执行 regime 和 raw-BF16 结果；若 route 或输出语义必须可复现，简单地允许任意合批或全局隔离都不可取。
- **Must-solve bottleneck**：必须先证明存在一个同时满足“逐 row exact”与“非零合批空间”的 operating region；否则 SemanticFence 要么不安全，要么退化为全部 `M=1`。
- **Core mechanism**：canonical route-determining domain 先产生不可变 route ledger；随后只在版本绑定、calibration 认证、未知即回退的 executor equivalence class 内稳定合批，并保持 row identity 与 scatter/combine 顺序。
- **Target outcome**：在保持选定 canonical semantics 的前提下，降低 expert-stage 延迟，最终才讨论 request-level latency / throughput。
- **Non-goals**：本轮不证明端到端模型质量、后续 route 传播、vLLM/DeepEP 集成、EP/NCCL/RDMA、多租户隔离、跨模型或跨 GPU 泛化，也不声称经验 calibration 是形式正确性证明。
- **Constraints**：本地为 Darwin arm64、无 CUDA；RTX 5090 当前连通性和实际 UUID 未验证。GPU Pilot 必须创建新 config、acceptance、lock 和 output root，不能继承旧 Spectator lock 的硬件身份。

## 2. 当前证据账本

### Confirmed

- 当前权威状态仍是：没有已正式验证的 MoE 系统机制。
- 本地 profile 已观察到 MoE blocks 和 expert loop 占 decode profile 的较大部分；这只支持问题重要性，不证明 SemanticFence 收益。
- 现有 calibration 8 docs 与旧 sealed 32 docs 的 canonical full-text SHA-256 文档级交集为 0，manifest SHA 分别为 `bfb8912539806d2948595eb7ba42cfb7d09aae0b31c7c00dfbc62136abc82630` 与 `469e5da28dc794e50f9e3d8b1d6b2b13dfb7079d1bb6fdf9d00cd41b7c4d0d11`。
- 现有 Spectator runner 已具备真实 pretrained capture、raw-BF16 比较、cuBLASLt trace、环境/污染检查、hard deadline 和 `COMPLETE-last` 基础能力。

### Suggestive

- BCRD router-boundary 与 SpectatorRoute 曾出现 batch/execution-shape 相关数值差异线索。
- 这些线索的正式证据均无效或不完整，因此只能用于提出本 Pilot，不能写成已观察科学结论。

### Negative

- 旧 `x.repeat(M)` / row-0-only 设计不能认证真实 heterogeneous batch 中的每个 row。
- 旧 N05 `reference_m=64` 不适合作为 SemanticFence 的语义真值；它仅是旧 capability gate 的 comparator。

### Unresolved

- 在 fresh rows 上是否存在可从 calibration 泛化的 `M>1` exact executor class。
- 该 class 的覆盖是否足以产生至少 10% 的本地 expert-stage 延迟改善。

## 3. 当前因果链与本轮位置

跨请求 expert 合批改变真实 `M` 及执行 regime
→ raw-BF16 expert row 可能相对隔离语义变化 **[只有无效诊断线索]**
→ calibration 可识别版本绑定的等价 class **[待验证]**
→ SemanticFence 只放行 class 内 `M>1`，其余回退 `M=1` **[可实现，效果待验证]**
→ held-out expert rows 保持 exact 且减少隔离调用 **[本轮唯一验证目标]**
→ full-layer / request-level latency 改善 **[本轮不验证]**。

当前阶段是 **B→C：机制分解到可控性验证**。Pilot 同时保留 unrestricted baseline，用合格的新 artifact 重新确认“问题实例非空”，但不跨越到 serving 系统化。

## 4. Claim map

### C1：主 claim（本轮可授权或削弱）

在固定 OLMoE、BF16、单 RTX 5090 stack 和 sealed route rows 下，仅由 calibration 建立的 versioned executor contract，能在 fresh evaluation 上放行至少两个自然 `M>1`，并使所有 admitted real rows 与各自隔离 `M=1` raw-BF16 完全一致。

### C2：支持性 claim（本轮只看局部）

在相同 real-row multiset 下，contract-constrained packing 相对 all-row `M=1` isolation 的 paired median expert-stage CUDA latency 至少降低 10%，且不被固定 `M=64` padding control 在 exactness 与 latency 上同时覆盖。

### 反 claim / 必要区分

- 若 unrestricted natural-M 对 `M=1` 完全无差异，则当前 stack 下没有需要 fence 的问题实例。
- 若 fixed `M=64` padding 与 SemanticFence 同样 exact 且不更慢，则简单 fixed-shape baseline 覆盖当前机制。
- 若只有看过 evaluation 输出后才能决定 allowlist，则不是可部署 contract。

## 5. 下一轮唯一核心问题

> 在固定 OLMoE route ledger、BF16 stack 和完全相同的 fresh real-row multiset 下，只使用 calibration 建立的 versioned executor contract，是否能让至少两个自然 `M>1` 的 heterogeneous expert batches 对 all-row isolated `M=1` 保持逐 row raw-BF16 exact，并带来至少 10% 的 expert-stage CUDA latency 降低？

## 6. 唯一最小实验

### 6.1 Canonical reference

Canonical 不是单个 victim 的 row 0，也不是 `M=64`。对每个 routed real row 单独调用对应 expert（`M=1`），再按冻结的 row identity 顺序拼成 **all-row canonical vector**。本 Pilot 把它限定为 decode-style expert-stage 的 single-request isolation proxy。

每个 row identity 至少绑定：`split / text_sha256 / offset / token_position / layer / expert_id / topk_rank / hidden_raw_sha256`。比较 `expert(batch)[i]` 的全部 BF16 uint16 bits，禁止只比较 `[0]`。

### 6.2 数据与 workload

- **Calibration**：只读现有 8-doc calibration manifest；offset `{0,256}`。为增加真实异质 pack 覆盖，使用每个 16-token window 的全部 16 个 hidden positions 建 contract；这些 row 仅用于 calibration。
- **Fresh evaluation**：新生成 32-doc manifest，使用固定新 salt；显式排除 historical registry 以及现有 calibration / sealed / smoke 共 42 docs 的 full-text hashes。生成后先冻结 manifest hash，再运行；不得回看结果修改 contract。
- **Evaluation logical requests**：32 docs × 2 offsets × victim position 15 = 64 rows；每层使用 frozen native top-8 route ledger，在同一 `(layer, expert_id)` 内形成自然 heterogeneous batches。
- **M grid**：`{1,2,4,8,16,32,64}`。Evaluation 每层最多 64 logical rows，因此 unrestricted/fixed arm 不需要超出该 grid。

旧 32-doc sealed 只允许作为开发 replay，标签必须是 `not_fresh_holdout`，不得替代 fresh evaluation 或进入本轮 verdict。

### 6.3 自变量与四组 arm

| Arm | 定义 | 作用 |
|---|---|---|
| A — isolated `M=1` | 每个真实 routed row 独立执行，按 row ID 拼接 | canonical reference + full-isolation latency |
| B — native unrestricted | 每层每 expert 将全部自然 rows 一次合批，`M` 为真实队列长度 | 验证问题实例是否非空；吞吐侧 baseline |
| C — fixed `M=64` | 使用同一真实 rows，不足 64 以 zero rows padding；只比较真实 row 输出 | 排除 D 只是 fixed padding / stable shape |
| D — SemanticFence | 仅按 calibration 冻结的 descriptor allowlist 选择最大可用自然 `M`；禁止 padding；未知类、余数或 signature 不匹配回退 `M=1` | 被测机制 |

四组必须保持相同 model weights、route ledger、routing weights、BF16 flags、row multiset 和逻辑 row 顺序。各 arm 只改变 expert packing policy。

### 6.4 Contract 规则

- Pre-call descriptor 固定包含 model/weight hash、runner/source hash、GPU/driver/CUDA/torch/transformers/cuBLASLt digest、math flags、layer、expert、`M`、dtype、三投影 shape 和稳定 row-slot policy。
- 对每个 `(layer, expert, M)`，至少有 3 个不重叠 calibration packs 且涉及至少 3 个 calibration docs，才有资格评估。
- 每个 pack 运行 10 次；只有所有真实 row 均 10/10 stable、且每次都与各自 `M=1` raw-BF16 exact，才加入 allowlist。
- Contract 同时保存 calibration 中预期的三 GEMM tactic signatures；evaluation admission 只能读取 pre-call descriptor，不能读取当前输出。独立 trace replay 发现 actual signature 漂移时 fail closed，本 Pilot判“无法判断”，不得事后把该 row 吸收入 allowlist。
- Evaluation artifact 不得反向扩充、拆分或修改 allowlist。

### 6.5 因变量、主指标和必要诊断

仅两个主指标：

1. **Exactness**：相对 A 的 admitted real-row raw-BF16 mismatch count/rate；支持要求 D 为 `0`。
2. **Local latency**：同一 row multiset 下 D 相对 A 的 paired median expert-stage CUDA-event latency；支持阈值为至少 `10%` 降低。

必要但非主指标：B 的稳定 mismatch victim 数、D 的 `M>1` victim/row 覆盖、自然 `M>1` 种类数、C 的 exactness/latency、每次 call 的 actual `M` 和三 GEMM signature、人工 padding row 数。

### 6.6 重复与计时

- 每个 arm 3 次 warmup、10 次正式 repeat；A 的每个真实 row 必须 10/10 raw-bit stable。
- 用冻结的轮转顺序对 A/B/C/D 做 paired repeats，避免固定 arm 顺序漂移。
- CUDA Events 包围完整的预建 pack dispatch + expert calls + scatter；planner 的 CPU 时间单列记录，不混入 GPU 主指标。
- Numeric/timing worker 禁用 cuBLASLt logging；独立 trace worker用同一 call index 重放并绑定输出 hash，避免 trace 开销污染 timing。

### 6.7 执行顺序

1. **Local preflight**：生成 fresh-32 manifest；实现纯 contract/packer；CPU tests 验证无泄漏、无漏/重 row、unknown→M1、任一 mismatch 撤销 `M`、tamper fail-closed。
2. **GPU acceptance**：在实际可用 RTX 5090 上观测并冻结新 stack identity；禁止复用旧 UUID/driver lock。
3. **Atomic Pilot**：一次模型加载完成 calibration capture → A references → calibration real packs/trace → 写入并 hash-seal contract → fresh evaluation 四臂 numeric/timing → 独立 trace replay → parent 重算。
4. **Completion**：只有所有 raw artifacts、hashes、contract、environment、failure authority 和 summary 完整后，最后写 `COMPLETE.json`。

## 7. 预定义结果解释

### 支持当前机制

同时满足：

- A 全部 real rows 10/10 bitwise stable；
- B 至少 8/64 evaluation victims 出现可重复的 raw-BF16 mismatch；
- D admitted rows 为 0 mismatch，覆盖至少 8/64 victims、至少两个不同的自然 `M>1`，且 artificial padding 为 0；
- D 相对 A 的 paired median CUDA latency 至少降低 10%；
- C 未在 exactness 与 latency 上同时不差于 D。

这只授权下一步做 full-layer 传播实验，不授权 serving、跨模型或形式正确性 claim。

若 exactness/coverage 满足但 latency 未达 10%，只支持“非平凡等价类存在”，削弱当前局部收益 claim；下一步先修改 pack selection / operating region，不扩大系统实现。

### 削弱当前机制

以下任一结果会削弱当前 formulation：

- B 对 A 为 0 mismatch：当前 stack/region 没有观察到需要 fence 的问题实例；
- D 只能使用 `M=1`，或只能依赖一个固定 `M` / padding 才 exact：机制退化；
- D 任一 admitted fresh row 出现稳定 mismatch：calibration-derived safe admission 不成立；
- C 在 exactness 与 latency 上稳定同时不差于 D：简单 baseline 覆盖；
- D exact 但稳定慢于 A，或收益低于 10%：当前 action/operating region 的系统价值不足。

这些结论绑定单 OLMoE、单 BF16 5090 stack，不自动否定完整 semantic-isolation 问题。

### 无法判断

- A 自身任一 real row 不是 10/10 stable；
- row/route/weight/dtype/order 不守恒，actual `M` 或 signature 未按计划发生；
- calibration 与 fresh evaluation 泄漏；
- raw bits、row identity、trace-output closure、环境或 artifact 完整性缺失；
- GPU 污染、hard deadline、未完成输出或 stack signature 漂移。

## 8. 复用、新增与禁改范围

### 可复用

从 `docs/ideas/spectatorroute/experiments/run_phase0a_5090.py` 迁移通用模式：hash/exclusive-write、OLMoE capture、raw-BF16 compare、cuBLASLt trace parser、GPU/environment validation、watchdog、numeric/trace 双 worker、snapshot 和 `COMPLETE-last`。

### 最小新增

- `docs/ideas/semanticfence/experiments/prepare_eval_manifest.py`
- `docs/ideas/semanticfence/experiments/executor_contract.py`
- `docs/ideas/semanticfence/experiments/run_pilot_5090.py`
- `docs/ideas/semanticfence/experiments/test_executor_contract.py`
- `docs/ideas/semanticfence/experiments/test_run_pilot.py`
- `docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json`
- 新的 content-addressed `FROZEN_RUN_LOCK.json` 与新 output root

预计新增约 500–700 行 runner/contract Python，另加 180–250 行 tests；这是一个 Pilot 的实现，不是完整 serving runtime。

### 禁止修改

旧 N05 protocol、Spectator config/runner/tests/locks、旧 run01 outputs、RouteGuard 原 manifests/provenance/history，以及当前已有用户修改的 `experiments/shared/capture_moe.py` 均保持原样。

## 9. 资源与预算

- Mac/CPU：实现、manifest、unit tests 与静态核对，预计 30–45 分钟。
- RTX 5090：hard cap 90 分钟；acceptance 10m、calibration 20m、fresh evaluation 40m、聚合及最多一次针对性复测 20m。
- 当前现实 blocker：可连接且干净的 RTX 5090 尚未在本轮验证；因此本文件不声称 Pilot 已运行。

## 10. 仅三个 P0

1. **Fresh canonical**：A 必须对全部真实 rows 10/10 raw-bit stable；否则无语义真值。
2. **可归因 treatment**：同一 rows/route/weights/dtype/order，逐 call 绑定 actual `M` + actual signature；calibration 与 fresh evaluation 严格隔离。
3. **非循环 contract**：D 只能用 pre-call 字段 admission，逐 row 判 exact，并保留 C；禁止输出泄漏、重复 filler 或 padding 冒充安全合批。

## 11. 非阻塞项

- **P1**：positive Pilot 后再做 full-layer combine 与下一层 route propagation。
- **P1**：positive Pilot 后接入一个真实 continuous-decode serving harness，核对 request-level 传播与 planner overhead。
- **P2**：第二模型、第二 backend/GPU、EP/NCCL、租户隔离、形式化误差界和投稿级统计。

## 12. Run order 与裁剪规则

- **Must-run**：本文件唯一 calibration→fresh-eval Pilot。
- **Nice-to-have**：旧 sealed replay；默认裁剪，不进入 verdict。
- **Stop after negative**：若 B 为 0 mismatch，或 D 任一 admitted fresh row mismatch，停止 full-layer/serving 扩展，先判断 operating region 或 admission formulation。
- **Proceed after positive**：只进入相邻的 D→E full-layer propagation，不直接搭完整系统。

当前实验已经足以推进下一轮探索，停止继续扩展审计项。

