# SemanticFence run03 实验完整性审计

## Overall verdict

- **Overall verdict:** `PASS_WITH_LIMITATIONS`
- **Evidence status:** `partially_valid`
- **Acceptance status:** `provisional`
- **Review independence:** `same-family`
- **Final experimental decision:** `WEAKEN`
- **Experiment type:** 单个预训练 OLMoE、单张 RTX 5090、BF16、decode-style 本地 expert-stage executable replay；calibration-to-fresh-evaluation 的计算等价性 Pilot。不是 full-layer、continuous serving、质量评测、EP 或生产实验。

核心结论可信：run03 确实完整执行并得到 `WEAKEN`，但它没有验证一个非平凡 SemanticFence 机制。冻结 contract 的 4,237 个 entry 中允许项为 0，D 组因而完全退化为 A 组的逐 row `M=1` 隔离执行。

## Verified artifact closure

对交付树的闭包核验通过：

- `run/` 共 102 个文件，所有 JSON/JSONL/源码/日志可解析，所有二进制文件完成流式 SHA-256 核验。
- `COMPLETE.json` 精确覆盖 39 个顶层文件，文件集合和散列均无差异。
- `evaluation_raw_output_index.json` 精确覆盖 40 个 raw BF16 文件；每个文件均为 33,554,432 bytes，路径、大小和 SHA-256 全部匹配。
- `run_request.json.snapshot_sha256` 精确覆盖 22 个 `frozen_inputs/` 文件。
- 三层覆盖合计为 `1 COMPLETE + 39 top-level + 40 raw BF16 + 22 frozen inputs = 102`，无未闭合文件。
- 原始 acceptance 与 frozen copy 完全一致：
  - `ACCEPTANCE.json`: `c842be78111cd5596e5f66d2c0dcc820daba019c5cff1d1e3fb760d22028c0e7`
  - `ACCEPTANCE_COMPLETE.json`: `00f20d057d49b0ee31202a1138d3813e186cd198ffa09d09dc6f59b1dae6d0d7`
- 11 个 source binding 均与 frozen source 文件一致。
- lock 文件散列、lock canonical content hash、contract file hash、contract canonical hash和 contract seal 均通过。
- calibration calls、numeric、trace 与 `CONTRACT_SEAL.json` 的绑定均通过。
- 未发现 `failure.json` 或其他 failure artifact。
- 四个 worker log 均只有模型分片加载输出，没有异常或 foreign-process 文本。
- `wall_seconds=791.900057`；相对 5,400 秒 deadline，记录上的剩余量为 `4,608.099943` 秒。

闭包限制：

- 8 个模型文件本身不在 artifact 中；只能确认 acceptance 中记录的模型散列及运行时校验逻辑，不能在本地重新散列模型权重。
- 没有连续的 `nvidia-smi` process snapshot/monitor journal；“无 foreign GPU process”由 runner 代码路径和成功完成状态证明，不是可独立回放的原始遥测。
- 没有独立 `finished_at_epoch`；deadline 只能由 `started_at + wall_seconds` 和 runner 的 watchdog 路径交叉判断。
- `sealed_manifest.jsonl` 与 `smoke_manifest.jsonl` 没有冻结进 artifact。当前 workspace 副本仍与 exclusion report 中的散列一致，重算 overlap 为 0，但 portable artifact 本身缺这两个排除源。
- `.pt` 文件完成内容散列闭包；决定性结论不依赖反序列化 `.pt`，而是从可移植 raw BF16 和 JSONL 重算。

相关实现证据：completion-last 逻辑位于 `run_pilot_5090.py:1137-1156`；输入快照位于 `2220-2267`；完整 orchestrator 位于 `2800-2976`。

## Ground-truth and provenance

### Canonical reference

参考不是外部质量 ground truth，而是实验明确定义的计算等价性 reference：

- 每个 routed row 单独调用相同 expert，`M=1`。
- A 的 pair 0 是每个 row 的 canonical raw-BF16 reference。
- 后续 9 个 A repeat 用于稳定性检查。
- 每个 row 比较完整 2,048 个 BF16 uint16 元素，包括 signed zero。

这适合回答“不同 packing 是否 bitwise 等同”这个问题，但不授权模型质量或语义正确性 claim。

设计定义见实验计划 `79-83`、`113-127`；实际 raw reference 产生见 `gpu_execution.py:924-1119`。

### Row, route, weight and order

独立核验结果：

- evaluation manifest：32 个唯一文档，索引为 `0..31`。
- 每文档 offset `{0,256}`，共 64 个 capture window。
- 每个 window 使用 token position 15。
- `64 × 16 layers × top-8 = 8,192` 个唯一 row。
- 每个 window/layer 恰有 rank `1..8`，expert ID 在 top-8 内无重复。
- 同一 window/layer 的 8 个 route rank 共享同一个 hidden-state hash。
- 所有 routing weight 均为正且有限。
- row ID 重新按冻结 payload 计算后全部匹配。
- 四个 arm 的 logical row traversal 完全一致，遍历散列为  
  `fb3b50b703f14a4deee4ba67cbfb467dc365c758700a77ecca3914a362983095`。

Row identity 定义见 `executor_contract.py:52-103`；route materialization 见 `gpu_execution.py:330-392`；跨 arm order 守恒见 `645-667`。

### Calibration/evaluation separation

- calibration：8 个唯一文档。
- evaluation：32 个唯一文档。
- 两者 full-text SHA-256 交集为 0。
- evaluation 对 historical registry、calibration、sealed、smoke 四个源重新核验后的交集为 0。
- 排除集合重算大小为 1,137，与 exclusion report 一致。
- evaluation manifest、provenance、固定 salt 和 selection hash 一致。
- contract 只允许 `split=calibration` 的 row，见 `executor_contract.py:309-424`。
- contract/seal 在 evaluation capture 前生成并重新认证，见 `run_pilot_5090.py:1614-1803`、`1806-1825`、`2899-2925`。

没有发现 evaluation output 反向进入 allowlist 的路径，也没有 score self-normalization。M=1 是显式的计算代理 reference，不是伪装成外部标签的模型自生成 ground truth。

## Independently recomputed decisive metrics

### Packing and treatment

| Arm | Calls | Execution M | Real rows | Padding |
|---|---:|---|---:|---:|
| A isolated | 8,192 | 全部 M=1 | 8,192 | 0 |
| B unrestricted | 980 | 自然 M=1..59 | 8,192 | 0 |
| C fixed | 980 | 全部 M=64 | 8,192 | 54,528 |
| D SemanticFence | 8,192 | 全部 M=1 | 8,192 | 0 |

D 与 A 的 `(layer, expert, row IDs, M, padding)` call sequence 完全相同。因此 D 的 treatment 没有产生任何非平凡 packing 变化。

### Raw-BF16 exactness

以 `pair_00_A_isolated_m1.bf16` 为逐 row reference，直接按 `<u2` 解码 40 个文件：

| Arm | Mismatch rows | Unstable rows | Stable mismatch victims |
|---|---:|---:|---:|
| A | 0 / 8,192 | 0 | 0 |
| B | 7,584 / 8,192 | 0 | 64 / 64 |
| C | 8,192 / 8,192 | 0 | 64 / 64 |
| D | 0 / 8,192 | 0 | 0 |

补充数据：

- B 的 mismatch-row 数在十个 repeat 中均为 7,584。
- B 每个 mismatch row 的单次 mismatch element 数范围为 1–1,785，中位数为 97。
- C 每个 repeat 的 8,192 个 row 全部 mismatch。
- D 十个 repeat 的所有 row 均与 A pair 0 完全相同。
- A 十个 repeat 完全稳定，因此 `reference_all_stable=true`。

Unrestricted victim 公式为：

```text
B_victims =
count(distinct window_id(row)
      where B row is stable across 10 repeats
      and every repeat has mismatch_count > 0)
= 64
```

### Contract and coverage

`CONTRACT.json` 独立聚合：

- 总 entry：4,237
- allowed entry：0
- `all_repeats_exact=true` entry：0
- 同时达到 pack/document support 门槛的 entry：2,612
- 同时达到 support 且 all-exact 的 entry：0

因此：

```text
D M>1 covered rows       = 0
D covered victims        = 0
D distinct admitted M>1  = 0
D padding rows           = 0
```

### Paired latency

记录的 CUDA-event latency 样本重新计算：

- A median：`480.8722839355469 ms`
- B median：`64.04075241088867 ms`
- C median：`68.4991683959961 ms`
- D median：`482.00927734375 ms`

冻结公式：

```text
reduction = 1 - median(D_i / A_i)
          = 1 - 1.002364013194172
          = -0.00236401319417201
```

即 D 相对 A 为 `-0.2364%` reduction，实质上是轻微变慢且落在测量噪声范围内。

Fixed-M64 dominance：

```text
C_dominates =
(C mismatch rows <= D mismatch rows)
AND
(median latency C <= median latency D)

= (8192 <= 0) AND (68.4992 <= 482.0093)
= false
```

C 虽显著更快，但不 exact，所以不支配 D。

### Frozen final decision

冻结 SUPPORT 条件要求：

- reference stable；
- B mismatch victims ≥ 8；
- D mismatch rows = 0；
- D covered victims ≥ 8；
- D distinct M>1 ≥ 2；
- D padding = 0；
- latency reduction ≥ 10%；
- fixed M64 不支配。

实际失败项：

- `D covered victims = 0 < 8`
- `D distinct M>1 = 0 < 2`
- `D latency reduction = -0.002364 < 0.10`

因此机械裁决为：

```text
WEAKEN
```

与 `decision_input_numeric.json`、`parent_recompute.json` 和 `summary.json` 一致。规则实现见 `run_pilot_5090.py:1099-1134`，冻结解释见计划 `136-170`。

## Contract effectiveness

通过项：

- calibration-only split 强制执行。
- duplicate calibration pack 不能增加 support。
- contract canonical hash 和 seal 有效。
- contract 在 fresh evaluation 前完成 seal。
- evaluation calls 和 paired schedule 在执行前 hash-seal。
- call descriptor 绑定 config、stack、model/source hashes、math state、layer、expert、M、dtype、projection shape、row IDs 和 padding。
- 每个 arm 恰好覆盖全部 8,192 row 一次，无丢失、重复或跨 expert call。
- unknown stack、未知 M 和 ambiguous signature 在 planner 中回退 M=1。
- trace signature drift会令最终 `evidence_complete=false`，而不是事后扩充 allowlist。

限制：

- production planner `plan_arm_d` 在 `gpu_execution.py:604-642` 独立实现 admission；单元测试覆盖的 `choose_pack_size` 不是实际 planner 的直接调用路径。
- `unique_allowed_signature_lookup` 未进入 run03 主路径。
- artifact 中只有冻结测试源码，没有单元测试实际运行日志。
- 本次 D 没有任何 M>1 call，所以 fail-closed 行为得到的是退化路径验证，不是 positive admission 验证。

## Trace integrity

交付记录显示：

- calibration numeric/trace：29,803 个 call。
- evaluation numeric/trace：18,344 个 call。
- numeric call identity、trace call identity及 full-output SHA-256 全部闭合。
- `evaluation_trace_validation.json` 的四项字段均为 true。

但有两个重要边界：

1. `semanticfence_signature_match=true` 对本次 D 是**真空成立**。检查只针对 `D && M>1`，见 `run_pilot_5090.py:2749-2755`；本次不存在这种 call。

2. numeric worker 禁用 logging，trace worker在另一个进程、另一次模型加载中启用 cuBLASLt logging。相同 output hash 证明 replay 输出一致，不证明 numeric timing worker 使用了同一个 tactic。日志 observer effect、cache state 或进程级 tactic selection 仍可能不同。

因此允许声称“trace replay 与 numeric representative output 闭合”，不允许声称“numeric timing call 的实际 tactic 已被直接观测”。

## Findings ordered by severity

### 1. Major scientific limitation: D 完全退化为 A

证据：

- `CONTRACT.json`: 4,237 entries，0 allowed。
- `evaluation_calls.jsonl`: D 有 8,192 个 call，全部 M=1。
- D 与 A packing sequence 完全相同。
- `summary.json.decision_input.semanticfence_covered_victims=0`。
- Planner 逻辑位于 `gpu_execution.py:604-642`。

影响：本实验没有验证“安全 M>1 rebatching”；它验证的是“contract 无法认证任何 M>1 后安全回退为全隔离”。这是有效负结果，不是机制成功。

### 2. Calibration M>1 raw bytes 未持久化

`calibration_raw_outputs.pt` 的名称容易产生误解。实际 `CalibrationPackExecution` 只保存：

- `repeat_row_exact`
- `repeat_row_sha256`
- `representative_full_output_sha256`

不保存十个 repeat 的 M>1 BF16 raw bytes，见 `gpu_execution.py:732-744`、`1122-1191`。

因此 auditor 能从 hashes/flags 重建 empty allowlist，但不能像 evaluation 那样从 raw uint16 独立重算全部 calibration mismatch。此项是 `partially_valid` 而非 `valid` 的主要原因。

### 3. D trace signature 验证真空成立，且 trace 与 timing 分进程

证据见 `run_pilot_5090.py:127`、`1440-1565`、`2721-2784`。

影响：不损害本次 `WEAKEN`，因为没有 admitted M>1；但不能把 trace 字段当成 positive tactic-certification 证据。

### 4. Contamination/deadline 是运行逻辑证明，不是完整原始遥测闭包

runner 在 worker 前后及每 0.5 秒查询 GPU process，见 `run_pilot_5090.py:2161-2217`。但查询结果没有逐次写入 artifact。

影响：未发现污染证据，且成功路径与 deadline 一致；不过第三方无法仅凭 artifact 回放全部 foreign-process 检查。

### 5. 排除源 portable closure 不完整

`sealed_manifest` 与 `smoke_manifest` 只在 provenance/report 中保存散列，没有复制到 `frozen_inputs/`。当前 workspace 副本匹配这些散列且 overlap 重算为 0，因此本次没有发现 leakage；但 artifact 单独搬迁后无法完成同样审计。

### 6. 统计范围有限

- 十个 latency sample 是同一 workload、同一 GPU、同一次运行的 paired repeats，不是十个独立实验。
- 8,192 rows 由 64 个 window 派生，不能作为 8,192 个独立样本。
- 两个 offset 来自同一文档，64 victim 也不等于 64 个独立文档。
- 没有跨 run、跨 GPU 或置信区间。
- D/A 是相同 call plan，因此 `-0.2364%` 只能解释为“无可测收益”，不应解释为稳定回归幅度。

冻结 paired rotation 和 3 warmup/10 repeat 实现正确，见 `gpu_execution.py:670-676`、`924-1119`。

## Authorized claims

1. 在该固定 OLMoE revision、BF16、单 RTX 5090 stack 的本地 expert-stage replay 中，A 的全部 8,192 routed rows 在十次 repeat 中 raw-BF16 稳定。
2. Native unrestricted packing 相对隔离 M=1 在 7,584/8,192 rows 上产生稳定 raw-BF16 差异，并覆盖 64/64 evaluation windows。
3. Frozen calibration-derived contract 记录了 0 个 allowed entry；run03 的 SemanticFence arm 因而全部回退 M=1。
4. D 与 A raw-BF16 exact，但覆盖 0 个 M>1 victim，没有非平凡 batching，paired latency reduction 为 `-0.2364%`。
5. Fixed M64 较快但全部 8,192 rows 都不 exact。
6. run03 按冻结规则正确裁决为 `WEAKEN`；当前 formulation 在该 stack 上退化，不授权进入 positive full-layer gate。

## Prohibited claims

不得声称：

- SemanticFence 已成功发现安全、非平凡的 M>1 equivalence class。
- SemanticFence 改善了 latency、throughput、TPOT、P99 或 serving efficiency。
- raw-BF16 mismatch 必然造成 token、logit、质量或用户可见语义差异。
- 已验证 full-layer combine、下一层 route propagation 或 autoregressive accumulation。
- 已验证 continuous decode、真实跨请求 serving 或 planner overhead。
- 已验证多租户隔离、EP、NCCL、RDMA 或多 GPU。
- 已验证跨模型、跨 GPU、跨 driver/backend 泛化。
- 已获得形式正确性证明或 production readiness。
- trace worker 直接证明 numeric timing worker 使用了同一 cuBLASLt tactic。
- 8,192 rows 或 64 windows 是相互独立的统计样本。

## Required fixes / next gate

1. 为 calibration 的每个 M>1 call 和每个 repeat 持久化 raw little-endian BF16，配套 row order、shape、size 和 SHA-256 index；由 parent 从 bytes 重建 contract。
2. 将 sealed/smoke exclusion manifests 一并复制到 frozen inputs。
3. 保存带时间戳的 GPU process monitor ledger、worker PID/UUID snapshots 和 `finished_at_epoch`。
4. 增加 frozen unit-test 执行报告及其 source/test-result hash closure。
5. 若 tactic identity 是 admission 条件，需要在 numeric execution 本身采集 tactic，或专门证明 logging 不改变 tactic；仅靠独立 replay output hash 不够。
6. 下一科学 gate 不是 full-layer，而是修改 calibration/admission operating region 后重新冻结一次 fresh run。只有出现至少两个 M>1、覆盖至少 8 victims、D raw-exact 且 latency 达阈值，才能进入 full-layer propagation。

## Concise JSON

```json
{
  "overall_verdict": "PASS_WITH_LIMITATIONS",
  "evidence_status": "partially_valid",
  "acceptance_status": "provisional",
  "review_independence": "same-family",
  "experiment_type": "single_RTX5090_pretrained_OLMoE_BF16_decode_style_expert_stage_calibration_to_fresh_eval_replay",
  "verified_artifact_closure": {
    "delivered_tree_complete": true,
    "run_file_count": 102,
    "complete_top_level_files": 39,
    "raw_bf16_files": 40,
    "frozen_input_files": 22,
    "uncovered_files": 0,
    "acceptance_copy_match": true,
    "failure_artifact_present": false,
    "model_binaries_present": false,
    "continuous_contamination_telemetry_present": false
  },
  "recomputed_metrics": {
    "evaluation_rows": 8192,
    "evaluation_victims": 64,
    "reference_all_stable": true,
    "arm_a_mismatch_rows": 0,
    "unrestricted_mismatch_rows": 7584,
    "unrestricted_mismatch_victims": 64,
    "fixed_m64_mismatch_rows": 8192,
    "semanticfence_mismatch_rows": 0,
    "semanticfence_covered_victims": 0,
    "semanticfence_distinct_m_gt_1": 0,
    "semanticfence_padding_rows": 0,
    "contract_entry_count": 4237,
    "allowed_contract_entry_count": 0,
    "arm_a_median_latency_ms": 480.8722839355469,
    "arm_b_median_latency_ms": 64.04075241088867,
    "arm_c_median_latency_ms": 68.4991683959961,
    "arm_d_median_latency_ms": 482.00927734375,
    "semanticfence_latency_reduction_fraction": -0.00236401319417201,
    "fixed_control_dominates": false
  },
  "final_decision": "WEAKEN",
  "decision_reason": "The sealed calibration contract admitted no M>1 class, so SemanticFence was identical to all-row M1 isolation, covered zero victims, admitted zero distinct M>1 values, and provided no latency reduction.",
  "authorized_claim": "On this exact single-GPU expert-stage replay, unrestricted packing produced stable raw-BF16 differences, while the calibration-derived SemanticFence contract degenerated to M1-only isolation and was weakened.",
  "prohibited_claims": [
    "nontrivial SemanticFence success",
    "full-layer correctness",
    "serving latency or throughput improvement",
    "quality preservation",
    "multi-tenant isolation",
    "EP_NCCL_RDMA",
    "production readiness",
    "formal correctness",
    "cross-model or cross-hardware generalization"
  ],
  "next_gate": "persist raw calibration M>1 outputs, close telemetry and exclusion artifacts, revise admission, freeze a new fresh run, and require nonzero exact M>1 coverage before full-layer work"
}
```

## Fresh child review trace appendix

The extra closure child was interrupted before producing a final response. No child opinion was incorporated.

**Exact prompt sent:**

```text
You are an adversarial experiment-integrity reviewer. Read the referenced files directly; do not rely on any executor summary or interpretation. Do not edit any files.

OBJECTIVE: Audit whether the SemanticFence run03 experiment is scientifically and operationally trustworthy, independently recompute its decisive result from raw artifacts, and state the narrowest claims it authorizes.

FILES TO READ:
- Entire completed artifact tree (all 102 files, including raw BF16 files, captures, traces, contract, status, summary, and frozen inputs): /Users/leandrozhao/Desktop/毕设论文资料/artifacts/semanticfence_remote_20260810_run03/run/
- Original acceptance copy for cross-check: /Users/leandrozhao/Desktop/毕设论文资料/artifacts/semanticfence_remote_20260810_run03/acceptance/
- Primary runner source: /Users/leandrozhao/Desktop/毕设论文资料/artifacts/semanticfence_remote_20260810_run03/run/frozen_inputs/docs__ideas__semanticfence__experiments__run_pilot_5090.py
- Contract/execution source: /Users/leandrozhao/Desktop/毕设论文资料/artifacts/semanticfence_remote_20260810_run03/run/frozen_inputs/docs__ideas__semanticfence__experiments__executor_contract.py and docs__ideas__semanticfence__experiments__gpu_execution.py
- Frozen tests: every docs__ideas__semanticfence__experiments__test_*.py file in run/frozen_inputs/
- Frozen config and research plan: run/frozen_inputs/config.json and run/frozen_inputs/refine-logs__EXPERIMENT_PLAN_20260809_202112.md

AUDIT CHECKLIST:
1. Ground-truth/reference provenance: verify M=1 canonical rows, calibration/evaluation separation, fresh-manifest exclusion, row identity, dtype, route, weight, order, and leakage/circularity.
2. Decision correctness: independently recompute every decisive metric from raw BF16 files and call/row artifacts, including reference stability, unrestricted mismatch victims, SemanticFence mismatch rows, coverage, distinct admitted M>1, padding, paired latency reduction, fixed-M64 dominance, and final SUPPORT/WEAKEN/UNABLE rule.
3. Contract effectiveness: prove the calibration-only allowlist is sealed before evaluation, pre-call descriptors drive admission, unknown/signature drift fails closed, all rows are neither lost nor duplicated, and treatment actually changes packing.
4. Artifact authenticity/closure: verify COMPLETE.json hashes every required top-level file, raw-output index hashes/sizes, contract seal, acceptance/lock/source/model/stack hashes, worker/trace/parent closure, deadline, and absence of failure/foreign GPU evidence. Cross-check the original acceptance copy against the frozen copy.
5. Trace integrity: expected versus actual cuBLASLt signatures, numeric/trace worker separation, trace-output binding, and any observer-effect ambiguity.
6. Statistical/measurement integrity: paired schedule, warmups/repeats, units/denominators, pseudo-replication, timing noise, target selection, and whether thresholds were frozen before run.
7. Scope and experiment type: distinguish local expert-stage executable replay from full-layer propagation, continuous serving, quality, multi-tenant, EP/NCCL/RDMA, or production claims.
8. Detect dead code, uncalled checks, missing raw evidence, self-normalization, post-hoc thresholds, or summary fields that cannot be recomputed.

OUTPUT SCHEMA:
- Overall verdict: PASS / PASS_WITH_LIMITATIONS / FAIL / INCONCLUSIVE
- Evidence status: valid / partially_valid / invalid
- Acceptance status: provisional
- Review independence: same-family
- Experiment type
- Verified artifact closure
- Independently recomputed decisive metrics, formulas, and final decision
- Findings ordered by severity with exact file/line or artifact/field evidence
- Authorized claims
- Prohibited claims
- Required fixes / next gate
- Final concise JSON object with the same core fields

Be adversarial. Assume the evaluation is compromised somewhere and try to falsify it. Trust no summary number until recomputed from raw evidence. Read every frozen test and source file line by line, and enumerate all 102 artifact paths so the assertion of complete-tree review is auditable. Return a self-contained audit only; no progress commentary.
```

**Full verbatim child response:** `ABSENT — child was interrupted before returning a final response.`
