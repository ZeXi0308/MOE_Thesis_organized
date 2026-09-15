# RouteShield raw capsule contract v1

> 状态：`DEVELOPMENT_RECOMPUTE_IMPLEMENTED / FORMAL_APPROVAL_LOCKED`
> 适用协议：`routeshield-gate0-v1`

## 1. 结论边界

raw capsule 的用途是让评估器从不可变 request/block 行重算 TTFT、P99、goodput、queue growth 和 paired-block bootstrap，而不是信任生产者填入的 aggregate 数字。

当前代码只允许开发态诊断：

- `DEVELOPMENT` capsule 最高输出 `RAW_RECOMPUTE_DIAGNOSTIC_ONLY`；
- 小样本夹具最高输出 `RAW_RECOMPUTE_SMOKE_ONLY`；
- `FORMAL` capsule 先完成完整性检查，然后固定停在 `BLOCKED_FORMAL_RAW_EVALUATOR_NOT_APPROVED`；
- 所有路径的 `formal_result` 均为 `false`。

不得将 `ALL_THRESHOLDS_PASS` 的开发态诊断分支改写为 Gate 通过、`NO-GO` 或 8×A100 授权。

## 2. capsule 目录

`manifest.json` 必须与 artifact 位于同一目录树：

```text
capsule/
├── manifest.json
├── requests.jsonl
└── blocks.jsonl
```

默认拒绝绝对路径、`..`、symlink、重复路径、未列入 manifest 的文件、重复 JSON key、`NaN/Infinity`、空 JSONL 行以及 hash/size/row-count 不一致。

manifest 字段严格为：

```json
{
  "schema": "routeshield-raw-bundle-v1",
  "mode": "DEVELOPMENT",
  "config_sha256": "<gate0_v1.json bytes>",
  "evaluator_source_sha256": "<raw_recompute.py bytes>",
  "artifacts": {
    "requests": {
      "path": "requests.jsonl",
      "sha256": "<bytes>",
      "size_bytes": 0,
      "row_count": 0,
      "format": "jsonl",
      "schema": "routeshield-raw-request-v1",
      "config_key": "required_evidence.raw_request_ledger_sha256"
    },
    "blocks": {
      "path": "blocks.jsonl",
      "sha256": "<bytes>",
      "size_bytes": 0,
      "row_count": 0,
      "format": "jsonl",
      "schema": "routeshield-raw-block-v1",
      "config_key": "required_evidence.raw_block_ledger_sha256"
    }
  }
}
```

`config_key` 必须指向冻结 config 中的 `*_sha256` 字段；列表下标用点分隔的十进制索引，例如 `models.0.tokenizer_sha256`。`FORMAL` 模式要求 config 里每个 `*_sha256` 路径在 capsule 中恰好绑定一次，且 artifact digest 必须等于冻结值。

## 3. raw request v1

每行只允许下列字段：

```text
model / load_cell / traffic_class / scenario / block_id / pair_id
tenant_id / role / request_id / document_id / document_cluster_id
prompt_hash / input_tokens / max_new_tokens
arrival_ns / first_token_ns / completion_ns
output_token_count / output_hash / terminal_reason
```

关键规则：

- TTFT 只由 `first_token_ns - arrival_ns` 重算，不接受自报 TTFT。
- formal denominator 中的每个 request 必须为 `COMPLETED`；`DROPPED/CANCELLED/TIMED_OUT` 返回 `CENSORED_REQUEST`，不得从分母中静默删除。
- `max_new_tokens` 固定为 1，时间戳必须单调，completed request 必须有 output hash。
- victim 在各配对臂中的 tenant/request/prompt/document/arrival/budget/output 必须一致。
- `A/O/S` 的 attacker identity、prompt、budget、arrival 和 output 必须一致；`B` 只允许把 attacker prompt/role 替换为 matched-benign cotenant，tenant 与资源预算不变。
- 每个 paired block 必须恰好只有一个 victim tenant 和一个 cotenant/attacker tenant；`B`/负控的非 victim 角色只能是 `cotenant`，`A/O/S` 只能是 `attacker`，不得用多 tenant 暗中扩大 Sybil 预算。

## 4. raw block v1

每行只允许下列字段：

```text
model / load_cell / traffic_class / scenario / block_id / policy_id
request_world_sha256 / arrival_trace_sha256
victim_manifest_sha256 / cotenant_budget_sha256
window_start_ns / window_end_ns
queue_service_work_start / queue_service_work_end / queue_service_work_arrived
oracle_status / oracle_gap
```

评估器会从 request 行使用排序后的 canonical JSON 重算四个 provenance hash，再与 block 值比较。因此，复制一个相同的自报 hash 不能隐藏 request world 漂移。

primary cell 的每个 block 必须恰好有 `B/A/O/S`：

```text
B = MATCHED_BENIGN
A = ATTACK_BASELINE
O = LEGAL_ORACLE
S = STRONGEST_SIMPLE
```

`30% NAT_BENIGN` 和 `70% NAT_PATHOLOGICAL` 每个 block 必须恰好有 `CONTROL_DEFAULT/CONTROL_ISOLATION`。任何缺臂或额外 cell 都是 invalid artifact。

`S` 必须是只用 calibration 选出并冻结的 strongest-simple policy。`O` 必须报 `OPTIMAL` 和 `gap=0`；timeout/state limit 只能返回 `UNSOLVED_EXACT_STATE_LIMIT`。当前评估器尚不会独立验证 solver certificate，所以这一项始终是 formal blocker。

## 5. 统计重算

固定规则是：

1. 每个 model/cell 单独计算，不跨模型 pooling。
2. request-level P99 使用 `nearest_rank_v1`，索引是 `ceil(0.99N)`。
3. 用同一个 multiplicity vector 联合重采样整个 paired arrival block；每个 bootstrap replicate 内重新合并 request 并重算 B/A/O/S P99，不对预先算好的 block P99 取平均。
4. formal 参数是 10,000 次、base seed `20260729`；每个 cell 的 seed 由 evaluator version/model/cell 稳定派生。
5. 95% interval 使用 percentile two-sided Type-7 的 0.025/0.975 quantile。
6. goodput 按每个 block 的 `completed input tokens / (last completion - first arrival)` 建立 ratio-of-sums；不使用 block 自报 duration 改变税负。
7. queue stability 由 `sum(end-start) <= 0.02 * sum(arrived service work)` 重算，不接受布尔自报。

小样本 `--smoke` 会降低样本门并将 bootstrap 限为 200 次，它只测代码路径，不能用于任何科学结论。

## 6. 信任与人工门

代码已锁定的是 schema、配对规则、estimator、threshold、config/evaluator/artifact hash 和非 canonical 输出边界。

它尚不能从 capsule 自证的是：

- 采集过程真的是 append-only，且没有在 seal 前替换行；
- route producer 真的来自 target native continuous-prefill backend；
- `EXECUTED_DISPATCH` 是物理执行事件，而不是 placement membership 的再标记；
- full request-DAG、tensor exactness 和 exact legal-Oracle certificate 已独立验证；
- sealed evaluation 的创建、开启和 baseline 选择没有受结果泄漏影响。

因此 `formal_execution_authorized` 和 `FORMAL_RAW_EVALUATOR_IMPLEMENTED` 必须由人工审查后分别打开；准备 artifact 或跑通 smoke 都不自动授权 formal run。

## 7. 命令

```bash
python3 docs/ideas/routeshield/experiments/run_gate0.py \
  --config docs/ideas/routeshield/experiments/configs/gate0_v1.json \
  --raw-bundle /path/to/capsule/manifest.json \
  --output /tmp/routeshield-raw-diagnostic.json
```

只有专用合成夹具允许加 `--smoke`。真实数据不得为了越过 30 blocks / 10,000 completed victim requests 门而使用 `--smoke`。
