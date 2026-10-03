## Overall verdict: WARN

`WEAKEN_STATIC_KEYED_RISK_TRANSFER` 的机械判定正确，所有汇总数字均能从绑定的 SemanticFence 原始文件独立重算得到；但证据只支持“单次、事后解盲、描述性的 CPU transfer probe”，不能当作前瞻性验证。

- Evidence status: `partially_valid`
- Review independence: `same-family`
- Acceptance status: `provisional`
- Evaluation type: `self_supervised_proxy`，M=1 隔离执行是差分参考，不是真实质量 GT
- 输入哈希与 `summary.json` 中四个哈希完全一致
- 单测：4/4 通过

### 独立重算结果

- 自然 B-arm、M>1：904 calls、8116 rows
- eligible grid：240 calls、1264 rows，占全部自然 M>1 rows 的 15.57%
- matched：235 calls、1232 rows
- conditional matched coverage：`1232/1264 = 0.9746835443037974`
- unknown：5 calls、32 rows，其中 30 mismatch rows
- matched labels：1123 positive、109 negative
- eligible 自然 mismatch：1153/1264

三种 row-level AUC：

- `(layer, expert, M)`：`0.538943851250337`
- `(layer, M)`：`0.5204154991136127`
- `M-only`：`0.5313870938753502`
- keyed 相对 M-only 增益：`0.007556757374986733`

Policy curve，格式为 `(threshold, admitted_calls, admitted_rows, mismatch_rows, launch_proxy, launch_reduction)`：

```text
0.00  0    0     0     1264  0
0.05  0    0     0     1264  0
0.10  0    0     0     1264  0
0.25  0    0     0     1264  0
0.50  1    2     2     1263  0.0007911392405063333
0.75  14   42    35    1236  0.022151898734177222
0.90  97   476   426   885   0.2998417721518988
0.95  165  882   801   547   0.567246835443038
1.00  235  1232  1123  267   0.7887658227848101
```

机械 gate：

- coverage 足够且两类均存在；
- support 条件失败：key AUC `<0.60`、增益 `<0.03`、primary launch reduction 为 0；
- key AUC `0.538943851250337 <= 0.55`；
- 因而精确落入 `WEAKEN_STATIC_KEYED_RISK_TRANSFER`。

### 关键限制

1. 无直接 row-label leakage：risk 只由 calibration 的 `1-exact_checks/total_checks` 构成。但脚本和 gate 是在 aggregate verdict 已知后编写、冻结的，不能视作预注册确认实验。[risk_transfer.py:2-7](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/errortoken/experiments/risk_transfer.py:2>) [risk_transfer_v1.json:3](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/errortoken/experiments/configs/risk_transfer_v1.json:3>)

2. 1232 rows 不是独立重复：同一 call 的所有 rows 共用一个 score。更严格地看，235/235 matched calls 都至少有一个 mismatch，因此 call-level “安全/不安全”AUC没有负类、不可计算。

3. primary threshold `0.25` 是退化 no-op：全部 calibration risk 最小值为 `0.5`，所以必然拒绝所有 call，得到零 mismatch 同时也得到零 launch reduction。

4. `launch_count_proxy` 只是“一个 admitted call 对一个 launch、fallback 每 row 一个 launch”的计数模型，不是 CUDA kernel 数、GPU latency、吞吐或成本测量。[risk_transfer.py:85-108](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/errortoken/experiments/risk_transfer.py:85>)

5. 所有 4237 个原 SemanticFence contract entries 均为 `allowed=false`；分析重新利用连续风险值，且 44/235 matched calls 的 calibration key 不满足原 `min_documents/min_packs` 支持要求。只保留满足支持要求的 191 calls 后，key AUC 进一步降至 `0.5229225016037927`。

6. 没有发现未调用的指标函数；但测试没有覆盖完整 `build_analysis` 集成路径。M=64 虽列入 eligible grid，本次自然 calls 中没有 M=64。

### Claim boundary

允许：

> 在 SemanticFence run03 的单次 retrospective、eligible-grid row-level 分析中，静态 `(layer, expert, M)` calibration risk 对 fresh mismatch 的排序能力接近随机，且冻结规则机械返回 WEAKEN。

禁止：

- 宣称静态 keyed risk 在其他模型、GPU、数据或 serving workload 上普遍无效；
- 宣称 ErrorToken、动态 selector 或其他风险信号已被否定；
- 宣称有实际 latency、吞吐、GPU launch、质量或 serving 收益；
- 将 1232 rows 当作独立样本或称为前瞻性泛化证据。

Required next step：在任何新结果出现前冻结 v2 协议，在新的 held-out split/run 上以前瞻方式验证；主终点改为 call-level zero-mismatch safety，并按 call 聚类报告不确定性。若要做系统收益声明，必须另测真实 GPU latency/throughput。

```json
{
  "overall_verdict": "WARN",
  "evidence_status": "partially_valid",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "evaluation_type": "self_supervised_proxy",
  "mechanical_verdict": {
    "reported": "WEAKEN_STATIC_KEYED_RISK_TRANSFER",
    "correct": true
  },
  "denominators": {
    "natural_m_gt_1_calls": 904,
    "natural_m_gt_1_rows": 8116,
    "eligible_calls": 240,
    "eligible_rows": 1264,
    "matched_calls": 235,
    "matched_rows": 1232,
    "unknown_calls": 5,
    "unknown_rows": 32,
    "matched_positive_rows": 1123,
    "matched_negative_rows": 109
  },
  "auc": {
    "key": 0.538943851250337,
    "layer_m": 0.5204154991136127,
    "m_only": 0.5313870938753502,
    "key_gain_over_m_only": 0.007556757374986733
  },
  "primary_policy": {
    "threshold": 0.25,
    "admitted_calls": 0,
    "launch_reduction_fraction": 0.0,
    "mismatch_row_fraction": 0.0
  },
  "claim_ceiling": "single-run retrospective row-level transfer description"
}
```
