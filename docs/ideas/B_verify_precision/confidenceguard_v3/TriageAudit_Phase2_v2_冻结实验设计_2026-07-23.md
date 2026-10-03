# TriageAudit-MoE Phase 2 v2 冻结实验设计

状态：**EXPERIMENT DESIGN FROZEN / NO FORMAL RUNNER / GPU NOT APPROVED / NO SCIENTIFIC RESULT**  
日期：2026-07-23

## 0. 与 v1 的必须修正

v2 保留 same-state audit 和 unique canonical KV，但修正三个混杂：

1. `hash_budget_matched`不再独立随机分配 period。它使用与 triage 完全相同的 `{2,4,8}` period multiset，仅按 document hash 重排；
2. 主成本指标改为 `total_candidate_forward_calls=high+low`。`audit_events`只是成本分解，不可单独触发 GO；
3. calibration 新增稳定性硬门。分层不稳定时不得打开 sealed。

v1 不得用于正式 verdict。

## 1. Scientific question

> 在不允许 prefill predictor 直接选择低精度的前提下，一个仅用 post-prefill 特征的风险排序，能否在与风险无关的严格同预算 audit 策略相比时，以不更差的 document-tail quality，降低 same-state shadow verification 的总候选 forward 数？

本轮检验的是 **risk ranking 对 audit allocation 的信息价值**，不是 INT4 加速、Energy-SLO、continuous serving 或多 GPU 结论。

## 2. Falsifiable hypotheses

### H0: calibration stability

对 calibration documents 做 2,000 次 document bootstrap：

- 每篇文档被分到原 full-calibration stratum 的概率中位数 `>=0.70`；
- 至少 `80%` 文档的 assignment probability `>=0.60`；
- predicted score 与 calibration label 的 Spearman `rho` 95% LCB `>0`。

任一失败：`NO_GO_UNSTABLE_TRIAGE_SIGNAL`，不打开 sealed。

### H1: quality non-inferiority

sealed 上，`triage_2_4_8` 相对 `hash_budget_matched_2_4_8` 的逐文档 document-CVaR90 token-KL paired ratio 中位数 95% UCB `<=1.05`。零 KL 比值用 `(triage+1e-12)/(baseline+1e-12)` 定义。

### H2: total verification work reduction

在 H1 通过时，`triage_2_4_8` 相对 hash-budget-matched baseline 的逐文档 `total_candidate_forward_calls` paired reduction 中位数 95% LCB `>=10%`。

### H3: strong-baseline Pareto increment

与 `fixed_2/fixed_4/fixed_8` 及 `hash_budget_matched` 的 policy-level quality-cost Pareto envelope 相比，triage 在不更高 document-CVaR90 下的 total forward reduction 95% LCB `>=10%`。每次 document bootstrap 内用各 policy 的文档均值重建 envelope，禁止为每篇文档事后选不同 baseline。若 triage 质量优于所有 baseline，仍保守地要求其相对最低成本 baseline 有 `>=10%` 降低，不因无 eligible point 自动 PASS。

### H4: safety is not obtained by missing dangerous steps

dangerous-step recall 的 paired difference `recall_triage-recall_hash` 95% LCB `>=-0.05`，且 threshold-violation fraction 的 paired difference 95% UCB `<=0`。

`dangerous-step recall` 精确定义为 dangerous steps 中最终 served BF16 的比例（protection recall）；`threshold-violation fraction` 为 served low 的 dangerous steps 占全部 decode steps 的比例。两者都用逐文档 paired difference 中位数做 bootstrap。

`dangerous step` 不得使用 full-shadow 或 always-low 的另一条 KV 轨迹定义。对每个 candidate arm 的每一 step，从该 arm 自己的 pre-step canonical KV 做 **diagnostic-only same-state fork**，执行未 served action 以获得 high/low discrepancy。该标签只用于事后 recall/violation 计算，不得被 policy 读取；其 forward/clone 单列记账，不进入 candidate deployable-cost 估计。因此 Gate M 仍只是机制 probe。

## 3. 实验变量

### 3.1 自变量

- audit allocation policy：`triage / hash-budget-matched / fixed / full-shadow`；
- 模型：OLMoE、LLM-jp；
- 执行 action：BF16 与 frozen W4A16 RTN quality proxy；
- audit period：`2/4/8`；
- 风险 stratum：`high/medium/low`。

### 3.2 因变量

- document mean/CVaR90/P95 token KL；
- threshold violation fraction；
- dangerous-step recall；
- high/low/total candidate forward calls；
- audit、cache clone、lockout、served high/low counts；
- diagnostic high/low forward calls 与 diagnostic clone events，与 candidate cost 分列；
- calibration assignment stability；
- peak GPU memory 与 OOM status（工程可行性，不是系统收益）。

### 3.3 控制变量

- 模型/tokenizer revision；
- 同一 document manifest、prompt=64、teacher-forced decode=32；
- 同一 action scope、quantization rule、threshold、lockout=3；
- 同一 random seed、audit phase rule、token sequence；
- 每 arm fresh prefill 和独立 canonical KV；
- 相同 document-level bootstrap indices。

## 4. Data and split

- dataset：`wikitext-103-raw-v1:train`，article 为独立单元；
- calibration：32 documents；
- sealed：64 documents；
- 两模型消费同一 manifest；
- `sha256(seed || canonical_text_sha256)` 冻结选择；
- canonicalization 只统一 CRLF/CR 为 LF；
- calibration/sealed/历史实验 text hash 零交集；
- sealed manifest 必须加密或独立保存，只有 H0 PASS + calibration lock + Code Review 后才可见。

数据不足或历史排除不闭合输出 `BLOCKED_DATA`，不得换数据集寻找阳性。

## 5. Predictor and budget-matched control

### 5.1 Predictor

只用原 v1 的 8 个 route summary + prefill NLL，frozen ridge `alpha=1.0`，目标为 `log10(document CVaR90 same-state discrepancy + 1e-12)`。不搜特征、alpha、非线性模型或 stratum 数。

### 5.2 Exact budget matching

对每个 model/split：

1. triage 按 frozen score/cuts 产生每篇 document period；
2. 记录 period multiset `M={2:n2,4:n4,8:n8}`；
3. hash baseline 按 `sha256("budget-control" || model || split || document_hash)` 排序 documents；
4. 把完全相同的 `M` 按 canonical `2 -> 4 -> 8` 分配给 hash 序列；
5. runner 必须断言两 arm 的 period histogram 逐值相等。

该 baseline 可读 sealed 的 document identity 和 frozen prefill score，不读 sealed quality label、decode route、logits 或 outcome。

所有 periodic arms 的 phase 统一使用
`sha256("audit-phase-v2" || document_hash || period) mod period`。policy name 不得进入 phase hash；相同 document/period 在不同 arm 中必须具有相同 phase。

## 6. Runtime semantics

每 policy/document：

1. fresh BF16 prefill；
2. always-BF16 reference 与 candidate 独立执行；每篇文档只生成一份不可变 reference logits，可被该文档的 arms 只读共享，candidate KV 不得共享；
3. candidate 维护唯一 served canonical KV；
4. low/high single-action step 只执行对应 action；
5. audit 前从同一 KV 分叉出 non-aliased high/low caches；
6. 两路消费相同 input token，cache length 均 `+1`；
7. 计算 `KL(high || low)`，由 threshold 选择唯一 served branch；
8. 另一 branch 释放，禁止进入下一 step。

所有 arm 独立运行；不得在预先生成的 always-low trajectory 上 post-hoc mask。

## 7. Arms

1. `always_bf16`；
2. `always_low`；
3. `triage_2_4_8`；
4. `hash_budget_matched_2_4_8`；
5. `fixed_2`；
6. `fixed_4`；
7. `fixed_8`；
8. `full_shadow`，作昂贵上界。

`hash_budget_matched` 已经是保留 exact budget、删除 risk-document 对应的负对照；不再添加与它数学等价的 score-shuffle arm。

## 8. Metrics and statistics

独立样本是 document，step/forward 不得作为独立样本。

- calibration stability：2,000 document bootstraps；
- sealed primary comparisons：5,000 paired document bootstraps；
- 相同 bootstrap draw 同时用于 quality/cost/recall；
- H1/H2/H3/H4 使用 Holm family-wise correction；
- H1/H2/H4 主报逐文档 effect 的 median 与 95% paired-document bootstrap CI，禁止用 ratio-of-means 代替；H3 是 policy-level Pareto 特例，按 H3 的 bootstrap 内均值点重建。
- 不删异常 document。NaN/Inf/cache failure/hook mismatch 直接使该 model `INVALID_RUN`；
- wall time 仅诊断，不进入 Gate M。

## 9. Gate M decision

### 9.1 Per-model

同时通过 H1–H4 且全部 closure checks，才是 model-level GO。

### 9.2 Cross-model

- LLM-jp 必须 GO；
- OLMoE 至少 H1/H4 PASS，允许 H2/H3 收益为 0；
- OLMoE 若 quality degradation 或 dangerous recall 失败，整体 NO-GO。

通过时最高结论仅为：

`GO_TO_NATIVE_LOW_PRECISION_GATE_S`

失败时：

`NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION`

禁止更换 predictor、改 split、调 period palette、放宽阈值或仅对 LLM-jp 写跨模型 claim。

## 10. Level 0–3 plan

### Level 0: calibration stability

- 输入：32 calibration documents/model 的 frozen features 与 same-state discrepancy label；
- 实现：ridge fit、cuts、bootstrap assignment stability；
- 输出：`calibration_lock.json`、stability report；
- 停止：H0 失败；
- 进入条件：两模型 H0 PASS。

### Level 1: policy trace dry replay

- 输入：frozen periods/thresholds + synthetic discrepancy traces；
- 实现：验证 budget matching、audit/lockout counters、negative controls、bootstrap/verdict；
- 输出：CPU dry-run artifacts；
- 停止：period histogram/counter/statistic 不闭合；
- 进入条件：全部 invariant tests PASS。

### Level 2: single-GPU mechanism probe

- 输入：两模型、calibration 后的 sealed 64 documents；
- 实现：W4A16 proxy + canonical-KV same-state audit；
- 输出：raw per-step/per-document results、paired bootstrap、Gate M decision；
- 停止：任一 model OOM/closure failure，或 Gate M NO-GO；
- 进入条件：`GO_TO_NATIVE_LOW_PRECISION_GATE_S`。

### Level 3: native precision system gate

本文档不冻结 Level 3 实现。只有 Level 2 GO 后才新建协议，必须包含 native kernel、KV fork/switch、废弃 branch、HBM、board energy、TPOT/P99 全账本。

## 11. Memory feasibility pre-gate

正式 calibration 前必须单独执行 OLMoE tiny smoke：

- 1 document，prompt 8，decode 2，至少 1 audit；
- 保持完整 16 layers/64 experts/3 linears，不得缩 scope；
- 记录 model load、prepared proxy、prefill、fork 和 dual forward 后 peak allocated/reserved memory；
- 要求峰值 `<=min(30.25 GiB, detected_total_GiB-1.5 GiB)`，保留至少 1.5 GiB guard；
- OOM 或超 guard：`BLOCKED_MEMORY`，必须重新设计 layer-streamed proxy，不得自动减少 action scope。

## 12. Required artifacts

每个 stage 新建 no-overwrite run directory，至少保存：

```text
config.json
protocol_sha256.json
source_manifest.json
data_manifest.json
environment.json
calibration_lock.json
raw_results.jsonl
document_metrics.csv
paired_bootstrap.json
decision.json
stdout.log
summary.md
status.json
```

`status.json` 必须区分 `BLOCKED / INVALID / NO_GO / GO`；不得把工程失败写成科学 No-Go。

## 13. GPU 前 Code Review 必查

1. exact budget matching 与 period histogram closure；
2. calibration bootstrap stability 没有读 sealed；
3. same-state fork、storage non-alias、cache length +1；
4. 每 arm fresh KV，唯一 served branch；
5. INT4 hook 命中 OLMoE 3072 / LLM-jp 1536 linears；
6. always-BF16 的 proxy call 为 0；
7. audit 确实计 high+low 两次 forward；
8. 每 arm 的 dangerous label 来自自身 same-state diagnostic fork，不读另一 arm trajectory，不反馈 policy；
9. diagnostic/candidate counters 分列，并证明关闭 diagnostic 不改变 action trace、served logits 或 canonical KV hash；
10. paired document bootstrap + Holm 正确；
11. OOM 不得触发 silent scope/length reduction；
12. no-overwrite、resume、source/config/data hash 闭合；
13. CPU dry run 和 tiny GPU smoke 均通过。

只有 review 明确输出 `GPU Run Approved: MECHANISM PROBE ONLY`，才能执行 calibration GPU。
