# RCBA Oracle Gate Protocol — Preparation-Blocked Draft

> 状态：`BLOCKED_PROTOCOL_AMBIGUITY`  
> 日期：2026-08-10  
> Formal Gate：`NOT_RUN`  
> 本文角色：记录已恢复合同、冻结 fallback metric 和必须先关闭的歧义；不是 run lock，也不是实验结果。

## 1. 研究问题

要验证的是：在自然 continuous-decode workload 中，MoE route identity 与 expert-tail 分布是否通过真实 runtime barrier 被放大为 full-request critical-path latency；保持相同 arrivals、计算 work、resource capacity、request population 与全部数据依赖时，只删除非必要 barrier edge 的 capacity-constrained Oracle，能否在两个模型的共同自然 regime 中保留至少 10% charged full-request headroom。

必须区分：

1. `local tail`：单层或单 expert service-time 差异；
2. `barrier amplification`：local tail 经 layer/step/batch/iteration synchronization 传播的额外等待；
3. `full-request leverage`：删除 barrier 后真实提前 request completion。

`local overlap opportunity != full-request critical-path headroom`。

## 2. 已恢复并冻结的字段

### 两个模型

| Key | Model | Revision | Dtype |
|---|---|---|---|
| `olmoe` | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | BF16 |
| `llmjp` | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M` | `1d5983076dfc67aee4a77ec06a27027f5bab6055` | BF16 |

来源是 `docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json` 与两个 frozen workload manifests；不得替换模型、revision 或 dtype。

### Natural trace source contract

- Dataset：WikiText-103 raw test，固定 revision、Arrow hash 与 first 128 nonempty rows；
- Arrivals：BurstGPT_1 固定 repository revision/blob/CSV hash，header 后 first 128 rows；
- 已有变换：source time origin 5 s、factor 1000、单位 microseconds；
- Decode：greedy，最多 16 steps；max batch size 8；workload seed `20260725`。

这些只定义输入来源，不证明 canonical runtime trace 已存在，也不自动定义 RCBA 的 common natural load regime。

### Frozen fallback primary metric

仓库没有唯一 canonical charged RCBA objective，因此采用本轮显式给定且必须在任何 formal result 前冻结的定义：

`flow_r = request_completion_r - request_arrival_r`

`J = mean_r(flow_r)`

`H = (J_real_barrier - J_no_barrier) / J_real_barrier`

诊断指标为 median/P95 full-request flow、completion distribution、makespan、total work、resource utilization 与 per-request barrier wait。Route attribution 为：

`A = H_original_route - H_route_decorrelated`

Formal qualification threshold 保持：同一个 common natural regime 中两个模型都满足 `H >= 10%`，且 `A >= 3 percentage points`；三个 replay 的 arrivals、work、capacity 与 request population 必须完全相同。本轮不计算 `H` 或 `A`。

## 3. 实现前必须关闭的协议歧义

| Field | Current state | Why it changes the Gate |
|---|---|---|
| Common natural regime | `AMBIGUOUS` | Jury 只写共同自然 regime；旧 BCRD cells 混合 phase、concurrency、virtual replicas 与 policy，未说明 RCBA 是否继承。选择不同 cell/population 会直接改变 `J` 和 `H>=10%`。 |
| Removable barrier whitelist | `AMBIGUOUS` | 当前没有可移除 layer/step/batch barrier 的逐类定义与 provenance；误删 DATA_DEPENDENCY 会虚增 H，漏删合法 runtime barrier 会压低 H。 |
| Full edge semantics | `PARTIAL` | 历史只约束 exact-once contribution、step recurrence 与 route/dispatch/expert/combine 顺序，没有全 DAG 的 edge-by-edge identity/classification。 |
| Resource capacity | `AMBIGUOUS` | 1×RTX 5090 和 max batch 8 不是 Oracle capacity vector；resource pools、operator sharing、occupancy/concurrency 与 capacity units 未锁，可能退化成无限并发或错误串行化。 |
| Route-decorrelation scope/seed | `AMBIGUOUS` | 未冻结在哪个 layer/step/request population 内置换 route/expert-tail bundle，也没有 permutation seed；post-result 选择会改变 A。 |

在以上字段形成单一、可审计合同前，不实现 evaluator 或 schema。这里不是工程洁癖：每项都能翻转正式 10% verdict。

## 4. Full-request DAG 最低合同（待歧义解除后实现）

每个 node 至少需要 `request_id`、`decode_step`、`layer_id`、task type、route/expert identity、service duration、resource requirement、original execution identity、input/output dependency identity 与 service-surface join key。DAG 必须覆盖 arrival、可观测的 prefill/decode boundary、step recurrence、layer dependency、router、dispatch、expert、combine/join、non-MoE work、sampling/completion 和真实 runtime barrier。

每条 edge 只能是 `DATA_DEPENDENCY`、`RESOURCE_ORDER` 或 `RUNTIME_BARRIER`。只有具备预冻结 whitelist 与 trace provenance 的 `RUNTIME_BARRIER` 可删除；真实数据依赖、expert contribution、work、arrival、token/decode length 与 capacity 均不可改变。

## 5. 三种 replay（均未实现）

- `REAL_BARRIER_REPLAY`：保留三类 edge，并在误差合同内重现 input trace completion、ordering 与 work。
- `CAPACITY_CONSTRAINED_NO_BARRIER_ORACLE`：只删除 whitelisted `RUNTIME_BARRIER`，保留 work/dependency/arrival/capacity，输出合法 earliest-feasible schedule。正式规模若不用 exact search，必须标注 bound direction；tiny fixture 必须与 exhaustive exact 对齐。
- `ROUTE_DECORRELATED_REPLAY`：只在预冻结 scope/seed 内重新关联 request/token 与 route/expert-tail identity，同时保持 per-layer/step routed-token count、expert histogram、duration multiset、work、capacity 与 barrier structure。

## 6. 已确认的后续输入缺口

- OLMoE identity-complete canonical trace：`NO`；只有 `INCOMPLETE / scientific_result_eligible=false` failure capsule。
- LLM-jp identity-complete canonical trace：`NO`；只有 frozen input manifest，没有 execution artifact。
- OLMoE complete measured service surface：`NO`。
- LLM-jp complete measured service surface：`NO`。

已有单 expert curves、少量 selected experts 或 aggregate MoE-stage timing均不能覆盖 router/dispatch/expert/combine/non-MoE/sampling 的 full-DAG durations，不能插值或复制补齐。

## 7. 解除条件

只有以下全部完成后才允许实现 evaluator：

1. 唯一 common natural regime 与 population rule 写入 protocol；
2. runtime barrier whitelist、不可删除 dependency mapping 与 provenance 规则逐项冻结；
3. resource pool/capacity/concurrency model 冻结；
4. route-decorrelation scope 与 seed 冻结。

之后仍必须先修复双模型 trace pipeline 和 measured service surface。当前 candidate 保持 `PRIMARY_NEXT_CANDIDATE / UNVALIDATED`，不得写成成立、READY 或机制。
