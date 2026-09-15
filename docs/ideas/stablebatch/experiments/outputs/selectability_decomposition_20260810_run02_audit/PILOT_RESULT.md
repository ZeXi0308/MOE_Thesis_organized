# StableBatch Selectability Decomposition Gate — Formal Result

**Verdict**：`STOP_PREACTION_STABLEBATCH`  
**Evidence tier**：fresh request-disjoint、单 OLMoE / 单 RTX 5090 / BF16 eager / self-supervised route proxy  
**Integrity**：`WARN / P0=0 / P1=0`；两次独立重算 PASS

## 一句话结论

action space 的 hindsight opportunity 在新请求上很强，但 calibration 后冻结的 static compatibility map 与 online observable ridge 都无法在执行前把它选出来；因此 StableBatch 不能继续作为 compatibility-aware coalescing 或 row-conditioned scheduling 的论文主机制。

## Frozen result

| Policy | Actions | Recovered | Harmed | Reward | Positive victims | Recovered Oracle Gap |
|---|---:|---:|---:|---:|---:|---:|
| Outcome Oracle | 33 | 57 | 0 | 57 | 12 | 1.0000 |
| Static map | 33 | 7 | 14 | -7 | 4 | -0.04918 |
| Online ridge | 33 | 7 | 14 | -7 | 4 | -0.04918 |
| Matched shuffle | 33 | 5 | 9 | -4 | 3 | 0.0000 |

- unprotected route distance：`84`
- Oracle recovery fraction：`57/84 = 67.86%`
- Oracle–shuffle gap：`61`
- deterministic uniform-expectation diagnostic：`-55/32 = -1.71875`
- static LODO：`0/16`
- online LODO：`0/16`

## Mechanical decision

- Oracle opportunity 四项全 PASS。
- Static 的 reward-positive、above-shuffle、30% gap 三项均 FAIL。
- Online 的同三项均 FAIL。
- full-cohort 已失败；LODO 没有提供任何救援证据。

因此符合预冻结分支：

> Oracle strong，static/online both near or below shuffle → stop original pre-action StableBatch.

## Integrity closure

- run02：240/240 cells，1,920/1,920 actions，formal status COMPLETE。
- run01：GPU 并发污染，已中止并排除，不进入结论。
- selector lock 在 outcome 文件创建前封存。
- fresh 16 documents 与 calibration 的全文/window overlap 均为 0。
- run manifest 全文件 hash/size 本地复核通过。
- 独立 aggregation recompute：`mismatch_fields=[]`。
- raw-route recompute：重新推导 240 个 U arms 与 1,920 个 action arms，route mismatch `0`、summary mismatch `0`。

## 结论边界

**成立**：execution shape 可以传播；fresh cohort 中存在很强的 hindsight intervention opportunity。  
**不成立**：当前静态 compatibility graph 或冻结浅层 online observables 能提供可执行选择。  
**未证明**：不存在任何未来 selector；serving/quality/跨模型/多 GPU/EP 收益。  
**研究动作**：不做 MaxGate-v2、第三个手工 selector、partition planner 或 controller。若研究 execution-after shadow verification，必须作为新机制重新立项，不能继续包装为 StableBatch。

## Evidence

- Raw authority：`../selectability_decomposition_20260810_run02/`
- Independent aggregation：`INDEPENDENT_RECOMPUTE.json`
- Raw-route verification：`RAW_ROUTE_RECOMPUTE.json`
- Integrity audit：`EXPERIMENT_AUDIT.md`

