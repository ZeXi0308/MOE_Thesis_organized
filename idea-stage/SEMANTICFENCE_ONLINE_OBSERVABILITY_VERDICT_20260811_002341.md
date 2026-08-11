# SemanticFence SFV2-O1 Fresh Online-observability Verdict

## 1. Gate verdict

正式机械裁决为 **`PIVOT_TO_SHADOW_VERIFY`**。Fresh natural Semantic Oracle 同时跨过 5% saving 与 5% row-coverage floor，故冻结定义下的 route/top-k-stability proxy action space 没有被否决；frozen witness-v1 则出现 4/5 unsafe executed pairs，只有 3.9216% matched-row coverage，且计入原型在线开销后 net projected saving=-183.3504%，因此不能进入 `GO_SEMANTIC_WITNESS_GATE`。`COMPLETE.json` 为 `SUCCESS_COMPLETE`、`paper_result=false`；无 project-module import 的 raw-ledger verifier 从 per-edge 记录重算得到同一 verdict。Fresh GPT-5.6-Sol ultra 审计为 `WARN / P0=0 / P1=3`（same-family provisional），三项 P1 均限制证据边界，不改变数值或机械 verdict。

## 2. Fresh/document-disjoint evidence

数据来自冻结的 WikiText-103 raw revision `b08601e04326c79dfdd32d625aee71d232d685c3`，dataset fingerprint 为 `051a1127882eb518`。扫描历史来源后形成 1,313 个 hash 的 defensive exclusion union；最终 12 个全文 hash 均不在该 union 内，并按 document 而非 row 做 6/2/4 train/validation/test split：

| Split | Source article indices | Full-text SHA-256 prefixes |
|---|---|---|
| Train | 1738, 16175, 12254, 23818, 289, 10781 | `81360b`, `c15a10`, `a3de33`, `40be42`, `ea4dcb`, `bd3a22` |
| Validation | 1484, 13422 | `5f0472`, `629afa` |
| Test | 15132, 25816, 1737, 28398 | `1f350d`, `bc592f`, `1a680c`, `54b42b` |

`PRE_OUTCOME_LOCK.json` SHA-256 为 `36a0595bb40296277f9fa31bcf35bcc0e54547f442aa88490315fc02ba3c9d38`，在任何 M1/M2 semantic outcome 前绑定 document/split、W=8、每 document 最多 32 edges、384-edge schedule（192/64/128）、64D projection seed 20260810、feature/score/threshold 规则和 runner/test/config/input hashes。Validation threshold 与 test admission plan 都记录 test outcome count=0。全部 768 个 endpoint observations 的 native self-replacement bitwise no-op、native/M1/M2 两次 full-forward stability 均通过。Hash chain、exclusive-write 代码顺序与“mtime 不晚于 outcome”共同支持 chronology；但回传后文件为同秒时间粒度，独立 raw receipt 如实记录 `strict_mtime_order_observable=false`，不能把严格 `>` 写入先后伪称为外部独立证明。

## 3. Natural Semantic Oracle

Fresh test natural graph 上 77/128 edges 在受控 M1/M2 intervention 下保持 downstream ordered top-k（60.15625%）；maximum matching 恰为 77 pairs，覆盖 154/255 unique rows（60.3922%），4/4 test documents 均有正动作。以同表 test expert microcost 做 additive projection：all-M1 为 10.491544 ms，Oracle 为 7.420525 ms，节省 29.2714%。相较历史 exact-M2 baseline 的 3.4034%，该 fresh proxy projection 高 25.8680 percentage points（约 8.60x），但两者的 safety boundary 与 candidate population 不同，不能把比例写成系统 speedup。

| Metric | Exact baseline | Fresh semantic oracle | Frozen certificate |
|---|---:|---:|---:|
| Safe/admitted edges | 6,160/786,281 safe（0.7834%） | 77/128 safe（60.1563%） | 5/128 admitted；仅 1/5 safe |
| Matched/executed pairs | 1,132 | 77 | 5 |
| Row coverage | 7.0236% | 60.3922% | 3.9216% |
| Positive fresh test documents | N/A | 4/4 | 3/4 |
| Unsafe executed pairs | 0 by exact definition | 0 by Oracle definition | 4 |
| Expert-stage projection | +3.4034% | +29.2714% gross | +1.9007% gross / -183.3504% net |

## 4. Frozen witness certificate

Witness representation、train-only normalization、cell banks、distance score 与唯一 global threshold 均在 test outcome 前冻结。Validation 选择 threshold `0.060565232014776364`：admit 4 endpoints、0 unsafe endpoints、greedy 2 pairs；test 只执行一次。Test 中 44/255 rows 有同时包含 safe/unsafe bank 的 scorable cell，19 endpoints 通过 threshold，形成 5 条 candidate/executed pairs。事后真值显示 5 个 admitted endpoints unsafe、4 条 executed pairs unsafe；只有 1 条 executed pair safe。Witness-v1 同时失败于 zero-unsafe、至少 16 pairs 与至少 5% row coverage 三门。该失败否决的是冻结的 pre-execution witness certificate v1，而不是只否决某个 Python bug；它也不证明所有未来 online observability 都不可能。

## 5. Cost and risk-coverage

Test microcost 为 `c1=0.041143311 ms`、`c2=0.042403251 ms`。Witness-v1 的 5 pairs 只把 expert compute 从 10.491544 ms 降至 10.292127 ms；measured hidden D2H + Python projection/normalization/nearest-witness/greedy overhead 中位数为 19.435709 ms，故 net runtime=29.727836 ms、net saving=-183.3504%。该 overhead 是未优化 prototype measurement，不能外推生产实现；但即使把它归零，4/5 unsafe 仍会独立阻断 GO。

固定 test 结束后的 diagnostic 显示：coverage@0%、1%、5% observed endpoint risk 都只有 2/255=0.7843%；仅 44/255 rows 可评分，AUROC=0.5333，AUPRC=0.8146，不能用高 AUPRC 掩盖低 scorable coverage 与近随机排序。Greedy token 在 4/256 test endpoint observations 上变化，仅作 diagnostic，未参与 semantic-safe 标签或 verdict。

## 6. System/paper implication

SemanticFence-v2 **在 SemanticFence 分支内继续作为 headline research problem，但不再是 witness-v1 headline method，更不是已成立的系统**；工作区全局 RCBA Oracle-first Primary 不因此被覆盖。被保留的是：在一个冻结的 OLMoE-1B-7B/BF16/single-RTX-5090 expert-stage natural workload 上，fresh document-disjoint downstream ordered-top-k-stability proxy action space 大且跨 4/4 documents。被否决的是：用当前 64D pre-expert witness distance 与单阈值，在执行前安全、足量、净正收益地识别它。

当前允许写入论文的最高 claim 是：**在上述单模型单栈窄域内，受控 M1/M2 contribution injection 显示 fresh natural candidates 存在 29.27% additive expert-stage downstream route/top-k-stability proxy Oracle headroom；但 frozen witness-v1 在 unseen documents 上无法零风险利用该空间。** 评价类型是 `self_supervised_proxy`，不是 task semantics 或 model-quality ground truth；不允许写成 serving throughput/TPOT/P99、vLLM integration、EP/NCCL、多卡泛化、production safety、形式化保证或 paper result。

## 7. 唯一下一步

只执行 **`SFV2-O2: Fresh No-Actuation Shadow-Verifier / Selective-Repair Gate`**：换一组新的 document-disjoint documents，outcome 前冻结一个 post-M2 verification/repair rule；M2 只做 tentative shadow execution，未经 verifier 通过不得 commit，失败时从冻结 checkpoint selective replay/fallback M1。唯一要回答的是：在 zero unsafe commit 前提下，verified/repaired coverage 是否至少达到 5%、pairs 是否至少 16，且计入 verifier 与 repair 实测成本后 net expert-stage saving 是否仍大于 0。不得在本次 test split 上重调 witness-v1，也不得开始第二个 classifier sweep。
