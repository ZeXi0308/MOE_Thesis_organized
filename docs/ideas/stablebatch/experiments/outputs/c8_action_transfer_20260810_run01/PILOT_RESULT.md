# StableBatch fixed-C8 Action-Transfer Gate

**状态**：`COMPLETE`  
**一句话 Gate 结论**：**`GO / CASE_A`**——在冻结的 33 个 M1-positive unique cells 上，fixed-C8 same-rank 保留 `31/37=83.78%` 的 M1 gross route recovery，净收益 `+25` 高于 exact-random rank 的 `+18.25`，且 8/8 LODO 均保持正的 rank-specificity gap。  
**审计**：`PASS / P0=0 / P1=0`，same-family provisional。  
**GPU**：单张 RTX 5090，单 OLMoE revision，BF16 eager，正式 run 100.52 秒。

## 本轮唯一问题

> fixed-C8 protection 是否能够继承 M1 oracle protection 的路由恢复能力，而且这种收益是否依赖保护正确的 rank？

## 核心路由对比

Baseline 有 41 个相对 all-M1 proxy 的原始 downstream route mismatches。

| 条件 | Recovered | Harmed | Net reward | 相对 R 剩余 distance |
|---|---:|---:|---:|---:|
| No intervention / U | 0 | 0 | 0 | 41 |
| M1 same-rank | 37 | 0 | **37** | 4 |
| C8 same-rank | 31 | 6 | **25** | 16 |
| C8 exact uniform-random rank | 22.25 (`89/4`) | 4 | **18.25 (`73/4`)** | 22.75 |
| C8 best-rank oracle | 36 | 0 | **36** | 5 |
| M1 exact uniform-random rank | 21.875 (`175/8`) | 2.375 (`19/8`) | **19.5 (`39/2`)** | 21.5 |

预冻结指标：

- **C8 transfer ratio**：`31/37 = 83.78%`。
- **Rank-specificity gap**：`25 - 18.25 = 6.75`。
- **C8 oracle gap**：`36 - 25 = 11`。
- Harm-aware net retention：`25/37 = 67.57%`；因此 C8 不是 M1 等价物，6 个新增 route mismatches 必须进入后续预算。

## Final-logit 元素级结果

| 条件 | Recovered elements | Harmed elements | Net | Residual mismatch elements |
|---|---:|---:|---:|---:|
| U baseline | 0 | 0 | 0 | 1,393,575 |
| M1 same-rank | 480,251 | 128,927 | 351,324 | 1,042,251 |
| C8 same-rank | 391,066 | 145,289 | 245,777 | 1,147,798 |
| C8 exact random | 302,596.25 | 141,938.625 | 160,657.625 | 1,232,917.375 |
| C8 oracle | 452,218 | 134,701 | 317,517 | 1,076,058 |

33/33 final-logit vectors在上述 intervention 后仍与 R bitwise 不同；没有 greedy-token 改变。既有 M1 sweep 只保留 final-logit hashes，因此 M1 exact-random 无法诚实重建元素级 recovered/harmed；其 33 个 vector-level 条件均未 exact restore。这里不把元素恢复写成模型质量提升。

## 三层聚合

### Action level

- 33 cells x 8 ranks = 264 个 C8 actions。
- 合计 route `178 recovered / 32 harmed / net +146`；这是完整 action surface，不是一个可部署 policy。
- positive / zero / negative net actions = `134 / 118 / 12`。
- 除以 8 得到 primary exact-random expectation：`22.25 / 4 / +18.25`。

### Unique-cell level

- Primary 始终是 33 unique cells 等权，不把 139 个正 rank 当独立样本。
- C8 same-rank：27/33 cells 有 recovery，4/33 有 harm；positive / zero / negative net cells = `24 / 7 / 2`。
- Full route restoration：M1 `31/33`，C8 same-rank `24/33`，C8 oracle `29/33`。
- C8 same-rank 相对 exact-random：`21` cells 更好、`5` 相同、`7` 更差。

Secondary sensitivity（会重复计算 multi-positive cells，不用于显著性）：全部 139 个 M1-positive rank actions 上，M1 为 `166/0/+166`，对应 C8 为 `149/11/+138`；C8 positive / zero / negative actions = `115/21/3`，方向与 primary 一致。

### Per-document 与 LODO

| Doc | Cells | M1 net | C8 same rec/harm/net | C8 random net | C8 oracle net |
|---:|---:|---:|---:|---:|---:|
| 17 | 8 | 8 | 7 / 1 / **6** | 3.5 | 8 |
| 18 | 1 | 1 | 1 / 0 / **1** | 0.625 | 1 |
| 19 | 7 | 8 | 5 / 2 / **3** | 1.25 | 8 |
| 22 | 1 | 1 | 1 / 0 / **1** | 0.625 | 1 |
| 26 | 4 | 7 | 7 / 0 / **7** | 5.375 | 7 |
| 28 | 1 | 1 | 1 / 0 / **1** | 0.875 | 1 |
| 30 | 6 | 6 | 4 / 1 / **3** | 3.625 | 5 |
| 31 | 5 | 5 | 5 / 2 / **3** | 2.375 | 5 |

C8 same-rank net 在 8/8 documents 为正，在 7/8 documents 高于 random；doc30 是唯一 per-document specificity 反例。

| LODO left-out doc | C8 same net | C8 random net | Specificity gap |
|---:|---:|---:|---:|
| 17 | 19 | 14.75 | 4.25 |
| 18 | 24 | 17.625 | 6.375 |
| 19 | 22 | 17 | 5 |
| 22 | 24 | 17.625 | 6.375 |
| 26 | 18 | 12.875 | 5.125 |
| 28 | 24 | 17.375 | 6.625 |
| 30 | 22 | 14.625 | 7.375 |
| 31 | 22 | 15.875 | 6.125 |

8/8 LODO 的 C8 same-rank net 与 specificity gap 均为正。

## 实验完整性

- Cohort 在读取 C8 outcome 前冻结：33 unique cells、8 docs、139 raw-positive ranks、27 个 multi-positive cells；SHA-256 `369106423a66236c35efb4f1c1ad92d9c153fb732093abb77ae53b119fdfd3`。
- Frozen lock SHA-256 `13cdda8f2a25a6913236ffedb6b46744534dc319d92e507276bca5e0cbac55e1`；runner/config/recompute SHA-256 分别为 `f9ded0c299bf5a2d24e0e999356835667b36468bb459cc4a4b4ad95d2e9bb7ed`、`046056af0850729184d64d39a805cfa7a0203b7445af133a7064bc226f633fed`、`247e8013e248ba07c5cdedb17789fe5e5f1f856cb51472560977d98ecd189ca4`。
- 两组本地与远端测试均为 6/6；远端运行前 GPU 为 2 MiB、0% 且无 compute process。
- Output manifest 的 13 个 bound artifacts 全部匹配；raw `cell_results.jsonl` SHA-256 `9d4a1742820c19d7f1e20cb6b16a4ff7d34b67dd9bf0846b7c210987420d2a7d`。
- 独立 recompute `mismatch_count=0`；本地第二次重算与密封版本 byte-identical，SHA-256 `673d2f884adaf149d44b5c1e351b5673ca1d45e6b59374d71d23d2e99e4e9da4`。
- Fresh integrity audit：`PASS / P0=0 / P1=0`。route 决策可从 raw top-k 独立重建；final-logit packed bitsets 只能内部验证，未保留 raw tensors 做 paper-grade 再生。

## 更新后的研究方向

这是预冻结的情况 A：**fixed-C8 是有用、需要选择、可作为稀疏预算候选的保护 primitive**。

主线收敛为一个系统，不再拆成两个 idea：

> **ShapeABI / ShapeLane execution primitive + StabilityBudget sparse scheduling policy**。

理由是：C8 same-rank 已继承大部分 gross M1 recovery，正确 rank 的净收益稳定高于精确随机 rank，而 C8 oracle 仍比 M1-rank surrogate 多 11 点净收益。后者说明后续 selector 应学习 C8-specific action value，不能把 M1 oracle rank 当成最终 ground truth。

当前仍未证明真实 serving 成本、SLO、自然 prevalence 或模型质量，所以只能升级为主论文 hypothesis 与下一 Gate，不能称为已完成系统。

## 唯一下一 Gate

**Outcome-naive C8 selector held-out Gate**：只把当前已密封的 33 x 8 C8 surface 当 calibration，用冻结的 action 前特征和唯一一个 L2 ridge rank scorer（无 feature/model search），在任何新 outcome 前密封它于 16 个 document-disjoint fresh windows 上选出恰好 `B=33` 个 cell-rank actions。随后才跑每 cell 的全 8-rank C8 surface，比较 budget-matched exact random 与 C8 oracle，报告 recovered/harmed/net、正负 action precision/recall、per-document 和 LODO。

它只排除两个竞争解释：

1. rank-specificity 只是 retrospective outcome oracle，action 前没有可用信号；
2. aggregate gain 只是特定 documents 的记忆，不能跨 document transfer。

该 Gate 前不做 serving prototype、更多 C、selector 大模型、feature search 或成本调参。
