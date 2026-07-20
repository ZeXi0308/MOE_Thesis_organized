# OLMoE Top-8 Final Report

## 1. 实验配置

- 模型：`allenai/OLMoE-1B-7B-0924`
- 架构：OLMoE
- MoE 层数：16
- hidden size：2048
- intermediate size：1024
- experts：64
- top-k：8
- dtype：bfloat16
- 设备：CPU-only
- profile 样本：4 条内置英文短文本
- seq len：64

这个模型比 Mixtral top-2 更贴近 Idea A 的关键假设，因为它是 **top-8 routing**。

## 2. Baseline 跑通情况

- 权重约 13 GB，已完整下载到 Hugging Face cache。
- 默认 xet 下载会卡住；使用 `HF_HUB_DISABLE_XET=1` 后普通 HTTP 下载成功。
- 缓存命中后加载很快：`load_seconds = 0.37`
- 单条 smoke forward：`0.32s`
- logits shape：`(1, 11, 50304)`

## 3. C1：top-k 内部 contribution 长尾

统计 `share = g * ||o|| / sum(g * ||o||)`。

总体结果：

- top-k：8
- rank-8 median share across layers：`0.055791`
- rank1/rank8 median ratio across layers：`4.304844`
- C1 verdict：**强成立**

代表层：

| layer | rank-1 median share | rank-8 median share | rank1/rank8 median |
|---|---:|---:|---:|
| 0 | 0.2138 | 0.0393 | 5.03 |
| 1 | 0.2843 | 0.0487 | 4.96 |
| 5 | 0.2059 | 0.0733 | 3.00 |
| 10 | 0.2383 | 0.0498 | 5.02 |
| 12 | 0.2701 | 0.0394 | 6.83 |
| 14 | 0.2398 | 0.0380 | 6.60 |

解释：

OLMoE 的 top-8 内部分布不是 12-layer Mixtral top-2 那种极端“一边倒”，但 lowest-rank contribution 仍显著小于 rank-1。rank-8 median share 约 5.6%，已经低于前置实验设定的 10% 强成立阈值。

## 4. Rank-aware 近似结果

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| full | 0.00000 | 0.0000 | 0.000000 | 0.0 |
| uniform_int4 | 0.75000 | 8.3315 | 0.070148 | 1317.1 |
| rank8_int4 | 0.09375 | 0.0737 | 0.000644 | -62.0 |
| rank1_int4 | 0.09375 | 6.7794 | 0.022005 | -41.6 |
| rank8_drop | 0.12500 | 0.3178 | 0.002516 | 445.5 |
| rank8_drop_renorm | 0.12500 | 46.0783 | 0.268972 | 7787.0 |
| rank1_drop | 0.12500 | 36.4960 | 0.580007 | 6376.9 |

可用结论：

1. `rank8_int4` 明显优于 `rank1_int4`：KL `0.0737` vs `6.7794`，local relative MSE `0.000644` vs `0.022005`。
2. `uniform_int4` 明显不稳：节省 75% bytes，但 KL `8.3315`。
3. `rank8_drop` 可以跑，但比 `rank8_int4` 伤得更多：KL `0.3178` vs `0.0737`。
4. `drop + renorm` 在 OLMoE 上失败：KL `46.0783`，不能写成默认修正。
5. PPL 仍只作辅助参考；当前样本数较小，主要看 KL 与 local relative MSE。

## 5. 对 Idea A 的影响

这轮 OLMoE top-8 结果是目前最关键的前置证据：

- C1 在真实 top-8 MoE 上成立。
- rank-aware 的必要性很强：处理 lowest-rank 与处理 rank-1 的误差差距巨大。
- `rank` 可以作为静态、低开销的重要性代理。
- 主线应写成 **rank-aware mixed precision combine**，尤其是 rank-k INT4。
- drop 可以保留为 aggressive ablation，但不应作为默认主策略。
- `drop + renorm` 不能继续当作“best-effort 修正一定有效”，它在不同模型上表现不一致。

建议更新 proposal：

1. 把 C1 从“待验证假设”提升为“前置实验支持，但仍需扩大样本确认”。
2. 主策略顺序改成：`BF16 / INT4 mixed precision` 优先，`drop` 次之。
3. 写明 OLMoE top-8 前置实验结果：rank-8 median share 约 5.6%，rank8 INT4 的 KL 远低于 rank1 INT4。
4. 不要再强调 gate renormalization 是可靠修正，只把它作为 ablation。

## 6. 产物

- `model_smoke_result.md`
- `rank_share_by_layer.csv`
- `c1_long_tail_report.md`
- `approx_results.csv`
- `rank_aware_report.md`
- `figures/rank_contribution_bar.png`
- `figures/rankk_share_by_layer.png`
- `figures/accuracy_byte_pareto_kl.png`
- `figures/accuracy_byte_pareto_ppl.png`

