# Idea A Mac Smoke Final Report

## 1. 实验配置

- 模型：`jamesdborin/tiny-mixtral`
- 架构：Mixtral
- MoE 层数：2
- hidden size：1024
- experts：8
- top-k：2
- 设备：CPU-only（当前 PyTorch 未检测到 MPS）
- 样本：8 条内置英文短文本
- seq len：96

本轮目标是验证代码链路与最小方法闭环，不把结果当成正式论文结论。

## 2. Baseline 跑通情况

- 首次模型加载：240.46s（包含 Hugging Face 权重下载）
- 单条 smoke forward：0.85s
- logits shape：`(1, 12, 32000)`

结论：Mac 上可以跑 tiny Mixtral 的前置实验。后续重复运行会走本地 HF 缓存，不需要重新下载权重。

## 3. C1：top-k 内部 contribution 长尾

统计 `share = g * ||o|| / sum(g * ||o||)`。

| layer | rank-1 median share | rank-2 median share | rank1/rank2 median |
|---|---:|---:|---:|
| 0 | 0.5923 | 0.3995 | 1.4528 |
| 1 | 0.5470 | 0.4525 | 1.2074 |

按预设标准：

- rank-k median share 没有低于 10%。
- rank1/rankk median 没有超过 3。
- 在这个 tiny top-2 模型上，C1 判定为：**不成立或证据不足**。

重要边界：

这个结果不能直接否定 Idea A。原因是当前模型只有 top-2，而 proposal 的关键先验更依赖 DeepSeek-V2-Lite top-6 或更大 top-k 模型。当前结论只能说明：在 tiny top-2 smoke 上，不能主张明显 top-k 内部长尾。

## 4. Rank-aware 近似结果

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| full | 0.000 | 0.0000 | 0.0000 | 0.0 |
| uniform_int4 | 0.750 | 0.1266 | 0.0204 | -719.1 |
| rankk_int4 | 0.375 | 0.0490 | 0.0070 | -642.6 |
| rank1_int4 | 0.375 | 0.0971 | 0.0135 | 112.2 |
| rankk_drop | 0.500 | 0.6728 | 0.3361 | -401.7 |
| rankk_drop_renorm | 0.500 | 1.0984 | 0.6601 | -2286.2 |
| rank1_drop | 0.500 | 1.5687 | 0.6554 | 1585.8 |

可用结论：

1. `rankk_int4` 比 `rank1_int4` 更稳：KL 约为 0.0490 vs 0.0971，local relative MSE 约为 0.0070 vs 0.0135。
2. `drop` 在这个模型上很激进：即使 drop rank-k，local relative MSE 也达到 0.3361，明显高于 INT4。
3. `rankk_drop_renorm` 没有改善，反而更差；说明简单 gate renormalization 不应默认写成可靠修正。
4. PPL 本轮不宜过度解读，因为 tiny model 的 baseline PPL 极高，且样本只有 8 条；KL 和 local relative MSE 更可信。

## 5. 对 Idea A 的影响

当前 smoke 支持：

- MoE combine 位置可以被 hook。
- 可以采集 `g * ||o||` contribution。
- 可以在 combine 前对指定 rank 做 fake INT4 / drop。
- rank-aware INT4 比错误 rank INT4 更稳，这支持“rank 是一个有用的重要性代理”的初步说法。

当前 smoke 不支持：

- 不能声称 top-k 内部 contribution 明显长尾。
- 不能把 drop 写成默认主策略。
- 不能声称 Mac 实验证明 serving TBT 或 receiver-port 拥塞收益。
- 不能把 tiny top-2 结果外推到 DeepSeek-V2-Lite top-6。

## 6. 建议下一步

优先路线：

1. 用同一套代码跑更大 top-k 的真实 MoE，例如 OLMoE 或 DeepSeek-V2-Lite 的小样本 profile。
2. 如果 Mac CPU 跑不动大模型，只在 Mac 上保留 tiny Mixtral smoke，把正式 C1 实验迁移到服务器/GPU。
3. proposal 里暂时把 drop 降级为 aggressive option，主线写成 BF16 / INT4 mixed precision 更稳。
4. C1 的措辞从“长尾已成立”改为“需要在真实 top-k MoE 上 profile 验证；rank-aware INT4 smoke 显示方向有初步可行性”。

## 7. 正式产物

- `model_smoke_result.md`
- `rank_share_by_layer.csv`
- `c1_long_tail_report.md`
- `approx_results.csv`
- `rank_aware_report.md`
- `figures/rank_contribution_bar.png`
- `figures/rankk_share_by_layer.png`
- `figures/accuracy_byte_pareto_kl.png`
- `figures/accuracy_byte_pareto_ppl.png`

