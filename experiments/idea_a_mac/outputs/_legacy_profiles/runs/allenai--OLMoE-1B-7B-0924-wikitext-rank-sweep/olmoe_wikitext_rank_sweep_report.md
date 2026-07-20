# OLMoE WikiText-2 Rank Sweep Report

## 1. 实验定位

这是目前 Idea A 最强的本地前置实验版本，用来回答两个关键问题：

1. 在真实 top-8 MoE 上，top-k 内部是否存在可利用的 rank 长尾？
2. 在相同 byte saving 下，近似 lowest-rank expert output 是否显著优于近似 high-rank expert output？

结论：**成立，而且非常明显。**

## 2. 实验配置

- 模型：`allenai/OLMoE-1B-7B-0924`
- 架构：16-layer OLMoE，64 experts，top-8 routing
- dtype：`bfloat16`
- 设备：Mac 本地 CPU-only
- profile 数据：WikiText-2 validation，128 条样本，seq_len = 128
- rank sweep 数据：WikiText-2 validation，32 条样本，seq_len = 128
- 指标：
  - contribution share：`g * ||o|| / sum(g * ||o||)`
  - next-token logit KL vs full
  - local combine relative MSE
  - WikiText-2 perplexity delta
  - expert output byte saving

## 3. C1：top-k 内部长尾强成立

Profile 结果：

- rank-8 median share across layers：`0.049277`
- rank1/rank8 median ratio across layers：`5.451188`
- C1 verdict：**强成立**

解释：

rank-8 的中位 contribution share 约为 4.93%，低于 10% 强成立阈值；rank-1 的贡献中位数约为 rank-8 的 5.45 倍。这说明 OLMoE top-8 内部不是均匀贡献，rank 本身可以作为一个低成本的重要性 proxy。

## 4. Rank-aware INT4 全 rank sweep

所有 `rankN_int4` 策略都只把一个 rank 的 expert output 做 INT4 fake quantization，因此 byte saving 相同，都是 `0.09375`。这样比较是公平的：同样节省 9.375% expert output bytes，只改变被压缩的是哪个 rank。

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| full | 0.00000 | 0.0000 | 0.000000 | 0.0000 |
| uniform_int4 | 0.75000 | 27.1522 | 0.085818 | 6.6782 |
| rank1_int4 | 0.09375 | 20.9892 | 0.029478 | 4.4878 |
| rank2_int4 | 0.09375 | 3.2981 | 0.011985 | 0.6440 |
| rank3_int4 | 0.09375 | 1.6709 | 0.008968 | 0.1684 |
| rank4_int4 | 0.09375 | 1.1275 | 0.006695 | 0.0203 |
| rank5_int4 | 0.09375 | 0.9036 | 0.004747 | -0.1579 |
| rank6_int4 | 0.09375 | 0.6251 | 0.003176 | -0.0420 |
| rank7_int4 | 0.09375 | 0.4553 | 0.001974 | 0.0406 |
| rank8_int4 | 0.09375 | 0.3614 | 0.001274 | -0.0291 |

关键对比：

- `rank8_int4` vs `rank1_int4`：
  - KL：`0.3614` vs `20.9892`，rank8 约低 `58.1x`
  - local relative MSE：`0.001274` vs `0.029478`，rank8 约低 `23.1x`
- `rank8_int4` vs `uniform_int4`：
  - KL：`0.3614` vs `27.1522`，rank8 约低 `75.1x`
  - local relative MSE：`0.001274` vs `0.085818`，rank8 约低 `67.4x`
- 从 rank1 到 rank8，KL 和 local relative MSE 基本单调下降。

## 5. 一锤定音的结论

这轮实验比之前 4 条 prompt 的 OLMoE smoke 更有说服力，因为它同时满足：

- 使用真实 top-8 MoE：OLMoE 16 layers、64 experts、top-8。
- 使用真实文本数据：WikiText-2 validation，而不是内置短 prompt。
- 使用更大 profile 样本：128 条文本验证 C1。
- 使用全 rank sweep：rank1 到 rank8 全部扫过，不是只挑 rank1/rank8 两个点。
- 使用同等 byte saving 的公平比较：所有 `rankN_int4` 都只压一个 rank。

最终结论可以写成：

> On OLMoE-1B-7B with top-8 routing, the lowest-rank expert output contributes only 4.93% median share across layers, while rank-1 contributes about 5.45x more. Under the same 9.375% expert-output byte saving, INT4 quantizing rank-8 reduces KL by 58.1x and local combine MSE by 23.1x compared with quantizing rank-1. This supports rank-aware mixed precision as a practical and low-overhead proxy for approximate MoE combine.

## 6. 对 Idea A 的写法影响

建议把 Idea A 主张收敛成：

**Profile-Guided Rank-Aware Mixed-Precision Combine for MoE Serving**

更具体地说：

- 主策略：rank-aware INT4 / BF16 mixed precision。
- LUT 维度：`layer x receiver_group x rank -> precision`。
- C1 证据：OLMoE top-8 上 rank 内部长尾强成立。
- C2 证据：同样 byte saving 下，压 rank8 远优于压 rank1。
- drop：只保留为 aggressive ablation，不作为默认主线。
- gate renormalization：不写成可靠修正，因为前一轮 OLMoE smoke 已经显示它会失败。

## 7. 产物

- `c1_long_tail_report.md`
- `rank_share_by_layer.csv`
- `approx_results.csv`
- `figures/rank_contribution_bar.png`
- `figures/rankk_share_by_layer.png`
- `figures/rank_sweep_kl.png`
- `figures/rank_sweep_local_mse.png`
- `figures/accuracy_byte_pareto_kl.png`
- `figures/accuracy_byte_pareto_ppl.png`
