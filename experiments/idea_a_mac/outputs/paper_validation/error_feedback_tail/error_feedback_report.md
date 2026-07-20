# Error-Feedback (Residual Compensation) for Tail-Rank Combine Quantization

## 目的

检验方向 3：把 tail-rank INT4/MXFP4/NVFP4 量化的舍入残差，沿 token 序列因果地
向前传递（类似压缩 SGD 里的 error feedback，搬到 combine 输出向量流），能否在
**同样的字节预算**下降低 held-out PPL/KL 退化。这是一个纯 sender 侧本地状态，
不产生任何额外通信字节，只改变"量化前送进量化器的是什么值"。

## 方法

- baseline: `full`（无量化）
- 每种 tail precision（int4, mxfp4, nvfp4）分别跑两种模式：
  - `plain`：逐 token 独立量化（现有 `fp8topN_rest_{prec}` 策略的行为）
  - `error_feedback`：残差在同一文档内，沿 token 顺序、按 (layer, tail-rank-slot)
    因果累积并补偿到下一个 token 的量化输入上；每篇文档开头残差重置为 0。
- head ranks（前 `4` 名）两种模式下都保持 FP8，不受影响。
- 测试集：`wikitext2:validation` offset=`128`，n=`32`，
  seq_len=`128`（与其余实验一致，避免引入新的评测口径差异）。

## 结果

| strategy | tail_precision | mode | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|---|
| full | - | - | 18.791504 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fp8top4_rest_int4_plain | int4 | plain | 19.238104 | 0.446600 | 0.030320 | 0.025771 | 0.035536 |
| fp8top4_rest_int4_error_feedback | int4 | error_feedback | 19.374362 | 0.582858 | 0.047373 | 0.041913 | 0.053055 |
| fp8top4_rest_mxfp4_plain | mxfp4 | plain | 18.734759 | -0.056745 | 0.006840 | 0.005638 | 0.008636 |
| fp8top4_rest_mxfp4_error_feedback | mxfp4 | error_feedback | 18.757067 | -0.034437 | 0.008517 | 0.007500 | 0.009806 |
| fp8top4_rest_nvfp4_plain | nvfp4 | plain | 18.669896 | -0.121608 | 0.005713 | 0.004736 | 0.007062 |
| fp8top4_rest_nvfp4_error_feedback | nvfp4 | error_feedback | 18.730955 | -0.060549 | 0.006617 | 0.005665 | 0.007758 |

## Plain vs Error-Feedback 直接对比

| tail_precision | plain_kl | ef_kl | kl_relative_change | ef_better |
|---|---|---|---|---|
| int4 | 0.030320 | 0.047373 | 0.562407 | False |
| mxfp4 | 0.006840 | 0.008517 | 0.245224 | False |
| nvfp4 | 0.005713 | 0.006617 | 0.158168 | False |

## 解读边界

- 这仍是 fake-quant PPL/KL 实验，不是通信或 kernel benchmark；error feedback 在
  真实系统里需要在 sender-side pack 路径里维护每个 (layer, tail-rank-slot) 的
  hidden_dim 残差状态，本实验未建模这部分状态维护和计算的延迟开销。
- 残差按"文档内 token 顺序"因果累积，只使用过去信息，不使用未来 token，理论上
  可以在真实 autoregressive 解码中原生实现（每步只需要维护和更新一个 hidden_dim
  向量），不需要额外通信。
- 若 `error_feedback` 在多个 precision 下都稳定降低 `mean_token_kl`（且降低幅度
  超出 bootstrap CI 范围），说明这是一个几乎零成本、可以直接叠加在现有
  tail-rank two-lane 方案上的质量改进，可作为论文里"tail-INT4 有用"这条仅存
  claim 的进一步加固点。
- 若两者差异很小或方向不稳定，说明 combine 输出向量之间（不同 token、同一
  rank slot）的相关性不足以支撑 error feedback 起作用，应如实报告为负结果。
