# Idea A 前置实验设计：Mac M5 Pro 可跑版本

## 0. 目标定位

这个前置实验不验证多机 serving，也不先做 receiver-port 拥塞优化。它只回答 Idea A 能不能继续推进的两个生死问题：

1. **C1 是否成立**：MoE top-k 内部的 combine contribution 是否真的长尾。
2. **rank-aware 近似是否有优势**：降精度 / drop 低 rank expert output 是否比处理高 rank 更不伤精度，并能形成初步 accuracy-traffic Pareto。

当前机器是 **Apple M5 Pro / 48 GB unified memory**，适合做单机离线 profiling。实验结论应定位为 characterization + feasibility evidence，不要包装成真实通信系统评估。

---

## 1. 最小闭环

### 1.1 模型选择

优先顺序：

1. **OLMoE-1B-7B 量级 MoE**：总参数较小，48 GB 内存更稳，适合作为第一版跑通 hook、contribution 统计和近似实验。
2. **Qwen1.5-MoE-A2.7B**：更贴近 proposal 中的 top-k MoE 目标，但总参数和实现复杂度更高，作为第二阶段。
3. **DeepSeek-V2-Lite**：更适合之后上服务器做正式实验；Mac 上只作为可选尝试，不作为前置实验依赖。

选择标准不是模型名本身，而是：

- transformers 里 MoE block 是 Python 可读实现，方便 hook / monkey patch；
- 能拿到 router logits、top-k expert id、gate weight、expert output；
- batch size = 1、seq len = 256 或 512 时能稳定 forward。

### 1.2 数据集

第一版用小规模语言建模数据即可：

- **WikiText-2 validation**：用于 perplexity 和 next-token logit KL；
- 可补充 **C4 小切片**：避免只在 WikiText 上过拟合结论；
- smoke test：16 条样本，每条 128 tokens；
- 正式前置实验：256 条样本，每条 256 tokens；
- 机器能承受时扩到 512 条样本，每条 512 tokens。

这一步不需要 MMLU / GSM8K。那些 benchmark 成本高、噪声大，适合后续正式实验。

---

## 2. 需要采集什么

对每个 MoE layer、每个 token、每个 top-k rank，采集：

| 记号 | 含义 |
|---|---|
| `l` | MoE layer id |
| `t` | token index |
| `r` | expert 在该 token top-k 中的 rank，`r=1` 表示 gate 最大 |
| `e_{l,t,r}` | rank `r` 对应 expert id |
| `g_{l,t,r}` | router softmax 后的 gate weight |
| `o_{l,t,r}` | expert output 向量，未乘 gate |
| `c_{l,t,r}` | contribution score，定义为 `g_{l,t,r} * ||o_{l,t,r}||_2` |
| `s_{l,t,r}` | normalized share，`c_{l,t,r} / sum_r c_{l,t,r}` |

核心图就是 `s_{l,t,r}` 按 rank 的分布。

为了省内存，不要把所有 `o_{l,t,r}` 全量落盘。第一版只在线累加：

- 每层每个 rank 的 share 均值 / median / P75 / P90；
- `rank1_share / rankk_share` 的分布；
- `rankk_share` 的 CDF；
- 后续近似实验需要的局部 MSE / KL 聚合值。

---

## 3. 实验 A：top-k 内部长尾验证

### 3.1 做法

跑原始 BF16/FP16 模型 forward，保持原始 combine 不变，只记录中间信息。

对每个 MoE layer 画：

1. **rank contribution bar plot**  
   横轴 rank，纵轴 `mean(s_{l,t,r})`，每层一条曲线或分组柱状图。

2. **rank-k contribution CDF**  
   横轴 `s_{l,t,k}`，纵轴 CDF，用来说明最低 gate expert 的贡献是否经常很小。

3. **tail mass by layer**  
   如果 top-k >= 4，统计 `sum_{r>=k-1} s_{l,t,r}`；如果 top-k = 2，只统计 `rank2_share`。

### 3.2 通过标准

这不是严格理论定理，但可以设一个工程上的 go/no-go：

- **强成立**：多数层里 `rank-k` median share < 10%，且 `rank1/rankk` median > 3。
- **弱成立**：`rank-k` median share 在 10% 到 20% 之间，仍可做差分量化，但 drop 要谨慎。
- **不成立**：top-k 内部 share 接近均匀，`rank-k` 经常超过 25%。这时不要主打 drop，转向 receiver-aware + layer-sensitive mixed precision。

这里的结果会直接决定 proposal 里的措辞：

- 成立：可以继续写 "top-k internal contribution is skewed"；
- 不成立：只能写 "rank is a coarse but deployable importance proxy"，并保留量化、去掉 aggressive drop。

---

## 4. 实验 B：rank-aware 近似是否更稳

### 4.1 近似算子

Mac 上不依赖真实 FP8 硬件，先用 fake quantization 模拟通信低精度：

| 档位 | 实现方式 | 字节假设 |
|---|---|---|
| BF16/FP16 | 原始 output | 2 B / element |
| INT8 proxy | per-token 或 per-block symmetric quant-dequant | 1 B / element |
| INT4 proxy | per-token 或 per-block symmetric quant-dequant | 0.5 B / element |
| drop | `o_{l,t,r}=0`，可选 gate renormalization | 0 B / element |

注意：这里的 INT8 proxy 不是严格 FP8，只是前置实验的 communication precision proxy。正式实验如果上 GPU，再替换成真实 FP8 / INT4 kernel 或更接近硬件的模拟。

### 4.2 策略组

至少跑这些策略：

| 策略 | 含义 |
|---|---|
| Full BF16 | 原始 baseline |
| Uniform INT8 | 所有 selected expert output 都 INT8 proxy |
| Uniform INT4 | 所有 selected expert output 都 INT4 proxy |
| Rank-k INT8 | 只把最低 gate rank 降到 INT8 |
| Rank-k INT4 | 只把最低 gate rank 降到 INT4 |
| Rank-k drop | 只 drop 最低 gate rank，带 / 不带 gate renormalization 两版 |
| Rank-(k-1,k) INT4 | 如果 top-k >= 4，额外压低最后两个 rank |
| Rank-1 INT4/drop | 反例实验，故意处理最高 rank，证明 rank-aware 的必要性 |

如果 `Rank-k INT4/drop` 明显比 `Rank-1 INT4/drop` 稳，这就是最早期、最有说服力的 rank-aware evidence。

### 4.3 指标

每个策略报告：

1. **local combine error**
   - `relative_mse = ||y_approx - y_full||_2^2 / ||y_full||_2^2`
   - 按 layer 和 rank 聚合。

2. **next-token logit KL**
   - 同一批输入，比较 full logits 和 approx logits；
   - 比 perplexity 更便宜，适合 Mac 上快速扫策略。

3. **perplexity delta**
   - WikiText-2 小切片即可；
   - 第一版只需要相对排序，不追求论文级精度。

4. **byte saving**
   - 按 BF16=2B、INT8=1B、INT4=0.5B、drop=0B 估算；
   - 画 `byte saving` vs `PPL delta / KL` 的 Pareto 图。

---

## 5. 实验 C：layer sensitivity 小表

Idea A 后续要有 `(layer, receiver_group, rank) -> precision` 的 LUT，因此 Mac 前置实验至少要证明 layer 维度有意义。

做法：

1. 每次只近似一个 MoE layer；
2. 固定策略，例如 `Rank-k INT4` 或 `Rank-k drop`；
3. 记录该层独占近似时的 `relative_mse / KL / PPL delta`；
4. 画 layer sensitivity heatmap。

通过标准：

- 如果不同层的损失差异明显，比如最敏感层比最低敏感层高 3x 以上，`WHICH-LAYER` 维度就站得住；
- 如果所有层几乎一样，LUT 可以先退化成 `(rank) -> precision`，不要强行讲 layer sensitivity。

---

## 6. 推荐执行顺序

### Phase 0：1 天内跑通

- 选一个小 MoE；
- batch size = 1，seq len = 128；
- 16 条样本；
- 只做 forward hook 和 contribution share 统计；
- 输出第一张 `rank contribution by layer` 图。

目标：确认 hook 位置正确，`g * ||o||` 能正常采集。

### Phase 1：2 到 3 天

- 样本扩到 256 条，seq len = 256；
- 完成实验 A；
- 输出：
  - `rank_share_by_layer.csv`
  - `rankk_share_cdf.png`
  - `rank_contribution_bar.png`

目标：判断 C1 是否成立。

### Phase 2：3 到 5 天

- 加入 fake INT8 / INT4 / drop；
- 完成实验 B；
- 输出：
  - `approx_results.csv`
  - `accuracy_byte_pareto.png`
  - `rank1_vs_rankk_ablation.png`

目标：判断 rank-aware 近似是否比 uniform 或错误 rank 更稳。

### Phase 3：可选，1 周内

- 完成实验 C；
- 输出：
  - `layer_sensitivity.csv`
  - `layer_sensitivity_heatmap.png`

目标：给后续 LUT 和 accuracy constraint 的 `delta_{l,R,p}` 提供雏形。

---

## 7. 预期结论模板

### 情况 1：最理想

观察到：

- rank-k contribution 明显小；
- rank-k INT4/drop 的 KL/PPL 损失显著低于 rank-1；
- 部分 layer 对近似明显更不敏感；
- rank-aware 策略在 Pareto 图上优于 uniform INT4。

结论：

> Idea A 的核心假设成立，可以继续推进 receiver-aware Rank-LUT。Mac 实验负责证明 top-k 内部重要性差异和 rank-aware 近似的可行性；多机实验再验证 receiver-port 拥塞收益。

### 情况 2：中等结果

观察到：

- rank-k 确实更小，但 drop 损失偏大；
- INT8/INT4 可以接受；
- layer sensitivity 存在。

结论：

> 保留 Rank-LUT，但把 drop 从主策略降级为 aggressive option。主线改成 BF16 / INT8 / INT4 mixed precision，不主打丢弃。

### 情况 3：不理想

观察到：

- top-k 内部贡献接近均匀；
- rank-k 近似和 rank-1 差不多；
- uniform INT4 已经足够好，rank-aware 没明显优势。

结论：

> Idea A 的 gate/rank-aware claim 需要收缩。可以保留 receiver-port congestion + layer sensitivity，但不要声称 top-k contribution long-tail 是主要来源。

---

## 8. 不建议在 Mac 前置实验里做的事

1. **不要先跑 AICB 当主实验**  
   AICB 能给通信 trace 和 receiver 频次，但不能证明 `g * ||o||` 的 top-k 内部长尾。AICB 是后续 receiver_group / freq 输入，不是 C1 的替代品。

2. **不要先做 MILP**  
   没有 `delta_{l,R,p}` 和 contribution 分布，MILP 只是形式正确，实验上没有说服力。

3. **不要先上 MMLU/GSM8K**  
   成本高、噪声大、迭代慢。前置实验先用 KL / PPL 小切片判断方向。

4. **不要把 Mac 结果包装成系统加速**  
   Mac 没有多 GPU all-to-all，也没有 receiver-port 真实拥塞。这里只能证明 approximation feasibility，不能证明 serving TBT。

---

## 9. 最小代码结构建议

```text
experiments/
  idea_a_mac/
    capture_moe.py          # hook / monkey patch MoE block
    fake_quant.py           # INT8 / INT4 quant-dequant proxy
    run_profile.py          # 实验 A：contribution share
    run_approx.py           # 实验 B：近似策略
    run_layer_sensitivity.py# 实验 C：单层敏感度
    plot_results.py
    outputs/
      rank_share_by_layer.csv
      approx_results.csv
      layer_sensitivity.csv
      figures/
```

第一版代码要优先保证可验证性，不要急着抽象成通用框架。每次 forward 后只保留聚合统计，避免把大 tensor 全量存盘。

