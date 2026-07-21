# Idea A 前置实验 Task List

> 来源：`IdeaA_前置实验设计_MacM5Pro.md`  
> 目标：在 Mac M5 Pro / 48GB 上完成 Idea A 的最小可验证闭环，判断 rank-aware approximate combine 是否值得继续推进。

## 当前执行状态

已完成一版 **tiny Mixtral smoke / 最小闭环**：

- 已创建 `experiments/idea_a_mac/` 实验目录和脚本。
- 已安装本地 `.venv` 依赖。
- 已下载并跑通 `jamesdborin/tiny-mixtral`。
- 已完成 contribution capture。
- 已完成 `full / uniform_int4 / rankk_int4 / rank1_int4 / rankk_drop / rankk_drop_renorm / rank1_drop` 对比。
- 已输出最终报告：`experiments/idea_a_mac/outputs/idea_a_mac_final_report.md`。

当前结论：tiny top-2 模型上没有观察到明显 top-k 内部长尾，但 `rankk_int4` 明显比 `rank1_int4` 稳；drop 暂时不适合作为主策略。

补充执行状态：

- 已额外跑完更大的 `NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1`。
- 该模型为 12-layer、8 experts、top-2、约 2.5GB 权重。
- 更大模型结果显示 C1 强成立，`rankk_int4` 明显优于 `rank1_int4`，`rankk_drop_renorm` 在本轮也很稳。
- 报告路径：`experiments/idea_a_mac/outputs/runs/NickyNicky--Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1/larger_mixtral_final_report.md`。
- 已扩展代码支持 OLMoE top-8 hook；OLMoE 权重约 13GB，下载较慢，本轮未完整跑完。

最新补充：

- 已完整下载并跑通 `allenai/OLMoE-1B-7B-0924`。
- 该模型为 16-layer、64 experts、top-8，是目前最贴近 Idea A C1 假设的本地前置实验。
- OLMoE top-8 结果：rank-8 median share across layers `0.055791`，rank1/rank8 median ratio `4.304844`，C1 强成立。
- `rank8_int4` 明显优于 `rank1_int4`：KL `0.0737` vs `6.7794`。
- `rank8_drop` 可作为 aggressive ablation，但弱于 `rank8_int4`；`drop + renorm` 在 OLMoE 上失败，不应写成默认修正。
- 报告路径：`experiments/idea_a_mac/outputs/runs/allenai--OLMoE-1B-7B-0924/olmoe_top8_final_report.md`。

更强补充实验：

- 已在 `allenai/OLMoE-1B-7B-0924` 上完成 WikiText-2 真数据 rank sweep。
- Profile：WikiText-2 validation 128 条，seq_len 128，rank-8 median share `0.049277`，rank1/rank8 median ratio `5.451188`，C1 强成立。
- Approx sweep：WikiText-2 validation 32 条，seq_len 128，完整扫描 `rank1_int4` 到 `rank8_int4`。
- 同样只压一个 rank、同样 byte saving `0.09375` 时，`rank8_int4` 比 `rank1_int4` 明显更稳：KL `0.3614` vs `20.9892`，local relative MSE `0.001274` vs `0.029478`。
- `rank8_int4` 相比 `rank1_int4`：KL 低约 `58.1x`，local relative MSE 低约 `23.1x`。
- 这轮结果是目前最适合写进 proposal 的前置实验主证据：主线应收敛到 rank-aware mixed precision combine，drop 只保留为 aggressive ablation。
- 报告路径：`experiments/idea_a_mac/outputs/runs/allenai--OLMoE-1B-7B-0924-wikitext-rank-sweep/olmoe_wikitext_rank_sweep_report.md`。

主实验包补充：

- 已新增 receiver-group profiling 与 serving simulation 脚本：
  - `experiments/idea_a_mac/run_receiver_profile.py`
  - `experiments/idea_a_mac/run_serving_sim.py`
- 已完成 OLMoE 主实验：WikiText-2 256 条 profile、4 receiver groups、32 条 approximation sweep。
- 已完成第二模型交叉验证：Mixtral-TinyMistral 128 条 profile、64 条 approximation sweep。
- OLMoE 结果：rank8 median share `0.049137`，rank1/rank8 ratio `5.434607`；`rank8_int4` KL `0.3614` vs `rank1_int4` KL `20.9892`。
- Mixtral-TinyMistral 结果：rank2 median share `0.000135`，rank1/rank2 ratio `14656.703125`；`rank2_int4` KL `2.8718` vs `rank1_int4` KL `118.7166`。
- Receiver group 结果：OLMoE group spread 中等，Mixtral group spread 更明显；建议把 receiver_group 写成部署/拥塞控制维度，而不是夸成唯一主因。
- Serving simulation 已输出 total bytes、bottleneck bytes、simulated latency saving 和 accuracy tradeoff。
- 主实验总报告：`experiments/idea_a_mac/outputs/main_experiments/main_experiment_report.md`。

## 总体完成标准

- [x] 跑通一个小 MoE 模型的 baseline forward。
- [x] 能采集每层、每个 top-k rank 的 `g * ||o||` contribution share。
- [x] 输出 top-k 内部 contribution 分布图，判断 C1 是否成立。
- [x] 跑完 rank-aware 近似策略，对比 uniform / rank-1 / rank-k。
- [x] 输出 accuracy-byte Pareto 图。
- [x] 给出 go/no-go 结论：保留 drop、只保留 quantization，还是收缩 Idea A claim。

---

## Phase 0：环境与模型跑通

### 0.1 创建实验目录

- [ ] 新建目录 `experiments/idea_a_mac/`。
- [ ] 新建输出目录 `experiments/idea_a_mac/outputs/`。
- [ ] 新建图表目录 `experiments/idea_a_mac/outputs/figures/`。
- [ ] 新建 `README.md`，记录实验目标、模型、数据集和运行命令。

验收产物：

- [ ] `experiments/idea_a_mac/README.md`
- [ ] `experiments/idea_a_mac/outputs/`

### 0.2 准备 Python 环境

- [ ] 确认 Python 版本。
- [ ] 安装 PyTorch。
- [ ] 安装 `transformers`、`datasets`、`accelerate`。
- [ ] 安装画图和统计依赖：`pandas`、`numpy`、`matplotlib`、`seaborn`。
- [ ] 写一个 `check_env.py`，打印 torch 版本、MPS 是否可用、当前设备信息。

验收产物：

- [ ] `experiments/idea_a_mac/check_env.py`
- [ ] 能看到 torch / transformers / MPS 状态。

### 0.3 选择并加载小 MoE 模型

- [ ] 优先尝试 OLMoE 小模型。
- [ ] 如果 OLMoE 加载不顺，再尝试 Qwen1.5-MoE-A2.7B。
- [ ] 写 `load_model_smoke.py`。
- [ ] 使用 batch size = 1、seq len = 128 跑一条文本 forward。
- [ ] 记录峰值内存、forward 时间、是否能稳定完成。

验收产物：

- [ ] `experiments/idea_a_mac/load_model_smoke.py`
- [ ] `outputs/model_smoke_result.md`

Go/No-Go：

- [ ] 如果 seq len = 128 都不能稳定 forward，先换更小模型。
- [ ] 如果 forward 可以跑，但很慢，后续样本数先压到 16 到 64。

---

## Phase 1：MoE Hook 与 Contribution 采集

### 1.1 定位 MoE block

- [ ] 打印模型结构。
- [ ] 找到 MoE layer / sparse MLP / router / experts 的模块名。
- [ ] 找到 router logits 输出位置。
- [ ] 找到 top-k expert id 和 gate weight 的计算位置。
- [ ] 找到 expert output 进入 combine 的位置。

验收产物：

- [ ] `outputs/model_structure.txt`
- [ ] `outputs/moe_hook_notes.md`

### 1.2 实现 capture 逻辑

- [ ] 新建 `capture_moe.py`。
- [ ] 采集 `layer_id`。
- [ ] 采集 top-k expert id。
- [ ] 采集 top-k gate weight。
- [ ] 采集每个 selected expert 的 output norm。
- [ ] 计算 `contribution = gate * output_norm`。
- [ ] 计算 `share = contribution / sum(contribution over top-k)`。
- [ ] 只保存聚合统计，不保存完整 expert output tensor。

验收产物：

- [ ] `experiments/idea_a_mac/capture_moe.py`
- [ ] 单条样本能打印每层每个 rank 的 contribution share。

### 1.3 Smoke Profile

- [ ] 新建 `run_profile.py`。
- [ ] 使用 16 条样本。
- [ ] seq len = 128。
- [ ] batch size = 1。
- [ ] 输出每层、每个 rank 的 mean / median / P75 / P90。
- [ ] 输出 `rank1_share / rankk_share` 的 median。

验收产物：

- [ ] `outputs/rank_share_smoke.csv`
- [ ] `outputs/profile_smoke_summary.md`

Go/No-Go：

- [ ] 如果 rank share 全是空值，说明 hook 点错了。
- [ ] 如果 top-k rank 无法区分，先修 capture，不进入下一阶段。

---

## Phase 2：C1 top-k 内部长尾验证

### 2.1 扩大 profiling 样本

- [ ] 使用 WikiText-2 validation。
- [ ] 样本数先设 256。
- [ ] seq len = 256。
- [ ] 跑完整 profiling。
- [ ] 保存每层每 rank 的统计结果。

验收产物：

- [ ] `outputs/rank_share_by_layer.csv`
- [ ] `outputs/rank_ratio_by_layer.csv`

### 2.2 画 contribution 图

- [ ] 新建 `plot_results.py`。
- [ ] 画 rank contribution bar plot。
- [ ] 画 rank-k contribution CDF。
- [ ] 画 tail mass by layer。
- [ ] 保存图片到 `outputs/figures/`。

验收产物：

- [ ] `outputs/figures/rank_contribution_bar.png`
- [ ] `outputs/figures/rankk_share_cdf.png`
- [ ] `outputs/figures/tail_mass_by_layer.png`

### 2.3 写 C1 判断报告

- [ ] 统计多数层的 `rank-k median share`。
- [ ] 统计多数层的 `rank1/rankk median ratio`。
- [ ] 判断 C1 是强成立、弱成立还是不成立。
- [ ] 写出对 proposal wording 的影响。

验收产物：

- [ ] `outputs/c1_long_tail_report.md`

判断标准：

- [ ] 强成立：多数层 `rank-k median share < 10%`，且 `rank1/rankk median > 3`。
- [ ] 弱成立：多数层 `rank-k median share` 在 `10%` 到 `20%`。
- [ ] 不成立：top-k 内部接近均匀，`rank-k` 经常超过 `25%`。

---

## Phase 3：Fake Quantization 与 Drop 算子

### 3.1 实现 fake quantization

- [ ] 新建 `fake_quant.py`。
- [ ] 实现 symmetric INT8 quant-dequant。
- [ ] 实现 symmetric INT4 quant-dequant。
- [ ] 支持 per-token scale。
- [ ] 支持 per-block scale，block size 作为参数。
- [ ] 写简单单元测试：shape 不变、无 NaN、误差可计算。

验收产物：

- [ ] `experiments/idea_a_mac/fake_quant.py`
- [ ] `outputs/fake_quant_test.md`

### 3.2 实现 drop 策略

- [ ] 支持指定 rank drop。
- [ ] 支持不做 gate renormalization。
- [ ] 支持 gate renormalization。
- [ ] 确认 drop 后 combine 输出 shape 不变。

验收产物：

- [ ] `capture_moe.py` 或 `run_approx.py` 中支持 drop policy。
- [ ] `outputs/drop_policy_smoke.md`

---

## Phase 4：Rank-Aware 近似实验

### 4.1 实现策略配置

- [ ] 新建 `run_approx.py`。
- [ ] 支持 Full BF16 baseline。
- [ ] 支持 Uniform INT8。
- [ ] 支持 Uniform INT4。
- [ ] 支持 Rank-k INT8。
- [ ] 支持 Rank-k INT4。
- [ ] 支持 Rank-k drop，不带 renormalization。
- [ ] 支持 Rank-k drop，带 renormalization。
- [ ] 支持 Rank-1 INT4。
- [ ] 支持 Rank-1 drop。
- [ ] 如果 top-k >= 4，支持 Rank-(k-1,k) INT4。

验收产物：

- [ ] `experiments/idea_a_mac/run_approx.py`
- [ ] `outputs/strategy_smoke_result.csv`

### 4.2 实现指标

- [ ] 计算 local combine relative MSE。
- [ ] 计算 next-token logit KL。
- [ ] 计算 WikiText-2 perplexity。
- [ ] 计算 byte saving。
- [ ] 每个策略输出一行结果。

验收产物：

- [ ] `outputs/approx_results.csv`

### 4.3 跑正式近似实验

- [ ] 使用与 Phase 2 相同的数据切片。
- [ ] 先跑 16 条样本 smoke。
- [ ] 再跑 256 条样本正式结果。
- [ ] 对每个策略记录运行时间。
- [ ] 保存失败策略和报错信息。

验收产物：

- [ ] `outputs/approx_results_smoke.csv`
- [ ] `outputs/approx_results.csv`
- [ ] `outputs/approx_run_log.md`

---

## Phase 5：Pareto 与 Rank-Aware 结论

### 5.1 画 Pareto 图

- [ ] 横轴使用 byte saving。
- [ ] 纵轴使用 KL。
- [ ] 另画一张纵轴使用 PPL delta。
- [ ] 标出 Full BF16、Uniform INT4、Rank-k INT4、Rank-k drop、Rank-1 INT4/drop。

验收产物：

- [ ] `outputs/figures/accuracy_byte_pareto_kl.png`
- [ ] `outputs/figures/accuracy_byte_pareto_ppl.png`

### 5.2 画 rank ablation 图

- [ ] 对比 Rank-1 INT4 vs Rank-k INT4。
- [ ] 对比 Rank-1 drop vs Rank-k drop。
- [ ] 如果 top-k >= 4，对比 Rank-k vs Rank-(k-1,k)。

验收产物：

- [ ] `outputs/figures/rank1_vs_rankk_ablation.png`

### 5.3 写 rank-aware 判断报告

- [ ] 判断 Rank-k 是否明显优于 Rank-1。
- [ ] 判断 Rank-k 是否明显优于 Uniform。
- [ ] 判断 drop 是否还能保留为主策略。
- [ ] 判断 INT4 / INT8 哪个更适合作为主线。

验收产物：

- [ ] `outputs/rank_aware_report.md`

---

## Phase 6：Layer Sensitivity 小表

### 6.1 单层近似实验

- [ ] 新建 `run_layer_sensitivity.py`。
- [ ] 固定策略为 Rank-k INT4。
- [ ] 每次只对一个 MoE layer 启用近似。
- [ ] 记录该层独占近似时的 KL / PPL delta / local MSE。
- [ ] 如果 Rank-k drop 在 Phase 4 中表现可接受，再加一组 Rank-k drop。

验收产物：

- [ ] `experiments/idea_a_mac/run_layer_sensitivity.py`
- [ ] `outputs/layer_sensitivity.csv`

### 6.2 画 heatmap

- [ ] 画 layer vs metric 的 heatmap。
- [ ] 标出最敏感 20% layer。
- [ ] 标出最低敏感 20% layer。
- [ ] 计算最高敏感层 / 最低敏感层的损失倍数。

验收产物：

- [ ] `outputs/figures/layer_sensitivity_heatmap.png`
- [ ] `outputs/layer_sensitivity_report.md`

判断标准：

- [ ] 如果最高敏感层损失至少是最低敏感层的 3x，layer sensitivity 维度成立。
- [ ] 如果层间差异很小，后续 LUT 先退化成 `(rank) -> precision`。

---

## Phase 7：最终 Go/No-Go 总结

### 7.1 汇总所有结果

- [ ] 汇总 C1 长尾结果。
- [ ] 汇总 rank-aware 近似结果。
- [ ] 汇总 Pareto 图。
- [ ] 汇总 layer sensitivity。
- [ ] 写出最推荐的下一步方向。

验收产物：

- [ ] `outputs/idea_a_mac_final_report.md`

### 7.2 三种结论分支

- [ ] **理想分支**：C1 成立，Rank-k 明显优于 Rank-1，Rank-LUT 继续推进。
- [ ] **中等分支**：C1 弱成立，drop 损失大，主线改成 BF16 / INT8 / INT4 mixed precision。
- [ ] **收缩分支**：C1 不成立，rank-aware 优势不明显，收缩 gate/rank-aware claim。

最终必须回答：

- [ ] `g * ||o||` 是否在 top-k 内部长尾？
- [ ] lowest-rank expert output 是否真的更适合被近似？
- [ ] drop 能不能留在主策略里？
- [ ] rank-aware 是否比 uniform low precision 更有意义？
- [ ] layer sensitivity 是否值得放进 LUT？
- [ ] Mac 实验能支撑 proposal 里的哪些 claim，不能支撑哪些 claim？

---

## 任务依赖关系

```text
Phase 0 环境与模型跑通
  -> Phase 1 Hook 与 contribution 采集
  -> Phase 2 C1 长尾验证
  -> Phase 3 Fake quant / drop 算子
  -> Phase 4 Rank-aware 近似实验
  -> Phase 5 Pareto 与 rank-aware 结论
  -> Phase 6 Layer sensitivity 小表
  -> Phase 7 Go/No-Go 总结
```

Phase 6 可以在 Phase 5 之后做，也可以在 Phase 4 的策略稳定后并行做。

---

## 最小版本任务

如果时间只够做最小闭环，保留这些任务即可：

- [ ] Phase 0.3：加载一个小 MoE 并跑通 forward。
- [ ] Phase 1.2：实现 contribution capture。
- [ ] Phase 2.1：输出 `rank_share_by_layer.csv`。
- [ ] Phase 2.2：输出 `rankk_share_cdf.png`。
- [ ] Phase 3.1：实现 INT4 fake quant。
- [ ] Phase 4.1：跑 Full BF16 / Uniform INT4 / Rank-k INT4 / Rank-1 INT4。
- [ ] Phase 5.1：画 KL vs byte saving Pareto。
- [ ] Phase 7.1：写 `idea_a_mac_final_report.md`。

这个最小版本已经足够判断 Idea A 能不能继续写下去。
