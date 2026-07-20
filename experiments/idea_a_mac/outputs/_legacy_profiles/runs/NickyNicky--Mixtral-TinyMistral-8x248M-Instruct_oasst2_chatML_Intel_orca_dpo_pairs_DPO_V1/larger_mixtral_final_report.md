# Larger Mixtral Run Final Report

## 1. 实验配置

- 模型：`NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1`
- 架构：Mixtral
- MoE 层数：12
- hidden size：1024
- intermediate size：4096
- experts：8
- top-k：2
- 权重体积：约 2.5 GB
- 设备：CPU-only
- dtype：float32
- 样本：8 条内置英文短文本
- seq len：96

这比上一轮 `jamesdborin/tiny-mixtral` 更大：层数从 2 增到 12，权重从约 0.94 GB 增到约 2.5 GB。

## 2. Baseline 跑通情况

- 首次下载 / 加载：444.72s
- 单条 smoke forward：0.42s
- logits shape：`(1, 12, 32005)`

后续重复运行会走本地 Hugging Face 缓存。

## 3. C1：top-k 内部 contribution 长尾

统计 `share = g * ||o|| / sum(g * ||o||)`。

总体结果：

- top-k：2
- rank-k median share across layers：`0.000152`
- rank1/rankk median ratio across layers：`10221.473877`
- C1 verdict：**强成立**

典型层结果：

| layer | rank-1 median share | rank-2 median share | rank1/rank2 median |
|---|---:|---:|---:|
| 0 | 0.9699 | 0.0269 | 32.24 |
| 1 | 0.9953 | 0.0044 | 213.95 |
| 3 | 0.9997 | 0.0003 | 3735.55 |
| 6 | 1.0000 | 0.000008 | 106394.48 |
| 11 | 1.0000 | 0.000014 | 52951.09 |

结论：

这个 12-layer top-2 MoE 里，router/gate 极度集中，rank-2 contribution 在多数层几乎可以忽略。它明显支持 “top-k 内部 contribution skew” 这个方向，但仍需注意：它是 top-2 模型，不等价于 DeepSeek-V2-Lite top-6。

## 4. Rank-aware 近似结果

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| full | 0.000 | 0.0000 | 0.000000 | 0.0 |
| uniform_int4 | 0.750 | 35.0038 | 0.207482 | -6833.7 |
| rankk_int4 | 0.375 | 0.5281 | 0.000633 | -181.8 |
| rank1_int4 | 0.375 | 32.7029 | 0.193634 | -7042.0 |
| rankk_drop | 0.500 | 1.6425 | 0.006023 | 122.6 |
| rankk_drop_renorm | 0.500 | 0.0219 | 0.000072 | 342.7 |
| rank1_drop | 0.500 | 148.3756 | 0.946336 | 105497611.6 |

可用结论：

1. `rankk_int4` 明显优于 `rank1_int4`：KL `0.5281` vs `32.7029`，local relative MSE `0.000633` vs `0.193634`。
2. `rankk_drop` 比 `rank1_drop` 稳很多，但裸 drop 的 KL 仍高于 `rankk_int4`。
3. `rankk_drop_renorm` 在这轮结果里非常稳：KL `0.0219`，local relative MSE `0.000072`，但这个结论需要在更多模型/数据上复核，不能只凭一个 top-2 模型就写成通用规律。
4. `uniform_int4` 虽然节省 75% bytes，但 KL 很高，说明 uniform 低精度不是可靠 baseline；rank-aware 的价值很明显。
5. PPL 仍不作为主判断依据，因为样本很少且模型/文本域不匹配；KL 和 local relative MSE 更可信。

## 5. 对 Idea A 的影响

这轮更大模型结果支持：

- C1 在更深的 Mixtral top-2 模型上强成立。
- rank-aware 近似明显比处理 rank-1 稳。
- `rank` 可以作为一个 deployable importance proxy。
- `uniform INT4` 不够稳，rank-aware mixed precision 有必要性。

这轮仍不能支持：

- 不能证明 DeepSeek-V2-Lite top-6 也长尾。
- 不能证明 receiver-port TBT 收益。
- 不能直接把 `rankk_drop_renorm` 写成通用最优策略。

建议 proposal 当前写法：

- 主线：`rank-aware mixed precision combine`。
- drop：从“主策略”改成 “aggressive option / ablation”，但可以保留 `drop + renorm` 作为值得继续测试的候选。
- C1：可以写 “larger top-2 Mixtral smoke shows strong skew; top-6 model validation remains necessary”。

## 6. OLMoE top-8 状态

已完成：

- 读取 OLMoE config：16 layers、hidden size 2048、64 experts、top-k 8。
- 已扩展 hook 支持 `OlmoeSparseMoeBlock`。
- 已验证磁盘空间充足。

未完成：

- OLMoE 权重约 13 GB，首次下载较慢。
- 默认 xet 下载卡住；禁用 xet 后普通 HTTP 可以下载，但预计需要较长时间。
- 本轮已中断 OLMoE 下载，避免把当前任务拖成超长下载任务。

后续命令：

```bash
HF_HUB_DISABLE_XET=1 ./.venv/bin/python experiments/idea_a_mac/load_model_smoke.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --dtype bfloat16

HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python experiments/idea_a_mac/run_profile.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --dtype bfloat16 \
  --num-samples 1 \
  --seq-len 64
```

