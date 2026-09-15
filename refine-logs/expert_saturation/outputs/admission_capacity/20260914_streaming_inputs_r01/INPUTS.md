# 本地完整自然文章输入

`candidate/` 已完成 CPU 生成；`frozen/` 是主会话随后一次选定的配置，见 [FREEZE.md](FREEZE.md)。无 GPU、模型权重下载或生成结果。

| 输入事实 | 当前结果 |
|---|---|
| 已有完整 token IDs | natural_prefill_tails_r01 的16篇，733–3011 tokens；与旧160文档identity/hash/row均不重叠，原任务记录GPU_UNRUN |
| 本次候选 | 固定source order选择64篇，排除旧160；复用上述16，其余在本地CPU tokenization |
| 固定筛选 | 完整文章256..3072 tokens；不截断、不重复document、不按生成或性能筛选 |
| 实际64篇长度 | min334、max3011、mean1659.90625、median1542.5，总106234 |
| 本次source interval | row[1,8173)，128篇完整文章；64旧文档被排除，其余64均满足固定长度域 |
| 该interval自然长度 | 所有128篇min334、max17756、mean4132.9453、median3053.5；这是局部语料统计 |
| 连续32请求上界 | 33个窗口的Σceil((P+1024)/16)范围5052–5873块 |
| 任意32请求上界 | 最大32项之和6551块；该值覆盖非连续resident集合，仍不是真实EOS占用 |

旧三个 preparer 的边界：`prepare_rotation_holdout.py` 与 `prepare_rotation_fresh_inputs.py` 会对原文完整tokenize，但只保存3072前缀，并固定32请求、1024输出、50ms arrival；它们的identity/hash检查可参考，输出不能冒称全文 token IDs。`prepare_heterogeneous_length.py` 生成1024切片并人为分配输出长度，不能用于这里的完整自然prompt/未知EOS输入。

数据来自本地 WikiText-103-raw-v1 train shard0，revision `b08601e04326c79dfdd32d625aee71d232d685c3`，Arrow SHA `0c22278eac8186f5dd6060fcf84e4bcb4aa2266c51c77a279169764814f9a99e`。模型与Tokenizer均为 `allenai/OLMoE-1B-7B-0924`、revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`。本地三个tokenizer文件与模型config逐一校验已有SHA；只调用tokenizer，不加载模型权重。

完整文章按原始标题行、空白相邻行和下一有效标题边界重建，保留原文所有字符。记录包含源row区间、document hash、完整token hash及实际长度。候选`arrival_traces_s=null`，没有提前知道输出长度；冻结版才加入明确的0.5s arrival及1024最大输出限制。

复跑候选命令（拒绝覆盖已有输出，使用新的输出目录）：

```sh
cd /private/tmp/moe-window-model-20260914
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  '/Users/leandrozhao/Desktop/、++++++++/.venv/bin/python' \
  refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_inputs_r01/prepare_candidate.py \
  --source-root '/Users/leandrozhao/Desktop/、++++++++' \
  --dataset-arrow '/Users/leandrozhao/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow' \
  --output-dir /private/tmp/moe-streaming-inputs-recheck/candidate
```

原6656 usable块预算足以容纳任意32个请求的全部声明上限（余105块），因此不能用这个输入在该预算下测试KV压力恢复增量；此处结论依赖无共享前缀、单组完整attention、无额外保留GPU块且最多32个resident的当前后端。主会话已一次固定4096 usable块档；没有根据GPU效果选择样本或压力。

目前`native_capture.py`仍强制`min_tokens=max_tokens`、`ignore_eos=True`并断言length结束。输入文件不会自动更改这些行为；主runner须按冻结合同接受EOS/实际变长完成。执行、重复、计时及host预算由主protocol负责。
