# 自然长文本输入：GPU 未运行

准备了 32 篇 WikiText-103 train 原始文章的成对输入：短 prompt 128 tokens，长 prompt 3072 tokens，两臂均固定输出 1024 tokens。请求身份、文章顺序和到达时间完全相同；steady 到达为 `i × 0.05 s`，最后一次到达在 1.55 s。两套配置只保留参考 SLO `TTFT=5 s / mean TPOT=0.2 s`；这些输入不定义方法收益或新的主 SLO。

Pinned OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5` 的本地 `config.json` 确认 `max_position_embeddings=4096`。长臂 `3072+1024=4096`，短臂 `128+1024=1152`，均合法。

3072/1024 是 GPU 执行前的域设计：使先接纳的请求在后续长 prefill 接纳期间继续生成，增加观察持续 active 请求和 KV 压力的机会。原候选 3968/128 可能使早期请求在后续请求接纳前完成；没有生成或运行这个版本。该选择不保证真实 HBM 压力存在，仍须 GPU 观测。

来源为同 dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3` 的本地 `wikitext-train-00000-of-00002.arrow`，SHA256 `0c22278eac8186f5dd6060fcf84e4bcb4aa2266c51c77a279169764814f9a99e`。程序按一级标题重建完整文章：仅逐字串接该文章已有 row 字符串，不添加分隔符，不跨文章，不重复或补齐文本。随后选择源顺序中首 32 篇至少 3072 tokens 的完整文章；第一个文章标题在 row 1，最后一个在 row 3640、结束于 row 3717（不含）。原文长度范围为 3154–14472 tokens。这是运行域资格化输入，不声明 fresh holdout。

产物：

- `prepared/short/{config,workload}.json`
- `prepared/long/{config,workload}.json`
- `prepared/inputs_report.json`：文章边界、标题、来源和输入检查
- `prepare_inputs.py`：119 行，可离线重跑；不加载模型权重

已实际检查：32 个唯一文章身份与唯一短前缀；短 token IDs 严格等于长 token IDs 的前 128；两臂同请求顺序及到达；逐篇对照 Arrow 重建原文且不存在内部一级标题；prompt/output 总长度合法；pinned tokenizer 文件及 workload 摘要匹配。检查通过只说明输入准备正确，GPU 容量、调度和性能均未测。

重跑时指定一个不存在的新输出目录，程序拒绝覆盖：

```bash
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_memory_pressure_r01/inputs_preparation/prepare_inputs.py \
  --dataset-arrow /Users/leandrozhao/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow \
  --long-prompt-tokens 3072 --output-tokens 1024 \
  --output-dir /private/tmp/moe-memory-inputs-reproduction
```
