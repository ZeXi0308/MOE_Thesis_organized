# 自然长度完整文章输入

状态：`INPUTS_PREPARED_GPU_UNRUN`。已在本地CPU离线生成 [config.json](prepared/natural/config.json) 和 [workload.json](prepared/natural/workload.json)，未运行GPU。

固定Wikitext-103-raw-v1 train shard0、revision `b08601e04326c79dfdd32d625aee71d232d685c3`。采用完整原始标题行 `r' = [^=\n]+ = \n'` 且前后原始行为空的结构规则，从该标题到下一有效标题逐字串接原row；不修改空白，不接受分片末尾未闭合文章。该规则排除了前轮补充说明中的七个表格注释伪标题。

排除前轮16个document IDs后，按源顺序取首16篇全文token数1..3968的文章。使用原pinned OLMoE tokenizer、`add_special_tokens=True, truncation=False, padding=False`，不截断、不增加长度下限、不按性能选样。request_id沿用document_id；50ms到达、固定128输出、context4096、seed20260905。单cohort为`natural`，`prompt_tokens=null`，实际逐请求长度写在config和workload中。

| 源起始行 | 结束行，不含 | 全文tokens |
|---:|---:|---:|
| 271 | 287 | 790 |
| 287 | 316 | 1572 |
| 406 | 433 | 1846 |
| 433 | 464 | 1117 |
| 464 | 480 | 733 |
| 557 | 610 | 2602 |
| 610 | 628 | 1521 |
| 748 | 799 | 3011 |
| 913 | 944 | 2663 |
| 944 | 976 | 1739 |
| 1038 | 1107 | 2546 |
| 1380 | 1401 | 1012 |
| 1582 | 1633 | 2736 |
| 1633 | 1648 | 1379 |
| 1648 | 1690 | 2404 |
| 1690 | 1710 | 1432 |

document ID为`memory-train-article-`加七位补零起始行。16个IDs和长度均与前次只读预期逐项一致，总计29103 tokens，范围733–3011；13条超过1024、3条不超过1024，零条为1024整倍数。此处只描述输入长度，实际prefill分块仍需原生执行记录。

实际执行命令（仓库根目录，已成功退出0；脚本拒绝覆盖已有输出目录）：

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python refine-logs/independent_ideas_20260911/natural_prefill_tails_r01/prepare_inputs.py \
  --dataset-arrow /Users/leandrozhao/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow
```

本地Python3.9.6、datasets4.5.0、transformers4.57.6、tokenizers0.22.2、pyarrow21.0.0。模型及tokenizer revision均为`6d84c48581ece794365f2b8e9cfb043c68ade9c5`。Arrow、模型config、三个tokenizer文件均通过既有hash校验；输出回读后16/16原文与所记录完整row范围一致，原文hash、tokenhash、长度、排除集合及上下文约束均通过。

准备脚本123行。config文件3344字节，SHA256 `5d1d1ab648c51b7c5a80a40fb7eb05d87e0bdc00472f8d7c3dfb5781f91bca81`；workload文件484522字节，SHA256 `08d2a54b1ab48f682f0298a66bf0f9c91e5505f0c314fba06e8ca03f29bb91a8`。config中的`workload_sha256`沿用旧格式，计算对象为排序键序列化后的逻辑JSON，区别于这里记录的文件字节hash。
