# Energy-SLO / LLM Serving：公开负载数据集清单

> 2026-07-20。用途：支撑「under SLO 选 (batch, precision)」控制器评估；解决「没有工业内部 trace」问题。

## 首选（直接用）

### 1. Azure LLM Inference Trace 2024（最推荐）

| 项 | 内容 |
|---|---|
| 来源 | [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) |
| 文档 | [AzureLLMInferenceDataset2024.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) |
| 下载 | [code 一周](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv)、[conversation 一周](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv) |
| 字段 | 请求时间、input/output tokens（隐私剥离后的负载形态） |
| 论文对口 | **DynamoLLM (HPCA'25)**：性能 + **能耗** 集群设计——与 Energy-SLO 叙事同族 |
| 许可 | CC-BY |
| 用法 | 按 RPS 缩放到达强度 → 驱动你们的 `(batch, fp8/bf16)` 代价表 / 真 serving |

另有 2023 版（Splitwise / ISCA'24）：同仓库 `AzureLLMInferenceDataset2023`。

### 2. BurstGPT

| 项 | 内容 |
|---|---|
| 来源 | [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT) |
| 下载 | [Releases v2.0](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)（`BurstGPT_without_fails_*.csv`） |
| 字段 | `Timestamp`、`Request tokens`、`Response tokens`、`Model`、`Log Type`（API/Conversation）、部分含 Session |
| 规模 | 百万级请求，跨多月，含 burst |
| 论文 | [arXiv:2401.17644](https://arxiv.org/abs/2401.17644) |
| 用法 | 取若干天窗口 + 缩放 RPS；对比平稳 vs burst 下控制器行为 |

### 3. Mooncake Trace（长上下文 / 缓存场景）

| 项 | 内容 |
|---|---|
| 来源 | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) → `FAST25-release` |
| 字段 | `timestamp`、`input_length`、`output_length`、`hash_ids`（匿名块哈希） |
| 特点 | 长 input、可模拟 prefix cache；偏 KV/disagg，也可当到达+长度分布 |
| 用法 | 作「长 prompt」压力场景，与 Azure/BurstGPT 短对话互补 |

## 辅助（长度分布 / bench 文本，不是到达过程）

| 数据集 | 链接 | 用途 |
|---|---|---|
| ShareGPT（多镜像） | 如 HF `anon8231489123/ShareGPT_Vicuna_unfiltered`；vLLM/SGLang `--dataset sharegpt` | 真实 prompt 文本与长度；**没有**生产到达时间戳 |
| LMSYS-Chat-1M | [HuggingFace LMSYS](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 大规模对话长度/多模型；需遵守许可 |
| WildChat | HF `allenai/WildChat` | 更野、更长尾；许可更严，投稿前读 license |

**组合打法（推荐）：**  
到达过程用 **Azure 2024 或 BurstGPT**；若要在 GPU 上真跑 MoE，用 ShareGPT/WikiText **按 trace 的 (in,out) 长度截断/采样** 填真实 token。

## 不建议当作 Energy-SLO 主负载

- 纯随机 `random-input-len`（无真实到达相关）  
- 仅 WikiText 文档流（无 serving 并发形态）  
- Azure LMM 2025（多模态，和你们 OLMoE/LLM-jp 文本 MoE 不完全同域）

## 最小落地步骤

1. 下载 Azure `conv` + `code` 各一周 CSV（约百 MB 级）。  
2. 解析为统一 schema：`arrival_s, prompt_tokens, output_tokens, workload_tag`。  
3. 选 1 小时高峰 + 1 小时低谷；RPS 缩放到单卡/8 卡能打满的区间。  
4. 控制器：查表或在线测 `(batch, precision) → TPOT, J/token`；SLO = P99 TPOT 上限。  
5. 对照：固定 batch+bf16、固定 batch+fp8、仅调 batch、Oracle（事后最优）。

## 引用时注意

- Azure / BurstGPT **不含原始文本**，只有 token 计数与时间——正好适合能耗/调度论文。  
- 在 A100 上跑 OLMoE 时，用公开对话语料 **匹配长度**，不要声称「重放了 Azure 原文」。  
- DynamoLLM 已占「Azure trace + 能耗」叙事；你们的差异应写清：**MoE expert 真 FP8 计算质量门 + batch×precision 交互 + 单机多卡 MoE serving**，避免写成通用 DynamoLLM 复现。
