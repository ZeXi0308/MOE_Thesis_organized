# C：原生并发上限的默认值与下一步边界（2026-10-01）

**结论。** 当前 `max_num_seqs=32` 是实验手动约束，沿袭旧 G64 长文档协议；它不是此运行时的 vLLM 默认值。若 16 条 Instruct GSM8K 自然 EOS 资格结果支持继续，下一项仅改变并发上限的**同入口**原生对照应明确冻结 `max_num_seqs=128`，保留显式 `max_num_batched_tokens=1024` 和其他资源配置，并记录运行时解析后的值。此文不冻结或启动新的 GPU 格。

## 证据与精确含义

| 范围 | 可核实事实 |
| --- | --- |
| 既有实验约束 | [`20260930_g64_fresh_gate_dev_v1/pkg/inputs/config.json`](20260930_g64_fresh_gate_dev_v1/pkg/inputs/config.json) 第 99、105–107、137 行记录原始长文档提示 334–3011 tokens、`cap=32`、`engine_max_num_seqs=32`、batch 1024，以及固定 4096 个可用 KV 块的容量点；其第 117–126 行指向 2026-09-14 的 streaming candidate。现行 [`C_INSTRUCT_GSM8K_CELL_V3.py`](C_INSTRUCT_GSM8K_CELL_V3.py) 第 214、222 行显式传入 `max_num_seqs=32`，再调用 `LLMEngine.from_engine_args(EngineArgs(**kwargs))`。这些记录证明 32 是继承的显式实验设置；最初选择数值 32 的动机尚无足够证据。 |
| 同入口默认 | vLLM v0.26.0 [`EngineArgs`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py#L526-L534) 的 `max_num_seqs` 初值为 `None`；[`LLMEngine.from_engine_args`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py#L160-L171) 未另传参数时使用 `ENGINE_CONTEXT`。[`get_batch_defaults` 与解析函数](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py#L2430-L2512) 的字典没有该 context，故在[第 2643–2652、2706–2712 行](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py#L2643-L2712)回退到 [`SchedulerConfig.DEFAULT_MAX_NUM_SEQS=128`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/scheduler.py#L41-L64)，再与显式 batch 1024 取 `min`，结果仍为 **128**。默认 performance mode 为 [`balanced`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/vllm.py#L387-L393)，不触发 throughput 翻倍。 |
| 公共离线入口 | [`vllm.LLM()`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/llm.py#L336-L339) 明确传入 `LLM_CLASS`。RTX 5090 的 32GB 显存落入 [`<70 GiB` 分支](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py#L2452-L2482)，该入口默认 `max_num_seqs=256`、默认 batch 8192；若只显式保留本研究 batch 1024，序列解析值仍为 `min(256,1024)=256`。**256 是另一入口的默认，不应称作当前直接 EngineArgs 入口的默认。** |
| 当前容量边界 | [`instruct_gsm8k_corpus_geometry_v1.json`](instruct_gsm8k_corpus_geometry_v1.json) 第 3–10、1402 行对 1319 条官方 test 源序列的已知提示长度和输出上限作 CPU 结构上界：任意至多 32 条请求各自按 1024 输出 cap 独立计块，至多 3586 个块，小于 4096 个可用块。这不包含实际模型输出、到达时序和运行时调度。因此现行 32 上限下没有 KV 容量压力；提高到 128 后会否产生压力、抢占或完整请求收益，均不能预先断言。 |

**安装源码核验。** 在授权备用机上只读取得 venv `.../site-packages/vllm/` 五个文件的 Git blob hash，均与上述官方 `v0.26.0` tag 完全一致：`engine/arg_utils.py` `72dfc28688d6d1170e2061f0ca69afc2bdb6f041`；`config/scheduler.py` `3773a47ec94f51bc557fd8bd8bd7b97e26b01854`；`v1/engine/llm_engine.py` `ff86a1dffd941f8c99778a5cfd5522396245f4b2`；`entrypoints/llm.py` `b3205728e4963dd21b306a15e8a0b4dee14eb8a0`；`config/vllm.py` `4b4a97f41b6b3e67fb46b1407b501ec90072b0f9`。未导入 vLLM、初始化 GPU 或运行新实验。

**决策条件。** 先读取既定 16 条自然 EOS/输出资格结果。只有该门槛通过且确需同入口原生比较时，才将 128 作为唯一预先指定的新并发值；同时核验 `engine.vllm_config.scheduler_config.max_num_seqs == 128`，并保存完整输入、模型修订、资源及 EOS 证据。该比较不自动构成策略增益或容量压力的证据。
