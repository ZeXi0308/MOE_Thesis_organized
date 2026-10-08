# 异构请求上限的历史删失：未选定的备选问题

**只有源码事实和CPU合成反例；没有新控制器、原生动作或服务收益。当前同cap1024实验不受此反例解释。** 同5%余量下cap与历史采样的固定四格消融现已完成，结果见[消融报告](C_NATIVE_PAST_FUTURE_CAP_ABLATION_RESULT_20261001.md)，所有原始参数和结果保留。

## 可观察信息与丢失位置

同一版本的作者AE [io_struct](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/io_struct.py#L242-L253)区分正常停止与达到max_output_len；[router manager](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/manager.py#L246-L256)却只把已完成请求的output_ids长度传到[历史deque](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/req_queue.py#L140-L212)。stop/length与旧请求cap没有进入此历史。运行中请求仍按当前进度条件化并附加自己的cap端点，输出tuple也被自己的cap截断；不能说该实现完全没有cap保护。

若旧请求和新请求cap相同，观测min(自然输出长度,cap)本来就是相同cap任务的有界输出需求样本，不能把当前大量length结束直接定为估计错误。只有请求cap不同或发生分布变化时，混用这些长度才产生这里描述的可识别性问题。

## 最小合成状态

[脚本](C_PAST_FUTURE_CENSORING_COUNTEREXAMPLE_V1.py)和[CPU结果](past_future_censoring_counterexample_v1.json)构造两组各40条长度128的历史：一组stop、旧cap1024；另一组length、旧cap128。当前17个resident均P2700/O100/M1024，队头P3000/M1024。两组使用相同固定RNG，得到完全相同的5个峰值51087，低于62259.2而通过。已知cap的共同峰值则为67215，超过65536token容量。

第二组没有观察到128以后的自然EOS，长尾可能仍很长。该例只证明现有API无法利用两类历史的区别；全cap是允许的未来上界，不是已知真实未来。它不是当前GPU轨迹，没有实际容量违例，更不反驳统计策略从未承诺的确定性保证。

## 直接近邻与不该声称的贡献

[TIE §4](https://arxiv.org/html/2604.00499v1#S4)已经将预测输出分布按每个请求max_tokens截断，再计算期望与尾部风险进行排序。未来预测的cap截断不同于训练/在线历史中区分自然终止与右删失观测；其正文的分布拟合公式没有展示这里的PF历史事件处理，但不能因此断言所有实现均未处理。TIE还涉及语义预测、等待老化和异步预测路径，不是本反例的完整同后端基线。

Kaplan–Meier是已有的[删失数据统计方法](https://doi.org/10.1080/01621459.1958.10501452)。单纯增加finish_reason字段或使用KM不足以成立独立系统贡献；当所有记录都在128被截断时，128以后的尾部不能从这些记录恢复。还需明确可识别范围、未知尾部的代价与回退、自然出现的异构cap输入以及完整请求服务增量。窄范围原论文/代码检查未找到完全相同的LLM准入实现，不构成不存在声明。

**决定：保留为一个有依据但未选定的备选，暂不据此启动GPU或改变当前输入。** 当前消融闭环已完成。若继续该问题，先建立自然运行域与可执行动作相对强简单/最近方法的差异，不能只把标准统计修正命名为新算法，也不能靠为它构造必胜cap顺序补救旧候选。

## 公开输入字段的窄范围核查

只检查了三个原始来源。Azure 的[2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md#schema)与[2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md#schema) LLM trace 只公布 `TIMESTAMP`、`ContextTokens`、`GeneratedTokens`，无请求上限或结束原因。[WildChat-1M 的字段说明](https://huggingface.co/datasets/allenai/WildChat-1M#data-fields)有用户/助手文本、时间和 HTTP 请求头；未列逐请求 `max_tokens` 或完成原因。这两类不能判定长度恰好等于上限时是否右删失。

[Exgentic agent-llm-traces-v2 的官方 schema](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2#schema--chat-only-minimal-attribute-set)在每个 chat span 同时列出 `gen_ai.request.max_tokens`、`gen_ai.usage.output_tokens` 和 `gen_ai.response.finish_reasons`，是字段层面的候选；其顶层 `max_tokens` 是会话汇总，不能替代 span 字段。该数据是六种 agent benchmark 的真实 API 执行记录，不是自然到达的生产服务流量。初查时数据查看服务超时，当时尚未逐行验证成功调用三字段共同非空、上限确实多值、以及 `length` 结束的数量，因此不能称其为已验证的自然异构 cap 评估集。当前同 cap1024 实验仍不受此备选影响。

## 首分片阶段记录（已被下节全量统计补足）

固定到官方页面所示提交 `4b8ad4ab198438e5a170f9171c19c6a2cf7c1814`。本地官方站超时，远端官方站不可达；公开镜像的固定版本文件下载可用，首分片2,281,215字节的SHA `f7cd079798f47b7610963c08105055b935316c6223f74d3fc03c401348590011` 与[官方blob页](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2/blob/main/data/train/0000.parquet)一致。镜像目录API返回403，之后依据官方列出的九个文件名直接取固定版本文件；没有跳过TLS校验。

[字段统计脚本](C_EXGENTIC_CAP_FIELDS_V1.py)只投影身份、模型、上限、输出长度、结束原因和状态元数据；不读取提示词/输出内容作为研究输入。成功口径是整数status.code≠2且error.type为空；另单列缺失值。首分片实际150会话、6743个span，3192个成功，2626个同时具有正cap、非负输出长度及非空结束原因。该完整子集有2614个cap8192、12个cap21333；0个字面length结束，全部输出低于cap。DeepSeek-V3.2完整子集2279个全为8192；Kimi-K2.5为335个8192和12个21333。一个会话内同一模型有不同cap，说明混合cap确实在这一基准调用样本中出现；未观察到截断，故不能据此支持删失问题的发生率或服务价值。

首分片结果见[JSON](exgentic_cap_fields_first_shard_v1.json)，使用脚本版本SHA `26434280585f5a11f880b6f533b9d0c66cfccd303a6d0bb4d62787b0acfceb02`。随后脚本仅修正完成标签保留规则，保留各提供方的短标识符、仍不把非length标签自动当截断；首分片结果不覆盖。全量九分片于11:28UTC下载完成，随后已复制到本地；本段仍只描述首分片。全量只统计字段和分布，不把后处理的基准时间戳当生产到达，也不据此改变既有GPU输入或上限。

## 上限与结束原因的构建来源

[官方构建脚本（固定版本）](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2/blob/4b8ad4ab198438e5a170f9171c19c6a2cf7c1814/scripts/build_agent_llm_traces.py)把 `gen_ai.request.max_tokens` 和 `gen_ai.response.finish_reasons` 列入保留字段，并以 `attrs.get(k)` 从上游 span 投影；此步没有为两字段生成默认值，缺失仍为 null。脚本说明上游 `Exgentic/traces-v2` 此前已由 `fix_traces_v2` 规范化。[官方提交记录](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2/commits/main)记载了 provider、时间戳、tool/message 的修复，没有说明两字段的回填；由于上游原始库未能公开读取，仍不能断定这两字段必然直接来自原始 API 响应或请求参数。

## 全九分片结果：异构cap存在，但删失假说缺可用检验样本

[全量字段JSON](exgentic_cap_fields_v1.json)来自固定版本的全部九个Parquet，合计231,705,307字节。当地文件逐个与下载回执SHA相同；首分片另有官方公布SHA核对。实际10,056个会话、241,473个span，没有重复或缺失身份；这些是文件实计数，不沿用数据卡的较旧10,057/241,674汇总。脚本版本SHA `7a78e95008d3ea2a3ca1e83261b8b7aa847dd4cdcb5a49d32df18d90266346f3`，全部统计在本地执行。

| 范围 | 调用数 | 成功调用 | 上限/长度/结束原因共同有效的成功调用 | 完整子集中的length结束 |
|---|---:|---:|---:|---:|
| 全部 | 241473 | 220085 | 20925 | 0 |
| DeepSeek-V3.2 | 67601 | 63719 | 14006 | 0 |
| Kimi-K2.5 | 55142 | 51244 | 6919 | 0 |
| Claude Opus 4.5 | 55789 | 42991 | 0 | 0 |
| Gemini 3 Pro preview | 34466 | 33656 | 0 | 0 |
| GPT-5.2 | 28475 | 28475 | 0 | 0 |

完整字段子集仅占成功调用的9.508%，全部来自claude_code框架中的前两个模型。其cap分布为8192:20548、20000:1、21333:366、32000:10；结束原因是1578个stop和19347个tool_calls。全部输出长度严格低于对应cap，无超cap记录。22个会话的成功完整子集中，同一模型使用不同cap，因而混合cap的存在获得了这个公开派生数据集层面的支持，仍不是生产流量或原始API字段的独立验证。

全数据的字面length结束共5个，均为Gemini/claude_code/BrowseCompPlus的成功调用，输出8189、请求cap缺失。不能从8189猜测请求上限，更不能把缺失cap补成8192后计算删失比例。保留各提供方的实际短结束标签（含tool_use、malformed_function_call等），未将它们自动当EOS或截断。缺失覆盖高度依赖模型/框架，完整子集的零截断不能外推为全部调用没有截断。

**本轮决定：这个输入源不能支持异构cap删失控制器的原生pilot，备选维持未选定。** 它证明有混合cap元数据，但没有同时保留cap的长度截断样本；所以目前既不能量化该估计偏差，也不能验证其服务代价。该结论仅约束当前数据源/假说检验，不是否定删失问题整体。重新进入实验需要自然任务中同时保留请求cap、终止原因和输出长度的截断样本，以及超出简单按cap分组/保守回退的可执行动作差异；不靠补cap、造cap顺序或更换seed制造正例。当前GPU统一cap1024的所有结果和判断保持原样。
