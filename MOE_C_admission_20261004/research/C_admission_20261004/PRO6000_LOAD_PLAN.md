# PRO6000：64 GiB KV 下的既有文章输入与压力点

状态：CPU 输入检查完成，本文压力点尚未运行。未下载、未重新分词、未修改 runner 或 controller。旧小 KV 结果只作机制调试证据，不能作为正常显存预算下的主结果。

## 资源与容量边界

固定同一 BF16 OLMoE base 模型及 tokenizer revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，`max_model_len=4096`，不做 RoPE 扩展。保持自然 EOS，输出上限 1024；输入最大 3066，因此每条请求合法。固定 `engine_max_num_seqs=256`、64 GiB **可用** GPU KV，以及所有臂相同的 host KV 预算、token budget、victim、恢复执行和服务量子。实际启动后的 GPU 总占用与 KV pool 数必须记录；本文没有验证模型、图与 64 GiB KV 在设备上的完整占用。

本模型每 KV token 为 131072 B（128 KiB），block size 16，因此每块 2097152 B（2 MiB）。本计划统一为 64 GiB **可用** KV：32768 个 usable pages，另加一个 null page，总计 32769 pages、`kv_cache_memory_bytes=68721573888`。启动后必须验证这一实际口径，不能将含 null 的 64 GiB 总池混作 64 GiB 可用。

`128 × 4096 × 131072 B = 64 GiB` 恰好等于本计划可用量。在已限定的无 speculative/lookahead、单 full-attention group 路径中，最多 128 条合法上下文全部满长也可容纳，不产生该口径的 KV 容量抢占。当前 C 的实际 128 条全文 prompt 加 1024 输出上限，经逐请求向上取整更只需 **23434 块 = 45.76953125 GiB**。继续压缩 KV 来创造恢复不能回答正常预算下的问题。

256 个在途请求平均只能占 2048 个 rounded KV token。但下述开发和测试各 192 条，不把同一文档跨两集合复用来制造并发。若 192 条同时驻留，平均容量为 2730.67 token；开发集合全部到输出上限需 69.6484 GiB，测试集合需 68.4414 GiB，所以容量压力在数学上可达。因动态 prefill、EOS 和提前完成，这不保证实际积压；必须观察 native 抢占和真实恢复状态。maxseq256 是统一编译/引擎容量，不意味着这 192 条的运行峰值能达到 256。

## 无需下载的 384 个唯一全文输入

原仓库根目录：`/Users/zhaozhenyu/Desktop/毕业设计/MOE_Thesis_organized`。

以下 `R` 指原仓库下 `refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck`。三个文件都已存在，保持 `actual_prompt_token_ids` 原样：

| 顺序 | workload 路径 | 请求数 | prompt 长度范围 | prompt 总 token | 文件 SHA256 |
|---|---|---:|---:|---:|---|
| A | `R/spare_followup_fresh_inputs_r01/workload.json` | 128 | 444–3066 | 241675 | `d336841dbaf24bd75d1bc8eb16f6f31aa9171325b7a537ecdf80fed639f7f730` |
| B | `R/native_residency_fresh_inputs_r01/workload.json` | 128 | 500–3062 | 242985 | `4529a6f680ec0e67ca2082078e583786f8115ea99b101d9a5dc0ba00e7c66e80` |
| C | `R/native_current_guard_fresh_inputs_r01/workload.json` | 128 | 616–3040 | 250676 | `f6e5d288fa54012851be92cacb77480e4d6601e92d53315e65c8c5d04f7b9974` |

B 与当前 `research/C_admission_20261004/inputs/workload.json`、`recovery_quantum_20261002/candidate_native_recovery_lease_r01/pkg/inputs/workload.json` 是同一载荷。A 在 B 的构建器 `build_native_residency_fresh_inputs_r01.py:92` 明确作为前序输入；C 的构建器 `build_native_current_guard_fresh_inputs_r01.py:92` 明确列 A、B 为前序输入。这是同一输入构建链，无需遍历其他 idea。

三者均为 WikiText-103 raw v1 train、dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`。来源 shard `train-00000-of-00002.parquet` 的已记录 SHA256 为 `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212`。模型与 tokenizer 配置三者一致；不使用 Instruct 健康题输入。

本次 CPU 检查了 384 个逐条 `prompt_token_ids_sha256`，并验证 token 数等于 `prompt_token_count` 和 `original_document_token_count`、`prompt_length+1024<=4096`。384 个 document ID、全文 hash、prompt token hash 均各有 384 个唯一值。它们是全文而非裁剪/补齐，不必重复请求。A/B/C 都有历史使用，不能声称这是严格独立测试集或全局未见数据。

固定分割规则：每个来源中原始偶数索引 0、2、…126 进入 dev，奇数索引 1、3、…127 进入 test；每个保留索引按 A/B/C 轮转，每个集合恰好各取三个来源 64 条，总计 192。选择不读取未来 EOS、生成长度或服务结果。原 request ID/document ID 不变，另保留来源标签、原索引和源文件路径。dev/test 的全文 hash 和完整 token hash 均不重叠，不能称为全局未见或严格盲测。

已生成压紧 JSON `pro_inputs.json`（约 3.75 MB），顶层结构为 `{dev:标准workload, test:标准workload, provenance:{...}}`；每个 workload 有 `source_requests`、`actual_prompt_token_ids`、`arrival_traces_s={'steady':[0.0]*192}`，供 runner 按同一低/近/高负载规则重新冻结外部到达。输入 token 不变，移除重复全文正文以减小包，但保留正文 hash。复现命令如下（输出必须为新路径，不覆盖已冻结文件）：

```sh
python3 research/C_admission_20261004/prepare_pro_inputs.py \
  --repository /Users/zhaozhenyu/Desktop/毕业设计/MOE_Thesis_organized \
  --output /absolute/path/to/NEW_pro_inputs.json
```

| 输入集合 | prompt 均值 | 短 <1536 | 中 1536–2559 | 长 ≥2560 | 所有 prompt rounded KV | 全部到输出上限的 rounded KV |
|---|---:|---:|---:|---:|---:|---:|
| dev 192 | 1940.5104 | 65 | 89 | 38 | 45.6484 GiB | 69.6484 GiB |
| test 192 | 1889.3646 | 65 | 92 | 35 | 44.4414 GiB | 68.4414 GiB |

最后两列是假定各集合全部同时保留的上界/载荷描述，不是测得峰值。未获准开始 prefill 的请求必须在可见的 scheduler waiting 队列里等待。

## 最少三个待验证到达压力点

先在相同 64 GiB usable KV、maxseq256 下用 native 比较三个预选负载点；三点都从相同空 connector cache 开始，暖机后 drain/reset，所有 due arrivals 如期提交 native scheduler。low 用所选集合前 64 条，near/high 用完整 192 条；这是显式请求数和到达强度变化，不能当作等工作量速度对照。它们是预先指定的到达负载点，真实压力等级需要用观测确认，不能预先写成已证实低/中/高压。

| 点 | 请求数 | 外部到达 `arrival_s[i]` | 到达窗口 | 目的 |
|---|---:|---|---:|---|
| low | 64 | `0.50*i` | 31.50 s | 正常预算低到达量控制；该请求数的容量上界明显可容纳 |
| near | 192 | `0.10*i` | 19.10 s | 增加在途量，寻找正常预算下有恢复的代表点 |
| high | 192 | `0.02*i` | 3.82 s | 有限密集到达，检验是否触发容量抢占及积压消退 |

这三次压力定位不必先铺三策略全矩阵。固定基线仅在 dev 比较少量 cap：96、128、192，参考 native cap256；引擎编译容量始终为 256。选择有竞争力的固定 cap 和一个确实存在恢复、能完整排空的代表点后冻结，进入 test 的最佳固定/KV/recovery 及反序重复。因为单个集合只有 192 条，native256 与固定192 的人数上限可能完全相同；这应作为预期等价性检查，不包装成独立有力对手。若三个点都无恢复，应如实报告正常预算下尚无 action space；不回退到小 KV，也不将 host 输出停顿直接当恢复积压。

若需补充 burst，在完整 192 条 test 上预先冻结每组 64 条同刻到达、组间 2 秒的 trace；不能根据某臂完成时间改变 burst。dev 的任何规则修订不得读取 test 服务结果；正式确认前固定 SLO 和规则。当前 test 只表示与 C 的 dev 文档分离，旧方向已使用过这些文章，仍需公开其历史。

每点至少保留 peak running/active、实际最小 free KV、真实 PREEMPTED 与恢复型 WAITING_FOR_REMOTE_KVS 计数及等待变化、抢占数、全部到达请求 TTFT/完成/最大生成间隔、输出长度与 stop reason、排空时间。队列与客户端提交 lag 都计入从外部 arrival 开始的延迟；输出上限不作为未来实际输出长度特征。观察截止需覆盖到达窗口与后续排空，未完成/失败/超时不得排除。

## 重复 prompt 与 host cache 的边界

当前已具备 384 个唯一输入，优先使用它们。所有策略保持相同 GPU prefix-cache 配置和 host offload/cache 配置，不通过给某臂清缓存来制造差异。全文/完整 prompt 去重不保证所有文章没有相同短前缀，应记录原生 connector 的真实 cache-hit/transfer 行为。

若未来确需复用超过 384 条，必须公开复用次数、顺序、独立 request ID 与重复 prompt hash；GPU prefix caching 关闭并不自动关闭 native host cache，重复内容可能使 H2D/store/匹配路径显著不同。重复产生的收益或无压力不能外推到唯一文章工作负载，不应隐藏为“更多独立请求”。
