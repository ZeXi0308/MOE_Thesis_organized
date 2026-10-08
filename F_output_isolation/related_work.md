# F：有界查新与定位（2026-10-08）

结论：**分块、让出事件循环、去掉中间对象、线程池或有界通道本身，都不足以作为 F 的方法贡献。** 候选只有在真实速率的异质请求混合中，按实际 CPU 成本分配服务，并明显超过调好的固定配额/公平轮转或 worker 隔离，才值得继续。以下仅核查最接近的四项，不是全面优先权检索。

| 最接近项 | 已核实事实与状态 | 对 F 的约束 |
|---|---|---|
| [RFC #57479](https://github.com/vllm-project/vllm/issues/57479) | Open；2026-09-18 建立，2026-10-07 重写。现在主要研究 RL 同时 abort 后的大量非流式 top-k 输出。Python 提出数组存储与无中间对象渲染；Rust 提出 64-position 分块和容量为 2 的 channel。文末说明新栈仍是作者 fork 中的草案，不能认作上游已合入。 | “有界可恢复渲染”已有非常近的设计。其目标与证据不是异质 streaming 的轻请求交付尾延迟；F 必须证明跨请求成本隔离的独立增量。RFC 自身也记录网络/客户端解析主导时渲染加速不改善全程，不能把其巨大响应 CPU 结果外推到本轮。 |
| [上游 AsyncLLM 输出循环](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/engine/async_llm.py) | 当前 main 的 `_run_output_handler` 已按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 切 EngineCoreOutput 批；chunk 间 `await asyncio.sleep(0)`，条件是批内仍有下一 chunk。单个 `process_outputs` 调用仍是同步的。 | 按消息数固定配额与 batch chunk 是必备强基线，不能称新方法。实际 CPU 成本异质性是否跨越这个边界，要由本轮 profiling 判定。 |
| [PR #58314：非流式 completion 原生 JSON](https://github.com/vllm-project/vllm/pull/58314) | Draft，未合入；普通非流式 completion 用 Pydantic `model_dump_json()` 替代先建字典再 JSON 编码，复杂字段走兼容 fallback。该 PR 不改 streaming。 | 选择大非流式 completion 时，应把这一简单渲染优化纳入强基线，避免只击败重复序列化。它优化单位工作成本，未给出跨请求公平隔离策略。 |
| [PR #52875：普通 completion streaming 模板](https://github.com/vllm-project/vllm/pull/52875) | 可访问页面为 Open，未合入；缓存 SSE 固定部分，`logprobs`、echo、token IDs 等仍走模型对象路径。[当前 completion serving](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/completion/serving.py) 的一般 streaming 路径本来就用 Pydantic `model_dump_json(exclude_unset=True)`。 | 轻流的对象构造也是潜在成本，但模板化已是现成简单优化。复现不得人为把 streaming 替换成更慢的 `dict` + Python JSON 弱基线。 |

版本说明：[GitHub main API](https://api.github.com/repos/vllm-project/vllm/commits/main) 此次返回 SHA `ba77c4c13018aa19244545cdc7ae8e016d66955e`（2026-10-08）。代码页面为同次查新的 `main` 页面；固定 SHA 的 raw URL 抓取失败，因此不声称所读页面已经按字节绑定该 SHA。实验 A 必须另记**实际安装版本、源码哈希和补丁状态**，不得用“最新 vLLM”代称。

仅查状态的旁支：[PR #58278](https://github.com/vllm-project/vllm/pull/58278) 独立 API worker reuseport listener 为 Open、未合入，解决连接归属问题；本轮单 frontend 不把它作为方法。#54050（token decode cache）抓取失败，状态未知，不声称已合入或未合入。

对 msgspec/Pydantic 的判断：RFC 的引擎协议使用真实 msgpack/msgspec，响应路径仍须按实际 API 核对；**“换 serializer”不能等同于“按 CPU 成本隔离”**。若原生 serializer、固定配额或单独 worker 已足够解决轻流尾延迟，本轮应停止论文探索。

证据边界：网页中的性能数字是作者报告，未在本项目复验；本文只据其界定最近邻设计及状态，不将那些数字填入 F 的主结果表。GitHub 网页状态可能有缓存；以上是本次可访问页面状态。

## 复审补充：必须补入的缓存组件对照

2026-10-08重新核查[RFC #57479](https://github.com/vllm-project/vllm/issues/57479)及当前[原生解码helper](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/tokenizers/detokenizer_utils.py)。RFC还包含有界单token解码缓存思路；它是当前logprobs热点的必要简单对照，不能因先前#54050抓取失败就略去。#54050状态仍未知；本次D是思想级独立组件适配，不声称完整复现该PR或RFC新栈。

实验使用实际安装helper的逐ID原生decode结果，缓存4096项不可变字符串；原生`_verify_tokens`可能修改返回列表，因此每次返回新列表，保留原有上下文修正。远端helper副本及SHA保存在`review_20261008/detokenizer_utils.py`和`runtime_identity.json`，未修改共享安装。原生与缓存路径所有已测输出摘要相同。候选词高度重复使本回放命中99.58%，不能外推自然候选集。

新P为已有静态类别优先的调度适配，未提出新状态；D为减少单位工作成本的近邻组件；DP检验二者是否可组合，不是新算法。RFC的非流式大输出栈、线程offload、有界channel及数组渲染未在本实验完整复现，其证据也不是异质chat streaming目标。该范围差异不自动构成F贡献；需要超越这些强简单原则且改善目标服务取舍的独立方法，本次尚未建立。
