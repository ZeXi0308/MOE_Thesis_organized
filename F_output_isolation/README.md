# F：CPU 输出服务隔离

本目录沿用[工作区科研基准](../AGENTS.md)，复用方向 F 的原生CPU输出原型与证据包。**最新判决：停止C1并收束当前投入；保留简单分类优先作为有代价的对照，公平取舍与新方法价值尚未确定。逐请求审查发现重请求子群退化，不能视为隔离问题已解决。** 不使用GPU，不修改共享安装、GPU调度、KV或准入。

最新入口：[一页研究卡](research_card.md)、[判决](verdict.md)、[假设与模型](hypotheses.md)、[主结果与副作用](review_20261008/endpoint_results.md)。`main_results.md`和根目录`summary.json`仍为历史C1阶段结果；不与复审数据混成同组配对。旧20%门槛仅是历史投资协议，不是最新科学判据。

## 本次复审的最小复现

原环境、tokenizer路径和依赖版本见下节。远端在`/root/F_output_isolation`运行，CPU三核预算和实际原生路径保持一致：

```sh
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/miniconda3/bin/python review_20261008/run_endpoint.py \
  --trace inputs/high.json --output review_20261008/reproduce_endpoints
```

默认顺序固定为`fixed1,priority,cache,cache_priority,cache_priority,cache,priority,fixed1`，输出目录必须不存在。一次约8个35秒测量窗口，另加原生组件启动时间；无需GPU或模型权重。`endpoint_server.py`使用未改的`native_adapter.py`和原生单条配额handler：P在取批后稳定轻流优先；D为4096项在线LRU，缓存原生单token解码字符串、返回新的列表给原生上下文修正。缓存从空开始，排序与缓存开销进入计时，完整冷窗口保留。它们均为简单原则/组件适配，非新方法。

从本地已保存的结果重算（仅Python标准库，在工作区根目录）：

```sh
python3 F_output_isolation/review_20261008/audit_existing.py
python3 F_output_isolation/review_20261008/analyze_endpoints.py
python3 F_output_isolation/review_20261008/tail_audit.py
python3 F_output_isolation/review_20261008/request_effects.py
```

新跑结果可用`python3 review_20261008/analyze_endpoints.py --runs review_20261008/reproduce_endpoints --output review_20261008/reproduce_analysis`分析（在F目录运行）。脚本核验冻结high输入、8次顺序、完成与输出摘要、原始时间账目，输出所有运行和副作用；源码清单只证明保存清单与本地文件匹配，新运行应重新记录执行身份，不能借旧清单声称新运行已被见证。原始记录在`review_20261008/endpoint_high/`；冻结协议见`review_20261008/protocol.md`；源码及安装路径哈希在`execution_hashes.sha256`、`runtime_identity.json`；归档传输校验在`download_verification.json`（均在review目录）。最初采集于新实验开始时，12个安装源码与原阶段一致。

本次按更新科研基准继续时，新增的只有[逐请求损益分析](review_20261008/request_effects.md)：复用既有A/P两对，不新增服务实验，不改主指标。120个轻请求两对均受益，但8个384-token重请求两对均有较大退化，已更新判决。

最新8run只复用high轨迹，不扩负载；全部保留，近邻有界缓存命中99.58%受合成候选分布影响。P的收益不能称为完整GPU服务收益，DP长尾归因未定。原始SSE正文在运行时完成语义摘要后释放；离线脚本核验摘要与账目，不声称重新逐字检查已未保存的正文。

## 历史C1阶段的最小复现

原环境：Linux x86_64，Python 3.12，vLLM 0.26.0 (`gffd46bfab`)，模型 tokenizer 已在 `/root/autodl-tmp/moe-research-20261002/model`。依赖精确版本与输出源码 SHA256 见 `environment.json`。不需要加载权重。`bootstrap.py` 只在当前进程绕过此环境 Torch 升级残留的三个孤儿模块，不修改安装包。

在原服务器 `/root/F_output_isolation` 中，或将本目录程序复制到同配置的新目录后：

```sh
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/miniconda3/bin/python smoke_native.py
/root/miniconda3/bin/python run_replay.py \
  --trace inputs/high.json --output reproduce_high \
  --arms native,fixed1,fixed8,cost,cost,fixed8,fixed1,native
/root/miniconda3/bin/python run_replay.py \
  --trace inputs/medium.json --output reproduce_medium \
  --arms native,fixed1,fixed8,cost,cost,fixed8,fixed1,native
```

输出目录必须不存在。默认 frontend/client/producer 分别绑定同 socket 的独立物理核 2/4/6，可用 `--server-core`、`--client-core`、`--producer-core` 修改，但所有臂须一致。默认 HTTP 仅监听 `127.0.0.1:18761`。脚本强制 `CUDA_VISIBLE_DEVICES=''`；不获取 GPU 锁、不启动模型引擎。

已有 `inputs/*.json` 可以直接回放。若从原服务器既有 D 数据重新生成：

```sh
/root/miniconda3/bin/python prepare_replay.py \
  --source-root /root/autodl-tmp/moe-d-prefill-20261004
```

此程序应生成 `frozen_workloads.json` 中一致的 SHA256。本地研究工作区可直接 `python3 prepare_replay.py` 读取相邻 D 目录。输入没有重新采样、重复或时间缩放。

## 历史C1阶段的复用边界与方法

- `native_adapter.py` 直接导入原生 tokenizer、OutputProcessor、detokenizer、LogprobsProcessor、RequestOutputCollector、ChatCompletionRequest、chat streaming generator、Pydantic JSON；没有重写一个简化响应前端。
- 引擎替身只供应预构建的原生 EngineCoreOutputs。独立时钟进程按逐批实际时序发送批号。原生对象构建在计时前完成，对应已由 engine 产生的 CPU 数据边界；引擎 IPC 的 msgpack transport 和输入渲染不在本次计时范围。
- FastAPI/Starlette/uvicorn 提供真实 SSE HTTP，包含原生 role、content、finish、usage 和 DONE。预先打开全部传输连接，保留轨迹的逻辑请求到达与 token ready 时间；测量不是完整 HTTP 请求输入/模型生成服务。
- A：未改动的 `AsyncLLM._run_output_handler`，原生输出配额 128。B1/B8：同一个函数，仅设置上游支持的 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1/8`。C1：唯一候选，累积实际输出处理 thread CPU 到 100 μs 后让出，保存 batch cursor；以原生单 token 输出为原子单位。C不改变对象构造或 serializer，也不声称严格字节积压上界。
- 全部臂关闭统计日志（`log_stats=False`），禁用工具/推理解析器，复用原生 API startup 的 `freeze_gc_heap()`。未关闭 GC，未添加轻请求优先级，未延迟重请求到实验之外，未修改 stream_interval=1。
- 原生 collector 可以合并尚未消费的多个 token。不同方法可能导致 SSE chunk 个数不同；它们走相同原生规则。完整文本、逐 token logprobs 顺序与内容、终止、usage 均验证，不通过截断减少输出。最初协议的“相同SSE切分”要求据此修正为相同原生切分规则与完整响应语义，实际字节/chunk 数保留在结果中。

## 指标与限制

主指标是轻请求每个 token 从独立 producer 实际发布时间到客户端完整 JSON 解析后的 P99 等待。一个含多个 token 的 SSE 中，这些 token 共享可见时间。另报 planned-ready 等待、客户端 token-event gap、重响应、所有请求完成后的 request/s 与 token/s、CPU与峰值RSS。

这里的 throughput 是给定真实产出轨迹下 CPU 输出服务的全量排空吞吐，不是模型完整推理吞吐。端到端论文主张仍需真实 GPU 闭环。client 的输出哈希在正式计时结束后计算。服务端仅一个工作核，client和producer各一个核；RSS包含 Python/vLLM导入、预建回放对象，以及完整响应的原始字节存储。

两条轨迹来自已完成的单卡真实实验，保留 prompt/output IDs、到达、批次时间与固定长度分布；不含自然 EOS、真实 logprobs 或原始批内列表顺序。batch 内采用固定输入请求顺序。重响应占请求的25%，为确定性哈希选择；top-20 候选 token 与数值为固定构造，遵循真实 sampler 数组布局。真实生成 token 来自原轨迹；该GPU轨迹没有请求top-20，故不能声称此速率已在top-20混合真实服务达到。来源和限制见 `trace_audit.md`。

`diagnostic_high`、`pilot_high`、`pilot_gc_high` 是接线诊断，排除正式判决：前两者遗漏原生GC冻结，后一组客户端为离线验证保留了嵌套对象，导致GC污染。它们仅解释为什么不能把回放器问题当成研究收益。`formal_high`、`formal_medium` 才是冻结原型的正式反序对照，所有运行都保留。

查新范围、RFC状态及来源见 `related_work.md`；半页先验研究主张、否定条件和收敛规则见 `research_contract.md`。本目录不保存 SSH 密码或私钥。

## 从历史C1结果重算

本地只需 Python 标准库，不需要 vLLM 或 GPU：

```sh
python3 F_output_isolation/analyze_results.py
```

脚本重建主结果表和 `summary.json`，核验16次正式记录的语义哈希，保留时序污染标记。`download_verification.json` 记录远端归档与全部原件的哈希核验；原始逐token可见时间、collector时间、producer实际与计划时间均在各run目录。

## 本轮交付与证据覆盖

本轮以有范围限定的收束结论完成，未产出已成立的独立论文。科研假设的“未确定”与交付的“已完成”分开：

| 要求 | 当前证据与结论边界 |
| --- | --- |
| 服务对象、部署、目标、合法动作 | research_card.md；无产品SLO时明确研究取舍，未据事后阈值宣称成功 |
| 问题/模型/方法分离及竞争解释 | hypotheses.md；含可推翻条件、局部前驱推论及其异步失效边界 |
| 先判断空间并执行决定性行动 | 冻结四端点8run；继而仅复用原记录做逐请求损益，发现并补报集中代价 |
| 当前运行时、强简单与最近邻 | 历史native/fixed1/fixed8/C1，新P/D/DP；D为近邻组件适配，未声称完整RFC复现；未宣称公平基线集合完备 |
| 完整策略公平评价 | 同输入/产出时序/内容/三核预算，原生frontend、tokenizer与serializer；冷窗口、排空、全部请求和失败记录口径见协议 |
| 独立重复、资源与全部副作用 | 历史16run和新8run全部保留；旧medium时序失配保留标记；新DP坏尾及重background退化不删除，不把请求当独立运行 |
| 可运行原型、冻结配置、复现 | 上述命令、输入、command.json、执行源码身份及原始数据；已有真实运行证明接线，离线摘要复算可用标准库完成 |
| 贡献与收束 | verdict.md：工程适配与探索证据可复用，C1未获支持；公平取舍/DP原因/GPU仍未确定，不伪造正向论文 |

最终核验确认当前历史测量/代码/输入与保存哈希一致，新8run归档及逐请求分析来源一致。核验覆盖来源完整性与记录内一致性，不能补足自然top-k、真实GPU批内顺序、应用价值或因果归因；这些仍是明确未证明事项。局部模型不构成完整服务上界。当前不追加WRR/aging参数点的决策依据见hypotheses.md，不以缺SLO作为单独停止理由。
