# 结果与证据边界

当前安装运行时：vLLM 0.26.0，Python 3.12，Torch 2.11；沿用原生OutputProcessor、tokenizer、collector和OpenAI chat/SSE生成器。CPU侧C++直接编码用当前服务器g++。新GPU组件以nvcc 12.8、sm_120编译通过，未运行CUDA。RFC提案不视为已合入运行时；原生对象构造路径仍被实际语义检查调用。

| 必须比较的路径 | 轻流客户端交付P99 / engine→visible | 完整完成吞吐 | 重流尾部/饥饿 | 总CPU/峰值内存 | 当前状态与代价 |
|---|---|---|---|---|---|
| A 原生vLLM 0.26.0 | 未测 | 未测 | 未测 | 未测 | 原生语义参考；旧F合成性能不移作本轮结果 |
| B CPU原生直接编码＋独立交付 | 未测 | 未测 | 未测 | 未测 | CPU依赖/完整语义检查通过；已就绪工作仍由单worker FIFO处理 |
| C GPU直接编码＋独立交付 | 未测 | 未测 | 未测 | 未测 | 新组件编译及接口检查通过；CUDA/模型接线未运行；固定padding、D2D及每请求launch可能抵消收益 |

“未测”是未运行，不是0或失败。不存在本轮收益百分比、成本交叉点或端到端主张。两次采集尝试分别见[capture.log](capture.log)及[capture_attempt2.log](capture_attempt2.log)，均为`LOCK_BUSY_NO_GPU_INITIALIZED`。未取得GPU锁，未采集真实候选。

| 有限验证 | 结果 | 原始证据 | 支持范围 |
|---|---|---|---|
| 同批heavy编码人为阻塞 | plain不等heavy；heavy1不等heavy2 | [dependency_check_v2.result.json](dependency_check_v2.result.json) | 删除CPU编码的整批完成依赖 |
| 两个独立metadata future按heavy2→heavy1完成 | plain和heavy2均可先交付，完整语义相同 | 同上 | 设备元数据未就绪不占用CPU worker；用CPU Future模拟，未使用CUDA |
| Unicode/context回退、rank越界、重复候选、stop/abort等8组原生比较 | 完整内容及每步生成状态相同 | [async_semantics_final.result.json](async_semantics_final.result.json) | 构造语义用例；不是实际候选分布或性能负载 |
| GPU每请求异步编码器 | nvcc编译通过；CUDA执行未运行 | [gpu_async_build.json](gpu_async_build.json)、[源码清单](source_manifest.json) | 仅工具链/静态可构建性 |

完整语义比较保留JSON数值、字段、请求内顺序、停止/终止和usage；不要求相同浮点字面串或相同SSE分块边界。人工快速feed时，原生collector会自然合并积压delta：部分heavy的分块数因此不同，未把减少分块计为性能收益。78行中38次fallback也是有意构造的上下文用例，不能当作真实fallback比例。

`native_semantics.result.json`是旧同步桥验证。第一次异步测试在测试脚本引用已删除的`prepared`字段处失败，保留[async_semantics.log](async_semantics.log)；修改这项过时内部断言后得到v2结果；metadata future接口加入后的最终回归也通过，见上述final结果。新桥另按原生规则只消费实际token对应的logprobs，处理零token abort合并时附带历史的情形。

CPU与GPUtransport首版共享固定池，CPU模式不跑编码kernel/不传padding，但仍预留GPU缓存和展开槽；这不是CPU最低内存配置。模型对照必须报此驻留、CPU-only可节省的部分，以及常规CPU原生直接编码成本。GPU扩张字节、固定padding、回退raw、producer快照、轮询和host复制不得只报有效JSON。取消仅能停止交付/清理请求；已运行的native任务和DMA须完成后再释放资源。
