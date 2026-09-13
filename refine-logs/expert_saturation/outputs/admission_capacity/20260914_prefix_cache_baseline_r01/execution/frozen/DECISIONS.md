# 原生前缀缓存强基线：CPU 准备，GPU UNRUN

唯一问题：同固定 KV 预算下，原生 APC 是否改变抢占恢复成本及完整请求结果？
这是后续强基线准备；当前唯一 GPU 实验仍为四项同路径时间重复。本包不上传、不占队。
复用已观察 cohort2；32 requests × 3072 prompt / 1024 output，50 ms steady。
OLMoE BF16、vLLM0.26、原编译模式、FCFS、full-history reservation、cap32、token budget1024。
固定 KV 16089350144 bytes；初始化核实 7671 可用块。四个独立引擎顺序 off/on/on/off。
两臂不安装 headroom/rotation 策略；native_capture、memory_telemetry、metrics 保持逐字不变。
三个公共 warmup 输入/顺序/输出长度不变，保留全部 warmup；APC 可能改变其实际缓存命中。
warmup 后完全排空，两臂均调用 LLMEngine.reset_prefix_cache(False, False)，保存返回值及块状态。
reset 必须返回 True；非 null 块 ref_cnt 必须为0，全部 hash 为空，空闲块仍7671，否则测量 UNRUN。
每格初始及测量前检查 GPU；占用或查询失败 ABORT，不终止其他进程。每格结束回传再继续。
按 block 比较 off→on；两次同角色重复仅描述。四格全部合格才做全组比较，不事后挑选或替换。
请求独立推进，保留全部 token、schedule、缓存跳变、重算、失败、长调用和环境信息。
主指标：完整请求吞吐、平均完成、max-ITL；同时保留 TTFT、抢占数、实际重算 token 和等待跨度。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls；恢复跨度可含其它请求有用 decode。
APC computed_adjustment 是缓存复用跳变，不是执行 token；只累计成功 engine.step 内的正跳变。
首次准入与抢占后的正跳变分开记录；不累计 waiting 失败重试的 cache lookup 命中。
原 capture 的 high-water 取成功调用后的 computed_after，包含继承的 APC 前缀；其 recompute_tokens
表示重建此前已建立的前缀，不能一概解释为该请求此前亲自执行过的 GPU token 再执行。
严格 request-specific 重复执行量应另用成功执行区间并集分析；本包不改采集公式。
cohort2 的32个首16-token块均不同；在原 block_size=16 且测量前冷缓存下，应无首次跨请求命中。
memory_trace free 包含可驱逐缓存块，used 是非空闲物理块，不是“带有效 KV 内容的块”。
APC 共享可使各请求块表长度之和大于物理占用；禁止借此推导错误守恒或仍用独占块的旧分析器。
所有策略保持独立 token/KV/完成状态，不从旧 trace 构造反事实；输出一致性不等于任务质量等价。
收益/退化/无变化均报告。若 APC 吸收原生恢复成本，先更新强基线；否则定位缓存存活与排队约束。
不据本四格声称 method GO、业务 SLO、跨模型显著性或 APC-aware 轮转成立。
停止：四格结束或任一资源/身份/会计条件失败；失败原样保留。此实验上限 NATIVE_SERVING/MEASUREMENT_ONLY。

源码 API 核验（官方 v0.26.0）：
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py : reset_prefix_cache 返回 bool，默认不重置运行请求/connector。
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/block_pool.py : reset 在非 null 块全 free 后清除 hash。
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py : APC 恢复查 request.num_tokens−1 内完整块。
