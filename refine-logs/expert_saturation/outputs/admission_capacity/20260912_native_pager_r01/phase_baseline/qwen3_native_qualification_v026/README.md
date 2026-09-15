# Qwen3 BF16 原生加载与执行资格化

唯一问题：在当前 5090 / 92 GiB 主机内存限制下，既有 vLLM 0.26 + WiSP expert 执行链能否加载并正确执行真实超显存模型？本目录是一次四请求资格化，不是性能对照。

冻结配置见 [protocol.json](protocol.json)。Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，BF16 tensor payload 56.871 GiB；48 层、128 专家、top-8、expert cap48、KV512 MiB、token64、maxseq4、prefill32。GPU 常驻权重/专家 scratch/KV 小计约 23.621 GiB，另有 CUDA、激活、allocator 与最多单层全权重的加载/参考瞬时空间；实际峰值以整轮记录为准。

四个既有公开文档使用本 revision tokenizer 重新分词，每条64输入/8输出，50ms间隔；原文和来源保留在 [prepared/workload.json](prepared/workload.json)，元数据和 tokenizer 的官方 Git blob/LFS 校验见 [metadata_manifest.json](metadata_manifest.json)。没有复用 OLMoE token IDs。另有单请求2输出预热，不计入正式32输出。

加载器在原生构造与后处理之间仅调用一次 `model.load_weights`。逐片下载、冻结 SHA/大小/源 key 校验，CPU tensor clone 后交给原生 Qwen 权重装载函数。消费完毕才删除本轮私有副本；异常保留临时分片及 receipt。原始 target 返回集合经实际 MoERunner 别名转换后检查，不能由默认 tracker 的补名代替初始化证明。CPU 证据见 [loader readiness](../qwen3_streaming_readiness/LOADER_READINESS.md)；CPU 夹具不等于真实 GPU 加载。

运行保留每层一次 same-precall full-weight fused-kernel 参考比较、逐请求/位置账、pager payload、显存与主机内存采样及完整进程状态。参考沿用实际已选 top-k，不能独立证明 router 正确性或任务质量；参考复制/计算含在请求时间中，本轮不报告性能收益。参考复制的 tensor payload 与 pager payload 分开，均不能称 PCIe wire bytes。

入口是 [run_remote.py](run_remote.py)，源码在 [source](source)，冻结上传包在 `code_bundle.tar.gz`。仅在连续3次GPU空闲检查后加载；外部GPU进程或查询失败时停止本轮进程，保留失败记录。`execution_attempt01.json` 是 SSH 命令状态，真实子进程终态以回读 `launch/launch.json` 为准。下载大分片留在远端本轮私有工作区，不纳入结果回读包。

原始 protocol 状态保留 `PREPARED_UNRUN`，最终实测状态将在结果报告中单列。准备与运行不改变旧 OLMoE/phase32 裁决，也不构成 scheduler GO。
