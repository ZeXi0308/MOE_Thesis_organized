# 真实KV往返与现成后端资格

前报告CPU预算与UNRUN是执行前状态，原文保留。本次GPU探针COMPLETE；raw、24次全部试验、失败为0，回读SHA一致，GPU已释放。

| 块数 | 全块单向字节 | D2H含gather中位 | H2D含scatter中位 | 往返中位 |
|---|---:|---:|---:|---:|
|205|429916160|8.151ms|8.373ms|16.500ms|
|207|434110464|8.307ms|8.430ms|16.743ms|
|256|536870912|10.148ms|10.385ms|20.539ms|

各列独立取中位，和不必等于roundtrip中位。每尺寸1 warmup+7测量，全部原始记录保留。24次原始payload恢复校验通过；校验不计时，alloc不计时，打包/拷贝/scatter/Python派发/同步计时。设备UUID GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9，torch2.11.0+cu130。完整模拟KV池13,960,740,864字节、最大单层staging33,554,432字节、最大pinned536,870,912字节；没有模型权重/并发decode，集成后必须在同显存总预算内计费这些资源。

首次自然抢占的205块往返16.50ms，小于冻结模型该请求重算token项38.13ms。差额约21.63ms只是局部测量对模型项的比较，不是请求级收益、统计保证或严格可删除成本。真实恢复窗口110/129ms还包含其他请求工作，不用于夸大节省。

## 安装源码中的复用路径

已保存当前机器安装源码native_offload_source.json：
- config/cache.py:180–189定义kv_offloading_size（GiB）与native backend。
- config/vllm.py:873–880配置OffloadingConnector或由环境选择SimpleCPUOffloadConnector；cpu_bytes_to_use显式计费。
- offloading/scheduler.py:720–775具有异步matched-token查询，in-flight transfer可延后请求；1122–1158 flush被抢占和重新分配块的未完成传输。
- scheduler.py:401–405有prefix caching开/关分支，但只证明实现考虑该配置，不替代当前模型/adapter兼容性测试。
- 现有冻结rotation_native.py:63明确拒绝connector，不能直接打开offload并宣称轮转已兼容。保持原guard；首先跑纯native同资源offload对照。

因此不新写pager，不将KV swapping本身包装为新颖贡献；若现成backend已捕获收益，就作为最强简单策略，研究剩余问题必须以它为基线。它可能主动保存未被抢占的KV，真实D2H总量与主机占用可能高于按39次victim预算；必须测实际保存/命中/加载与请求完成。

Verdict: MEASUREMENT_ONLY，物理传输路径有继续资格化的空间。
Evidence: ISOLATED_GPU_RUNTIME，非native完整请求。
Unmeasured: offload实际命中/驱逐、worker集成、并发影响、主机与staging税、模型输出与完成收益。
Strongest baseline: 现成native OffloadingConnector优先于自写swap。
Oracle/headroom: 仅局部成本空间，无full-request Oracle。
Failure category: 尚待系统集成资格，不是NO-GO。
Claim ceiling: 单设备、三个尺寸、合成已知payload的布局往返正确性与成本。
One next smallest experiment: 保留现有native runner，新增固定主机offload预算的on/off对照与transfer观测；先证明加载与恢复路径真实生效，再考虑轮转兼容。
