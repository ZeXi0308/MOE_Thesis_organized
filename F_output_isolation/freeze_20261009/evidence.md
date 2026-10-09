# F 冻结证据状态表

2026-10-09。本轮只整理已有记录；未运行新服务实验、GPU、候锁、参数搜索或指标替换。F定位为保存完好的工程候选与测量案例。表中的“未运行”不是负结果，“检查通过”不是性能收益，“旧候选停止”不否定新直接编码分支。

| 路线 / 比较对象 | 证据状态 | 已保存事实 | 支持的判断与明确边界 | 原始入口 |
|---|---|---|---|---|
| 旧C1实际CPU成本配额 vs fixed1等简单配额 | 已有CPU回放结果；C1未获稳定增量支持，已停止 | high/medium共16次正式记录，时序污染标记保留；相对fixed1的high方向不一致 | 限定为当时C1及其负载；不能否定异质输出问题，更不能转作直接编码负结果 | [C1及复审判决](../verdict.md)、[最初主表](../main_results.md)、[历史判决](../review_20261008/historical_verdict.md) |
| 旧P类别优先 vs A单条配额FIFO | 已有CPU回放取舍；未证明公平隔离 | 固定high的A/P/D/DP正反序共8次；P轻流合并token P99为18.22/17.80ms，A为32.44/30.93ms；120轻请求两对均受益，8个重background两对均退化 | 轻流改善真实存在于该回放，同时有集中代价；类别成本不是服务优先级，不能称所有请求改善或真实GPU收益 | [四方案结果](../review_20261008/endpoint_results.md)、[完整P/A请求分析](../review_20261008/request_effects.md) |
| 旧D解码缓存、DP缓存加类别优先 | 已有局部工作与交付反例 | D局部OutputProcessor CPU下降而交付未稳定改善；DP两次重P99为61.32/62.39ms，额外尾等待主要在取批前 | 局部加速不能自动外推客户端收益；DP具体归因未定，不能把它归因于新编码器 | 同上 |
| 旧输出物化L vs 同释放规则S及五个其他静态点 | 已有CPU回放结果；候选已停止 | 两固定负载×7方案×2重复=28次；high轻P99相对S下降16.66%/12.90%，重P99增加4.440/4.823ms；完整server+client CPU增加0.60%/0.13% | 是交付取舍，未建立服务权益及独立方法价值；停止并非“CPU必须下降”。light内部及heavy群体的得失完整列于本次固定案例 | [物化主表](../materialization/results.md)、[判决](../materialization/verdict.md)、[逐请求案例](request_case.md) |
| 新直接编码：C++编码、异步前端桥及原生回退 | CPU完整语义检查通过；性能未测 | 最终8组构造用例，原生/直接路径的文本、ordered logprobs/bytes/数值、finish/stop/usage/DONE及每步状态相同 | 支持这些用例的实现正确性；构造fallback频率和人工快速feed分块不能外推真实分布、P99或CPU节省 | [最终语义结果](../direct_encoding_20261009/async_semantics_final.result.json) |
| 新直接编码：按请求完成依赖 | CPU受控阻塞与Future检查通过；不是CUDA测试 | plain可在heavy被阻塞时完成；heavy1编码完成可不等heavy2；只完成heavy2的metadata future时heavy2可先交付；取消检查范围明确 | 支持删除原型自行引入的整批编码完成屏障；单worker仍串行处理已就绪工作，未证明完整HTTP断连链、硬界或生产公平性 | [依赖检查](../direct_encoding_20261009/dependency_check_v2.result.json) |
| 新直接编码：GPU异步组件与首次D2H前接线 | CUDA组件编译通过；设备执行和模型接线未运行 | nvcc 12.8 / sm_120成功编译；有每请求槽、event及compact-only接口源码；`device_execution:false` | 仅静态可构建性和已保存实现；不证明CUDA数值正确、每请求真实独立完成、传输成本或推理干扰 | [编译记录](../direct_encoding_20261009/gpu_async_build.json)、[采样接线源码](../direct_encoding_20261009/sampling_branch.py) |
| 新分支真实top-20候选 / 自然批次采集 | 未运行、无数据 | 前轮两次启动记录均为LOCK_BUSY_NO_GPU_INITIALIZED；本轮未重试、未查询或等待GPU资源 | 无真实候选分布、fallback率或真实负载成本结论。旧GPU token轨迹没有开启top-20，不能替代 | [两次历史启动记录](../direct_encoding_20261009/capture.log)、[第二次](../direct_encoding_20261009/capture_attempt2.log) |
| 新分支完整CPU/GPU路径及客户端交付 | 未运行、未知 | 没有新分支的客户端P99、engine→visible、完整吞吐、重流尾部/饥饿、总CPU/峰内存或下一步推理时间结果 | CPU最合适的区域、GPU增量、成本交叉点、执行位置选择及论文贡献均未知；不从旧路线迁移成败 | [新分支状态表](../direct_encoding_20261009/results.md)、[复现与待验证项](reproduction.md) |

旧性能数据是开放环CPU输出回放：保留既有生成token与产出时序，top-20候选合成，原GPU批内顺序缺失而使用固定回放顺序。使用真实原生输出组件与HTTP/SSE客户端，但不含真实top-20推理、完整输入处理及engine IPC。每方案两次运行是重复单位；token或请求不能当作新增独立实验。固定产出轨迹下的完整排空吞吐不是容量上限。旧原始SSE正文已在计算摘要后释放，离线核对的是保存摘要和账目，不是重新逐字验证未保存正文。

新分支的前端同步屏障是其初版桥接引入的设计缺陷；CPU修复检查不能当作又发现了一个生产系统瓶颈。完整响应被保留也不自动意味着满足应用交付要求。
