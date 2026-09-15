# 同字段显存观测的实现开销

2026-09-13；**MEASUREMENT_ONLY / ADOPT_NESTED_OBSERVER**。四格完成，支持将nested读法作为本执行链的共同观测实现；不构成调度机制收益。

同一admit2_32、三请求完整工作量、OLMoE/vLLM0.26、expert cap16、KV512MiB、token64。顺序flat、nested、nested、flat，均无profiler。只将expert _apply中的三处显存标量查询改为同一底层统计树的直接字段读取；位置、次数及current/peak语义保留，两臂都有共同helper。五段形状暖机与轻量暖机仍全部flat；每格capture前验证实际安装Torch源码SHA及flat/nested/flat读数相等，finally恢复flat。

| 观测读法 | 完整wall s | 旧maxITL ms | 新TTFT s | 新maxITL ms |
|---|---:|---:|---:|---:|
| flat_before | 2.350358 | 54.676 | 1.203297 | 40.562 |
| nested_1 | 2.272188 | 51.044 | 1.132746 | 46.865 |
| nested_2 | 2.259651 | 50.999 | 1.136247 | 37.435 |
| flat_after | 2.333764 | 54.586 | 1.188351 | 40.493 |

两对nested−flat的wall为−78.171/−74.112ms（−3.326%/−3.176%），engine观测窗口thread CPU为−78.072/−73.809ms。旧完成提前41.953/42.627ms，新TTFT减少70.552/52.103ms，但新请求maxITL变化+6.303/−3.058ms，不能称逐请求/逐指标支配。同模式wall差flat −16.595ms、nested −12.536ms，只是两次观测差，非噪声界或统计确认。

12测量请求/160输出完成；每格27次engine调用、974组、91.55859375GiB实现内权重copy payload。逻辑前态、cache、分配、分组/加载、post-event逐行top-k与输出均相同。每格368个expert layer-call的三个allocator字段，共1104个数逐项一致；物理KV IDs另存，不声称KV tensor bytes相同。完整逐请求指标、phase账本、所有正负差见[summary.json](summary.json)。

host_apply差−26.095/−28.383ms，GC真重合差−10.037/−13.385ms，CUDA load span差−3.128/−2.239ms。这些范围包含或重叠，不相加、不从wall中扣除。该对照测的是共同采集实现的成本变化；它未定位前轮约578ms的同策略漂移。前轮四份异常cProfile仍全部INVALID，不能援引其函数时间解释这里的节省。

Verdict为本域观测实现采用nested，科学状态仍MEASUREMENT_ONLY；证据是人工专家预算下的native完整请求对照。最强基线为同政策flat观测；未测真实超显存模型、持续到达、质量、SLO或exact Oracle。未发现本轮身份/会计失败，统计外推未验证。若其他环境字段语义或源码变化，重新检查后再使用。

唯一下一实验回到请求调度：固定共同nested观测，在8个独立评价文档、异构输入/输出与时钟到达上，比较cap3-static32、cap2-static32、cap2-phase32及反序；普通cap2-static32是phase规则的强基线，所有请求等待和完整工作量计费。不得把此处观测修正的收益归入后续scheduler。

worker79915已退出；进程92.717s、初始化后65.455s；24次预热64请求/808输出均完成。11次GPU边界通过，184个硬件样本只见本worker；这是采样与边界记录，不保证连续独占或CPU隔离。[完成回执](readback_completion.json)保留源码与archive SHA e505ca53…，原件不改。
