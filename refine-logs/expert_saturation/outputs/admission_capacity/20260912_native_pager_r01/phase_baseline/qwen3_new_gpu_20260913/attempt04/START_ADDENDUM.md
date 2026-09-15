# Qwen r04 已首次启动

2026-09-14，1789397400.51546。前序funding六格及staged-store两格均COMPLETE/exit0、控制进程不存在；现场GPU空/共同flock可取。新GPU bd5e9bb9，90GiB共享容器，host anon330579968B；root分片空间12309188608B、data结果空间2538971136B。包367cb7e6…5540a的40成员/38输入/11安装源码核验，17runtime/30payload不变。monitor6954/receipt detached-qgcfu5ns。

本组独占锁覆盖加载、layer47资格、条件32/16/16/32和退出，6h总上限/600s每episode；性能数据不得跨旧设备拼接。仅STARTED，不是数值或性能通过。完整原件、失败与分片保留；不因SSH观察断开重启。


2026-09-15 加载完成：loader COMPLETE于1789410481.648671，16分片、18867源张量、435目标名全部记录。1789410614.495原worker6967仍存活/GPU25016MiB，numerical_qualification目录已生成reset，数值及四格性能尚无终态。原组不重启，继续原资格门槛。

2026-09-15 终态补记：原组四格已全部完成，parent/child均exit0，parent结束1789410741.7905633；1789410851.460459现场确认6954/6955/6967均不存在、GPU计算进程为空、共同锁可取。唯一回收归档c81b9d56…557d、144成员/141payload/38冻结输入核验完成，原始输入与前序失败不变。

结果见[REPORT.md](REPORT.md)及[analysis_r01.json](analysis_r01.json)：16请求执行/416输出/1552计算位置；小分块降低观察maxITL同时增加TTFT与专家payload，平均完成方向翻转，MEASUREMENT_ONLY。原47/48全参考allclose保留，layer47相同分组reference的逐位归因门槛通过不等于质量通过。总runner寿命13313.962秒及全部资格/预热/IO计入分账。当前为COMPLETE_MEASUREMENT_ONLY，B支线交接，不再以本页历史STARTED/在途状态触发重跑。
