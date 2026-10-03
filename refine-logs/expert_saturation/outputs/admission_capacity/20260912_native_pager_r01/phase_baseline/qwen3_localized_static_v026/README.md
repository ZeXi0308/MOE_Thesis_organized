# 单次加载：数值定位与条件静态对照

**REMOTE_STATE_UNVERIFIED。** CPU准备及冻结已完成，r01通过空闲检查并实际启动；SSH在第9/16片加载期间断开，两次重连在认证前关闭。当前无法核实远端进程或结果，详见[断连记录](CONNECTION_INTERRUPTION.json)；没有回读到本次资格化或性能结果，也没有重启或终止远端任务。原Qwen资格化已完成加载与请求会计，但layer47的原allclose诊断未通过，见[实际报告](../qwen3_native_qualification_v026/attempt03/REPORT.md)和[定向审计](../qwen3_native_qualification_v026/attempt03/EXPERIMENT_AUDIT.md)。本目录保留该事实，不修改原阈值。

本次冻结流程只加载一次相同Qwen BF16权重。先重跑原资格化输入与暖机，在layer47相同precall上保存每组实际输出，与同一分组、全权重的kernel参考逐项比较。只有全部48层finite、专家集合完整且不相交、每组输出及最终分组求和均按位一致，才执行已冻结的static32/16/16/32。未达到条件时保留完整失败证据，性能格数为0。这是运行前规则，不代表当前已观察到任何数值判定。

旧hidden张量未保存；新运行中的原请求/位置、top-k、分组和诊断模式对照另记，不把模式一致叫作旧hidden按位一致。FP32-cast和反序求和只作描述，两个partial不必因这些操作而改善。局部分组等价也不证明任务质量或整个resident引擎等价。

数值作用域独立封存后，仅重置同一runtime的计数、context、验证标记和输出目录，保留模型、CPU master、GPU scratch、KV及map分配，逐项检查地址和尺寸。正式四格仍使用source4–7、输入64/64/128/32、输出32/32/16/24、到达0/0/2/4秒；每格共同暖机16→32、排空并重置pager。参考计算和快照复制不进入性能格。

固定cap48、KV预算512MiB（实际341block/511.5MiB）、token64、maxseq4、BF16/expert/LRU/nested。输入SHA、检查条件和成本边界见[运行前补充](LOCALIZATION_ADDENDUM.json)。完整engine session包括加载、数值资格化、记录切换、四格和关闭；每格capture与warmup/reset/flush分别记账，不从总成本中删去资格化或等待。

入口为`source/run_native_pager.py --qwen-static-prefill --qualification-prepared qualification_prepared`，其余参数继承原静态准备。原始四格目录保持未执行记录；本目录是明确增加一次数值定位后的新运行范围。GPU入口仍在初始化前、阶段边界及运行中检查占用，只终止本次进程组。

本次上限是局部数值归因与一个真实超显存模型、有限到达episode的同预算权衡；没有新controller、质量/SLO/显著性、全局最优静态、Oracle或method GO。完成这四格也不自动完成总研究目标。
