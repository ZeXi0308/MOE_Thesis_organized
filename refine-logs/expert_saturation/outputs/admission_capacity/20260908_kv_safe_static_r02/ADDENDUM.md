# r02：修正实际 coordinator 类型识别

r01首个引擎在资格检查退出，状态 `UNRUN / QUALIFICATION_FAILED`，完整保留于
`../20260908_kv_safe_static_r01/gpu_results/repeat0-baseline16/`；0次warmup、0次请求测量。
不是容量实验失败，也没有已测性能结果可继承。

错误来自资格代码要求 `UnitaryKVCacheCoordinator`，但vLLM0.26在prefix caching
关闭时实际选择 `KVCacheCoordinatorNoPrefixCache`。root已核实工厂分支和基类分配
逻辑：NoPrefixCache的单组manager共用同一pool，块数只计算一次。

r02仅修正该类型检查，改从单组manager读取block_size，并与spec及coordinator的
scheduler_block_size核对；静态公式、完整长度预留、全部输入、四cell顺序、共同暖机、
采集、保护和退出语义保持。r01所有源码、执行包、资格原始记录均不修改。

`source.patch` 记录相对r01的修正；`r01-source.patch` 保留原始实现增量。
新请求测量仍需live资格通过。此修正没有连接远端或运行GPU。
