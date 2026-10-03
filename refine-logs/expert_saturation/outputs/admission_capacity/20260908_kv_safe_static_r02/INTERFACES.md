# 已核实的 vLLM 0.26 接口与实现边界

2026-09-08，root 对既有远端环境做只读源码核验，没有为此创建GPU引擎。
本文件记录核实结果，不复制整份后端源码，也不将CPU夹具当作live资格。

- `scheduler.kv_cache_config` 与 `kv_cache_manager.kv_cache_config` 暴露 `num_blocks`
  和 `kv_cache_groups`；manager 还暴露组数、prefix caching、EAGLE和watermark状态。
- 一个 `KVCacheGroupSpec` 的 `layer_names` 共用同一张 KV block table，在manager中
  作为一组处理。当前计算只接受一个非EAGLE的 `FullAttentionSpec`，其 `block_size`
  为每个逻辑块的token数，`sliding_window` 和 `attention_chunk_size` 均须为None。
- 禁用prefix caching时，工厂实际返回 `KVCacheCoordinatorNoPrefixCache`，而非
  `UnitaryKVCacheCoordinator`。NoPrefixCache沿用基类的block_pool与single_type_managers；
  单组时只分配/计数一次。`single_type_managers[0].block_size` 来自该spec，需与
  `coordinator.scheduler_block_size` 相等。此次仅接受dcp/pcp均为1。
- `BlockPool` 创建 `num_gpu_blocks` 个块，再从空闲队列取出block_id0作为
  `null_block` 并设 `is_null=True`。`get_num_free_blocks()` 返回空闲队列数量。
  所以新引擎空池的真实可用数为total−1；公式不把层数再乘进池块数。
- 当前不支持非零 `watermark_blocks`，也不支持prefix共享、speculative/EAGLE或
  非零lookahead。未知布局直接返回资格失败，没有推测性的回退公式。

上一轮同机天然上下文实验的实际四个后端哈希如下；本次每cell仍记录实际后端哈希。
来源：`../20260906_native_memory_pressure_r01/gpu_results/repeat0-long-cap16/environment.json`。

```text
v1/core/sched/scheduler.py       2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941
v1/core/kv_cache_manager.py      3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf
v1/core/block_pool.py            202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661
v1/worker/gpu_model_runner.py    81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0
```

相对natural实验的代码变更：新增 `safe_static.py` 的live静态公式与资格字典；`run_probe.py` 在
初始化后调用公式、保存资格并选择固定cap；启动层改为四个冻结标签。其余捕获、内存
会计、抢占保护、指标、全部输入和三次共同暖机保持原实现。

本地针对性检查：语法、输入字节一致、四cell顺序及baseline/safe路由通过。临时
namespace夹具中5632可用块、每请求256块得到22；r01中的cap<=16、prefix共享、非零watermark
和多组拒绝逻辑保持。r02只重验NoPrefixCache的块大小读取位置。该数值仅为CPU公式夹具，
不是远端cap或性能结果，夹具未进入执行包。

每cell的 `environment.json` 保留四个原运行模块及 `safe_static.py` 哈希；
`safe-cap-qualification.json` 保留实际布局、可用块、每请求预留和推导cap。
若资格失败，保存 `UNRUN / QUALIFICATION_FAILED`，退出2，执行层回传后停止。
