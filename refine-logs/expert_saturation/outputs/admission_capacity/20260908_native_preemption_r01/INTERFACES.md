# 安装版 vLLM 抢占与重算接口

2026-09-08直接读取指定GPU服务器已安装vLLM0.26.0源码，无模型加载或GPU执行。
解释依据为安装源码，不从其它版本推断。新cell环境文件继续记录后端源码hash。

- `v1/core/sched/scheduler.py:1212–1234`：`_preempt_request`要求RUNNING，调用
  `_free_request_blocks`、释放encoder cache、移出inflight prefill，设PREEMPTED，
  `num_computed_tokens=0`，清空speculative IDs，递增num_preemptions，放入waiting队首，
  加入reset_preempted_req_ids。此方法不改request ID或既有output token IDs。
  调用者在进入此方法前已把victim移出running，事件里的before不是完整schedule前状态。
- `scheduler.py:1236–1256`：`_update_after_schedule`按scheduled tokens递增computed。
  因此schedule返回后，实际输入计算区间起点为computed_after减scheduled amount；
  `computed_before`只反映进入schedule前状态，不能忽略schedule中的重置。
- `v1/request.py:267–276`：num_tokens是_all_token_ids长度，num_output_tokens是
  _output_token_ids长度。最后一个已生成token可能尚未执行forward，因此输出长度
  本身不能充当已计算高水位；重算量应以已执行计算区间的再次覆盖判定。
- 继承上轮实际布局：关闭prefix caching时factory使用
  `KVCacheCoordinatorNoPrefixCache`；一个FullAttentionSpec组共享块表。
  block_size取单组manager与spec并和coordinator.scheduler_block_size核对。
  pool总块数包含一个null块；空池free=total−1，已占用=usable−free。

本轮两臂完整request/receipt指标来自同一host计时器。原生抢占可使request合法地
暂停；无新输出的engine.step仍是实际计算/等待时间。原保护钩子仅用于safe臂容量失败，
不能把原生事件混成capacity_boundary或将native臂标为严格非抢占。
