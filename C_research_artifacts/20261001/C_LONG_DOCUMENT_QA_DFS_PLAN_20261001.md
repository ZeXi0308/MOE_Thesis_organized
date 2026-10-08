# C：已知前缀 DFS 重排的单格参照

## 新证据与单一问题

原生完整150格已按预案关闭本配置的活跃KV容量方向：0分配失败/0抢占、peak3055/4096。后续[CPU工作核算](long_document_qa_cache_headroom_v1.json)发现，在仅共享prompt前缀、固定已有输出工作这一受限模型下，原生462946个scheduled token与理想390290之间仍有72656个token差额。聚合差值不能定位缓存逐出，也不能直接变成速度收益。这一数据支持检验**现成的简单前缀重排实际能回收多少工作，以及全部请求承担什么等待变化**；不重新打开已停止的活跃KV假说。

## 唯一动作与来源

使用[作者源码](20261001_c_peek_offline_source_v1/README.md)commit `3aca72b63c1dd6433193c0857b965acecdbfbb8a` 的 `reorder_for_prefix_sharing`，默认128深度及所有guard原样保留。它是PEEK离线第一阶段的纯DFS重排，不包含shadow-cache、eviction或在线公平lane，不称完整PEEK复现、新C方法或强于完整PEEK。

实际CPU运行已经得到完整150的置换，149个位置变化；共享coverage0.546667、平均共享深度128、完整重复率0.013333，作者guard均未退化为identity。固定置换在 `long_document_qa_peek_order_v1.json`。引擎仍为原生FCFS，仅在全部外部到达t=0后按作者结果提交；每条prompt、source ID、模型、输出cap与外部到达都不改变。在线计算只读已到达prompt IDs，不读gold、实际EOS或后续输出。重排在生成观察起点之后执行并计时；在GPU正式格内重新算出置换并核对冻结CPU结果。作者package init从不执行，不修改已安装引擎或公共环境。

## 固定执行与判断

唯一根 `c-instruct-longbench-dfs150-dev-v1`，同一模型修订、greedy seed20260905、EOS/无文本stop/max64、3500适配、128seq/batch1024/4096KV/APC/full-ISL、相同首尾暖机/reset和全部drain。复用已完成150格的cell与launcher，只添加纯重排及其记录，另用自有stage。现有非阻塞锁、360s owned child/180s正式deadline、child回收和GPU释放规则不变；忙即延后。

主问题为实际scheduled工作变化。在各自无抢占时用 `scheduled − Σ(output_tokens−1)`描述推导的prompt工作，明确输出变化会改变真实运行轨迹；若发生抢占，不能用该减法冒充独立prefill统计。全部150逐请求报告内容/IDs、EOS/cap、全文F1、TTFT、外部到达至完成、每请求最大不同host返回间隔和改善/恶化数量。不给定新的SLO阈值，不以全局工作减少掩盖受损请求。

参照是已完成 `c-instruct-longbench-full150-dev-v1`，本格更晚运行，时序/JIT/输出差异全部保留；这是一格开发策略级描述，不是交错重复或等工作量加速。若作者默认重排没有减少推导prompt工作，停止这条简单重排探针，不改depth、guard或顺序补救；若减少了工作，也仅确认已有方法在本域的作用。是否值得继续研究必须由强基线之后实际留下的动作问题决定，不能把复现收益变为新颖性。当前没有C新控制器或确认集。
