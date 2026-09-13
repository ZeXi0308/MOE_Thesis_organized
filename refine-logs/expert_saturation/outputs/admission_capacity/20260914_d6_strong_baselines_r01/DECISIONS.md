# d6 既有简单策略对照（探索性，未运行）

问题：固定6656可用KV块时，既有headroom或most_output是否能降低least_progress轮转的完成时间代价，同时保住长停顿改善？这不是新预测器或新控制器。

依据：上一压力扫描d6中least_progress最大ITL约2.81秒，native约14.2–14.4秒，但平均完成慢4.49%–8.90%。本轮保持原32请求、3072/1024 token、50ms到达、vLLM0.26、固定KV 13,960,740,864字节、相同预热。沿用压力扫描输入，属于同工作负载探索，不是新holdout。

四臂：native、headroom、least_progress轮转、most_output轮转。block0按上述顺序；block1完全反向。每格新引擎，全部结果保留；不复用上一扫描的native作本轮配对。不调等待、冷却、驻留或保护阈值，不选择有利重复。

主要判断：逐block报告平均完成变化、最大ITL、完整wall/吞吐；同时报告每请求损益、全部调用、混合重算及有效输出。预指定对照为least_progress相对native、headroom相对least_progress、most_output相对least_progress；列出所有臂原值。报告多目标权衡，不由一项胜出宣布GO；不把两block差值当总体噪声界或非劣阈值。最长停顿若恶化必须同时呈现，不用平均完成改善掩盖。

实现：压力扫描run_probe/采集/metrics/headroom/输入保持原来源；两份rotation模块取已执行strong-baseline版本，仅接入其已有victim_order参数。运行配置限制为本d6点。不是FastServe复现，也不声称MoE特有。

## 执行条件：先吸收正在运行的APC强基线

另一会话的 `20260914_prefix_cache_baseline_r01` 正在对比同资源原生APC off/on/on/off。该组排在本组之前。其原始结果与结论必须先读取：若APC已经显著改变容量等待/恢复路径，先重新界定默认强基线，本组不得自动以关闭缓存的native代表默认最强配置。轮转现有适配器不支持prefix sharing，不能直接改开关声称支持APC；也不能因这一实现限制判死问题。

当前仅完成CPU运行包准备与CLI接线检查，GPU0、上传0、无controller。执行仍须先完成上述科学判断、读取GPU_COORDINATION并现场核对整组交接。任何无动作、初始化失败或未完成请求都单列，不当完整性能结果。

结论上限：单GPU、原生vLLM进程内性能测量；不涉及质量、自然EOS、外部服务部署、统计显著性或一般停顿保证。

## APC条件已核对，执行范围冻结

已读取所属会话APC analysis/analysis.json：d2两block开启缓存复用1008 token，重执行7685→6677；恢复首服务步骤仍1030/1026，最长停顿4.686/4.709秒，对应off4.663/4.687秒。因此继续本d6关闭缓存的四臂机制对照，回答高压下已有简单策略是否改变平均完成与停顿权衡。APC d6未测，不外推缓存收益、不宣称击败默认最强配置。原APC数据只读引用，不重复主分析。
