# 同池恢复机制四臂：执行前约定

状态 PREPARED_UNRUN；本文件不表示已上传或执行。
问题：固定 KV 下减少已有请求长暂停，同时保住完整有效服务量。
继承 headroom-fast 已测原生 vLLM0.26/OLMoE BF16/RTX5090：固定实际 KV 16089350144 bytes/7671 usable blocks，32 篇3072输入/1024输出，50ms到达，budget1024，no-prefix、同步、同三项预热。

顺序：native32 / safe29 / headroom-fast / rotate-c20，再反序。各cell新引擎，所有原始输出、失败、预热和日志保留。四臂engine最大能力32；safe29仅限制准入29。

轮转参数沿用现有代码：min_absence_steps=30，min_steps_between_swaps=20，min_residency_steps=30，protect_progress_fraction=0.90，max_absences_per_request=8；c20指交换间隔，旧0.39s算术不保证该实现的max-ITL。
新动作明确包括：强制原生preempt一次、目标排waiting前、完整恢复历史的KV余量保护。保护结束于首个新输出，期间held的running请求和其全部延迟计入。它不是只改队列顺序；不把原生122步周期/缺席总量当不变量。

在线输入仅当前请求/KV与声明输出上限。不同策略独立推进路由、队列和完成。未增加资源、改变精度或模型算法。

比较吞吐/墙时、全体完成/失败、每请求max-ITL p50/p90/max和>1s计数、TTFT/完成时间分布、受损请求数、held/forced与natural抢占、重算位置和含重算调用跨度。没有统一业务SLO；5s/.2s仅保留历史参考，不用其全通过宣称贡献。

先资格验证真实forced释放/重发块表/新输出恢复路径，再看完整差值。无实际动作记INVALID_NO_ACTION；实现或身份/会计错误记INVALID/INCOMPLETE，不能判方法失败。实际负结果用于修正成本模型，不判问题族NO-GO。相对同轮native/headroom/safe29展示全部代价；无跨轮百分比相减。

GPU初始化及每cell边界查询并留痕；占用/查询失败即退出，不杀其他任务，不自动重试或重启。当前已见PID36942占用；先完成CPU接入。

不确定性口径：本八项是探索性动作/成本测量，不是显著性或非劣性确认。按每个策略的两次执行展示运行差异，但不把一个差值或多个共享cell的pairwise差值当独立噪声样本；也不从其它all_short运行域移植噪声上界。最大ITL与p99保留原读数，不因未观察到一致损害宣称安全。若出现有利权衡，下一确认使用同目标负载、独立重复的零动作对照，并预先选定实际意义阈值及非劣性容忍量；不使用3×观测最大差值作为统计阈值。原始输出不一致时区分自由生成整体效果与固定token工作量诊断。
