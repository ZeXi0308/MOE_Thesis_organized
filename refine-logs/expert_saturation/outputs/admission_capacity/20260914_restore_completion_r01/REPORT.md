当前执行调整：近邻源码新核对显示original LTR遇首个不合格候选break，而当前共同后端continue fit-scan。先执行原组件方的fit-scan/rank-prefix强简单对照；本恢复完成包暂停CPU扩展，尚未GPU执行，不把准备方案称为结果。若prefix仍有实际残留事件再恢复本组。下文保留此前准备合同。

# 同资源后端的恢复完成义务对照

PREPARED_CPU_IN_PROGRESS / GPU_UNRUN。

Repository HEAD c4ae4f5daaea929e1a4358862296d832f1cc67ab，共享dirty工作区。已读docs/current/README、docs/ideas/README、RESULT_LEDGER、GPU_COORDINATION、LTR原生四格结果及gap_localization；权威入口不修改。

继承实测：LTR原生on中3571的311→312输出间隙为4.928/4.868秒。其518–520重算2985位置后被保留14步，再在535抢占；646–649重新恢复才返回新输出。两段未调度分别113、125步，部分重算会清零idle，未达到200提权门槛。另一个请求3640的10次提权量子已经产出7新token，不能称量子全部被重算吞掉。

唯一研究问题：在同KV预算、相同LTR200/10计数及同一资源后端下，让已经获选的恢复持续到首个新输出，是否能避免未完成恢复被丢弃，并改善全体暂停与完整服务量？

最弱因果环节：当前每次调度的完整历史预留，在下一次普通分配/排序中没有保持为持续到有效输出的义务。一次获选和部分重算不自动建立持续恢复保证。

唯一实验：LTR均开启，在同后端仅切换complete-restores off/on/on/off。两臂记录同样的义务账本；on对尚未返回新输出的已开始恢复优先排序，off仅观察。释放条件为下一次调度实际观察到输出数增加或请求完成。保留原native allocation、全历史需求、同6656 KV块/APC off、固定旧32文档3072/1024与50ms到达、1024 token预算。每臂独立推进所有状态，全部运行保留。

时间守恒仍为wall=scheduler_inclusive+engine_excluding_scheduler+outside_engine；decision在scheduler内。资源预留不是额外显存；真实held请求及新增抢占/重算需要全计。义务仅覆盖已有输出且开始重算的请求，初始prefill和普通decode不由此获得新优先级。

支持条件：on真的避免已开始恢复在首新输出前再被抢占，且强基线同资源完整请求结果能解释其收益与代价。否定当前实现的条件：原生资格/计划实际不一致、有效义务不可行，或避免重算仍被其它请求代价抵消。任何失败只落在这个干预/运行域；不判死整个问题。

证据上限：一个旧cohort下NATIVE_SERVING进程内/MEASUREMENT_ONLY；没有新颖性、全局Oracle、SLO或质量保证。已有轮转adapter与近邻工作已包含恢复/有效输出保护，这是在真实残留事件上的同底座消融，不能重新命名为独立新机制。另一worktree的funded-resume是在least轮转的noop分支前置/保护并检查恢复后下一步，未执行；本组不代跑或改写该包。

主结果固定为最大ITL与完整请求吞吐，并列TTFT、平均完成及逐请求损益、重复计算、held/调度成本。原始LTR四格指标变号，不挑更好的block当基线；此新组重跑同后端baseline。成功后再用新文档/到达过程确认和补同组最强native/least/most比较；不以旧数据宣布方法GO。

当前仅CPU实现与封包，无上传/GPU、无窗口保留。真正封包后读取最新队列并现场检查，整组持共同flock，不能借换格空隙启动。下一唯一动作是完成这个四格，无额外controller或阈值扫描。
