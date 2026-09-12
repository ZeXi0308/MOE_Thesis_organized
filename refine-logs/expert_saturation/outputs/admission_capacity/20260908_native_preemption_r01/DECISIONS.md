# 原生抢占重算与完整长度预留：执行前冻结

2026-09-08，HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`，分支
`agent/publish-current-moe-code`；准入实验文件未提交，实际执行源码与必要patch随包保留。
权威入口 `docs/current/README.md`、`docs/ideas/README.md`、准入实验README以及
`20260908_kv_safe_static_r02/REPORT.md` 已读。前一轮属于进展：真实GPU证明cap29
可完成，较cap16吞吐提高3.75%/4.53%，但尾部TTFT和平均TPOT恶化。

唯一问题：在相同长上下文压力负载下，默认cap32允许原生抢占/重算后的完整请求
代价，相对公式cap29预留如何？最弱链路是此前cap32保护终止遗漏了后续恢复与完成。
本轮不把该终止数据作为默认吞吐，也不实现新控制器。

## 冻结配置与比较

沿用相同32篇WikiText103 train文章、3072-token输入、固定1024-token输出、50ms到达，
BF16 pinned OLMoE、vLLM0.26、单RTX5090、engine32、context4096、token budget1024、
gpu_memory_utilization0.90、FCFS、chunked prefill、无prefix共享、同步in-process。
每个新引擎相同暖机：short32/cap16、short32/cap32、long2/cap2，全部16输出，原始暖机保留。

四个新进程，固定次序 `repeat0-native32 → repeat0-safe → repeat1-safe → repeat1-native32`。
native32调用原生 `_preempt_request`，记录前后状态，允许合法等待和重算。
safe仍按live块池与每请求完整长度推导并保留原保护：
`min(32, floor(usable_blocks / ceil((3072+1024)/block_size)))`。
两臂都执行同样布局资格检查；本轮固定比较cap29，若公式不是29则测量UNRUN并定位布局变化，
不改变已选比较。输入、其他引擎参数、采集器版本和每项max_seconds=120一致。

## 指标与会计

主指标为全部请求完成、完整episode吞吐、TTFT、每请求平均TPOT、请求完成延迟、
实际token间隔和原生等待/重算事件。完整分母为host观测终点减起点，已包含等待、
重算、采集及调度；不把各stage时间再次累加。
5s TTFT/200ms平均TPOT仅作继承的参考SLO，不以它取代连续指标或按新结果调阈值。

每项保留相同请求身份、累积输出前缀和完整token时间。抢占不等于请求失败；
重算step可无新token receipt，不能套用每步必须生成token或已有decode必须推进的
非抢占invariant。记录实际调度计算区间，重复覆盖已计算区间才算重算量；
历史生成token的重算不能混为新生成工作。请求输出质量仍未评估。

## 解释与停止规则

证据上限：当前单模型、单GPU、固定长度域的原生in-process完整请求对照。
最强已知简单对照为公式cap29；没有Oracle、专家动作或论文方法GO。
全部完成才可比较完整吞吐；超时/失败保留全部数据并停下一cell查原因，不静默重启。
若原生未发生抢占，说明本次未复现压力暴露，不能宣称测到了抢占代价。
若跨重复变号，保留不稳定结论，不追加同配置campaign或调cap来选正结果。
若两种普通策略只是转移不同请求的成本，先解释受损请求和发生时点，再判断是否还有
值得研究的动作空间；不增加专家预测器。只有普通KV/queue策略之外存在可测残差且有
实际专家动作时，才重开MoE特定机制。

## 执行与留存

只串行运行一个cell；每项退出后回传目录、stdout、stderr、执行状态及launcher日志，
比较一个归档SHA256并读取核心raw后才开始下一项。保留失败、暖机、全部重复和远端原件。
观察超时不是进程结束；重新观察同一PID，不因短暂连接失败重启实验。
不修改旧raw或权威入口、不push；凭据从仓库外临时读取，运行包不含凭据或模型权重。

执行前状态：GPU测量 `UNRUN`。
