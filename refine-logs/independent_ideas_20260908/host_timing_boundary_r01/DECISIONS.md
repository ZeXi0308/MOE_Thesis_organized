# 本线程独立问题：host边界能否解释请求SLO变化？

2026-09-08，thread `01a07d4b-eeac-7730-b55e-927bff4e9135`。
这是本线程认领的独立测量问题。只写本目录；其他线程的档位对齐、档位重复、
单步宽度可行性、自然KV压力和Verify Precision材料均作为已有工作，不另行实现。
工作区未提供所有Codex进程的可查询身份，因此独立范围按当前可见实验材料核对，
不声称已与所有进程通信或取得独占GPU锁。

HEAD `2a37765`，指定分支，已有未跟踪准入代码及结果。继承当前权威入口“没有已验证
系统主机制”的结论；本线程复核了其他进程的paired原32测量及30预热全量回传。
不把已有实验认领为本线程新GPU执行，不重复其32个测量或已准备的受控复测。

唯一问题：原生同步in-process driver的请求TTFT/TPOT，有多少时间处于前端提交/循环、
调度器wrapper和其后的backend/交付区间？这些观测边界是否改变steady结果的解释？
最弱环节：host测量是否把明显的frontend开销误当作调度或模型执行变化。

本次实验是对既有paired-r01的32个测量episode做互斥时间分解，0个新GPU执行。
输入集合固定为forward/reverse全部16 cells；不读repeat结果选取样本，不调整SLO。
不采用拟合pure-decode基线扣除“prefill税”，不模拟请求或策略收益。

全局同钟分桶：

```
episode wall time
= frontend_gap（0到首次schedule、前次receipt到下次schedule、末次receipt到结束）
+ scheduler_prefix（schedule start到end，结束时间戳之前的wrapper）
+ post_scheduler_stamp（schedule end到该步receipt）
```

最后一桶按“该步是否含prefill tokens”再分mixed/prefill与pure decode。
它还包含end时间戳之后的wrapper字段收集及feedback检查、模型执行、采样、
引擎/输出处理，不能叫GPU kernel时间或纯backend时间。
frontend_gap可能含到达空闲、请求提交、输出记录及反馈观测；其总和不能叫可回收开销。
字段名`frontend_gap`只为代码记号；准确边界是步间host区间，还包含engine.step进入
scheduler之前的内部路径。初始区间也含capture初始化。这给frontend份额的是宽松边界，
不能将整个桶归于frontend。receipt记录在输出批次返回之后、逐请求Python处理之前，
不是每个请求实际发送给外部客户端的完成时间。

按每请求[arrival,first_token]和[first_token,last_token]截取这些互斥区间，
分别核对TTFT与平均TPOT分母。另报告arrival→admission→add_return→first_schedule
的同host钟分解；first_schedule后的时间不全是该request的独占执行。
native first/last token只作同native钟差值，与host首末差值比较交付端点偏差；
禁止将native绝对值直接减host相对值。负控是这两个时钟域的同请求首末间隔对照。

完整保留32个episode与每请求结果。各step仅用于会计，不充当独立统计样本。
不存在raw则保留UNRUN；无法一一对应step/receipt或守恒失败则INVALID，不解释收益。
原始文件不写入。输出保留分析脚本、来源/代码hash、指标和一个报告。

判读：若frontend占请求TPOT <5%且跨重复增量不足以解释TPOT变化，只关闭“大量
frontend空隙是主要原因”的解释；不能据此排除9ms阈值附近的影响。
若frontend有明显份额，只定位边界，不能直接将扣掉后的SLO计数称为可实现收益。
观察到实际SLO计数对微小时间分量敏感时，应报告连续延迟及margin分布，不改运行阈值。
static32/shadow32仅作相同cap参考，不将跨episode差值全部归给采集税。

证据上限：REQUEST_LEVEL / OBSERVATIONAL_ACCOUNTING，单模型单卡既有raw。
未测新frontend干预、GPU kernel时间、专家信号增量、动态Oracle或可部署方法。
唯一下一步由本次分解确定：host桶大则定位最大具体边界；小则停止host优化候选，
不重新占用其他线程的档位或宽度问题。
