本轮停在 **SOURCE_LOCATION_CEILING**：两次 on 的约20.9 ms planner 确实包含同线程 GC，但不能把整个包络或两次运行差异归因于 GC。仅解释 host_cost_r01 本轮，不追认旧 r02 根因。

两格均为 measurement step 1 / call_id 137 / layer 9 / planner event_id 52。`gc_events` 无固有ID，下文 `[5]` 指零基数组索引：

| on格 | planner wall / thread CPU ms | GC[5]重合 ms | 重合占比 | 包络未覆盖 ms |
|---|---:|---:|---:|---:|
| 1_observer_on | 20.923255 / 20.913059 | 14.096518 | 67.372% | 6.826737 |
| 2_observer_on | 20.890244 / 20.889247 | 14.102331 | 67.507% | 6.787913 |

两次 GC 均为 generation 2，collected=9361、uncollectable=0，完整落在 planner 内。其余约6.8 ms 的来源未测；上述差值只是区间补集，不是可扣除开销或关闭GC的收益。

| on格 / step | engine wall / thread CPU ms | planner wall / GC重合 ms | kernel host wall / GC重合 ms |
|---|---:|---:|---:|
| 1 / 0 | 169.802 / 169.801 | 3.870 / 0.075 | 11.179 / 0.179 |
| 1 / 1 | 182.319 / 182.309 | 25.339 / 14.914 | 6.833 / 0.059 |
| 1 / 2 | 168.695 / 168.694 | 4.057 / 0.155 | 6.935 / 0.145 |
| 2 / 0 | 167.817 / 167.816 | 4.252 / 0.075 | 14.907 / 0.178 |
| 2 / 1 | 189.706 / 189.704 | 25.366 / 15.001 | 11.957 / 0.062 |
| 2 / 2 | 195.463 / 195.462 | 6.720 / 0.157 | 7.830 / 0.146 |

两次测量前16层slots/LRU/clock及上次warmup最终map完整相同；step 0–10的归一化请求进度、逐行top-k、完整plan/copy/final-cache和已输出token序列相同，未测hidden/KV字节。时间差先出现：第一格step0前observer snapshot为16.575374 ms，第二格0.099371 ms，均无GC重合；snapshot内部CPU/sysfs/RSS各段没有独立计时，不能命名具体等待源。

前三步第二格engine累计增加 32.170030 ms，GC重合仅增加 0.090576 ms。互斥host区间差分为planner +3.071052、kernel +9.746647、engine其余 +19.352331 ms；最后一项包含copy/route/调度及包装等，未进一步定因。

具体非GC长kernel host调用：on1 step0/call120/layer8/event18为4.704769 ms；on2 step0/call123/layer11/event24为8.199500 ms，step1/call134/layer6/event47为5.473074 ms，三者均零GC重合。layer13/14前3步planner为0.239–0.257 ms、kernel为0.411–1.251 ms；旧位置step1/layer11的外层host_plan仅0.248212/0.240415 ms，本轮长planner出现在layer9。

首次实际调度/route分叉在step11/call288/layer0：step10返回1.975139 vs2.066788 s，跨过第9请求2.000 s到达点；下一步分别70/102行。step0–2均为32/32/64行，差异发生在已记录状态仍相同时。

解释边界：engine包含planner/kernel，GC也嵌套，不能互加；thread CPU≈wall仍包含busy wait，process CPU含其他线程。两格sched_schedstats=0，原始schedstat仍有非零counter delta；口径未资格化，不能据此判定无排队或归因某种等待。kernel计时是host调用而非纯GPU时间；既有CUDA load段含D2D/H2D/map及提交间隙，不与host段相加。代码边界见`attempt01/instrumentation/run_host_cost.py:35–73`、`source/native_capture.py:272–296`、`source/runtime_variation_observer.py:30–55,83–98`。

本轮不提出新GPU实验：已定位GC重合、其他长host调用和到达反馈边界；现有结果不足以选定一个能关闭新高价值不确定性的干预。保留完整成本，不升级为稳定因果收益、噪声底或旧r02根因。精确事件/CPU/状态键见同目录`PREFIX_LOCALIZATION.json`。
