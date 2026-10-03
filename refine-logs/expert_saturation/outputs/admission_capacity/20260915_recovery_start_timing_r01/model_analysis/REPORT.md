本次CPU结构比较确认：仅将 `min_absence_steps` 从30改0，会提前首次恢复动作，但没有形成所有完整服务指标同向改善的结构预测。

从diag-on最早完整32请求均pure-decode的**before step98**切入：F403/6656、waiting/skipped均空、tracker历史为空，此前无抢占或store。初态只取当前computed/prompt/output/held数量和声明输出上限1024，不能从step329截取而漏掉首次提前。32个请求各自保留当前状态，物理block ID未建模，仅以占位数表达实占；两动作随后独立演进。

只用 `Immediate(AbsenceRotation)` 子类改min_absence=0；cooldown20、residency30、progress guard .9、最多8次absence及most-output victim均不变，采用同一已修正pending-load guard的共享model。两条轨迹均save-on、load delay固定2，不传真实后续load完成通知。

首次动作：共同发生natural preempt299（target0003640）。门槛30首次prepare329→commit330→目标首输出333；门槛0为prepare300→commit301→首输出304。victim均0000001，准备保存的完整前缀分别3392/3360位置。首次实际schedule/free轨迹分叉在301，首次prepare动作已在300分叉。

| 模型量 | 门槛30 | 门槛0 |
|---|---:|---:|
| 最后完成step（0起算） | 1311 | 1308 |
| 包含共同前缀的engine调用数 | 1312 | 1309 |
| 32请求平均完成step | 1167.125 | 1167.46875 |
| 抢占 / 恢复 | 44 / 44 | 47 / 47 |
| forced commit | 38 | 41 |
| 恢复重执行token位置 | 3731 | 3755 |
| load数 / load前缀token位置 | 43 / 160096 | 46 / 170160 |
| 增量store前缀token位置 | 95376 | 95584 |
| 模型同时存活的saved前缀峰值（16-token块） | 5961 | 5974 |
| 零输出 / 1–2输出后再失效段 | 0 / 0 | 0 / 0 |

首个target均恢复后822输出并完成；更早首输出不自动转化为所有请求更早完成。门槛0少3次总调用，但平均完成step稍晚、3次额外加载及更多重执行；这只说明启动提前有额外代价，不能以此表预告毫秒收益。经共享主线门槛定位后，本absence对照已取消GPU执行；唯一下一动作为cooldown20/0，以CURRENT_EXPERIMENT.json为准。

验证与边界：门槛30从step98开始的1214个模型步骤，在schedule顺序/起始computed/执行tokens、schedule后free及每批新输出数上与原diag-on逐步完全吻合，最终1312calls相同。此处未来raw仅用于单列baseline验证，不输入候选。门槛0尚无真实执行；这是同一已用cohort上的结构预测，非新数据确认。模型假设store fence及host分配成功、无host eviction、固定delay2；native主机缓存可能保留已完成请求，表中saved峰值不是实测host RSS或完整cache上限。复制位置数不是实际字节计量，调用数不是墙钟，未预测route/token identity/质量或未知EOS。

文件：`analysis.json`含当前32请求初态、source hashes、首分叉、逐请求完成step及恢复段；`compare.py`可重跑：从仓库根执行 `python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_recovery_start_timing_r01/model_analysis/compare.py --output /tmp/absence-start-recheck.json`（目标文件必须尚不存在）。源码随本目录source提供；路径改造后的复算结果为recheck-relative.json，原analysis.json未覆盖。共享代码和全部raw未改，无GPU运行。


本会话交接（继承用户统一分工）：职责为保存语义、请求测量与恢复生命周期解释，主实验选择/执行由CURRENT_EXPERIMENT指定long-task root负责。已完成保存六格主表与四时间点；本次新增证据是absence0在独立结构演进中虽提前首输出却增加恢复/加载并使10请求更晚，排除“去掉额外等待必然全面改善”的简单解释。该候选不抢占GPU；新增62早段/488较广cooldown机会的主线定位由另一会话完成，直接复用。当前决策接受统一cooldown20/0，等待唯一执行方原件后仅分析分配与受损请求代价，不重跑主表或另造controller。复算已用仓库相对入口和随附精确源码验证，所有科学字段与原结果完全一致。
