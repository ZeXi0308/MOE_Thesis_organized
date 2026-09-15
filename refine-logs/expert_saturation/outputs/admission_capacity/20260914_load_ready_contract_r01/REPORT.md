# Native异步恢复：通知边界与状态回放

Verdict: OBSERVED_EVENT_CONDITIONED_VALIDATION。纯CPU源码/既有真实事件回放；不是未来完成预测、反事实策略收益或新GPU结果。

原生Scheduler现场源码SHA2ed2a550…3941与封存版本一致。新增源码快照native_source.json SHA d643e0157b8e231240bd52028d03394c08ecd326dcd3a9f99228ef4efc7a2535；只读访问未初始化GPU，Qwen长加载未受本方操作干扰。

## 源码规定的边界

1. `offloading/worker.py:305–316` 提交load并记录job→request；`offloading_connector.py:95–98`只转发start_load，wait_for_layer_load为空。
2. `kv_connector_model_runner_mixin.py:91–109`在上下文开始提交，在finally查询get_finished并收集worker_meta。worker.py:328–365对完成load同时标completed_jobs与finished_recving；store不发finished_recving。
3. `scheduler.py:978–998`先占KV块、填写computed，但保持WAITING_FOR_REMOTE_KVS；该computed值不能证明已加载完成。
4. `scheduler.py:2622–2645`收到worker输出后填finished_recving_kv_req_ids；`:2586–2601`下一调度扫描才依据这个集合解除阻塞，恢复PREEMPTED或WAITING。完成通知不是同次调用内可用的事前信息。

模型因此加入可选observed_load_completions：通知只在对应调用末尾消费，最早下一步可调度；缺通知继续占块并等待，未知/重复完成拒绝。此前固定restore_delay_steps模式保留为未通过的校准近似。默认least/most/defer预测与原封存结果逐字段不变。

## 真实日志回放

对两次保存臂，使用worker completed_jobs的原生perf_counter时间与每步memory_trace.host_start/end时钟匹配，验证通知落在本步schedule结束后、下一步开始前；没有用请求恢复时间反推通知。

| 两次重复均相同 | 完成通知所在调用 | 首个允许恢复调用 |
|---|---:|---:|
| 首次load，job116 | 331 | 332 |
| 第二次load，job117 | 1040 | 1041 |

把这些实际发生的通知作为外生回放输入后，329–1866共1538步/重复的有序调度、空闲KV块、返回token数量均匹配，无首次差异。验证结果event_replay.json；脚本validate_load_completion_replay.py。模型源码封存model_source.py。

这不是修好了预测器：完整未来通知序列来自已执行的各自轨迹，只能验证状态机。不能把该序列复用于另一个未执行的候选策略，也不能写成action→完成时间Oracle。首次两步/第二次一步由查询时刻与异步完成共同决定，物理搬运为何恰在某次查询前完成尚未测定。

当前获得的是合法在线动作可复用的状态边界：将已占块但未ready的请求与可运行请求分开；只有当时已可见的完成通知才能推进恢复。新机制仍需实际使用这些状态并相对强简单策略产生完整请求收益。

下一唯一CPU工作：在同一保存/恢复动作空间中，比较“优先恢复原缺席请求”与当前资格实现“刚抢占的victim仍排队首”的状态转移，先量化后者是否把保存收益消耗在立即恢复自身。这直接改变恢复排序；不再尝试用固定步数预测异步完成，也不重跑单次净效应矩阵。分支若需未来通知，只能保持条件场景，不能伪装实测。
