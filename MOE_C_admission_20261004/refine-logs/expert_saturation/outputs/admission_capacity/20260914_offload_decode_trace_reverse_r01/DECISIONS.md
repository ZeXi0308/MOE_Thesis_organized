# 反序共同decode诊断

问题保持固定KV资源下恢复成本是否抵消服务收益。本轮仅检验旧共同decode窗口额外219ms是否随offload状态反序复现。旧on/off GPU活动均约512ms，总窗口868/649ms，顺序及主机漂移未隔离。

次序on16GiB/off0，500..531，复用原pkg及expected_window，不改变模型/资源/精度/调度。新driver两臂一致每秒只读/proc、cgroup CPU统计、MHz及亲和性，不绑定核心或设频率。该观测有成本且旧组没有，不能声称频率已控制或直接合并为性能重复。

须匹配32CPU标记、实际CUDA事件、调度签名及完整输出。若差异翻转或消失，先归为环境/顺序不稳定；若复现再定位具体主机路径。失败保留，不改窗口或自动重试。API/kernel/CPU重叠不可相加。

前序恢复预算六格已封包，遵守共同GPU顺序，不凭空闲插队。整组flock/每格现场GPU检查，忙或查询失败ABORT。证据上限NATIVE_PROFILER_DIAGNOSTIC，无方法净收益或新workload确认。

CPU smoke正常返回、观测写入、语法通过。macOS缺Linux接口记错误字段；真实Linux采样仍待运行验证。
