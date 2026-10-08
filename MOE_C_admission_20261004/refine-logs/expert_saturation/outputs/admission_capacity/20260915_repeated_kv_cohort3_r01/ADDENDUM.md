# cohort3 元数据解释补充

原始 `raw.json` 的 `policy_hooks_modified=false` 是 `pkg/request_measurement.py:128` 固定写入的helper字段。`pkg/run_probe.py` 在调用该helper前已安装 `staged_store_rotation.py` 的调度适配器。因此本实验执行的是 **vLLM原生进程内引擎 + 自定义staged调度适配器**，不是未经修改的stock调度器。

该字段不得用作“未改变调度策略”的证据。它也不表示保存执行成本被排除：同步engine.step的host返回计时包含策略、保存、加载与正常执行成本。诊断关闭与调度机制关闭是不同开关。

本补充不修改raw、冻结执行包或性能数字。未来runner应分别表达measurement helper是否加hook与caller已安装的policy，避免沿用歧义字段。测量组件的正式变更由原负责人处理，不改他人已接受的执行包。

审阅等级为WARN / same-family / provisional，无P0/P1；详见 [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)。
