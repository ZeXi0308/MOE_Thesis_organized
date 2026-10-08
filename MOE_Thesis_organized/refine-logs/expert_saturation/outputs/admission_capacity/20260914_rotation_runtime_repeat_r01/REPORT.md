# 同执行路径时间差：四项重复已准备

`BLOCKED_AUTO_APPROVAL / UNRUN`，上传0、GPU0。此前八项中，两次most_output同为1,162调用、43,471重算位置，但wall相差0.971142秒；具体同路径恢复调用为0.766318/0.031701秒。没有测量窗口日志证明冷编译。

下一问题限定为：相同输入、runtime、公共预热下，该长调用是否再次出现，以及native/most完整服务量方向是否重复。四格native/most、most/native，复用cohort2，不是新文档holdout。原21个runtime/输入等文件逐字节相同，仅manifest和本轮协议改变，另保存父manifest。没有新controller或阈值调整。

执行方复用run_frozen_kv_remote.py；本轮rotation_runtime_repeat.py只准备四格编排、验证父来源并复用已有分析/资格/成本函数。实际输入及四格顺序检查通过；无GPU时4/4 UNRUN、数值配对关闭。全部四格合格后只形成两个同block主配对及两个同角色描述差。完整四格保留，不删除长调用、不减前轮差、不按结果自动追加。

GPU顺序为现有F/X→A-review压力16→本四项。当前不上传或初始化，避免干扰正在执行的性能组；以共享协调记录和现场查询为准。没有后台提交器。

冻结协议：[DECISIONS](preparation/source/DECISIONS.md)；准备来源/不变文件及包hash：[收据](preparation/status.json)；无GPU分析：[UNRUN](preparation_checks/unrun_analysis/analysis.json)。

命令：[上传](STAGE_COMMAND.sh)、[四格执行](RESUME_COMMAND.sh)、[全量分析](ANALYZE_COMMAND.sh)。一旦已有cell活动，不重复执行这些入口，先核对真实终态。证据上限NATIVE_SERVING/MEASUREMENT_ONLY；业务SLO、质量和根因尚未验证。

自动审批记录：原暂存命令因具体新四项包未获直接授权被拒。已补充21文件原样、两处协议/manifest变更及父manifest副本的差分证明，再次执行同一命令仍被拒；理由为工具记录的全局授权不被当作可信直接用户证据。两个请求均未创建execution或上传。现已向用户提出这个包到westc:53036上传并执行四项的明确授权问题；不通过其它入口绕过。见[差分证明](preparation_checks/upload_scope_proof.json)。
