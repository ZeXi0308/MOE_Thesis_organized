# 恢复KV保留与执行预算分开：准备合同

Status：CPU_PREPARING / GPU_UNRUN。没有净收益或新颖性结论。

Repository：本轮封包前重查HEAD为de64dae5ea4fa3c92ec605e9847e817d8e68f3ac，共享dirty；前序三组准备时为c4ae4f5daaea929e1a4358862296d832f1cc67ab，各自冻结源与原件SHA不变。已读current/ideas权威、RESULT_LEDGER、GPU_COORDINATION及前三个LTR/packing/restore实际bundle，不修改科学权威入口。

继承事实：首输出恢复四格中每个on的27义务全部兑现、0次未兑现抢占；最大ITL下降，但重算74982→96955，25→27抢占，27/32请求maxITL恶化。首个共同动作分叉406，当前-2恢复拿走全部1024 token预算。4个后续恢复仅产1新token就又被抢占，不能将完成性等同于成本摊销。

唯一研究问题：同样保留恢复KV至首输出，是否可以继续服务原本resident且pending=1的工作，降低当前实现对其它请求造成的额外停顿并保住完整服务量？最弱因果环节是KV保留与执行优先级捆绑，可能将可共批的一token工作挡在恢复之外。

最小动作：-2受保护恢复的当步token上限为当前总预算减去其后仍未作为victim的resident pending=1候选数，至少保留1个token使恢复可推进；仍受pending和实际chunk上限约束。该预留数从可见状态直接得到，无扫描阈值。候选仍需通过原资源可行性，不保证这些候选全部能执行，也不能借出额外KV/token预算。首次prefill结束只剩1位置的resident也属于该明确定义，不能将所有pending=1都称为已有输出的decode。

不改变首输出释放、原LTR200/10、历史块保留或实际pool。reserve-off重现原guard_all；reserve-on改变当步token分配，未来状态完全独立重新执行。第一不同动作只在CPU核对合法性，不沿旧future trace推导请求收益。

唯一实验：三臂fit_scan（无义务，仅原200/10）、guard_all（rank-prefix/首输出义务）、guard_residual（同guard+一token工作预留），反序六格F/G/R/R/G/F。同旧d6的32文档3072/1024、50ms、6656 KV块/APC off、token预算1024。fit是同组强简单策略，原guard是机制消融；此组仍不代表完整LTR或胜过native/least/most。

主目标不变：最大请求ITL与完整请求吞吐；同时列TTFT、平均完成/TPOT、所有逐请求损益、重算、held、调度/观测税。wall = scheduler inclusive + engine excluding scheduler + outside engine；决策和观测嵌套部分不另加。所有反序repeat和失败保留，不择优。

支持：原准备状态的动作合法，实际保护依旧兑现，受影响resident暂停减少，并能解释完整请求与fit的代价。停止当前实现：真实budget/计划/义务守恒失败，或实际可共批空间不存在，或损失抵消收益。只否定这个实现/运行域，不扩写问题NO-GO。没有实际预算干预则INVALID_NO_ACTION。

Allowed claim ceiling：NATIVE_SERVING/MEASUREMENT_ONLY on old cohort；无业务SLO/质量/生产P99/全局Oracle/原创方法GO。已有decode优先与chunked prefill是最近邻动作，这属于修正集成后的执行成本，不将已有机制重新命名为论文贡献。若出现可靠增量，才做新文档/到达过程及同组native/least/most确认；若fit已覆盖收益，就不包装本修正为新方法。

GPU协调：仅CPU准备，排当前A native-offload-cost四格及已登记B fresh_cohort八格之后；未上传/无driver/不持GPU窗口。真正冻结后再查现场进程与共同flock，不能在任何组间隙插入。

执行接续：原准备合同保留。六格已于1789329264.065首次启动，controller44070/shell44071，包与metadata保持原SHA；旧A decode两格已完整释放，现场检查和共同锁见GPU_COORDINATION。结果以完成后的独立 RESULTS_ADDENDUM 为准，目前 RUNNING，不再将本段之前的准备状态当当前状态。

本轮新增成本问题：冻结原生校准三项模型（cost_baseline.json SHA207ad5a0…f679f1），对实际六格从事先固定step299后的执行序列作成本迁移检查，不重拟合、不删异常。该诊断输入实际后续schedule，不能称为动作预测或Oracle；条件剩余时间与主表完整请求指标分别保留，系数不解释为物理kernel分项。脚本 analyze_frozen_cost.py。
