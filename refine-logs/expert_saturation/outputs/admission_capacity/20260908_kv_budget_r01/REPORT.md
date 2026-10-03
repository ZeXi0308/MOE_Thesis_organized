# KV预算对照：资源预检拦截，性能仍未测量

2026-09-08，HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`。
**本轮0.95/0.90对照没有进入模型加载、暖机或请求测量，不能回答预算收益问题。**
执行前方案、输入、代码和分析已完成；首项失败资料已完整回传，未启动剩余三项。

## 实际发生的事

[冻结合同](DECISIONS.md)只改变gpu_memory_utilization，固定cap32，四个新引擎
按95→90→90→95执行；0.95先核对live KV是否满足32条最大长度同时驻留。
执行包845001 bytes，SHA256
`1f77a8ea3eb1c5fda29339da9f53a0aff6a1cb7bb9c6934b9dccc82a2f3fcf56`，
[上传记录](UPLOAD_20260908.json)确认远端校验一致。上传时GPU计算进程列表为空。

首项launcher28623、child28624随后启动，于run_probe.py最初的gpu_state检查发现
另一计算进程并退出1；异常发生在import torch、环境记录和engine初始化之前。
因而没有live KV块数、实际增配字节、暖机或请求结果。原状态文件为INCOMPLETE，
这里将性能证据准确标为UNRUN，不把进程退出解读为0.95容量失败。

原函数在抛出异常时只保留“GPU has another compute process”文字，没有保存当时的
进程CSV，因此无法从此失败bundle识别该进程PID。随后两次只读SSH复查均收到
`Connection closed by 36.103.198.204 port 37116`；后续观察另存
`RESOURCE_OBSERVATIONS.jsonl`。不能推断该进程仍在运行、服务器已关机或发生了什么操作。
本任务自己的launcher/child有明确EXITED终态，不存在需要重新接管的活跃实验。

失败cell目录、配置、命令、stdout/stderr、退出记录和launcher日志全部在
[gpu_results/](gpu_results/)。无损归档2348 bytes、SHA256
`824ce3e8e865e84c9f6a7a7c8223993379f049954f46381fca4f1fb99fc45f6f`，
已核对本地可读；[执行状态](execution-state-20260908.json)保存唯一失败尝试。
0次暖机、0次测量请求；其余三个cell未启动。远端原件未删除。最终
[失败矩阵](analysis/report.md)将运行时源码/软件比较标为UNRUN，引擎比较为null，
没有用空集合产生PASS或性能数值。临时凭据读取助手已删除。

## 科学判断保持原边界

上一轮“原生cap32比cap29吞吐高约17%，但两条请求有秒级暂停”仍为已有证据，
本轮没有新数据支持或削弱它。更大KV预算是否解决暂停、实际需要多少额外容量、
完整TPOT/TTFT/吞吐如何变化，全部未测。

局部查新只核对三篇原文及一个官方仓库，见 [PRIOR_ART.md](PRIOR_ART.md)。WiSP已有
专家/KV联合分配，ELDR已有expert signature驱动的decoder选择，FluxMoE已有按层流式
residency腾出KV空间。当前工作尚缺MoE专家结构超出普通KV/queue配置的增量。
若之后检查稀疏专家回收，应先测实际batched expert union、跨step复用与KV压力交集；
单token未选中专家算术不构成可回收HBM。该结构诊断仍是条件候选，没有启动。

## 唯一下一步

资源恢复且确认GPU空闲后，在新r02目录执行原冻结四项预算对照。r01失败、源码和包
全部保留；r02仅补充占用异常中的实际进程CSV，成功采集路径、预算、输入和比较不变。
这项修复服务于已出现的定位缺口，不新增策略或检查门。不能因没有预算数据而切换
到专家控制器，也不把空闲观察当作连续隔离。

| 结束字段 | 当前结论 |
|---|---|
| Verdict | UNRUN：预算性能没有测量；进程层面为资源预检失败 |
| Evidence type | 本地执行包、远端启动/退出日志、回传校验；无新GPU模型性能 |
| What was measured | 启动前资源预检曾通过，首项检查随后报告其他计算进程；失败完整留存 |
| What was not measured | 0.95 live KV容量、任何暖机/请求、预算增量和性能收益 |
| Strongest baseline | 冻结的同cap32预算90对照也未运行；旧native32只作继承事实 |
| Oracle/headroom | 未测，没有同预算或容量Oracle |
| Claim ceiling | 无预算性能判断，无配置NO-GO，无MoE方法GO |
| Failure category | 外部资源占用导致预检退出，随后SSH不可观察；不是机制失败 |
| Resurrection condition | 同端点可连接且GPU空闲，再以新目录执行相同冻结方案 |
| One next smallest experiment | 完成cap32下95→90→90→95预算对照，先读取95臂live资格 |

直接回答本轮问题：**额外KV预算能否消除暂停仍未验证；此次只证明实验在资源预检处
被拦截，没有获得可比较的请求结果。**
