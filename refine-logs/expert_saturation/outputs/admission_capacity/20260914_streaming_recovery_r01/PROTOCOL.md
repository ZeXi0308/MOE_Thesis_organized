# 自然长度、持续到达与EOS下的强基线恢复诊断

Status: `PREPARED_GPU_UNRUN` at freeze; later execution/result receipts take precedence. 包SHA `1d0e5f050d620c4a84ebac6af4eb76d7fde109ce84317a74f24d4085b852cce6`。HEAD `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`，主工作区有其它会话修改；只增加本组新目录/源文件及已有分析器的零抢占兼容修复。

## 证据与唯一问题

`20260914_service_window_holdout_r01/REPORT.md` 已从cohort3八格证明：native/most没有此前定义的1–2输出短恢复，residual在Q(g)两轮各42个非零区间均未超过两条静态强基线的较优边界。most虽重算更多，仍有更优wall/max-gap；residual有更早平均完成，不能将两坐标覆盖扩大到全部约束。停止该固定3072/1024形状的residual窗口扩展，保留通用恢复问题。

本组只问：在完整自然异构文章、持续到达、实际EOS未知且明确GPU容量下，native/most是否仍出现“昂贵恢复后很少新输出又丢弃”的可见服务缺口？该缺口若不出现，不再用更长保护期或新阈值制造方法空间。

## 冻结配置与资源

64个新完整WikiText文章，排除旧160文档/原始行重叠；prompt334–3011，source-order每0.5s到达一次，共31.5s。max_output1024仅上限，min_tokens0、ignore_eosFalse。OLMoE-1B-7B-0924固定revision6d84c48581ece794365f2b8e9cfb043c68ade9c5；只用现有缓存和vLLM0.26.0/Torch2.11.0+cu130/Transformers5.15.1。

实际目标4096 usable KV blocks×2MiB=8GiB，加1个null block；engine参数8592031744 bytes，运行时核验实际池与张量存储。max_running32/max_batched_tokens1024/max_model_len4096，APC关闭，native recompute，host KV offload=0，FCFS与full initial-history reservation保持。原6656块档任意32请求声明上界6551，作为结构低压力边界，不占GPU重复验证。新4096档并不保证恢复出现；不按action频率调整文章、到达、KV档或EOS。

共享容器parent memory.max预期96636764160 bytes（90GiB），每格启动前严格读回。不修改cgroup/RLIMIT；约1Hz读取parent charge/events/swap、命令进程树PID/starttime/RSS。父峰值不归因本格，树RSS可能重复共享页且漏短命进程，明确为OBSERVED_ONLY，不能声称独立进程树硬预算。GNU timeout只控制本格owned command，监控异常仍等该命令结束后才让外层整组flock释放。

## 最小改动与执行

顺序native/most/most/native，四个新引擎独立KV、batch、queue、生成轨迹；相同三次预热。共同flock覆盖初始化与换格，逐格GPU查询失败或非空即ABORT，不终止他人进程。单格测量上限180s；全命令timeout600s，保留失败，不自动重试。

本轮新增代码只关闭旧runner无法表达的状态：开放请求集合、EOS终止、明确内存观测。复用原始cohort3 capture的exclusive first-prefill/decode/recompute会计、memory telemetry、AbsenceRotation和原生资源后端。open adapter允许终止退出与后来请求加入；30/20/30/.90/8参数、most victim规则不变，partial prefill不成为强制victim。native也经过同一adapter和capture，仅原生动作。没有新窗口、预测器、pager或在线全轨迹模拟器。

CPU准备通过32个absence/open adapter检查、3个EOS测量检查、4个host检查和3个新分析器检查；旧生命周期4测试仍通过。曾因测试PYTHONPATH漏已有fixture而失败，修正路径的记录一并保留。首次SSH旧socket失效发生在上传前，保留execution_attempt01_auth_unavailable；未产生GPU attempt。重新核验现有连接后只暂存原包，未修改包内容。

## 判读规则

时间边界统一为同步LLMEngine.step返回后host记录；不代表客户端收到。终止通知不产生新token时只记录completion，不重置last-output。TTFT/相邻输出gap按实际可定义人数报告，0/1输出不能硬填零。未知EOS下两策略可以返回不同长度和轨迹，逐请求保留；耗时差不自动等于等量工作加速，输出质量未测。

先报告实际stop/length、到达交叠、native/forced preemption、恢复服务量和丢弃/完成，保留0、1–2、3+全部段，不改变旧短段界限。恢复调用inclusive wall含共享batch、调度和host开销，不当作纯重算税；可复用内部工作与新输出分开。

本组完整插桩用于动作与生命周期资格，wall/完成/输出率只是带观测成本的描述性结果。若native/most均覆盖该缺口，停止本档窗口机制；若确有短恢复缺口，先从真实pre-resume状态检验一个资源可行且不借未来EOS的有限动作，再做同预算、计数消融和低插桩性能比较。没有动作或大部分length-capped均原样保留，属于适用域/数据结果，不是问题级NO_GO。

结果以新`RESULTS.md`追加，本文件为执行前合同，不用事后改写冻结条件。
