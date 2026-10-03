# 同 cap32 的 KV 预算干预：20260910 新硬件执行前冻结

2026-09-10，本地 HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`，工作区有未提交文件。
本次逐字继承 20260908_kv_budget_r02 的科学源码、输入、配置与执行归档；
仅更换新 attempt 的本地驱动端点、远端目录与状态文件日期。冻结包及来源 patch 保留。
本地迁移已读旧 COMMANDS.md、ADDENDUM.md、DECISIONS.md、两份驱动、分析与打包内容。
此前 native cap32 长暂停是选择该问题的历史依据，不作为新硬件的测量或资格。
新端点 `connect.weste.seetacloud.com:11155` 的 GPU UUID、显存、软件与缓存均须重新核实。
四项均应在同一新 GPU 上完成；不要求、也不声称它与旧 GPU 的 UUID 相同。

唯一问题：在本次新 GPU 的同一 cap32 下，将gpu_memory_utilization由0.90提高至0.95，
普通KV预算配置是否已能消除长暂停并改善完整请求？最弱链路是增加的真实KV容量
能否改变请求等待和完成。没有专家回收、路由预测或新控制器。

## 固定四项

新引擎顺序：`repeat0-budget95 → repeat0-budget90 → repeat1-budget90 → repeat1-budget95`。
复用同32篇WikiText103 train文章、3072输入、固定1024输出、50ms到达；
pinned BF16 OLMoE、vLLM0.26、RTX5090、engine32、context4096、token budget1024、
FCFS、chunked prefill、无prefix cache、同步in-process。两臂均允许原生抢占恢复，
不更换victim策略。唯一引擎参数变化是gpu_memory_utilization。

每项保持三次原暖机：short32/cap16、short32/cap32、long2/cap2，全部16输出、同时到达。
全部暖机raw保留。主测量cap32、max_seconds120；请求身份、到达和host计时相同。
采集器、互斥调度区间会计及输出前缀检查直接复用上一轮，不加路由采集。

## 测量前live资格

两臂均读取真实full-attention布局、单组block size、排除null的可用块、空池free，
保留每请求完整长度块数与物理KV storage字节。
0.95臂只有在 `usable_blocks >= 32 * ceil((3072+1024)/block_size)` 才执行暖机/测量；
若本次 live block_size 为 16，该值为 8192。8192是保证最大长度同时驻留的充分条件，
不是原生完成或无抢占的必要条件；资格失败只能说明这次足量预算干预未合格。
0.90臂记录真实池，不硬编码7677。两个重复分别报告实际块数及物理增配字节；旧 GPU 的块数和物理容量不继承。

0.95首先执行，因此若初始化失败或容量未合格，保留退出/环境/日志及已得到的状态，
该测量UNRUN并停止后续cell，不向上调0.96/0.99或改负载。原生0.90若不复现抢占，
仍保留完整结果，只能说本轮未形成相同压力对照。

## 会计与解释

主指标：全部请求完成、host episode吞吐、TTFT、每请求平均TPOT、completion延迟、
每请求最大ITL、pooled token ITL、实际抢占/重复计算与等待。5s/200ms仅为继承参考SLO。
完整请求分母已含等待、重算、调度和采集，不再叠加任何局部stage；成功但无新token的
engine.step仍保留。pool摘要含schedule边界、allocation和preemption事件，并区分采样范围。
Torch reserved/allocated、参数/专家子集和KV占用不可重复相加。

这不是相同显存预算下的策略比较：0.95明确给vLLM更大的预算。若它消除暂停并带来
完整收益，结论限于普通配置足以解决当前损害；若只消除抢占而完整指标不改善，
说明零抢占仍不是性能代理。不能据此主张MoE方法GO或可回收专家内存。
证据上限是当前单模型、单GPU、固定长度自然输入的原生in-process请求测量。
没有质量、第二模型、EP或生产SLO证据，也没有同预算Oracle。

若比较变号，保留两次结果，不扫预算或加同配置campaign来挑选正结果。
未完整结束的cell不参与完整吞吐比较；失败后先完整回传并定位，不静默重新运行。
只有普通KV/queue配置后还有可重复请求损害且存在实际专家动作，才重开专家机制。

## 留存与执行状态

每项退出后完整回传raw、warmup、配置、命令、环境、stdout/stderr、退出记录，
核对一个归档SHA256并读取核心数据后才启动下一项。观察超时继续查同PID，不能当作
实验终止或重启依据。所有原始尝试和远端原件保留，不覆盖旧bundle，不改权威入口，不push。
凭据不入库。当前上传、硬件资格、GPU 测量均 UNRUN；本地迁移不生成执行证据。
