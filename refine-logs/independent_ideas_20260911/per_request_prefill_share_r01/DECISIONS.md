# 总预算不变时，限制单请求 prefill 份额是否减少完整请求代价？

2026-09-11，任何新 GPU 运行前冻结。HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`；
工作区含本线程与其他线程未跟踪文件。已读目标文件、`docs/current/README.md`、
`docs/ideas/README.md`、此前 global budget512 的报告及本轮等待排序原始数据。

本轮只研究原生引擎内的 token 分配，独立于其他线程的 cap32/KV 预算和专家并集实验。
不复活等待越过次数调参，也不把原生配置差异当作新调度算法。

## 问题来自实际执行路径

前一轮全局 token budget 从1024减到512，减少极尾 token 间隔，却增加 TTFT 和完整墙钟。
新回传的两个 FCFS mixed block 又各有3个实际步骤：running只有7条、cap为8，
已有长 prefill 消耗1018 tokens，加6个decode tokens后用满1024预算，
已到达的128-token队首短请求没有被调度。每组共有16个超过512的prefill步骤，
其余步骤没有观察到上述完整组合，不能默认每次切小prefill都能把预算交给别人。
观察证据见上一实验 `analysis/prefill_sharing_opportunities.json`。

这里最弱的因果环节是：给单请求限额后，原生调度是否实际把剩余预算用于另一个请求，
以及较多分块、KV增长、batch变化和后续生成的成本是否抵消了等待收益。
这份观察不预测动作后的未来，也不把3个事件折算为可回收毫秒数。

## 唯一对照

| 策略 | 全局实际 token 预算 | 单请求 long_prefill_token_threshold | 等待顺序 |
|---|---:|---:|---|
| native1024 | 1024 | 0，原生不另限单请求 | FCFS |
| global512 | 512 | 0 | FCFS |
| per_request512 | 1024 | 512 | FCFS |

三臂的编译 token 容量均为1024、请求并发上限均为8。只在引擎完全排空后设置
`scheduler.max_num_scheduled_tokens` 和 `scheduler_config.long_prefill_token_threshold`。
冻结 vLLM scheduler 在running路径507–509行、waiting路径861–875行读取后者；
必须记录每步实际配置和实际分配，不能以字段设置成功替代动作生效。
两个预算维度不同：global512限制当步全部工作，per_request512仍允许剩余预算供其他请求使用。

模型 OLMoE-1B-7B-0924，revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；
BF16、vLLM0.26.0、单RTX5090、memutil0.70、context4096，graph启用，prefix cache关闭，
同步in-process、stream_interval1。复用前轮全部输入token IDs：all_short为16×128，
mixed为[128,2048]×8；16篇源文档、每50ms到达，固定128输出，seed20260905。

每个fresh engine先做6个共同warmup，再做6个正式episode。两个block正式顺序相反，
warmup顺序相同。各臂独立推进queue、KV、batch、生成与完成；不共享未来route。
all_short是单请求512上限的负控；对global512仍检查真实总token数，不预设它一定无动作。
global512是同引擎比较基线，旧主机的512结果只用于提出问题，不合并为本轮repeat。

## 度量和判据

主度量保留全部请求及长短组的TTFT、完成延迟、平均TPOT、每请求最大ITL、episode wall，
保留所有失败/未完成及预热。参考SLO仍为TTFT5s、平均TPOT0.2s；不根据结果换阈值。
统计实际每步prefill请求数、既有partial prefill与新请求合步、总token使用、首次入场和完成顺序。
不能用更高token利用率或更多合批请求替代完整请求收益。

- 若原生字段没有生效或动作不满足冻结语义，停止并保留失败；不解释为科学负结果。
- 若限额确实生效，却没有给其他请求形成实际服务机会，停止这个运行域的机制扩展。
- 若较少等待被更多分块或后续生成抵消，记录完整权衡，停止单请求限额扫描。
- 两个block均出现至少约3%的某项预定请求指标改善，且全体完成延迟、wall和最大ITL
  没有可重复损害，才考虑新输入上的受控确认。小于3%、变号或以一类请求受损换收益，
  保留为测量结论；不转入预测器或动态阈值搜索。

strongest simple baseline为同引擎native1024，同时保留global512用于区分两种预算动作。
Oracle未测；这是普通原生动作的请求级存在性/完整成本对照。没有专家信号增量、
任务质量、输出一致、HTTP生产服务或多卡EP结论。

## 候选取舍与停止范围

Primary是上述单请求份额对照。Conditional Backup仅在真实动作有效但成本抵消时，
把观察到的“空名额但无token预算”整理为容量边界测量，不再实现第三种限额控制器。
Wildcard是允许自然输出结束后的服务时间异质性；当前不运行，只有确有新的请求级
等待问题时才另立问题和负载，不拿它替换本轮不利结果。

同一单请求限额只在新的自然运行域出现不同的同一步预算竞争时重开；
不能仅换512阈值、seed或新名字。最近邻是chunked prefill与原生单请求chunk限额，
当前没有action-level novelty claim。

每块完成后立即归档、回传、核验，再开始下一块。保留全部原始输入/输出、代码、
配置、环境、命令、日志、实际退出状态与失败；本地完整确认前不删除远端唯一副本。
