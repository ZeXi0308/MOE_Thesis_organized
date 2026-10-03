# 四项受控重复：停顿下降，服务量方向仍变号

Verdict：`OPEN / MEASUREMENT_ONLY`。本轮四项完整，128/128请求；首execution仅初始化失败，无请求测量，单独保留。

Evidence type：NATIVE_SERVING，原生in-process主机输出时间；固定7671可用KV块，prefix cache关闭。

| block | native wall s | most wall s | 吞吐变化 | 平均完成变化 | native最大ITL s | most最大ITL s | 完成变差请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.438142 | 23.478504 | +4.087% | +0.258% | 4.885155 | 1.064623 | 9/32 |
| 1 | 23.927953 | 24.326836 | -1.640% | +6.310% | 4.710559 | 1.067163 | 32/32 |

两most的1162步实际执行选择、决策及输出完全相同，32/32输出一致；waiting计数因到达边界不同而变化，没有改变执行选择。两对跨策略32/32输出也相同，但不构成任务质量评估。原生各2次自然抢占，most各9次forced+2次natural；held=0。

首个forced恢复的末次调用仍为step839，3780位置历史恢复、末次829调度位置；本轮调用0.032029/0.032457秒。旧0.766318秒慢调用本轮未见，不能由此宣称根因消失或已确定JIT。

两most wall仍相差0.848332秒，其中scheduler增加0.163471秒、engine excluding scheduler增加0.555316秒、outside engine增加0.129545秒。它不是相同step839长调用的重现；不删、扣除或归因于未测硬件/编译/GC。原会话另做原八项与本四项的跨组事件对齐，原件只读。

Strongest baseline：本组同资源native；此前headroom、safe29与排序结果分别保留。默认APC=True基线仍UNRUN。

Oracle/headroom：已有真实动作空间；没有本轮全动作Oracle或吞吐非劣证明。

What was not measured / claim ceiling：没有新文档holdout、APC-on、业务SLO、任务质量、跨模型、持续到达、硬件/GC/JIT因果或方法GO。参考SLO全通过只重复吞吐，不能作为长停顿SLO-goodput证据。

Failure category：长暂停的分散有效；完整服务量仍受运行时间变化与策略成本共同影响，尚不能归因或宣布净正。单条历史长调用解释不足；不是研究问题NO-GO。

Resurrection condition：无需复活，问题OPEN；若后续强基线、完整成本或资源模型提供新证据，再评估最小机制。

One next smallest experiment：同资源、同cohort2的原生APC off/on/on/off四项，检查默认缓存是否改变抢占自恢复成本及完整请求结果。不改已跑轮转参数。

复核入口：[完整分析](analysis02_after_finite/analysis.json)、[本组路径对齐](most_same_path_after_finite.json)、[启动失败](INITIALIZATION_FAILURE_ADDENDUM.md)。fresh same-family完整性复核另见runtime_integrity_review（正在执行，不预报结果）。
