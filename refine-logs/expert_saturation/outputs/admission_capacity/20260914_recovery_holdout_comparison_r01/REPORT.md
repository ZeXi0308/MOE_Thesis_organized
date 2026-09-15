# 新文档上的恢复调度强基线比较

Status：CPU_PREPARING / GPU_UNRUN。本轮不设计新Controller。

Repository HEAD：de64dae5ea4fa3c92ec605e9847e817d8e68f3ac，共享dirty。已读current/ideas权威、RESULT_LEDGER、GPU_COORDINATION、T实际六格主分析与first-dispatch/最长gap定位、d6强基线实际engine/source。旧原件只读、不修改科学权威入口或push。

冻结事实：T修正将恢复KV保留与执行份额分开，54步改变预算且实际ready服务兑现；相对guard_all吞吐+4.351%/+1.926%，相对fit却−0.161%/−2.653%，32/32完成更慢；最长暂停97.42%在首恢复前。早期most_output在旧d6有较强暂停/服务量权衡，但没有与T同组、新文档比较。这些事实不支持延长保护或调200/10，也不否定整个请求资源问题。

One research question：在同一实际KV预算和新文档上，普通native/most_output是否已经覆盖fit与首输出+剩余预算保护的暂停—服务量权衡？

One weakest causal link：既有改进是否只胜过当前集成中的弱规则；换到未参与策略调整的文档后，强简单策略与当前修正的排序尚未测量。

One experiment：四臂native / most_output / fit_scan / guard_residual及精确反序，共八个独立新引擎。每格同OLMoE BF16/vLLM0.26.0、32×3072输入/1024输出、50ms steady、cap32/token预算1024、APC off、KV13,960,740,864 bytes/6656可用块。启动前逐组和逐格查GPU，整组共同flock，首失败停、原件保留、不自动retry。相同warmup，所有初始化与实际分配记录保留。

输入：../20260914_recovery_holdout_inputs_r01/PREPARATION.json SHA4775c84082d6afb6067b8060cda336e23fdb2a7d776fe70f72be4d09315c6c18；cohort3 workload semantic SHA34aeca96e0537736234aa8b4cba835dcbc54fec60869959b87ab391f0dbce554，rows17106:21131。128旧文档/token已回放排除，32新文档全文/前缀/行区间互斥；仍是同固定WikiText train shard，不称独立来源总体。短warmup与固定长warmup处理对全部臂一致，不在测量后改输入顺序。

动作冻结：native仅原FCFS执行与共同观察；most_output复用原absence rotation及victim排序，不重新调参：min_absence30、swap间隔20、min_residency30、progress保护0.90、最多8次absence、free_slack0；fit为原LTR200/10+fit_scan；guard_residual为相同200/10+rank_prefix、首新输出KV义务、pending1预算预留。每种动作独立重新产生KV/batch/route/输出/完成状态。所有实现税保留于完整wall。

模型：状态为当前请求进度、KV块、running/waiting、实际输出及过去动作历史；合法动作须满足当前资源与full-history恢复需要。完成/实际驱逐释放KV；降低cap不能立即释放资源。模型用于界定动作与计费，旧未来trace和拟合系数都不成为新臂的结果。资源F_next=F_now−allocations+frees；wall=scheduler inclusive+engine excluding scheduler+outside engine，嵌套decision不再加。

主指标保持完整请求吞吐和最大请求ITL，并列报告全部逐请求maxITL/TPOT/TTFT/完成分布、受损人数、重算、抢占及实现成本。SLO5s/.2s仅继承参考字段，不作为后选goodput赢家分数。每格32个请求的分位数不是生产P99。

Stop / continue：若普通most/native覆盖当前修正的暂停—服务量前沿，则停止将新增保护/预算组合作为独立方法包装，在同问题内保留已证实的资源/等待边界；若修正有重复、可解释且超过强基线的增量，再补另一个到达过程确认；若排序翻转，先解释新文档/完整执行路径差异，不更改参数择优。运行/会计无效则INVALID或UNRUN，不写NO-GO。两次反序不构成统计显著性或非劣证明。

Allowed claim ceiling：NATIVE_SERVING / MEASUREMENT_ONLY，单一新文档cohort、受控长度/steady到达。没有全局Oracle、自由EOS/质量、业务SLO、生产或MoE专属性、新颖性结论。完整LTR、所有资源反馈及多模型尚未覆盖，四臂不冒称穷尽调度家族。

Resurrection condition：后续代表性运行域或公平强基线出现当前动作未解释的完整请求residual，且有因果可执行动作；换阈值/挑seed不是复活证据。

GPU协调：仅CPU封包，未上传、无driver、不占窗口；排已登记A反序与B数值资格及当时已启动组之后。冻结包后重新读取GPU_COORDINATION与现场占用，当前快照不是持久空闲证明。

实现规模说明：本轮复用原native/rotation与T组件模块，不新增调度算法。原入口分别只支持其中一组策略，无法在同一输入与资源配置下运行四臂；新的封包器包含旧源码替换模板，新增运行逻辑仅做显式variant映射、共同入口及日志类型选择。两臂只能确认单一消融，无法同时判断原native、普通轮转与fit是否覆盖修正收益，因此这次四臂反序是所需的最小完整比较。分析器复用已检查的指标/组件会计，仅增加两类日志适配与严格共享配置检查，直接关闭强基线与输入独立性缺口。

执行接续：本合同之前的准备状态作为历史保留。原包b7557c2e…018516d、metadata28f7a20c…f883940已核验并于1789330770.798首次启动八格；controller49026/shell49027，整组持共同锁。当前RUNNING，执行完成后的结果另写RESULTS_ADDENDUM，不原地修改冻结包。

成本迁移预先固定：继续使用T未重拟合的native三项系数与step299条件起点，沿本组各自真实执行序列检查其排序/误差；U脚本只将同一已冻结分析扩展到8格。实际future schedule是该诊断输入，不能称候选动作预测；模型估计与完整请求实测分别报告，误差较大时不据此选择策略。

近邻补核：本轮most_output仅称仓库已测强简单策略，不称最强已知家族方法。[UniBoost MemGuard](https://arxiv.org/html/2606.18431v1#S3.SS3)已经以attained work参与优先级，并保护到下一几何token门槛；[Andes](https://arxiv.org/html/2404.16283v2#S4)联合考虑候选batch QoE、释放KV及swap/recompute成本，公开[solver](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/core/andes_utils/knapsack_solver.py)是简化实现；[LTR](https://arxiv.org/html/2408.15792v1#S4.SS3)提供预测长度rank、starvation量子和反向优先级SWAP资助。没有核实到完全相同的argmax observed-output单victim规则，不等于新颖性已成立，完整系统与本适配的运行域/目标仍有差异。

本轮需要保留的解释边界：所有prompt长度相同，较多已输出token可能同时意味着较大KV和较高重算成本。most_output当前先按输出量选择，再验证单victim是否能补足目标完整history；不能直接把收益命名为进度信号的独立价值。另一个旧d6前态CPU诊断仅量化这种排序/资金关系，绝不将其候选枚举当U的未来收益。
