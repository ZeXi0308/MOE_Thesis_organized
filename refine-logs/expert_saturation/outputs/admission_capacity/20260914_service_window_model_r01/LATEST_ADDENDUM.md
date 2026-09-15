# 模型结论接续

原REPORT保留创建时的模型/合成例与单步证书；其中“下一六格待运行”和fit_scan最强基线描述已由后续结果更新，不应再次执行。最新状态见../20260914_service_window_holdout_r01/REPORT.md：原六格与新cohort3八格均已有真实结果，native/most覆盖了已定义的1–2输出短恢复缺口；residual在两轮各42个Q(g)非零区间未超过两条基线的较优边界。无真实模型动作排序或窗口方法GO。

真实pre-resume证书采用pre_resume_window_certificate_v2.md；v1保留为被修正的诊断，不使用pool_after反推执行前可释放块。v2的212块证书只支持条件资源资格，未执行其未来窗口、不填合成毫秒。

最小模型仍是L=ceil(C_remaining/alpha)，受恢复峰值、增量KV、其它请求输出等待上界限制；未知EOS意味着输出机会，不是未来服务量保证。C_remaining未知时不进行摊销判定，保留后从当前未来可避免成本比较，不能拿已付恢复费强行延长保留。

下一唯一已选实验为../20260914_streaming_recovery_r01/PROTOCOL.md：自然异构、持续到达、允许EOS、固定单档KV，先查native/most是否仍有机制残差。01:31 UTC本组STAGED/GPU_UNRUN，前序context-victim为UNKNOWN_REMOTE，须确认终态后再按队列执行。


持续到达/EOS接续已完成：../20260914_streaming_recovery_r01/RESULTS.md记录冻结四格全部COMPLETE。自然抢占2/5/2/0，强制轮转均0；9恢复均完成并获996–1024新输出，无再次丢弃。每格6实际EOS末尾/58长度截断。原UNRUN与认证问题已被终态替代，不重跑本档或调阈值。窗口模型保留STRUCTURAL资格，未取得动作排序或完整收益；下一复用原组件已冻结funding-filter简单资格消融，不增加窗口矩阵。


原生ready状态接续见../20260914_service_window_ready_r01/REPORT.md：模型现显式区分pending load已占块与可执行前缀，8定向检查通过；首输出路径须含通知/下一调度边界。混合恢复调用时间不填C_remaining，在线接入和动作收益仍未验证；既有窗口/streaming verdict不变。


资金过滤原六格已完成：见../20260914_service_window_context_r01/funding_r01/REPORT.md。当前资金资格过滤虽消除18次未资助proposal，却每轮各出现2输出短段和0输出中断；前者首输出后native抢占，后者自然resume未纳入adapter guard、peer增长后缺1块。most/least在同组无此类段，不能把filtered新问题叫most残差。完整收益对least符号反转，对most仍是约5%吞吐代价与更小最大gap的交换。保留全部Q(g)区间；局部可资助不推出可持续服务，也未证明更长窗口有效。不新增窗口/GPU矩阵，接原生恢复底座已登记结果。


恢复前增长/cap接续见../20260914_service_window_ready_r01/DECLARED_CAPS_ADDENDUM.md。before1026有限计算预示27peer恢复4批需243>F242；FCFS prefix22可行但转移5peer×4输出机会，不假定未知EOS、不借同批完成后释放、不称GPU时延排序。模型现检查声明output/context上限，10项定向检查通过。原native两阶段保存后已有696输出完整服务，长gap主要发生在恢复启动前；C_remaining仍未知，不用3392少重算或60.6ms调用跨度代填。没有窗口方法GO，下一接A完整most保存结构检查，不重跑已完成六格/两臂。
