# prefix残留后的执行合同增补

2026-09-14。原REPORT的暂停条件已由实际四格关闭；本次恢复CPU实现，尚未上传/GPU。

同HEAD c4ae4f5daaea929e1a4358862296d832f1cc67ab/共享dirty，已读current/ideas权威与最新台账、GPU协调、LTR component及packing原件。继承prefix每次6次实际恢复后0新输出再抢占，fit为2；prefix最长ITL6.732/6.577s、吞吐相对fit变号。没有量子耗尽仍无输出事件。

唯一问题：固定实际6656 KV块与rank-prefix/200/10底座，保留已开始恢复至首个新输出能否减少重复恢复和长暂停，并保住完整请求服务量？最弱链路是每次调度所预留的完整历史，在下一次排序中可被更老的请求打断。资源预算与原native allocation不变；无新算法/控制器/全局Oracle主张。

状态机：在schedule前观察请求已有output计数；只有实际PREEMPTED恢复请求被本次native真正执行且已有输出历史，才建立义务。后续schedule前若可见output增加或请求完成则解除。干预臂给未解除义务priority=-2，高于原LTR -1；off只记录且允许实际抢占关闭该恢复段。输出为0的首次prefill不纳入，不改变等待200/量子10。首输出之后立即交回原调度。所有义务当前剩余完整history须已被真实free blocks覆盖，不能借虚拟预算；on不能抢占未完成义务。

实验：complete-restores off/on/on/off，同rank-prefix、同旧32文档/3072输入/1024输出/50ms steady/1024调度预算/APC off/精度。每臂独立生成/KV/到达队列/未来选择；不能复用baseline未来trace。共享状态机成本两臂均计入wall；decision嵌套scheduler不相加。先写合法状态及抢占边界测试，重用已运行native runner/allocator与指标，不重写引擎。

支持：真实首输出前中断下降且新数据完整指标与其它请求损失可解释。停止当前实现：资源资格失败、计划与实际不一致、义务守恒失败、或改善局部被其它请求代价抵消。任何失败保留并仅否定当前实现/运行域；若原件无实际保护则INVALID_NO_ACTION，不能称负效。

Claim ceiling=NATIVE_SERVING / MEASUREMENT_ONLY on old cohort。无质量保证/生产P99/业务SLO/新颖性。已有恢复保护与近邻方法必须作强基线，不能将常见first-output保护包装为原创。若正信号，唯一后续是新文档/到达过程及同组强native/least/most确认，而非继续调参。

所有四格与失败保留，不重试同remote。真正封包后新审批通道按本会话直接授权执行，再现场GPU/共同锁检查；排B host_cost完整终态之后。原paused草案/旧结果不覆盖。
