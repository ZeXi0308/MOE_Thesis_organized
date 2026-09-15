# 默认offload共同decode段CPU/CUDA诊断

问题：前序额外engine时间主要是kernel执行、CPU派发间隙、驱动等待，还是布局改变？不以更多全程host埋点替代该区分。

仅两条新引擎轨迹：native offload-off、native prompt-only offload-on16GiB。原OLMoE/固定旧d6/6656块/APC off/32×3072→1024/50ms到达全部保持。保留warmup、drain/cache reset、原raw/transfer记录。原四格500–531已验证schedule完全相同、无prefill/重算、每请求每步1token，宽度29/30；本包expected_window.json冻结具体逐请求执行签名。

Torch profiler仅在engine.step索引500进入、完成531后退出并export Chrome trace；每调用标decode_call_N。CPU/CUDA活动开启，shapes/memory/stacks关闭。profiler可同步和改变clock/后续请求时间，整个run不得作为性能收益结果。只比较实际活动和选段身份，调用span、CUDA API与kernel可能重叠，不能相加。

运行前两arms同硬件/容量、现场GPU独占，整组锁。二格任一失败立即停止，不原地重试或扩大窗口。原B fresh-cohort八格正在执行，本包仅CPU准备，之后还需按最新已登记顺序协调。

分析：新raw必须验证500–531实际schedule与冻结签名一致，记录输出前缀是否相同；若不匹配，只保留trace为未对齐诊断，不作动作因果比较。确认恰好32个用户标记、CPU与CUDA活动实际存在；缺CUDA采集不是GPU耗时为0。按时间区间并集计算GPU活动，核对flow/correlation后区分派发/等待；不得用CUPTI事件简单相加算完整request saving。

不预注册提升阈值，因为这是源定位。少量相同kernel的时间差只是诊断，无法确认总体性能；不增加GPU资源、精度变化或策略。

CPU检查：正常540调用仅记录500..531；第501调用异常时关闭/导出已采集部分、异常透传并恢复engine.step。语法/CLI检查通过，实际CUPTI/GPU路径UNRUN。

Evidence ceiling: NATIVE_PROFILER_DIAGNOSTIC，仅支持成本源定位，不是新机制或净收益。
唯一下一动作：按实际主导项设计同状态最小消融；证据不支持时不做猜测性优化。
