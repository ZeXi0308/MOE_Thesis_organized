# 保 KV 四项对照：完整性复核

**PASS；P0 = 0，P1 = 0。** Reviewer `/root/headroom_integrity`，fresh GPT-5.6-Sol / ultra，未继承对话；same-family / provisional，不是跨模型或外部复现。单轮只读复核，无新增 GPU 实验，无原始数据修改。

- **执行、资源与身份：** 四项各 32/32 完成，失败/未完成均为 0；顺序为 native/headroom/headroom/native，各使用独立进程和新引擎。EngineArgs 逐字节相同，workload、模型、实际 KV 与源码哈希一致。证据：[execution02/execution.json](execution02/execution.json)。
- **合法动作与状态：** 两个 headroom 臂各有 1,440 条决策，首次 held 在 step799，31 个请求共 9,598 request-steps 被暂缓。决策与实际调度、held 未被调度、当前/下一步状态连续性、WAITING 关闭、leader 每步调度及余量算术均无不一致。held 的精确 block-ID 所有权由运行时断言逐步检查；raw 保留块数/进度连续性，没有持久化完整 block-ID 序列。依据：[completion_headroom.py](../../../experiments/admission_capacity/completion_headroom.py)、[安装源码定位](source_probe.json)。
- **会计与指标：** 每臂首次工作 131,040；native 另有 7,685 重算位置及 2 次抢占，headroom 为 0/0。逐 token 时间直接复算墙时、吞吐、平均完成和最大 ITL，与分析一致。互斥成本分解精确闭合，decision 子区间没有重复相加。依据：[分析器](../../../experiments/admission_capacity/analyze_completion_headroom.py)、[cost_breakdown.json](analysis/cost_breakdown.json)。
- **结论范围：** 报告完整保留吞吐 −3.84%/−3.90%、平均完成 +8.28%/+8.46%、29/32 自身 max-ITL 更差及 31/32 完成更晚。各对 31/32 完整输出相同，不宣称质量等价。没有将移除已测计时区间当成反事实性能，没有 method GO 或整个问题 NO-GO。

允许结论：本次单模型、单 GPU、固定池、同 cohort 的两轮原生 in-process 对照中，保 KV 动作实际生效，削弱最严重暂停，但把等待分散给多数请求并损失吞吐。尚不支持质量、独立负载泛化、生产尾延迟或最优性；下一成本优化未实现/实测。
