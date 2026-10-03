# Trace 生命周期实验的口径补充（2026-09-13）

原始协议、源码、raw、分析和归档均不修改。

- 冻结协议 `reset` 字段沿用了“all runtime layer records retained across episodes”。它只适用于原 reset 本身不删除记录；本轮两臂的实际生命周期以 `action`、`order`、`baseline` 和已封存源码为准：memory 在 finalize 写出，episode 在每个排空的测量 episode 后写出并释放该轮及其预热记录。两臂都保留全部磁盘记录。
- `gc_by_window_s.cycle` / `gen2_cycle` 是已观测 GC 区间与 cycle 的交集。observer 在导出事件和 shutdown 前关闭，因此它们不是整个周期或进程的完整 GC 总量。进程 wall 仍计入这些未观测尾部的实际耗时；没有扣除 GC 或把重叠 CPU/CUDA 时间相加。
- process wall 覆盖子进程启动、初始化、全部预热/请求、trace 与结果写出、shutdown 和退出；post-init cycle 从首轮 reset 前至 shutdown 后，不含自身最终 marker 的写出，后者由 process wall 覆盖。四格结束后的统一 tar 压缩与网络回传不在这些数中，不主张端到端实验归档加速。flush wall 是同步及 Python 文件写入/close 的实测包络，不是物理介质持久化时间。
- 每模式只有两个新 engine，每 engine 复用相同三篇文章八次；这些 episode 相关。配对差值、GC 区间和请求尖峰范围都是本次观察，不是总体噪声界、显著性结论或调度方法收益。
